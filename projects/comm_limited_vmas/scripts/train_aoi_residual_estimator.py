"""Train an AoI-conditioned residual estimator from supervised samples."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

CONFIG_ROOT = PROJECT_ROOT / "configs"

from comm_limited_vmas.estimators.aoi_residual import AoiResidualNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an AoI residual estimator from collected samples."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", default="aoi_residual")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=1.0)
    parser.add_argument("--loss", choices=["mse", "nll", "nll_mse"], default=None)
    parser.add_argument(
        "--covariance-mode",
        choices=["fixed_kinematic", "learned_diag"],
        default=None,
    )
    parser.add_argument("--min-variance", type=float, default=None)
    parser.add_argument("--max-variance", type=float, default=None)
    parser.add_argument("--nll-weight", type=float, default=None)
    parser.add_argument("--mse-weight", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    estimator_config = load_yaml("estimator", args.config)
    dataset = torch.load(args.dataset.expanduser(), map_location="cpu")
    tensors = dataset["tensors"]
    options = training_options(args, estimator_config, dataset)
    train_indices, val_indices = split_indices(
        n_samples=tensors["cached_state"].shape[0],
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    model = AoiResidualNetwork(
        pe_dim=int(estimator_config.get("pe_dim", 8)),
        hidden_layers=estimator_config.get("hidden_layers", [128, 128]),
        u_max=float(estimator_config.get("u_max", 1.0)),
        covariance_mode=options.covariance_mode,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_loader = make_loader(tensors, train_indices, args.batch_size, shuffle=True)
    val_batch = batch_from_indices(tensors, val_indices, args.device)
    loss_weights = torch.tensor(
        [
            args.position_weight,
            args.position_weight,
            args.velocity_weight,
            args.velocity_weight,
        ],
        dtype=torch.float32,
        device=args.device,
    )

    best_val_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = tuple(item.to(args.device) for item in batch)
            loss = residual_loss(model, batch, loss_weights, options)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(
                residual_loss(model, val_batch, loss_weights, options).cpu().item()
            )
        train_loss = sum(train_losses) / max(1, len(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6g} val_loss={val_loss:.6g}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        metrics = evaluate_model(model, val_batch, options)

    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{options.loss}_{options.covariance_mode}_model.pt"
    result = {
        "model_state_dict": model.state_dict(),
        "config": {
            "pe_dim": int(estimator_config.get("pe_dim", 8)),
            "hidden_layers": estimator_config.get("hidden_layers", [128, 128]),
            "u_max": float(estimator_config.get("u_max", 1.0)),
            "covariance_mode": options.covariance_mode,
            "min_variance": options.min_variance,
            "max_variance": options.max_variance,
            "loss": options.loss,
            "nll_weight": options.nll_weight,
            "mse_weight": options.mse_weight,
            "sigma_p": options.sigma_p,
            "sigma_v": options.sigma_v,
            "process_noise_q": options.process_noise_q,
            "dt": options.dt,
        },
        "dataset": str(args.dataset),
        "history": history,
        "metrics": metrics,
    }
    torch.save(result, checkpoint_path)
    write_json(output_dir / "summary.json", json_safe(result, exclude_model=True))

    print(f"Wrote AoI residual checkpoint: {checkpoint_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def residual_loss(
    model: AoiResidualNetwork,
    batch: tuple[Tensor, ...],
    loss_weights: Tensor,
    options: "TrainingOptions",
) -> Tensor:
    cached, kin, true, aoi, comm_mask, delta_t = batch
    kin_cov_diag = kinematic_covariance_diag(aoi, options, device=kin.device, dtype=kin.dtype)
    pred, _, _, cov_diag = model(
        cached,
        kin,
        aoi,
        delta_t,
        comm_mask,
        kin_cov_diag=kin_cov_diag,
        min_variance=options.min_variance,
        max_variance=options.max_variance,
    )
    mse_loss = weighted_mse_loss(pred, true, loss_weights)
    if options.loss == "mse":
        return mse_loss

    assert cov_diag is not None
    nll_loss = gaussian_nll_loss(pred, true, cov_diag)
    if options.loss == "nll":
        return options.nll_weight * nll_loss
    return options.nll_weight * nll_loss + options.mse_weight * mse_loss


def evaluate_model(
    model: AoiResidualNetwork,
    batch: tuple[Tensor, ...],
    options: "TrainingOptions",
) -> dict[str, Any]:
    cached, kin, true, aoi, comm_mask, delta_t = batch
    kin_cov_diag = kinematic_covariance_diag(aoi, options, device=kin.device, dtype=kin.dtype)
    pred, _, rho, cov_diag = model(
        cached,
        kin,
        aoi,
        delta_t,
        comm_mask,
        kin_cov_diag=kin_cov_diag,
        min_variance=options.min_variance,
        max_variance=options.max_variance,
    )
    assert cov_diag is not None
    metrics = {
        "stale_mse": mse(cached, true),
        "kinematic_mse": mse(kin, true),
        "residual_mse": mse(pred, true),
        "kinematic_nll": nll(kin, true, kin_cov_diag),
        "residual_nll": nll(pred, true, cov_diag),
        "mean_kinematic_variance": float(kin_cov_diag.mean().detach().cpu().item()),
        "mean_pred_variance": float(cov_diag.mean().detach().cpu().item()),
        "mean_gate": float(rho.mean().detach().cpu().item()),
        "aoi_bins": binned_metrics(aoi, cached, kin, pred, true, kin_cov_diag, cov_diag),
    }
    return metrics


def binned_metrics(
    aoi: Tensor,
    cached: Tensor,
    kin: Tensor,
    pred: Tensor,
    true: Tensor,
    kin_cov_diag: Tensor,
    pred_cov_diag: Tensor,
) -> list[dict[str, float | str | int]]:
    bins = [
        (0.0, 1.0),
        (1.0, 3.0),
        (3.0, 8.0),
        (8.0, 15.0),
        (15.0, 30.0),
        (30.0, float("inf")),
    ]
    rows = []
    for low, high in bins:
        if high == float("inf"):
            mask = aoi >= low
            label = f"[{low:g},inf)"
        else:
            mask = (aoi >= low) & (aoi < high)
            label = f"[{low:g},{high:g})"
        if not mask.any():
            rows.append({"bin": label, "count": 0})
            continue
        rows.append(
            {
                "bin": label,
                "count": int(mask.sum().item()),
                "stale_mse": mse(cached[mask], true[mask]),
                "kinematic_mse": mse(kin[mask], true[mask]),
                "residual_mse": mse(pred[mask], true[mask]),
                "kinematic_nll": nll(kin[mask], true[mask], kin_cov_diag[mask]),
                "residual_nll": nll(pred[mask], true[mask], pred_cov_diag[mask]),
                "mean_kinematic_variance": float(
                    kin_cov_diag[mask].mean().detach().cpu().item()
                ),
                "mean_pred_variance": float(
                    pred_cov_diag[mask].mean().detach().cpu().item()
                ),
            }
        )
    return rows


def weighted_mse_loss(pred: Tensor, target: Tensor, loss_weights: Tensor) -> Tensor:
    return (((pred - target) ** 2) * loss_weights).mean()


def gaussian_nll_loss(pred: Tensor, target: Tensor, covariance_diag: Tensor) -> Tensor:
    return 0.5 * (
        torch.log(covariance_diag) + (target - pred).square() / covariance_diag
    ).sum(dim=-1).mean()


def mse(pred: Tensor, target: Tensor) -> float:
    return float(((pred - target) ** 2).mean().detach().cpu().item())


def nll(pred: Tensor, target: Tensor, covariance_diag: Tensor) -> float:
    return float(gaussian_nll_loss(pred, target, covariance_diag).detach().cpu().item())


def kinematic_covariance_diag(
    aoi: Tensor,
    options: "TrainingOptions",
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    delta_t = aoi.to(device=device, dtype=dtype) * float(options.dt)
    dt2 = delta_t.square()
    dt3 = dt2 * delta_t
    pos_var = (
        options.sigma_p**2
        + dt2 * options.sigma_v**2
        + options.process_noise_q * dt3 / 3.0
    )
    vel_var = options.sigma_v**2 + options.process_noise_q * delta_t
    return torch.stack([pos_var, pos_var, vel_var, vel_var], dim=-1).clamp(
        min=options.min_variance,
        max=options.max_variance,
    )


@dataclass(frozen=True)
class TrainingOptions:
    loss: str
    covariance_mode: str
    min_variance: float
    max_variance: float
    nll_weight: float
    mse_weight: float
    sigma_p: float
    sigma_v: float
    process_noise_q: float
    dt: float


def training_options(
    args: argparse.Namespace,
    estimator_config: dict[str, Any],
    dataset: dict[str, Any],
) -> TrainingOptions:
    loss = str(args.loss or estimator_config.get("loss", "mse"))
    if loss not in {"mse", "nll", "nll_mse"}:
        raise ValueError(f"Unknown loss={loss!r}. Expected mse, nll, or nll_mse")
    covariance_mode = str(
        args.covariance_mode
        or estimator_config.get("covariance_mode", "fixed_kinematic")
    )
    if covariance_mode not in {"fixed_kinematic", "learned_diag"}:
        raise ValueError(
            f"Unknown covariance_mode={covariance_mode!r}. "
            "Expected fixed_kinematic or learned_diag"
        )
    min_variance = float(
        args.min_variance
        if args.min_variance is not None
        else estimator_config.get("min_variance", 1.0e-4)
    )
    max_variance = float(
        args.max_variance
        if args.max_variance is not None
        else estimator_config.get("max_variance", 10.0)
    )
    if min_variance <= 0:
        raise ValueError("--min-variance must be positive for Gaussian NLL")
    if max_variance <= min_variance:
        raise ValueError("--max-variance must be greater than --min-variance")

    return TrainingOptions(
        loss=loss,
        covariance_mode=covariance_mode,
        min_variance=min_variance,
        max_variance=max_variance,
        nll_weight=float(
            args.nll_weight
            if args.nll_weight is not None
            else estimator_config.get("nll_weight", 1.0)
        ),
        mse_weight=float(
            args.mse_weight
            if args.mse_weight is not None
            else estimator_config.get("mse_weight", 0.1)
        ),
        sigma_p=float(estimator_config.get("sigma_p", 0.01)),
        sigma_v=float(estimator_config.get("sigma_v", 0.01)),
        process_noise_q=float(estimator_config.get("process_noise_q", 1.0)),
        dt=float(estimator_config.get("dt", 0.1)),
    )


def make_loader(
    tensors: dict[str, Tensor],
    indices: Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    batch = batch_from_indices(tensors, indices, device="cpu")
    dataset = TensorDataset(*batch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def batch_from_indices(
    tensors: dict[str, Tensor],
    indices: Tensor,
    device: str | torch.device,
) -> tuple[Tensor, ...]:
    keys = ["cached_state", "kinematic_state", "true_state", "aoi", "comm_mask", "delta_t"]
    return tuple(tensors[key][indices].to(device) for key in keys)


def split_indices(
    n_samples: int,
    val_fraction: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    if n_samples < 2:
        raise ValueError("Need at least two samples to split train/val")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_samples, generator=generator)
    n_val = max(1, min(n_samples - 1, int(round(n_samples * val_fraction))))
    return indices[n_val:], indices[:n_val]


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%y_%m_%d-%H_%M_%S")
    return PROJECT_ROOT / "outputs" / "estimators" / "aoi_residual" / stamp


def json_safe(result: dict[str, Any], exclude_model: bool) -> dict[str, Any]:
    safe = {}
    for key, value in result.items():
        if exclude_model and key == "model_state_dict":
            continue
        safe[key] = value
    return safe


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_yaml(group: str, name: str) -> dict[str, Any]:
    path = CONFIG_ROOT / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


if __name__ == "__main__":
    main()
