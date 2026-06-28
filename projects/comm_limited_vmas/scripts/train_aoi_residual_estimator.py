"""Train an AoI-conditioned residual estimator from supervised samples."""

from __future__ import annotations

import argparse
import json
import sys
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
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    estimator_config = load_yaml("estimator", args.config)
    dataset = torch.load(args.dataset.expanduser(), map_location="cpu")
    tensors = dataset["tensors"]
    train_indices, val_indices = split_indices(
        n_samples=tensors["cached_state"].shape[0],
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    model = AoiResidualNetwork(
        pe_dim=int(estimator_config.get("pe_dim", 8)),
        hidden_layers=estimator_config.get("hidden_layers", [128, 128]),
        u_max=float(estimator_config.get("u_max", 1.0)),
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
            loss = residual_loss(model, batch, loss_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(residual_loss(model, val_batch, loss_weights).cpu().item())
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
        metrics = evaluate_model(model, val_batch)

    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    result = {
        "model_state_dict": model.state_dict(),
        "config": {
            "pe_dim": int(estimator_config.get("pe_dim", 8)),
            "hidden_layers": estimator_config.get("hidden_layers", [128, 128]),
            "u_max": float(estimator_config.get("u_max", 1.0)),
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
) -> Tensor:
    cached, kin, true, aoi, comm_mask, delta_t = batch
    pred, _, _ = model(cached, kin, aoi, delta_t, comm_mask)
    return (((pred - true) ** 2) * loss_weights).mean()


def evaluate_model(model: AoiResidualNetwork, batch: tuple[Tensor, ...]) -> dict[str, Any]:
    cached, kin, true, aoi, comm_mask, delta_t = batch
    pred, _, rho = model(cached, kin, aoi, delta_t, comm_mask)
    metrics = {
        "stale_mse": mse(cached, true),
        "kinematic_mse": mse(kin, true),
        "residual_mse": mse(pred, true),
        "mean_gate": float(rho.mean().detach().cpu().item()),
        "aoi_bins": binned_metrics(aoi, cached, kin, pred, true),
    }
    return metrics


def binned_metrics(
    aoi: Tensor,
    cached: Tensor,
    kin: Tensor,
    pred: Tensor,
    true: Tensor,
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
            }
        )
    return rows


def mse(pred: Tensor, target: Tensor) -> float:
    return float(((pred - target) ** 2).mean().detach().cpu().item())


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
