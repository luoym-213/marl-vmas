"""Train a graph residual estimator from receiver-centered graph samples."""

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

from comm_limited_vmas.estimators.gnn_residual import (
    GnnResidualNetwork,
    build_base_estimator,
    receiver_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GNN residual estimator from collected graph samples."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", default="gnn_residual")
    parser.add_argument("--base-estimator-config", default=None)
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
    gnn_config = load_yaml("estimator", args.config)
    base_config = load_base_config(args, gnn_config)
    dataset = torch.load(args.dataset.expanduser(), map_location="cpu")
    tensors = dataset["tensors"]
    train_indices, val_indices = split_indices(
        n_samples=tensors["cached_states"].shape[0],
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    base_estimator = build_base_estimator(
        config=base_config,
        state_dim=4,
        device=torch.device(args.device),
        dt=float(base_config.get("dt", 0.1)),
    )
    model = GnnResidualNetwork(
        pe_dim=int(gnn_config.get("pe_dim", 8)),
        hidden_dim=int(gnn_config.get("hidden_dim", 128)),
        num_layers=int(gnn_config.get("num_layers", 2)),
        node_hidden_layers=gnn_config.get("node_hidden_layers", [128]),
        edge_hidden_layers=gnn_config.get("edge_hidden_layers", [128]),
        gate_hidden_layers=gnn_config.get("gate_hidden_layers", [128]),
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
            loss, _ = gnn_loss(model, base_estimator, batch, loss_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_loss, _ = gnn_loss(model, base_estimator, val_batch, loss_weights)
        val_loss_value = float(val_loss.detach().cpu().item())
        train_loss = sum(train_losses) / max(1, len(train_losses))
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss_value}
        )
        print(f"epoch={epoch} train_loss={train_loss:.6g} val_loss={val_loss_value:.6g}")

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, outputs = gnn_loss(model, base_estimator, val_batch, loss_weights)
        metrics = evaluate_outputs(val_batch, outputs)

    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    result = {
        "model_state_dict": model.state_dict(),
        "config": {
            "pe_dim": int(gnn_config.get("pe_dim", 8)),
            "hidden_dim": int(gnn_config.get("hidden_dim", 128)),
            "num_layers": int(gnn_config.get("num_layers", 2)),
            "node_hidden_layers": gnn_config.get("node_hidden_layers", [128]),
            "edge_hidden_layers": gnn_config.get("edge_hidden_layers", [128]),
            "gate_hidden_layers": gnn_config.get("gate_hidden_layers", [128]),
            "base_estimator": base_config,
        },
        "dataset": str(args.dataset),
        "history": history,
        "metrics": metrics,
    }
    torch.save(result, checkpoint_path)
    write_json(output_dir / "summary.json", json_safe(result, exclude_model=True))

    print(f"Wrote GNN residual checkpoint: {checkpoint_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def gnn_loss(
    model: GnnResidualNetwork,
    base_estimator: Any,
    batch: tuple[Tensor, ...],
    loss_weights: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    cached, true, aoi, comm_mask, receiver = batch
    with torch.no_grad():
        mu_self, p_self_diag = base_estimator.estimate_receiver_distribution(
            cached,
            aoi,
            comm_mask,
            receiver=None,
        )
    pred, delta_mu, gate, attention = model(
        mu_self=mu_self,
        p_self_diag=p_self_diag,
        aoi_steps=aoi,
        comm_mask=comm_mask,
        receiver=receiver,
    )
    peer_mask = ~receiver_mask(receiver, cached.shape[0], cached.shape[1], cached.device)
    target_residual = true - mu_self.detach()
    residual_pred = gate * delta_mu
    residual_error = ((residual_pred - target_residual).square() * loss_weights)
    loss = residual_error[peer_mask].mean()
    outputs = {
        "mu_self": mu_self,
        "p_self_diag": p_self_diag,
        "pred": pred,
        "delta_mu": delta_mu,
        "gate": gate,
        "attention": attention,
        "peer_mask": peer_mask,
    }
    return loss, outputs


def evaluate_outputs(batch: tuple[Tensor, ...], outputs: dict[str, Tensor]) -> dict[str, Any]:
    cached, true, aoi, comm_mask, receiver = batch
    peer_mask = outputs["peer_mask"]
    attention = outputs["attention"]
    gate = outputs["gate"]
    metrics = {
        "stale_mse": mse(cached[peer_mask], true[peer_mask]),
        "aoi_residual_mse": mse(outputs["mu_self"][peer_mask], true[peer_mask]),
        "gnn_residual_mse": mse(outputs["pred"][peer_mask], true[peer_mask]),
        "mean_gate": float(gate[peer_mask].mean().detach().cpu().item()),
        "mean_attention_max": float(attention.max(dim=-1).values.mean().detach().cpu().item()),
        "aoi_bins": binned_metrics(
            aoi=aoi,
            cached=cached,
            base=outputs["mu_self"],
            pred=outputs["pred"],
            true=true,
            peer_mask=peer_mask,
        ),
    }
    return metrics


def binned_metrics(
    aoi: Tensor,
    cached: Tensor,
    base: Tensor,
    pred: Tensor,
    true: Tensor,
    peer_mask: Tensor,
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
            mask = (aoi >= low) & peer_mask
            label = f"[{low:g},inf)"
        else:
            mask = (aoi >= low) & (aoi < high) & peer_mask
            label = f"[{low:g},{high:g})"
        if not mask.any():
            rows.append({"bin": label, "count": 0})
            continue
        rows.append(
            {
                "bin": label,
                "count": int(mask.sum().item()),
                "stale_mse": mse(cached[mask], true[mask]),
                "aoi_residual_mse": mse(base[mask], true[mask]),
                "gnn_residual_mse": mse(pred[mask], true[mask]),
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
    keys = ["cached_states", "true_states", "aoi", "comm_mask", "receiver"]
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


def load_base_config(
    args: argparse.Namespace,
    gnn_config: dict[str, Any],
) -> dict[str, Any]:
    if args.base_estimator_config:
        return load_yaml("estimator", args.base_estimator_config)
    return dict(gnn_config.get("base_estimator") or load_yaml("estimator", "aoi_residual"))


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%y_%m_%d-%H_%M_%S")
    return PROJECT_ROOT / "outputs" / "estimators" / "gnn_residual" / stamp


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
