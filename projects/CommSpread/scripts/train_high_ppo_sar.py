"""Train SAR high-level policy with standalone PyTorch PPO."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import torch
import torch.nn.functional as F

from comm_spread.async_smdp import AsyncSMDPCollector, HighLevelTransition
from comm_spread.env_factory import make_sar_env
from comm_spread.high_level_policy import HGSARActorCriticPolicy
from comm_spread.low_level_policy import BenchMARLLowLevelPolicy

try:
    torch.backends.nnpack.enabled = False
except AttributeError:
    pass


DEFAULT_CHECKPOINT = Path(
    "outputs/sar_low_full/"
    "mappo_sar_low_mlp__f4788a9a_26_06_17-04_08_10/"
    "checkpoints/checkpoint_50040000.pt"
)

CSV_FIELDS = [
    "update",
    "transitions",
    "high_reward_mean",
    "high_reward_sum",
    "duration_mean",
    "detected_targets_mean",
    "visited_targets_mean",
    "retired_agents_mean",
    "success_rate",
    "entropy_reduction_mean",
    "actor_explore_actions",
    "actor_target_actions",
    "actor_invalid_actions",
    "actor_nonfinite_logits",
    "actor_valid_target_actions",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-level-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--low-level-steps-per-batch", type=int, default=400)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-folder", type=Path, default=Path("outputs/sar_high_ppo"))
    parser.add_argument("--save-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.save_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.save_folder / "checkpoints"
    scalars_dir = args.save_folder / "scalars"
    texts_dir = args.save_folder / "texts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scalars_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)
    config = serialize_config(args)
    (texts_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    csv_path = scalars_dir / "train.csv"
    run_summary_path = texts_dir / "run_summary.txt"
    write_run_summary(
        run_summary_path,
        args=args,
        status="running",
        final_update=0,
        last_metrics={},
        latest_checkpoint=checkpoint_dir / "latest.pt",
    )

    env = make_sar_env(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        mode="high",
        emit_info=False,
        enable_high_level_state=True,
        enable_rrt_candidates=True,
        auto_resample_goals=False,
    )
    scenario = env.scenario
    low_policy = BenchMARLLowLevelPolicy(
        args.low_level_checkpoint,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        max_steps=scenario.max_steps,
    )
    high_policy = HGSARActorCriticPolicy(
        n_agents=scenario.n_agents,
        rrt_top_k=scenario.rrt_top_k,
        device=args.device,
        deterministic=False,
    )
    optimizer = torch.optim.Adam(high_policy.parameters(), lr=args.lr)
    start_update = 0
    if args.resume_checkpoint is not None:
        start_update = load_checkpoint(args.resume_checkpoint, high_policy, optimizer, args.device)

    collector = AsyncSMDPCollector(
        env,
        high_level_policy=high_policy,
        low_level_policy=low_policy,
    )

    try:
        for update in range(start_update + 1, args.updates + 1):
            high_policy.reset_stats()
            collector.reset()
            initial_entropy = entropy_total(scenario).detach().clone()
            transitions = collector.rollout(args.low_level_steps_per_batch)
            final_entropy = entropy_total(scenario).detach().clone()
            if not transitions:
                print(f"[update {update}] transitions=0, skipping")
                continue

            batch = transitions_to_batch(transitions, device=args.device)
            add_gae(batch, gamma=args.gamma, gae_lambda=args.gae_lambda)
            update_metrics = ppo_update(
                policy=high_policy,
                optimizer=optimizer,
                batch=batch,
                ppo_epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                clip_eps=args.clip_eps,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                max_grad_norm=args.max_grad_norm,
            )
            rollout_metrics = summarize_rollout(
                transitions=transitions,
                scenario=scenario,
                initial_entropy=initial_entropy,
                final_entropy=final_entropy,
                low_level_source=low_policy.source,
                actor_metrics=high_policy.metrics(),
            )
            print_update(update, rollout_metrics | update_metrics)
            all_metrics = {"update": update} | rollout_metrics | update_metrics
            append_train_csv(csv_path, all_metrics)

            if update % args.save_interval == 0 or update == args.updates:
                save_checkpoint(checkpoint_dir / f"checkpoint_{update}.pt", high_policy, optimizer, update, args)
            save_checkpoint(checkpoint_dir / "latest.pt", high_policy, optimizer, update, args)
            write_run_summary(
                run_summary_path,
                args=args,
                status="running" if update < args.updates else "complete",
                final_update=update,
                last_metrics=all_metrics,
                latest_checkpoint=checkpoint_dir / "latest.pt",
            )
    finally:
        low_policy.close()


def transitions_to_batch(
    transitions: list[HighLevelTransition],
    *,
    device: str,
) -> dict[str, torch.Tensor]:
    def stack(name: str) -> torch.Tensor:
        return torch.stack([getattr(transition, name) for transition in transitions]).to(device)

    return {
        "ego_node": stack("ego_node").float(),
        "teammate_nodes": stack("teammate_nodes").float(),
        "teammate_mask": stack("teammate_mask").float(),
        "explore_nodes": stack("explore_nodes").float(),
        "target_nodes": stack("target_nodes").float(),
        "target_mask": stack("target_mask").bool(),
        "map_channels": stack("map_channels").float(),
        "global_map_channels": stack("global_map_channels").float(),
        "global_agent_nodes": stack("global_agent_nodes").float(),
        "explore_edges": stack("explore_edges").float(),
        "target_edges": stack("target_edges").float(),
        "action_mask": stack("action_mask").bool(),
        "action": stack("action").long().view(-1),
        "old_log_prob": stack("log_prob").float().view(-1, 1),
        "old_value": stack("value").float().view(-1, 1),
        "reward": stack("reward").float().view(-1, 1),
        "done": stack("done").float().view(-1, 1),
        "duration": torch.tensor(
            [transition.duration for transition in transitions],
            dtype=torch.float32,
            device=device,
        ).view(-1, 1),
        "env_id": torch.tensor(
            [transition.env_id for transition in transitions],
            dtype=torch.long,
            device=device,
        ),
        "agent_id": torch.tensor(
            [transition.agent_id for transition in transitions],
            dtype=torch.long,
            device=device,
        ),
        "decision_start_t": torch.tensor(
            [transition.decision_start_t for transition in transitions],
            dtype=torch.long,
            device=device,
        ),
    }


def add_gae(batch: dict[str, torch.Tensor], *, gamma: float, gae_lambda: float) -> None:
    n = batch["reward"].shape[0]
    advantages = torch.zeros(n, 1, device=batch["reward"].device)
    returns = torch.zeros_like(advantages)
    env_ids = batch["env_id"]
    agent_ids = batch["agent_id"]
    starts = batch["decision_start_t"]

    for env_id in env_ids.unique():
        for agent_id in agent_ids[env_ids == env_id].unique():
            mask = (env_ids == env_id) & (agent_ids == agent_id)
            indices = torch.nonzero(mask, as_tuple=False).view(-1)
            if indices.numel() == 0:
                continue
            indices = indices[starts[indices].argsort()]
            next_value = torch.zeros(1, 1, device=batch["reward"].device)
            next_advantage = torch.zeros(1, 1, device=batch["reward"].device)
            for idx in reversed(indices.tolist()):
                done_mask = 1.0 - batch["done"][idx : idx + 1]
                discount = gamma ** batch["duration"][idx : idx + 1]
                delta = (
                    batch["reward"][idx : idx + 1]
                    + discount * next_value * done_mask
                    - batch["old_value"][idx : idx + 1]
                )
                advantage = delta + discount * gae_lambda * next_advantage * done_mask
                advantages[idx : idx + 1] = advantage
                returns[idx : idx + 1] = advantage + batch["old_value"][idx : idx + 1]
                next_value = batch["old_value"][idx : idx + 1]
                next_advantage = advantage

    batch["advantage"] = advantages
    batch["return"] = returns


def ppo_update(
    *,
    policy: HGSARActorCriticPolicy,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    ppo_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
) -> dict[str, float]:
    policy.train()
    n = batch["action"].shape[0]
    advantages = batch["advantage"]
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
    metrics: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
    }
    for _ in range(ppo_epochs):
        permutation = torch.randperm(n, device=batch["action"].device)
        for start in range(0, n, minibatch_size):
            mb_indices = permutation[start : start + minibatch_size]
            minibatch = index_batch(batch, mb_indices)
            mb_advantages = advantages[mb_indices]
            eval_out = policy.evaluate_actions(minibatch)
            ratio = torch.exp(eval_out["log_prob"] - minibatch["old_log_prob"])
            unclipped = ratio * mb_advantages
            clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(eval_out["value"], minibatch["return"])
            entropy = eval_out["entropy"].mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = eval_out["log_prob"] - minibatch["old_log_prob"]
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()
            metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
            metrics["value_loss"].append(float(value_loss.detach().cpu()))
            metrics["entropy"].append(float(entropy.detach().cpu()))
            metrics["approx_kl"].append(float(approx_kl.detach().cpu()))
            metrics["clip_fraction"].append(float(clip_fraction.detach().cpu()))

    return {key: float(torch.tensor(values).mean()) for key, values in metrics.items()}


def index_batch(batch: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    indexed = {}
    for key, value in batch.items():
        if key in {"env_id", "agent_id", "decision_start_t", "duration"}:
            indexed[key] = value[indices]
        elif isinstance(value, torch.Tensor) and value.shape[0] == indices.shape[0]:
            indexed[key] = value
        elif isinstance(value, torch.Tensor) and value.shape[0] == batch["action"].shape[0]:
            indexed[key] = value[indices]
    return indexed


def summarize_rollout(
    *,
    transitions: list[HighLevelTransition],
    scenario,
    initial_entropy: torch.Tensor,
    final_entropy: torch.Tensor,
    low_level_source: str,
    actor_metrics: dict[str, int],
) -> dict[str, float | int | str]:
    durations = torch.tensor([transition.duration for transition in transitions], dtype=torch.float32)
    rewards = torch.stack([transition.reward.float() for transition in transitions])
    detected = scenario.target_detected.any(dim=1).float().sum(dim=-1)
    visited = scenario.target_visited.float().sum(dim=-1)
    retired = (~scenario.active_agents).float().sum(dim=-1)
    entropy_reduction = initial_entropy - final_entropy
    metrics: dict[str, float | int | str] = {
        "low_level_action_source": low_level_source,
        "transitions": len(transitions),
        "duration_mean": float(durations.mean()),
        "high_reward_sum": float(rewards.sum()),
        "high_reward_mean": float(rewards.mean()),
        "detected_targets_mean": float(detected.mean().cpu()),
        "visited_targets_mean": float(visited.mean().cpu()),
        "retired_agents_mean": float(retired.mean().cpu()),
        "success_rate": float(scenario.success.float().mean().cpu()),
        "entropy_reduction_mean": float(entropy_reduction.mean().cpu()),
    }
    metrics.update(actor_metrics)
    return metrics


def print_update(update: int, metrics: dict[str, float | int | str]) -> None:
    print(
        "\n".join(
            [
                f"[high_ppo] update {update:04d}",
                (
                    "  rollout | "
                    f"transitions {fmt_int(metrics, 'transitions')}  "
                    f"duration {fmt_float(metrics, 'duration_mean')}  "
                    f"reward {fmt_float(metrics, 'high_reward_mean')} "
                    f"(sum {fmt_float(metrics, 'high_reward_sum')})  "
                    f"success {fmt_float(metrics, 'success_rate')}"
                ),
                (
                    "  targets | "
                    f"detected {fmt_float(metrics, 'detected_targets_mean')}  "
                    f"visited {fmt_float(metrics, 'visited_targets_mean')}  "
                    f"retired {fmt_float(metrics, 'retired_agents_mean')}  "
                    f"entropy_gain {fmt_float(metrics, 'entropy_reduction_mean')}"
                ),
                (
                    "  actor   | "
                    f"explore {fmt_int(metrics, 'actor_explore_actions')}  "
                    f"target {fmt_int(metrics, 'actor_target_actions')} "
                    f"(valid {fmt_int(metrics, 'actor_valid_target_actions')})  "
                    f"invalid {fmt_int(metrics, 'actor_invalid_actions')}  "
                    f"nonfinite {fmt_int(metrics, 'actor_nonfinite_logits')}"
                ),
                (
                    "  ppo     | "
                    f"policy {fmt_float(metrics, 'policy_loss')}  "
                    f"value {fmt_float(metrics, 'value_loss')}  "
                    f"entropy {fmt_float(metrics, 'entropy')}  "
                    f"kl {fmt_float(metrics, 'approx_kl')}  "
                    f"clip {fmt_float(metrics, 'clip_fraction')}"
                ),
            ]
        ),
        flush=True,
    )


def fmt_float(metrics: dict[str, float | int | str], key: str) -> str:
    value = metrics.get(key, 0.0)
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return f"{float(value):.4f}"
    return str(value)


def fmt_int(metrics: dict[str, float | int | str], key: str) -> str:
    value = metrics.get(key, 0)
    if isinstance(value, float):
        return str(int(round(value)))
    if isinstance(value, int):
        return str(value)
    return str(value)


def entropy_total(scenario) -> torch.Tensor:
    return scenario._compute_entropy(scenario.belief_maps).sum(dim=(-1, -2))


def save_checkpoint(
    path: Path,
    policy: HGSARActorCriticPolicy,
    optimizer: torch.optim.Optimizer,
    update: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update": update,
            "config": serialize_config(args),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    policy: HGSARActorCriticPolicy,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> int:
    checkpoint: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("update", 0))


def serialize_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    return config


def append_train_csv(path: Path, metrics: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: metrics.get(field, "") for field in CSV_FIELDS})


def write_run_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    status: str,
    final_update: int,
    last_metrics: dict[str, float | int | str],
    latest_checkpoint: Path,
) -> None:
    lines = [
        "SAR High-Level PPO Run Summary",
        f"status: {status}",
        f"updated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"command: {' '.join(sys.argv)}",
        f"low_level_checkpoint: {args.low_level_checkpoint}",
        f"device: {args.device}",
        f"num_envs: {args.num_envs}",
        f"low_level_steps_per_batch: {args.low_level_steps_per_batch}",
        f"updates: {args.updates}",
        f"ppo_epochs: {args.ppo_epochs}",
        f"minibatch_size: {args.minibatch_size}",
        f"gamma: {args.gamma}",
        f"gae_lambda: {args.gae_lambda}",
        f"clip_eps: {args.clip_eps}",
        f"entropy_coef: {args.entropy_coef}",
        f"value_coef: {args.value_coef}",
        f"lr: {args.lr}",
        f"max_grad_norm: {args.max_grad_norm}",
        f"final_update: {final_update}",
        f"latest_checkpoint: {latest_checkpoint}",
        "",
        "last_metrics:",
    ]
    if last_metrics:
        for key in CSV_FIELDS:
            if key in last_metrics:
                lines.append(f"  {key}: {last_metrics[key]}")
    else:
        lines.append("  none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
