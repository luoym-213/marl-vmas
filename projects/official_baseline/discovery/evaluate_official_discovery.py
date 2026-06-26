"""Evaluate an official BenchMARL VMAS discovery checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from torch import Tensor
from torchrl.envs.utils import ExplorationType, set_exploration_type


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate official BenchMARL MAPPO on VMAS discovery."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs/eval")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def build_experiment(episodes: int, device: str, seed: int) -> Experiment:
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.train_device = device
    experiment_config.sampling_device = device
    experiment_config.buffer_device = device
    experiment_config.evaluation_episodes = episodes
    experiment_config.render = False
    experiment_config.loggers = []
    experiment_config.create_json = False
    experiment_config.checkpoint_interval = 0
    experiment_config.checkpoint_at_end = False
    experiment_config.restore_file = None
    experiment_config.save_folder = None

    return Experiment(
        task=VmasTask.DISCOVERY.get_from_yaml(),
        algorithm_config=MappoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=seed,
        config=experiment_config,
    )


def load_checkpoint_weights(
    experiment: Experiment,
    checkpoint: Path,
    device: str,
) -> None:
    state = torch.load(checkpoint, map_location=device)
    for group in experiment.group_map:
        key = f"loss_{group}"
        if key not in state:
            raise KeyError(
                f"Checkpoint does not contain {key}. "
                f"Available keys: {list(state.keys())}"
            )
        experiment.losses[group].load_state_dict(state[key])


def run_rollout(
    experiment: Experiment,
    max_steps: int,
    deterministic: bool,
):
    exploration_type = (
        ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
    )
    with torch.no_grad(), set_exploration_type(exploration_type):
        return experiment.test_env.rollout(
            max_steps=max_steps,
            policy=experiment.policy,
            auto_cast_to_device=True,
            break_when_any_done=False,
        )


def summarize_discovery(rollout, max_steps: int) -> dict[str, float]:
    done = rollout["next", "done"].squeeze(-1).bool()
    episode_len = first_done_lengths(done, max_steps)
    valid = valid_step_mask(episode_len, done.shape[1])

    reward = rollout["next", "agents", "reward"].squeeze(-1)
    valid_reward = reward * valid.unsqueeze(-1)
    per_agent_return = valid_reward.sum(dim=1)
    episode_return = per_agent_return.mean(dim=-1)
    team_return = per_agent_return.sum(dim=-1)

    metrics = {
        "episodes": float(done.shape[0]),
        **stats("episode_len", episode_len.float()),
        **stats("episode_return", episode_return),
        "team_return_mean": tensor_mean(team_return),
        "coverage_per_step": tensor_mean(episode_return / float(max_steps)),
        "success_at_1": tensor_mean((episode_return >= 1).float()),
        "success_at_5": tensor_mean((episode_return >= 5).float()),
        "success_at_10": tensor_mean((episode_return >= 10).float()),
    }

    info_prefix = ("next", "agents", "info")
    if has_key(rollout, (*info_prefix, "targets_covered")):
        targets_covered = rollout[(*info_prefix, "targets_covered")]
        if targets_covered.ndim > valid.ndim + 1 and targets_covered.shape[-1] == 1:
            targets_covered = targets_covered.squeeze(-1)
        if targets_covered.ndim == valid.ndim + 1:
            targets_covered = targets_covered[..., 0]
        episode_targets_covered = (targets_covered * valid).max(dim=1).values
        metrics.update(stats("targets_covered", episode_targets_covered))

    if has_key(rollout, (*info_prefix, "collision_rew")):
        collision_rew = rollout[(*info_prefix, "collision_rew")]
        if collision_rew.ndim > valid.ndim + 1 and collision_rew.shape[-1] == 1:
            collision_rew = collision_rew.squeeze(-1)
        episode_collision_rew = (collision_rew * valid.unsqueeze(-1)).sum(dim=1).mean(
            dim=-1
        )
        metrics.update(stats("collision_rew", episode_collision_rew))

    return metrics


def first_done_lengths(done: Tensor, max_steps: int) -> Tensor:
    step_count = done.shape[1]
    step_indices = torch.arange(step_count, device=done.device).unsqueeze(0)
    first_done = torch.where(
        done,
        step_indices,
        torch.full_like(step_indices, step_count),
    ).min(dim=1).values
    return torch.where(
        first_done < step_count,
        first_done + 1,
        torch.full_like(first_done, max_steps),
    )


def valid_step_mask(episode_len: Tensor, step_count: int) -> Tensor:
    step_indices = torch.arange(step_count, device=episode_len.device).unsqueeze(0)
    return (step_indices < episode_len.unsqueeze(-1)).float()


def tensor_mean(tensor: Tensor) -> float:
    return float(tensor.float().mean().detach().cpu())


def tensor_std(tensor: Tensor) -> float:
    return float(tensor.float().std(unbiased=False).detach().cpu())


def stats(prefix: str, tensor: Tensor) -> dict[str, float]:
    tensor = tensor.float().detach()
    return {
        f"{prefix}_mean": tensor_mean(tensor),
        f"{prefix}_std": tensor_std(tensor),
        f"{prefix}_min": float(tensor.min().cpu()),
        f"{prefix}_max": float(tensor.max().cpu()),
    }


def has_key(tensordict, key: tuple[Any, ...]) -> bool:
    try:
        tensordict.get(key)
    except KeyError:
        return False
    return True


def write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    row = flatten_result(result)
    with (output_dir / "eval_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for section in ("metadata", "metrics"):
        for key, value in result[section].items():
            row[f"{section}_{key}"] = value
    return row


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    experiment = build_experiment(args.episodes, args.device, args.seed)
    try:
        load_checkpoint_weights(experiment, checkpoint, args.device)
        rollout = run_rollout(experiment, args.max_steps, args.deterministic)
        metrics = summarize_discovery(rollout, args.max_steps)
    finally:
        experiment.close()

    metadata = {
        "task": "official_vmas_discovery",
        "algorithm": "mappo",
        "checkpoint": str(checkpoint),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "device": args.device,
        "seed": args.seed,
        "deterministic": args.deterministic,
    }
    result = {"metadata": metadata, "metrics": metrics}
    output_dir = args.output_dir / f"{checkpoint.stem}_seed{args.seed}"
    write_outputs(output_dir, result)

    print(
        f"Eval official_vmas_discovery checkpoint={checkpoint.name} "
        f"episodes={args.episodes} deterministic={args.deterministic}"
    )
    for key in sorted(metrics):
        value = metrics[key]
        print(f"  {key}: {value:.6g}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"Wrote: {output_dir / 'eval_summary.json'}")
    print(f"Wrote: {output_dir / 'eval_summary.csv'}")


if __name__ == "__main__":
    main()
