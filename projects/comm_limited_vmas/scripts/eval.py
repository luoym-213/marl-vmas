"""Evaluate communication-limited VMAS checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from torch import Tensor
from torchrl.envs.utils import ExplorationType, set_exploration_type


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

CONFIG_ROOT = PROJECT_ROOT / "configs"

from comm_limited_vmas.benchmarl_task import build_comm_vmas_task


ALGORITHM_REGISTRY = {
    "mappo": MappoConfig,
    "ippo": IppoConfig,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate comm-limited VMAS checkpoints."
    )
    parser.add_argument("--task", required=True, help="Task config name.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--algorithm",
        default="mappo",
        choices=sorted(ALGORITHM_REGISTRY),
    )
    parser.add_argument("--comm", default="comm_full")
    parser.add_argument("--estimator", default="stale")
    parser.add_argument(
        "--eval-config",
        default=None,
        help="Eval config under configs/eval. Defaults to --task.",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use deterministic actions. Defaults to eval config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_name = args.eval_config or args.task
    eval_config = load_yaml("eval", eval_name)

    checkpoint = args.checkpoint.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    episodes = int(args.episodes or eval_config.get("episodes", 100))
    device = str(args.device or eval_config.get("device", "cpu"))
    deterministic = bool(
        eval_config.get("deterministic", True)
        if args.deterministic is None
        else args.deterministic
    )
    max_steps = int(
        eval_config.get("max_steps") or load_yaml("task", args.task)["max_steps"]
    )
    output_root = args.output_dir or PROJECT_ROOT / eval_config.get(
        "output_dir",
        "outputs/eval",
    )
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    experiment = build_experiment(
        task_name=args.task,
        algorithm_name=args.algorithm,
        comm_name=args.comm,
        estimator_name=args.estimator,
        episodes=episodes,
        device=device,
        seed=args.seed,
    )

    try:
        load_checkpoint_weights(experiment, checkpoint, device)
        rollout = run_rollout(
            experiment=experiment,
            max_steps=max_steps,
            deterministic=deterministic,
        )
        summary = summarize_rollout(args.task, rollout, max_steps)
        summary = filter_metrics(
            task_name=args.task,
            summary=summary,
            metric_groups=eval_config.get("metrics"),
        )
    finally:
        experiment.close()

    metadata = {
        "task": args.task,
        "algorithm": args.algorithm,
        "comm": args.comm,
        "estimator": args.estimator,
        "eval_config": eval_name,
        "checkpoint": str(checkpoint),
        "episodes": episodes,
        "max_steps": max_steps,
        "device": device,
        "seed": args.seed,
        "deterministic": deterministic,
    }
    result = {"metadata": metadata, "metrics": summary}
    output_dir = make_output_dir(output_root, metadata, checkpoint)
    write_outputs(output_dir, result)
    print_summary(result, output_dir)


def load_yaml(group: str, name: str) -> dict[str, Any]:
    path = CONFIG_ROOT / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def apply_overrides(config_obj: Any, overrides: dict[str, Any]) -> Any:
    for key, value in overrides.items():
        if hasattr(config_obj, key):
            setattr(config_obj, key, value)
        else:
            print(
                f"[Warning] Skip unknown config field: {key} "
                f"for {type(config_obj).__name__}"
            )
    return config_obj


def build_algorithm_config(name: str) -> Any:
    algorithm_cls = ALGORITHM_REGISTRY[name]
    algorithm_config = algorithm_cls.get_from_yaml()
    apply_overrides(algorithm_config, load_yaml("algorithm", name))
    return algorithm_config


def build_model_configs() -> tuple[Any, Any]:
    return MlpConfig.get_from_yaml(), MlpConfig.get_from_yaml()


def build_experiment(
    task_name: str,
    algorithm_name: str,
    comm_name: str,
    estimator_name: str,
    episodes: int,
    device: str,
    seed: int,
) -> Experiment:
    task = build_comm_vmas_task(
        task_name=task_name,
        task_config=load_yaml("task", task_name),
        comm_config=load_yaml("comm", comm_name),
        estimator_config=load_yaml("estimator", estimator_name),
    )
    algorithm_config = build_algorithm_config(algorithm_name)
    model_config, critic_model_config = build_model_configs()

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = device
    experiment_config.train_device = device
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
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
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


def summarize_rollout(task_name: str, rollout, max_steps: int) -> dict[str, float]:
    summarizers: dict[str, Callable[[Any, int], dict[str, float]]] = {
        "comm_navigation": summarize_comm_navigation,
        "comm_discovery": summarize_comm_discovery,
        "comm_dispersion": summarize_comm_dispersion,
    }
    if task_name not in summarizers:
        raise ValueError(
            f"No eval metric summarizer for task={task_name}. "
            f"Available: {sorted(summarizers)}"
        )
    return summarizers[task_name](rollout, max_steps)


def summarize_comm_navigation(rollout, max_steps: int) -> dict[str, float]:
    done = rollout["next", "done"].squeeze(-1).bool()
    success = done.any(dim=1)
    episode_len = first_done_lengths(done, max_steps)
    valid = valid_step_mask(episode_len, done.shape[1])

    reward = rollout["next", "agents", "reward"].squeeze(-1)
    valid_reward = reward * valid.unsqueeze(-1)
    per_agent_return = valid_reward.sum(dim=1)
    episode_return = per_agent_return.mean(dim=-1)
    team_return = per_agent_return.sum(dim=-1)

    metrics = {
        "episodes": float(done.shape[0]),
        "success_rate": tensor_mean(success.float()),
        "timeout_rate": tensor_mean((~success).float()),
        **stats("episode_len", episode_len.float()),
        **stats("episode_return", episode_return),
        "team_return_mean": tensor_mean(team_return),
    }

    info_prefix = ("next", "agents", "info")
    if has_key(rollout, (*info_prefix, "agent_collisions")):
        collision_penalty = rollout[(*info_prefix, "agent_collisions")].squeeze(-1)
        episode_collision_penalty = (
            collision_penalty * valid.unsqueeze(-1)
        ).sum(dim=1).mean(dim=-1)
        metrics.update(stats("collision_penalty", episode_collision_penalty))

    if has_key(rollout, (*info_prefix, "pair_collision_count")):
        pair_count = rollout[(*info_prefix, "pair_collision_count")].squeeze(-1)
        agent0_pair_count = pair_count[..., 0]
        episode_pair_count = (agent0_pair_count * valid).sum(dim=1)
        metrics.update(stats("pair_collision_count", episode_pair_count))

    if has_key(rollout, (*info_prefix, "mean_aoi")):
        mean_aoi = rollout[(*info_prefix, "mean_aoi")].squeeze(-1)
        metrics["mean_aoi"] = masked_mean(mean_aoi, valid)

    if has_key(rollout, (*info_prefix, "mean_comm_mask")):
        mean_comm_mask = rollout[(*info_prefix, "mean_comm_mask")].squeeze(-1)
        metrics["mean_comm_mask"] = masked_mean(mean_comm_mask, valid)

    return metrics


def summarize_comm_discovery(rollout, max_steps: int) -> dict[str, float]:
    done = rollout["next", "done"].squeeze(-1).bool()
    success = done.any(dim=1)
    episode_len = first_done_lengths(done, max_steps)
    valid = valid_step_mask(episode_len, done.shape[1])

    reward = rollout["next", "agents", "reward"].squeeze(-1)
    valid_reward = reward * valid.unsqueeze(-1)
    per_agent_return = valid_reward.sum(dim=1)
    episode_return = per_agent_return.mean(dim=-1)
    team_return = per_agent_return.sum(dim=-1)

    metrics = {
        "episodes": float(done.shape[0]),
        "success_rate": tensor_mean(success.float()),
        "timeout_rate": tensor_mean((~success).float()),
        **stats("episode_len", episode_len.float()),
        **stats("episode_return", episode_return),
        "team_return_mean": tensor_mean(team_return),
    }

    info_prefix = ("next", "agents", "info")
    if has_key(rollout, (*info_prefix, "targets_covered")):
        targets_covered = rollout[(*info_prefix, "targets_covered")]
        if targets_covered.ndim > valid.ndim + 1 and targets_covered.shape[-1] == 1:
            targets_covered = targets_covered.squeeze(-1)
        if targets_covered.ndim == valid.ndim + 1:
            targets_covered = targets_covered[..., 0]
        valid_targets = targets_covered * valid
        episode_targets_covered = valid_targets.max(dim=1).values
        metrics.update(stats("targets_covered", episode_targets_covered))

    if has_key(rollout, (*info_prefix, "collision_rew")):
        collision_penalty = rollout[(*info_prefix, "collision_rew")]
        if (
            collision_penalty.ndim > valid.ndim + 1
            and collision_penalty.shape[-1] == 1
        ):
            collision_penalty = collision_penalty.squeeze(-1)
        episode_collision_penalty = (
            collision_penalty * valid.unsqueeze(-1)
        ).sum(dim=1).mean(dim=-1)
        metrics.update(stats("collision_penalty", episode_collision_penalty))

    if has_key(rollout, (*info_prefix, "mean_aoi")):
        mean_aoi = rollout[(*info_prefix, "mean_aoi")].squeeze(-1)
        metrics["mean_aoi"] = masked_mean(mean_aoi, valid)

    if has_key(rollout, (*info_prefix, "mean_comm_mask")):
        mean_comm_mask = rollout[(*info_prefix, "mean_comm_mask")].squeeze(-1)
        metrics["mean_comm_mask"] = masked_mean(mean_comm_mask, valid)

    return metrics


def summarize_comm_dispersion(rollout, max_steps: int) -> dict[str, float]:
    done = rollout["next", "done"].squeeze(-1).bool()
    success = done.any(dim=1)
    episode_len = first_done_lengths(done, max_steps)
    valid = valid_step_mask(episode_len, done.shape[1])

    reward = rollout["next", "agents", "reward"].squeeze(-1)
    valid_reward = reward * valid.unsqueeze(-1)
    per_agent_return = valid_reward.sum(dim=1)
    episode_return = per_agent_return.mean(dim=-1)
    team_return = per_agent_return.sum(dim=-1)

    metrics = {
        "episodes": float(done.shape[0]),
        "success_rate": tensor_mean(success.float()),
        "timeout_rate": tensor_mean((~success).float()),
        **stats("episode_len", episode_len.float()),
        **stats("episode_return", episode_return),
        "team_return_mean": tensor_mean(team_return),
    }

    info_prefix = ("next", "agents", "info")
    if has_key(rollout, (*info_prefix, "food_eaten")):
        food_eaten = rollout[(*info_prefix, "food_eaten")].squeeze(-1)
        agent0_food_eaten = food_eaten[..., 0]
        episode_food_eaten = (agent0_food_eaten * valid).max(dim=1).values
        metrics.update(stats("food_eaten", episode_food_eaten))

    if has_key(rollout, (*info_prefix, "mean_aoi")):
        mean_aoi = rollout[(*info_prefix, "mean_aoi")].squeeze(-1)
        metrics["mean_aoi"] = masked_mean(mean_aoi, valid)

    if has_key(rollout, (*info_prefix, "mean_comm_mask")):
        mean_comm_mask = rollout[(*info_prefix, "mean_comm_mask")].squeeze(-1)
        metrics["mean_comm_mask"] = masked_mean(mean_comm_mask, valid)

    return metrics


def filter_metrics(
    task_name: str,
    summary: dict[str, float],
    metric_groups: list[str] | None,
) -> dict[str, float]:
    if not metric_groups:
        return summary

    group_keys = {
        "comm_navigation": {
            "reward": ("episode_return_", "team_return_mean"),
            "success_rate": ("success_rate", "timeout_rate"),
            "episode_len": ("episode_len_",),
            "collision_penalty": ("collision_penalty_",),
            "collision_count": ("pair_collision_count_",),
            "communication": ("mean_aoi", "mean_comm_mask"),
        },
        "comm_discovery": {
            "reward": ("episode_return_", "team_return_mean"),
            "success_rate": ("success_rate", "timeout_rate"),
            "episode_len": ("episode_len_",),
            "coverage": ("targets_covered_",),
            "collision_penalty": ("collision_penalty_",),
            "communication": ("mean_aoi", "mean_comm_mask"),
        },
        "comm_dispersion": {
            "reward": ("episode_return_", "team_return_mean"),
            "success_rate": ("success_rate", "timeout_rate"),
            "episode_len": ("episode_len_",),
            "food": ("food_eaten_",),
            "communication": ("mean_aoi", "mean_comm_mask"),
        },
    }
    task_groups = group_keys.get(task_name, {})
    selected = {"episodes"}
    for group in metric_groups:
        prefixes = task_groups.get(group)
        if prefixes is None:
            print(f"[Warning] Unknown eval metric group for {task_name}: {group}")
            continue
        for key in summary:
            if any(key == prefix or key.startswith(prefix) for prefix in prefixes):
                selected.add(key)

    return {key: value for key, value in summary.items() if key in selected}


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


def masked_mean(values: Tensor, valid: Tensor) -> float:
    while valid.ndim < values.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.expand_as(values)
    denominator = valid.sum().clamp_min(1.0)
    return float(((values * valid).sum() / denominator).detach().cpu())


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


def make_output_dir(
    output_root: Path,
    metadata: dict[str, Any],
    checkpoint: Path,
) -> Path:
    output_dir = output_root / (
        f"{metadata['task']}_{metadata['algorithm']}_{metadata['comm']}_"
        f"{checkpoint.stem}_seed{metadata['seed']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    json_path = output_dir / "eval_summary.json"
    csv_path = output_dir / "eval_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    row = flatten_result(result)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for section in ("metadata", "metrics"):
        for key, value in result[section].items():
            row[f"{section}_{key}"] = value
    return row


def print_summary(result: dict[str, Any], output_dir: Path) -> None:
    metadata = result["metadata"]
    metrics = result["metrics"]
    print(
        f"Eval {metadata['task']} checkpoint={Path(metadata['checkpoint']).name} "
        f"episodes={metadata['episodes']} deterministic={metadata['deterministic']}"
    )
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.6g}")
        else:
            print(f"  {key}: {value}")
    print(f"Wrote: {output_dir / 'eval_summary.json'}")
    print(f"Wrote: {output_dir / 'eval_summary.csv'}")


if __name__ == "__main__":
    main()
