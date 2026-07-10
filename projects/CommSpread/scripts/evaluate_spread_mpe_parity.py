"""Evaluate VMAS spread_mpe_parity checkpoints with old MPE metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from benchmarl.experiment import Experiment

from comm_spread.benchmarl_task import CommSpreadTask
from comm_spread.training_config import (
    TASK_VARIANTS,
    build_experiment_config,
    build_gnn_actor_config,
    build_mappo_config,
    build_mlp_configs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=None,
        help="BenchMARL checkpoint produced by spread_mpe_parity training.",
    )
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument(
        "--variant",
        choices=sorted(TASK_VARIANTS),
        default="spread_mpe_parity",
        help="Task variant to evaluate; defaults to the parity baseline.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", choices=["mlp", "gnn"], default="mlp")
    parser.add_argument("--comms-radius", type=float, default=1.0)
    parser.add_argument(
        "--random-baseline",
        action="store_true",
        help="Also run a random-action baseline with the same task config.",
    )
    return parser.parse_args()


def find_scenario(env: Any) -> Any:
    seen: set[int] = set()
    stack = [env]
    attr_names = ("scenario", "base_env", "_env", "env", "parent")
    while stack:
        item = stack.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        if all(hasattr(item, name) for name in ("matched_landmark_dists", "world")):
            return item
        for attr_name in attr_names:
            if hasattr(item, attr_name):
                stack.append(getattr(item, attr_name))
    raise RuntimeError(f"Could not find CommSpreadScenario under env type {type(env)!r}")


def make_experiment(args: argparse.Namespace, *, restore: bool) -> Experiment:
    task_config = dict(TASK_VARIANTS[args.variant])
    task_config["max_steps"] = int(task_config.get("max_steps", 50))
    if args.model == "gnn":
        task_config["comms_rendering_range"] = args.comms_radius
        model_config = build_gnn_actor_config(comms_radius=args.comms_radius)
        _, critic_model_config = build_mlp_configs()
    else:
        model_config, critic_model_config = build_mlp_configs()

    task = CommSpreadTask.SPREAD.update_config(task_config)
    output_dir = Path("outputs") / f"{args.variant}_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_experiment_config(
        train_device=args.device,
        sampling_device=args.device,
        quick=True,
        save_folder=str(output_dir),
        restore_file=str(args.checkpoint) if restore and args.checkpoint is not None else None,
        restore_map_location=args.device,
        max_n_frames=1_000,
    )
    config.collect_with_grad = True
    config.evaluation = False
    config.render = False
    config.evaluation_episodes = args.episodes
    config.loggers = []
    config.create_json = False

    return Experiment(
        task=task,
        algorithm_config=build_mappo_config(),
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=config,
    )


@torch.no_grad()
def run_eval(experiment: Experiment, *, random_policy: bool = False) -> dict[str, float]:
    env = experiment.test_env
    scenario = find_scenario(env)
    max_steps = int(experiment.task.config.get("max_steps", 50))
    num_envs = scenario.world.batch_dim

    td = env.reset()
    done_once = torch.zeros(num_envs, dtype=torch.bool, device=scenario.world.device)
    success_once = torch.zeros_like(done_once)
    episode_lengths = torch.full(
        (num_envs,),
        float(max_steps),
        dtype=torch.float32,
        device=scenario.world.device,
    )
    final_matched_mean = scenario.matched_mean_dist.detach().clone()
    rewards: list[torch.Tensor] = []

    exploration_type = ExplorationType.RANDOM if random_policy else ExplorationType.DETERMINISTIC
    with set_exploration_type(exploration_type):
        for step in range(1, max_steps + 1):
            if random_policy:
                td = env.rand_action(td)
            else:
                td = experiment.policy(td)

            td = env.step(td)
            reward = td.get(("next", "agents", "reward")).squeeze(-1)
            rewards.append(reward.detach().cpu())

            next_done = td.get(("next", "done")).squeeze(-1).bool()
            success = scenario.all_landmarks_covered.detach().clone()
            newly_done = next_done & ~done_once
            episode_lengths = torch.where(
                newly_done,
                torch.full_like(episode_lengths, float(step)),
                episode_lengths,
            )
            final_matched_mean = torch.where(
                newly_done | ~done_once,
                scenario.matched_mean_dist.detach(),
                final_matched_mean,
            )
            success_once |= success
            done_once |= next_done

            td = step_mdp(
                td,
                reward_keys=env.reward_keys,
                action_keys=env.action_keys,
                done_keys=env.done_keys,
            )

    reward_tensor = torch.stack(rewards)
    final_matched_mean = torch.where(
        done_once,
        final_matched_mean,
        scenario.matched_mean_dist.detach(),
    )
    return {
        "episodes": float(num_envs),
        "max_steps": float(max_steps),
        "success_rate": float(success_once.float().mean().cpu()),
        "mean_episode_length": float(episode_lengths.mean().cpu()),
        "median_episode_length": float(episode_lengths.median().cpu()),
        "mean_final_matched_distance": float(final_matched_mean.mean().cpu()),
        "median_final_matched_distance": float(final_matched_mean.median().cpu()),
        "mean_step_reward": float(reward_tensor.mean()),
        "mean_episode_reward_per_agent": float(reward_tensor.sum(dim=0).mean()),
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n[{title}]")
    for key, value in metrics.items():
        if key in {"episodes", "max_steps"}:
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.6g}")


def main() -> None:
    args = parse_args()
    if args.checkpoint is None and not args.random_baseline:
        raise SystemExit("Pass a checkpoint or use --random-baseline.")

    if args.checkpoint is not None:
        experiment = make_experiment(args, restore=True)
        try:
            print_metrics("trained_policy", run_eval(experiment))
        finally:
            experiment.close()

    if args.random_baseline:
        baseline = make_experiment(args, restore=False)
        try:
            print_metrics("random_baseline", run_eval(baseline, random_policy=True))
        finally:
            baseline.close()


if __name__ == "__main__":
    main()
