"""Evaluate whether a trained sar_low policy reaches assigned navigation goals."""

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
    build_mappo_config,
    build_mlp_configs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to a BenchMARL checkpoint produced by sar_low MAPPO training.",
    )
    parser.add_argument(
        "--variant",
        choices=("sar_low", "sar_low_mpe_physics"),
        default="sar_low",
    )
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-folder",
        type=Path,
        default=Path("outputs/sar_low_goal_eval"),
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--random-baseline",
        action="store_true",
        help="Also run a random-action baseline with the same SAR task config.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Save a GIF of the trained policy rollout.",
    )
    parser.add_argument(
        "--render-output",
        type=Path,
        default=Path("outputs/sar_low_goal_eval/goal_eval.gif"),
        help="Output path for the rendered GIF.",
    )
    parser.add_argument(
        "--render-env-index",
        type=int,
        default=0,
        help="Index of the vectorized evaluation environment to render.",
    )
    parser.add_argument(
        "--render-fps",
        type=int,
        default=30,
        help="Frames per second for the GIF.",
    )
    parser.add_argument(
        "--render-max-steps",
        type=int,
        default=None,
        help="Maximum rendered rollout steps. Defaults to the evaluation max steps.",
    )
    return parser.parse_args()


def find_scenario(env: Any) -> Any:
    """Find the VMAS scenario inside a possibly transformed TorchRL env."""

    seen: set[int] = set()
    stack = [env]
    attr_names = ("scenario", "base_env", "_env", "env", "parent")
    while stack:
        item = stack.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        if all(hasattr(item, name) for name in ("assigned_goals", "world")):
            return item
        for attr_name in attr_names:
            if hasattr(item, attr_name):
                stack.append(getattr(item, attr_name))
    raise RuntimeError(f"Could not find SarScenario under env type {type(env)!r}")


def make_experiment(args: argparse.Namespace) -> Experiment:
    task_config = dict(TASK_VARIANTS[args.variant])
    if args.max_steps is not None:
        task_config["max_steps"] = args.max_steps

    model_config, critic_model_config = build_mlp_configs()
    task = CommSpreadTask.SAR_LOW.update_config(task_config)
    args.save_folder.mkdir(parents=True, exist_ok=True)
    config = build_experiment_config(
        train_device=args.device,
        sampling_device=args.device,
        quick=True,
        save_folder=str(args.save_folder),
        restore_file=str(args.checkpoint),
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


def _stack_agent_pos(scenario: Any) -> torch.Tensor:
    return torch.stack([agent.state.pos for agent in scenario.world.agents], dim=1)


def save_gif(frames: list[Any], output: Path, fps: int) -> None:
    if not frames:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    from moviepy import ImageSequenceClip

    ImageSequenceClip(frames, fps=fps).write_gif(str(output), fps=fps)
    print(f"saved render: {output}")


@torch.no_grad()
def run_policy_eval(
    experiment: Experiment,
    *,
    random_policy: bool = False,
    render_output: Path | None = None,
    render_env_index: int = 0,
    render_fps: int = 30,
    render_max_steps: int | None = None,
) -> dict[str, float]:
    env = experiment.test_env
    scenario = find_scenario(env)
    max_steps = int(experiment.task.config.get("max_steps", scenario.max_steps))
    scenario.max_steps = max_steps
    num_envs = scenario.world.batch_dim
    n_agents = scenario.n_agents
    goal_radius = scenario.goal_radius
    render_enabled = render_output is not None
    render_limit = (
        max_steps if render_max_steps is None else min(render_max_steps, max_steps)
    )
    frames: list[Any] = []
    if render_enabled:
        if not (0 <= render_env_index < num_envs):
            raise ValueError(
                f"--render-env-index must be in [0, {num_envs - 1}], got {render_env_index}"
            )

    td = env.reset()
    if render_enabled:
        frames.append(env.render(mode="rgb_array", env_index=render_env_index))
    elapsed = torch.zeros(num_envs, n_agents, device=scenario.world.device)
    final_dist = torch.zeros(num_envs, n_agents, device=scenario.world.device)
    min_dist = torch.full((num_envs, n_agents), torch.inf, device=scenario.world.device)
    completed = torch.zeros(num_envs, n_agents, device=scenario.world.device)
    active_steps = torch.zeros(num_envs, n_agents, device=scenario.world.device)
    hit_durations: list[torch.Tensor] = []
    hit_distances: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []

    exploration_type = ExplorationType.RANDOM if random_policy else ExplorationType.DETERMINISTIC
    with set_exploration_type(exploration_type):
        for _ in range(max_steps):
            pre_goals = scenario.assigned_goals.clone()
            pre_active = scenario.active_agents.clone()
            elapsed = elapsed + pre_active.float()
            active_steps = active_steps + pre_active.float()

            if random_policy:
                td = env.rand_action(td)
            else:
                td = experiment.policy(td)

            td = env.step(td)
            if render_enabled and len(frames) <= render_limit:
                frames.append(env.render(mode="rgb_array", env_index=render_env_index))
            post_pos = _stack_agent_pos(scenario)
            dist_to_previous_goal = torch.linalg.vector_norm(post_pos - pre_goals, dim=-1)
            min_dist = torch.minimum(min_dist, dist_to_previous_goal)
            final_dist = dist_to_previous_goal

            hit = (dist_to_previous_goal <= goal_radius) & pre_active
            if hit.any():
                completed += hit.float()
                hit_durations.append(elapsed[hit].detach().cpu())
                hit_distances.append(dist_to_previous_goal[hit].detach().cpu())
                elapsed = torch.where(hit, torch.zeros_like(elapsed), elapsed)

            reward = td.get(("next", "agents", "reward")).squeeze(-1)
            rewards.append(reward.detach().cpu())
            td = step_mdp(
                td,
                reward_keys=env.reward_keys,
                action_keys=env.action_keys,
                done_keys=env.done_keys,
            )

    if render_enabled:
        save_gif(frames, render_output, render_fps)

    hit_duration = torch.cat(hit_durations) if hit_durations else torch.empty(0)
    hit_distance = torch.cat(hit_distances) if hit_distances else torch.empty(0)
    reward_tensor = torch.stack(rewards)
    attempts = float(num_envs * n_agents)

    return {
        "episodes": float(num_envs),
        "agents": float(n_agents),
        "max_steps": float(max_steps),
        "goal_radius": float(goal_radius),
        "completed_goals": float(completed.sum().cpu()),
        "completed_goals_per_agent_episode": float(completed.sum().cpu() / attempts),
        "agent_episode_success_rate_ge_1_goal": float((completed > 0).float().mean().cpu()),
        "mean_steps_per_completed_goal": float(hit_duration.float().mean()) if hit_duration.numel() else float("nan"),
        "median_steps_per_completed_goal": float(hit_duration.float().median()) if hit_duration.numel() else float("nan"),
        "mean_hit_distance": float(hit_distance.float().mean()) if hit_distance.numel() else float("nan"),
        "max_hit_distance": float(hit_distance.float().max()) if hit_distance.numel() else float("nan"),
        "mean_final_distance": float(final_dist.mean().cpu()),
        "median_final_distance": float(final_dist.median().cpu()),
        "mean_min_distance": float(min_dist.mean().cpu()),
        "median_min_distance": float(min_dist.median().cpu()),
        "active_step_hit_rate": float(completed.sum().cpu() / active_steps.sum().clamp_min(1).cpu()),
        "mean_step_reward": float(reward_tensor.mean()),
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n[{title}]")
    for key, value in metrics.items():
        if key in {"episodes", "agents", "max_steps"}:
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.6g}")


def main() -> None:
    args = parse_args()
    results: dict[str, dict[str, float]] = {}
    experiment = make_experiment(args)
    try:
        metrics = run_policy_eval(
            experiment,
            render_output=args.render_output if args.render else None,
            render_env_index=args.render_env_index,
            render_fps=args.render_fps,
            render_max_steps=args.render_max_steps,
        )
        print_metrics("trained_policy", metrics)
        results["trained_policy"] = metrics
    finally:
        experiment.close()

    if args.random_baseline:
        baseline = make_experiment(args)
        try:
            metrics = run_policy_eval(baseline, random_policy=True)
            print_metrics("random_baseline", metrics)
            results["random_baseline"] = metrics
        finally:
            baseline.close()

    if args.json_output is not None:
        import json

        if args.json_output.exists():
            raise FileExistsError(f"refusing to overwrite {args.json_output}")
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
