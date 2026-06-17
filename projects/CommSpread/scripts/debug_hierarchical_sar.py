"""Debug hierarchical SAR rollouts with a trained low-level checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from comm_spread.async_smdp import (
    AsyncSMDPCollector,
    FirstExploreNodePolicy,
    GreedyExplorePolicy,
    TargetFirstPolicy,
)
from comm_spread.env_factory import make_sar_env
from comm_spread.low_level_policy import BenchMARLLowLevelPolicy


DEFAULT_CHECKPOINT = Path(
    "outputs/sar_low_full/"
    "mappo_sar_low_mlp__f4788a9a_26_06_17-04_08_10/"
    "checkpoints/checkpoint_50040000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-level-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--high-policy",
        choices=["first_explore", "greedy_explore", "target_first"],
        default="first_explore",
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--render-output",
        type=Path,
        default=Path("outputs/hierarchical_sar_debug/debug_rollout.gif"),
    )
    parser.add_argument("--render-env-index", type=int, default=0)
    parser.add_argument("--render-fps", type=int, default=30)
    return parser.parse_args()


def build_high_policy(name: str):
    if name == "first_explore":
        return FirstExploreNodePolicy()
    if name == "greedy_explore":
        return GreedyExplorePolicy()
    if name == "target_first":
        return TargetFirstPolicy()
    raise ValueError(f"Unknown high policy: {name}")


def save_gif(frames: list, output: Path, fps: int) -> None:
    if not frames:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    from moviepy import ImageSequenceClip

    ImageSequenceClip(frames, fps=fps).write_gif(str(output), fps=fps)
    print(f"saved render: {output}")


def entropy_total(scenario) -> torch.Tensor:
    return scenario._compute_entropy(scenario.belief_maps).sum(dim=(-1, -2))


def print_metrics(metrics: dict[str, float | int | str]) -> None:
    print("\n[hierarchical_debug]")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6g}")
        else:
            print(f"{key}: {value}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
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
    collector = AsyncSMDPCollector(
        env,
        high_level_policy=build_high_policy(args.high_policy),
        low_level_policy=low_policy,
    )

    frames = []
    try:
        collector.reset()
        initial_entropy = entropy_total(scenario).detach().clone()
        if args.render:
            frames.append(env.render(mode="rgb_array", env_index=args.render_env_index))

        for _ in range(args.steps):
            dones = collector.step()
            if args.render:
                frames.append(env.render(mode="rgb_array", env_index=args.render_env_index))
            if bool(dones.all()):
                break

        final_entropy = entropy_total(scenario)
        metrics = summarize(
            collector=collector,
            scenario=scenario,
            initial_entropy=initial_entropy,
            final_entropy=final_entropy,
            low_level_source=low_policy.source,
            high_policy=args.high_policy,
        )
        print_metrics(metrics)
        if args.render:
            save_gif(frames, args.render_output, args.render_fps)
    finally:
        low_policy.close()


def summarize(
    *,
    collector: AsyncSMDPCollector,
    scenario,
    initial_entropy: torch.Tensor,
    final_entropy: torch.Tensor,
    low_level_source: str,
    high_policy: str,
) -> dict[str, float | int | str]:
    transitions = collector.transitions
    durations = torch.tensor(
        [transition.duration for transition in transitions],
        dtype=torch.float32,
    )
    rewards = torch.stack([transition.reward.float() for transition in transitions]) if transitions else torch.empty(0)
    detected = scenario.target_detected.any(dim=1).float().sum(dim=-1)
    visited = scenario.target_visited.float().sum(dim=-1)
    retired = (~scenario.active_agents).float().sum(dim=-1)
    success = scenario.success.float()
    entropy_reduction = initial_entropy - final_entropy

    metrics: dict[str, float | int | str] = {
        "low_level_action_source": low_level_source,
        "high_policy": high_policy,
        "transitions": len(transitions),
        "detected_targets_mean": float(detected.mean().cpu()),
        "visited_targets_mean": float(visited.mean().cpu()),
        "retired_agents_mean": float(retired.mean().cpu()),
        "success_rate": float(success.mean().cpu()),
        "entropy_initial_mean": float(initial_entropy.mean().cpu()),
        "entropy_final_mean": float(final_entropy.mean().cpu()),
        "entropy_reduction_mean": float(entropy_reduction.mean().cpu()),
    }
    if durations.numel():
        metrics.update(
            {
                "duration_min": float(durations.min()),
                "duration_mean": float(durations.mean()),
                "duration_max": float(durations.max()),
            }
        )
    else:
        metrics.update({"duration_min": 0.0, "duration_mean": 0.0, "duration_max": 0.0})
    if rewards.numel():
        metrics.update(
            {
                "high_reward_sum": float(rewards.sum()),
                "high_reward_mean": float(rewards.mean()),
            }
        )
    else:
        metrics.update({"high_reward_sum": 0.0, "high_reward_mean": 0.0})
    return metrics


if __name__ == "__main__":
    main()
