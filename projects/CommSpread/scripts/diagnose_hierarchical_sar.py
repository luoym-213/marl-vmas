"""Diagnose SAR high-level assignments, RRT candidates, and low-level goal tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from comm_spread.async_smdp import (
    AsyncSMDPCollector,
    FirstExploreNodePolicy,
    GreedyExplorePolicy,
    TargetFirstPolicy,
)
from comm_spread.env_factory import make_sar_env
from comm_spread.high_level_policy import HGSARActorCriticPolicy, HGSARHighLevelPolicy
from comm_spread.low_level_policy import BenchMARLLowLevelPolicy


DEFAULT_LOW_CHECKPOINT = Path(
    "projects/CommSpread/outputs/sar_low_full/low_safe_0.06/checkpoints/checkpoint_50040000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-level-checkpoint", type=Path, default=DEFAULT_LOW_CHECKPOINT)
    parser.add_argument("--high-level-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--high-policies",
        nargs="+",
        default=["greedy_explore", "target_first", "hgsar_untrained"],
        choices=["first_explore", "greedy_explore", "target_first", "hgsar_untrained", "hgsar_checkpoint"],
    )
    parser.add_argument("--sensor-radii", nargs="+", type=float, default=[0.3, 0.45, 0.6])
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("projects/CommSpread/outputs/hierarchical_sar_diagnostics/diagnostics.json"))
    parser.add_argument("--markdown", type=Path, default=Path("projects/CommSpread/hierarchical_sar_diagnostics.md"))
    return parser.parse_args()


def build_high_policy(name: str, *, args: argparse.Namespace, scenario: Any):
    if name == "first_explore":
        return FirstExploreNodePolicy()
    if name == "greedy_explore":
        return GreedyExplorePolicy()
    if name == "target_first":
        return TargetFirstPolicy()
    if name == "hgsar_untrained":
        return HGSARHighLevelPolicy(
            n_agents=scenario.n_agents,
            rrt_top_k=scenario.rrt_top_k,
            device=args.device,
            deterministic=True,
        )
    if name == "hgsar_checkpoint":
        if args.high_level_checkpoint is None:
            raise ValueError("--high-level-checkpoint is required for hgsar_checkpoint")
        policy = HGSARActorCriticPolicy(
            n_agents=scenario.n_agents,
            rrt_top_k=scenario.rrt_top_k,
            device=args.device,
            deterministic=True,
        )
        checkpoint = torch.load(args.high_level_checkpoint, map_location=args.device, weights_only=False)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.eval()
        return policy
    raise ValueError(f"Unknown high policy: {name}")


def entropy_total(scenario: Any) -> torch.Tensor:
    return scenario._compute_entropy(scenario.belief_maps).sum(dim=(-1, -2))


def candidate_metrics(scenario: Any) -> dict[str, float]:
    candidates = scenario.explore_candidates.detach()
    agent_pos = torch.stack([agent.state.pos for agent in scenario.world.agents], dim=1).detach()
    world_nodes = agent_pos.unsqueeze(2) + candidates[..., 0:2]
    half = float(scenario.world_size) / 2.0
    in_bounds = ((world_nodes >= -half) & (world_nodes <= half)).all(dim=-1)
    rel_dist = torch.linalg.vector_norm(candidates[..., 0:2], dim=-1)
    utility = candidates[..., 2]
    return {
        "rrt_in_bounds_fraction": float(in_bounds.float().mean().cpu()),
        "rrt_rel_distance_mean": float(rel_dist.mean().cpu()),
        "rrt_rel_distance_max": float(rel_dist.max().cpu()),
        "rrt_utility_mean": float(utility.mean().cpu()),
        "rrt_utility_max": float(utility.max().cpu()),
        "rrt_zero_utility_fraction": float((utility <= 1e-8).float().mean().cpu()),
    }


@torch.no_grad()
def run_case(args: argparse.Namespace, *, high_policy_name: str, sensor_radius: float) -> dict[str, float | int | str]:
    env = make_sar_env(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        mode="high",
        emit_info=False,
        enable_high_level_state=True,
        enable_rrt_candidates=True,
        auto_resample_goals=False,
        sensor_radius=sensor_radius,
    )
    scenario = env.scenario
    low_policy = BenchMARLLowLevelPolicy(
        args.low_level_checkpoint,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        max_steps=scenario.max_steps,
    )
    high_policy = build_high_policy(high_policy_name, args=args, scenario=scenario)
    collector = AsyncSMDPCollector(env, high_level_policy=high_policy, low_level_policy=low_policy)

    goal_hit_count = torch.zeros((), device=scenario.world.device)
    assigned_active_steps = torch.zeros((), device=scenario.world.device)
    min_goal_distance_sum = torch.zeros((), device=scenario.world.device)
    min_goal_distance_count = torch.zeros((), device=scenario.world.device)
    per_step_goal_distance_sum = torch.zeros((), device=scenario.world.device)
    per_step_goal_distance_count = torch.zeros((), device=scenario.world.device)

    try:
        collector.reset()
        initial_entropy = entropy_total(scenario).detach().clone()
        initial_candidates = candidate_metrics(scenario)
        min_dist = torch.full_like(scenario.previous_goal_dist, torch.inf)

        for _ in range(args.steps):
            pre_goal_done = scenario.goal_done.detach().clone()
            pre_active = scenario.active_agents.detach().clone()
            pre_pending = pre_active & ~pre_goal_done
            pre_goals = scenario.assigned_goals.detach().clone()
            dones = collector.step()
            post_pos = torch.stack([agent.state.pos for agent in scenario.world.agents], dim=1)
            dist = torch.linalg.vector_norm(post_pos - pre_goals, dim=-1)
            min_dist = torch.where(pre_pending, torch.minimum(min_dist, dist), min_dist)
            assigned_active_steps += pre_pending.float().sum()
            per_step_goal_distance_sum += torch.where(pre_pending, dist, torch.zeros_like(dist)).sum()
            per_step_goal_distance_count += pre_pending.float().sum()
            hits = (dist <= scenario.goal_radius) & pre_pending
            goal_hit_count += hits.float().sum()
            if hits.any():
                min_goal_distance_sum += min_dist[hits].sum()
                min_goal_distance_count += hits.float().sum()
                min_dist = torch.where(hits, torch.full_like(min_dist, torch.inf), min_dist)
            if bool(dones.all()):
                break

        final_entropy = entropy_total(scenario).detach()
        transitions = collector.transitions
        durations = torch.tensor([transition.duration for transition in transitions], dtype=torch.float32)
        rewards = torch.stack([transition.reward.float() for transition in transitions]) if transitions else torch.empty(0)
        detected = scenario.target_detected.any(dim=1).float().sum(dim=-1)
        visited = scenario.target_visited.float().sum(dim=-1)
        retired = (~scenario.active_agents).float().sum(dim=-1)
        target_detected_events = scenario.target_detected.float().sum(dim=(1, 2))

        finalized = max(len(transitions), 1)
        metrics: dict[str, float | int | str] = {
            "seed": args.seed,
            "high_policy": high_policy_name,
            "sensor_radius": sensor_radius,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "low_level_checkpoint": str(args.low_level_checkpoint),
            "transitions": len(transitions),
            "low_level_goal_hits": float(goal_hit_count.cpu()),
            "low_level_goal_hit_per_transition": float((goal_hit_count / finalized).cpu()),
            "assigned_active_steps": float(assigned_active_steps.cpu()),
            "assigned_active_step_hit_rate": float((goal_hit_count / assigned_active_steps.clamp_min(1)).cpu()),
            "mean_step_goal_distance": float((per_step_goal_distance_sum / per_step_goal_distance_count.clamp_min(1)).cpu()),
            "mean_min_distance_on_hit": float((min_goal_distance_sum / min_goal_distance_count.clamp_min(1)).cpu()),
            "detected_targets_mean": float(detected.mean().cpu()),
            "visited_targets_mean": float(visited.mean().cpu()),
            "retired_agents_mean": float(retired.mean().cpu()),
            "success_rate": float(scenario.success.float().mean().cpu()),
            "raw_target_detection_events_mean": float(target_detected_events.mean().cpu()),
            "entropy_initial_mean": float(initial_entropy.mean().cpu()),
            "entropy_final_mean": float(final_entropy.mean().cpu()),
            "entropy_reduction_mean": float((initial_entropy - final_entropy).mean().cpu()),
            "duration_mean": float(durations.mean()) if durations.numel() else 0.0,
            "duration_max": float(durations.max()) if durations.numel() else 0.0,
            "high_reward_mean": float(rewards.mean()) if rewards.numel() else 0.0,
            "high_reward_sum": float(rewards.sum()) if rewards.numel() else 0.0,
        }
        metrics.update(initial_candidates)
        if hasattr(high_policy, "metrics"):
            metrics.update(high_policy.metrics())
        return metrics
    finally:
        low_policy.close()
        if hasattr(env, "close"):
            env.close()


def write_markdown(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    headers = [
        "policy", "sensor", "success", "visited", "detected", "goal_hits/tr", "step_hit", "goal_dist", "entropy_gain", "rrt_util", "rrt_zero", "duration",
    ]
    lines = [
        "# Hierarchical SAR Diagnostics",
        "",
        "This report evaluates high-level assignment quality, RRT candidate quality, low-level goal reaching, and relaxed sensor radii without training new code.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            str(row["high_policy"]),
            f"{float(row['sensor_radius']):.2f}",
            f"{float(row['success_rate']):.3f}",
            f"{float(row['visited_targets_mean']):.3f}",
            f"{float(row['detected_targets_mean']):.3f}",
            f"{float(row['low_level_goal_hit_per_transition']):.3f}",
            f"{float(row['assigned_active_step_hit_rate']):.4f}",
            f"{float(row['mean_step_goal_distance']):.3f}",
            f"{float(row['entropy_reduction_mean']):.1f}",
            f"{float(row['rrt_utility_mean']):.3f}",
            f"{float(row['rrt_zero_utility_fraction']):.3f}",
            f"{float(row['duration_mean']):.1f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `goal_hits/tr` and `step_hit` measure whether the fixed low-level controller reaches the high-level assigned waypoint.",
        "- `visited` and `success` measure whether high-level assignment plus target handling solves the SAR task.",
        "- Large entropy gain with low visited targets indicates exploration is happening but target-to-rescue assignment is weak.",
        "- If larger `sensor` sharply improves visited/success, the current task is likely too sparse for validating the high-level objective early in training.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sensor_radius in args.sensor_radii:
        for high_policy_name in args.high_policies:
            row = run_case(args, high_policy_name=high_policy_name, sensor_radius=sensor_radius)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.markdown, rows)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
