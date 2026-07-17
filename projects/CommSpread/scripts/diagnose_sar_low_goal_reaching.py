"""Classify fixed MAPPO failures on one fixed navigation goal per agent."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from comm_spread.env_factory import make_sar_env
from comm_spread.low_level_policy import BenchMARLLowLevelPolicy


DEFAULT_CHECKPOINT = Path(
    "projects/CommSpread/outputs/sar_low_full/low_safe_0.06/"
    "checkpoints/checkpoint_50040000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--controller",
        choices=("fixed_mappo", "near_goal_hybrid", "safe_hybrid", "integrated_fallback", "proportional"),
        default="fixed_mappo",
    )
    parser.add_argument("--near-goal-threshold", type=float, default=0.25)
    parser.add_argument("--proportional-gain", type=float, default=2.0)
    parser.add_argument("--stagnation-steps", type=int, default=5)
    parser.add_argument("--stagnation-progress-epsilon", type=float, default=0.005)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _positions(scenario: Any) -> Tensor:
    return torch.stack([agent.state.pos for agent in scenario.world.agents], dim=1)


def _velocities(scenario: Any) -> Tensor:
    return torch.stack([agent.state.vel for agent in scenario.world.agents], dim=1)


def _pair_min_distance(pos: Tensor) -> Tensor:
    distances = torch.cdist(pos, pos)
    eye = torch.eye(pos.shape[1], dtype=torch.bool, device=pos.device).unsqueeze(0)
    return distances.masked_fill(eye, torch.inf).min(dim=-1).values


def _rate(mask: Tensor, denominator: Tensor | None = None) -> float:
    if denominator is None:
        return float(mask.float().mean().cpu())
    count = int(denominator.sum())
    return float((mask & denominator).sum().cpu() / max(count, 1))


def _group_rates(values: Tensor, success: Tensor, edges: list[float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (values >= low) & (values < high)
        count = int(mask.sum())
        result[f"[{low:.2f},{high:.2f})"] = {
            "count": count,
            "success_rate": float(success[mask].float().mean().cpu()) if count else None,
        }
    return result


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    env = make_sar_env(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        mode="high",
        emit_info=False,
        enable_high_level_state=False,
        enable_rrt_candidates=False,
        auto_resample_goals=False,
        done_when_all_targets_visited=False,
        retire_on_rescue=False,
        max_steps=args.max_steps,
    )
    scenario = env.scenario
    policy = None
    if args.controller != "proportional":
        policy = BenchMARLLowLevelPolicy(
            args.checkpoint,
            device=args.device,
            seed=args.seed,
            deterministic=True,
            max_steps=args.max_steps,
            enable_goal_reaching_fallback=args.controller == "integrated_fallback",
            goal_near_threshold=args.near_goal_threshold,
            fallback_proportional_gain=args.proportional_gain,
            fallback_stagnation_steps=args.stagnation_steps,
            fallback_progress_epsilon=args.stagnation_progress_epsilon,
        )

    # Policy construction consumes RNG; reseed so controller variants share layouts.
    torch.manual_seed(args.seed)
    env.reset()
    # This diagnostic holds the reset goal fixed instead of resampling it on contact.
    scenario.auto_resample_goals = False
    initial_pos = _positions(scenario).clone()
    goals = scenario.assigned_goals.clone()
    initial_delta = goals - initial_pos
    initial_distance = torch.linalg.vector_norm(initial_delta, dim=-1)
    initial_angle = torch.atan2(initial_delta[..., 1], initial_delta[..., 0])
    initial_pair_distance = _pair_min_distance(initial_pos)

    shape = initial_distance.shape
    success = torch.zeros(shape, dtype=torch.bool, device=scenario.world.device)
    hit_step = torch.full(shape, -1, dtype=torch.long, device=scenario.world.device)
    min_distance = initial_distance.clone()
    min_pair_distance = initial_pair_distance.clone()
    max_abs_position = initial_pos.abs().amax(dim=-1)
    previous_distance = initial_distance.clone()
    radial_sign_changes = torch.zeros(shape, dtype=torch.long, device=scenario.world.device)
    previous_radial_sign = torch.zeros(shape, dtype=torch.int8, device=scenario.world.device)
    recent_positions: list[Tensor] = []
    recent_distances: list[Tensor] = []
    hybrid_steps = torch.zeros(shape, dtype=torch.long, device=scenario.world.device)
    best_distance = initial_distance.clone()
    stagnant_steps = torch.zeros(shape, dtype=torch.long, device=scenario.world.device)
    fallback_latched = torch.zeros(shape, dtype=torch.bool, device=scenario.world.device)

    try:
        for step in range(args.max_steps):
            pos = _positions(scenario)
            delta = goals - pos
            distance = torch.linalg.vector_norm(delta, dim=-1)
            if args.controller == "proportional":
                action_tensor = torch.clamp(
                    delta * args.proportional_gain,
                    -1.0,
                    1.0,
                )
                actions = [action_tensor[:, i] for i in range(scenario.n_agents)]
            else:
                assert policy is not None
                actions = policy(env)
                if args.controller in {"near_goal_hybrid", "safe_hybrid"}:
                    fallback = torch.clamp(
                        delta * args.proportional_gain,
                        -1.0,
                        1.0,
                    )
                    improved = distance <= (best_distance - args.stagnation_progress_epsilon)
                    best_distance = torch.minimum(best_distance, distance)
                    stagnant_steps = torch.where(
                        improved, torch.zeros_like(stagnant_steps), stagnant_steps + 1
                    )
                    fallback_mask = distance <= args.near_goal_threshold
                    if args.controller == "safe_hybrid":
                        fallback_mask |= stagnant_steps >= args.stagnation_steps
                    fallback_latched |= fallback_mask
                    fallback_mask = fallback_latched & ~success
                    hybrid_steps += fallback_mask.long()
                    actions = [
                        torch.where(
                            fallback_mask[:, i].unsqueeze(-1), fallback[:, i], actions[i]
                        )
                        for i in range(scenario.n_agents)
                    ]
            actions = [
                torch.where(success[:, i].unsqueeze(-1), torch.zeros_like(actions[i]), actions[i])
                for i in range(scenario.n_agents)
            ]

            env.step(actions)
            post_pos = _positions(scenario)
            post_distance = torch.linalg.vector_norm(post_pos - goals, dim=-1)
            newly_hit = (post_distance <= scenario.goal_radius) & ~success
            hit_step = torch.where(
                newly_hit,
                torch.full_like(hit_step, step + 1),
                hit_step,
            )
            success |= newly_hit
            min_distance = torch.minimum(min_distance, post_distance)
            min_pair_distance = torch.minimum(min_pair_distance, _pair_min_distance(post_pos))
            max_abs_position = torch.maximum(max_abs_position, post_pos.abs().amax(dim=-1))

            radial_delta = post_distance - previous_distance
            radial_sign = torch.sign(radial_delta).to(torch.int8)
            changed = (
                (radial_sign != 0)
                & (previous_radial_sign != 0)
                & (radial_sign != previous_radial_sign)
            )
            radial_sign_changes += changed.long()
            previous_radial_sign = torch.where(
                radial_sign != 0,
                radial_sign,
                previous_radial_sign,
            )
            previous_distance = post_distance
            recent_positions.append(post_pos.clone())
            recent_distances.append(post_distance.clone())
            if len(recent_positions) > 20:
                recent_positions.pop(0)
                recent_distances.pop(0)
    finally:
        if policy is not None:
            policy.close()
        if hasattr(env, "close"):
            env.close()

    final_pos = recent_positions[-1]
    final_distance = recent_distances[-1]
    window_displacement = torch.linalg.vector_norm(
        final_pos - recent_positions[0],
        dim=-1,
    )
    window_progress = recent_distances[0] - final_distance
    failed = ~success
    contact_distance = 2.0 * float(scenario.agent_radius) + 1e-3
    close_shell = (
        failed
        & (min_distance <= 2.0 * float(scenario.goal_radius))
        & (min_distance > float(scenario.goal_radius))
    )
    close_pass_escape = failed & (min_distance <= 0.10) & ((final_distance - min_distance) >= 0.05)
    stalled = failed & (window_displacement <= 0.03) & (window_progress <= 0.01)
    oscillating = failed & (radial_sign_changes >= 6)
    collision_exposed = min_pair_distance <= contact_distance
    boundary_exposed = max_abs_position >= 0.95
    goal_near_boundary = goals.abs().amax(dim=-1) >= 0.90

    checksum = float(torch.cat([initial_pos, goals], dim=1).sum().cpu())
    flat_success = success.reshape(-1)
    flat_initial_distance = initial_distance.reshape(-1)
    direction_sector = torch.remainder(
        torch.floor((initial_angle + math.pi) / (math.pi / 4.0)),
        8,
    ).long()
    direction_metrics: dict[str, Any] = {}
    for sector in range(8):
        mask = direction_sector == sector
        direction_metrics[str(sector)] = {
            "count": int(mask.sum()),
            "success_rate": float(success[mask].float().mean().cpu()),
        }

    metrics: dict[str, Any] = {
        "config": {
            "controller": args.controller,
            "checkpoint": str(args.checkpoint),
            "num_envs": args.num_envs,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "goal_radius": float(scenario.goal_radius),
            "near_goal_threshold": args.near_goal_threshold,
            "proportional_gain": args.proportional_gain,
            "stagnation_steps": args.stagnation_steps,
            "stagnation_progress_epsilon": args.stagnation_progress_epsilon,
            "initial_layout_checksum": checksum,
        },
        "summary": {
            "agent_goal_success_rate": float(success.float().mean().cpu()),
            "all_three_goals_success_rate": float(success.all(dim=1).float().mean().cpu()),
            "mean_hit_step": float(hit_step[success].float().mean().cpu()),
            "p95_hit_step": float(torch.quantile(hit_step[success].float(), 0.95).cpu()),
            "mean_initial_distance": float(initial_distance.mean().cpu()),
            "mean_min_distance": float(min_distance.mean().cpu()),
            "mean_final_distance": float(final_distance.mean().cpu()),
            "failed_mean_min_distance": float(min_distance[failed].mean().cpu()) if failed.any() else 0.0,
            "failed_mean_final_distance": float(final_distance[failed].mean().cpu()) if failed.any() else 0.0,
            "hybrid_step_fraction": (
                policy.fallback_action_fraction
                if args.controller == "integrated_fallback" and policy is not None
                else float(
                    hybrid_steps.float().sum().cpu()
                    / (args.max_steps * args.num_envs * scenario.n_agents)
                )
            ),
        },
        "failure_rates_among_failures": {
            "stuck_just_outside_goal": _rate(close_shell, failed),
            "close_pass_then_escape": _rate(close_pass_escape, failed),
            "last20_stalled": _rate(stalled, failed),
            "radial_oscillation": _rate(oscillating, failed),
            "collision_exposed": _rate(collision_exposed, failed),
            "boundary_exposed": _rate(boundary_exposed, failed),
            "goal_near_boundary": _rate(goal_near_boundary, failed),
        },
        "conditional_success": {
            "collision_exposed": _rate(success, collision_exposed),
            "not_collision_exposed": _rate(success, ~collision_exposed),
            "goal_near_boundary": _rate(success, goal_near_boundary),
            "goal_not_near_boundary": _rate(success, ~goal_near_boundary),
        },
        "success_by_initial_distance": _group_rates(
            flat_initial_distance,
            flat_success,
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        ),
        "success_by_direction_octant": direction_metrics,
    }
    return metrics


def main() -> None:
    args = parse_args()
    metrics = run(args)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
