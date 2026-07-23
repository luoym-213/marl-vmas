"""Analyze SAR failure modes for trained high-level policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from comm_spread.async_smdp import AsyncSMDPCollector, TargetFirstPolicy
from comm_spread.env_factory import make_sar_env
from comm_spread.high_level_policy import HGSARActorCriticPolicy
from comm_spread.low_level_policy import BenchMARLLowLevelPolicy
from comm_spread.sar_module_ablation import (
    CoverageTrajectoryTracker,
    ModularAblationPolicy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOW_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/sar_low_full/low_safe_0.06/checkpoints/checkpoint_50040000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-level-checkpoint", type=Path, default=DEFAULT_LOW_CHECKPOINT)
    parser.add_argument("--enable-low-level-goal-fallback", action="store_true")
    parser.add_argument("--low-level-goal-near-threshold", type=float, default=0.25)
    parser.add_argument("--low-level-fallback-gain", type=float, default=2.0)
    parser.add_argument("--low-level-fallback-stagnation-steps", type=int, default=5)
    parser.add_argument("--low-level-fallback-progress-epsilon", type=float, default=0.005)
    parser.add_argument("--high-level-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--policy",
        choices=[
            "hgsar",
            "target_first",
            "rescue_greedy",
            "search_all_then_rescue",
            "search_then_target",
        ],
        default="hgsar",
    )
    parser.add_argument("--target-threshold", type=int, default=3)
    parser.add_argument("--sensor-radius", type=float, default=0.6)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--retire-on-rescue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-features", action="store_true")
    parser.add_argument("--target-assignment-features", action="store_true")
    parser.add_argument("--staged-rescue", action="store_true")
    parser.add_argument("--rescue-detected-threshold", type=int, default=2)
    parser.add_argument("--rescue-entropy-threshold", type=float, default=None)
    parser.add_argument("--dynamic-rescue-release", action="store_true")
    parser.add_argument("--dynamic-release-min-searchers", type=int, default=2)
    parser.add_argument("--dynamic-release-entropy-ratio-threshold", type=float, default=0.58)
    parser.add_argument("--dynamic-release-entropy-rate-threshold", type=float, default=0.0015)
    parser.add_argument("--dynamic-release-min-stagnation-step", type=int, default=25)
    parser.add_argument("--dynamic-release-search-steps-per-target", type=float, default=22.0)
    parser.add_argument("--dynamic-release-speed-per-step", type=float, default=0.035)
    parser.add_argument("--dynamic-release-time-margin", type=float, default=8.0)
    parser.add_argument("--dynamic-release-max-new-agents-per-event", type=int, default=1)
    parser.add_argument("--dynamic-rescue-only-after-all-detected", action="store_true")
    parser.add_argument("--redecide-on-detection-change", action="store_true")
    parser.add_argument("--redecide-on-assignment-change", action="store_true")
    parser.add_argument("--enable-finder-first-cascade", action="store_true")
    parser.add_argument(
        "--finder-cascade-mode",
        choices=("finder_only", "immediate", "one_event"),
        default="immediate",
    )
    parser.add_argument("--early-rescue-penalty", type=float, default=0.0)
    parser.add_argument("--early-rescue-detected-threshold", type=int, default=3)
    parser.add_argument("--search-capacity-discovery-bonus", type=float, default=0.0)
    parser.add_argument("--discovery-active-agents-threshold", type=int, default=2)
    parser.add_argument("--all-targets-detected-bonus", type=float, default=0.0)
    parser.add_argument(
        "--all-targets-detected-bonus-requires-no-rescue",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rescue-phase-explore-penalty", type=float, default=0.0)
    parser.add_argument("--rescue-phase-detected-threshold", type=int, default=2)
    parser.add_argument("--rescue-phase-min-search-agents", type=int, default=1)
    parser.add_argument("--unique-rescue-assignment-bonus", type=float, default=0.0)
    parser.add_argument("--duplicate-rescue-assignment-penalty", type=float, default=0.0)
    parser.add_argument("--detected-unassigned-target-penalty", type=float, default=0.0)
    parser.add_argument("--coordinated-target-selection", action="store_true")
    parser.add_argument("--enable-commitment-aware-actor", action="store_true")
    parser.add_argument("--enable-phase-policy", action="store_true")
    parser.add_argument("--phase-initial-rescue-logit", type=float, default=-1.5)
    parser.add_argument("--rescue-distance-logit-scale", type=float, default=0.0)
    parser.add_argument(
        "--enable-teammate-intention-coordination",
        action="store_true",
    )
    parser.add_argument("--intent-loss-coef", type=float, default=0.1)
    parser.add_argument("--coordination-beta", type=float, default=0.25)
    parser.add_argument("--coordination-margin", type=float, default=0.1)
    parser.add_argument("--coordination-temperature", type=float, default=1.0)
    parser.add_argument(
        "--low-level-controller",
        choices=("checkpoint", "proportional"),
        default="checkpoint",
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--diagnostic-speed-per-step", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("projects/CommSpread/outputs/hierarchical_sar_diagnostics/failure_modes.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("projects/CommSpread/outputs/hierarchical_sar_diagnostics/failure_modes.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("projects/CommSpread/sar_failure_modes.md"))
    parser.add_argument("--assignment-events-output", type=Path, default=None)
    parser.add_argument("--assignment-cost-matrices-output", type=Path, default=None)
    parser.add_argument("--crossing-log-output", type=Path, default=None)
    parser.add_argument("--enable-module-ablation", action="store_true")
    parser.add_argument(
        "--ablation-search-module",
        choices=("hgsar", "three_lane"),
        default="hgsar",
    )
    parser.add_argument(
        "--ablation-timing-module",
        choices=("finder", "all_detected"),
        default="finder",
    )
    parser.add_argument(
        "--ablation-assignment-module",
        choices=("actor", "exact"),
        default="actor",
    )
    parser.add_argument("--enable-coverage-metrics", action="store_true")
    parser.add_argument("--coverage-metrics-output", type=Path, default=None)
    return parser.parse_args()


class RescueGreedyPolicy:
    """Rescue immediately with per-env unique greedy target assignment."""

    def __call__(self, observation: dict[str, torch.Tensor]):
        target_mask = observation["target_mask"].bool()
        target_dist = torch.linalg.vector_norm(observation["target_nodes"][..., 0:2], dim=-1)
        greedy_explore = observation["explore_nodes"][..., 2].argmax(dim=-1)
        actions = greedy_explore.clone()
        rrt_top_k = observation["explore_nodes"].shape[1]
        env_ids = observation.get("env_id")
        if env_ids is None:
            env_ids = torch.arange(actions.shape[0], device=actions.device)
        for env_id in env_ids.unique(sorted=True):
            row_ids = torch.nonzero(env_ids == env_id, as_tuple=False).flatten()
            available_targets = target_mask[row_ids].any(dim=0).clone()
            unassigned_rows = row_ids.clone()
            while unassigned_rows.numel() > 0 and bool(available_targets.any()):
                dists = target_dist[unassigned_rows].masked_fill(
                    ~target_mask[unassigned_rows],
                    torch.inf,
                )
                dists = dists.masked_fill(~available_targets.view(1, -1), torch.inf)
                flat_index = dists.argmin()
                if not torch.isfinite(dists.flatten()[flat_index]):
                    break
                row_offset = flat_index // dists.shape[1]
                target_index = flat_index % dists.shape[1]
                row = unassigned_rows[row_offset]
                actions[row] = rrt_top_k + target_index
                available_targets[target_index] = False
                keep = torch.ones_like(unassigned_rows, dtype=torch.bool)
                keep[row_offset] = False
                unassigned_rows = unassigned_rows[keep]
        n = actions.shape[0]
        device = actions.device
        return actions.long(), torch.zeros(n, 1, device=device), torch.zeros(n, 1, device=device)


class SearchAllThenRescuePolicy:
    """Search until all targets are globally detected, then greedily rescue nearest targets."""

    def __init__(self, scenario: Any):
        self.scenario = scenario
        self.greedy = RescueGreedyPolicy()

    def __call__(self, observation: dict[str, torch.Tensor]):
        target_mask = observation["target_mask"].bool()
        env_ids = observation["env_id"].long()
        detected_count = self.scenario.target_detected.any(dim=1).float().sum(dim=-1)
        can_rescue = detected_count[env_ids] >= float(self.scenario.n_targets)
        greedy_actions, logp, value = self.greedy(observation)
        greedy_explore = observation["explore_nodes"][..., 2].argmax(dim=-1)
        has_target = target_mask.any(dim=-1)
        actions = torch.where(can_rescue & has_target, greedy_actions, greedy_explore)
        return actions.long(), logp, value


class SearchThenTargetPolicy:
    def __init__(self, *, threshold: int):
        self.threshold = threshold

    def __call__(self, observation: dict[str, torch.Tensor]):
        target_mask = observation["target_mask"].bool()
        detected_count = target_mask.sum(dim=-1)
        can_rescue = detected_count >= self.threshold
        first_target = target_mask.float().argmax(dim=-1)
        greedy_explore = observation["explore_nodes"][..., 2].argmax(dim=-1)
        rrt_top_k = observation["explore_nodes"].shape[1]
        actions = torch.where(can_rescue, rrt_top_k + first_target, greedy_explore)
        n = actions.shape[0]
        device = actions.device
        return actions.long(), torch.zeros(n, 1, device=device), torch.zeros(n, 1, device=device)


def build_policy(args: argparse.Namespace, scenario: Any):
    if args.policy == "target_first":
        return TargetFirstPolicy()
    if args.policy == "rescue_greedy":
        return RescueGreedyPolicy()
    if args.policy == "search_all_then_rescue":
        return SearchAllThenRescuePolicy(scenario)
    if args.policy == "search_then_target":
        return SearchThenTargetPolicy(threshold=args.target_threshold)
    if args.high_level_checkpoint is None:
        raise ValueError("--high-level-checkpoint is required for --policy hgsar")
    policy = HGSARActorCriticPolicy(
        n_agents=scenario.n_agents,
        rrt_top_k=scenario.rrt_top_k,
        device=args.device,
        deterministic=True,
        ego_features=9 if args.progress_features else 5,
        target_features=7 if args.target_assignment_features else 4,
        enable_commitment_aware_actor=args.enable_commitment_aware_actor,
        enable_phase_policy=args.enable_phase_policy,
        phase_initial_rescue_logit=args.phase_initial_rescue_logit,
        rescue_distance_logit_scale=args.rescue_distance_logit_scale,
        enable_teammate_intention_coordination=(
            args.enable_teammate_intention_coordination
        ),
        coordination_beta=args.coordination_beta,
        coordination_margin=args.coordination_margin,
        coordination_temperature=args.coordination_temperature,
    )
    checkpoint = torch.load(args.high_level_checkpoint, map_location=args.device, weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    if args.enable_module_ablation:
        return ModularAblationPolicy(
            policy,
            scenario,
            search_module=args.ablation_search_module,
            timing_module=args.ablation_timing_module,
            assignment_module=args.ablation_assignment_module,
        )
    return policy


def stack_agent_positions(scenario: Any) -> torch.Tensor:
    return torch.stack([agent.state.pos for agent in scenario.world.agents], dim=1)


def stack_target_positions(scenario: Any) -> torch.Tensor:
    return torch.stack([target.state.pos for target in scenario.targets], dim=1)


def estimated_parallel_rescue_steps(
    agent_positions: torch.Tensor,
    active_mask: torch.Tensor,
    target_positions: torch.Tensor,
    target_mask: torch.Tensor,
    speed_per_step: float,
) -> float:
    """Exact small-team geometric makespan estimate for diagnostic labels."""
    agent_ids = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
    target_ids = torch.nonzero(target_mask, as_tuple=False).flatten().tolist()
    if not target_ids:
        return 0.0
    if len(agent_ids) < len(target_ids):
        return math.inf
    distances = torch.cdist(
        agent_positions.unsqueeze(0), target_positions.unsqueeze(0)
    ).squeeze(0)
    best = math.inf
    for chosen_agents in itertools.combinations(agent_ids, len(target_ids)):
        for target_order in itertools.permutations(target_ids):
            makespan = max(
                float(distances[agent_id, target_id])
                for agent_id, target_id in zip(chosen_agents, target_order)
            )
            best = min(best, makespan)
    return math.ceil(best / max(speed_per_step, 1e-6))


def optimal_assignment_costs(cost_matrix: torch.Tensor) -> tuple[float, float]:
    """Return minimum sum cost and minimum makespan for a small square matrix."""
    n = cost_matrix.shape[0]
    if n == 0:
        return 0.0, 0.0
    if cost_matrix.shape[1] != n:
        raise ValueError("assignment diagnostic expects a square cost matrix")
    rows = torch.arange(n, device=cost_matrix.device)
    sum_best = math.inf
    makespan_best = math.inf
    for permutation in itertools.permutations(range(n)):
        chosen = cost_matrix[rows, torch.tensor(permutation, device=cost_matrix.device)]
        sum_best = min(sum_best, float(chosen.sum()))
        makespan_best = min(makespan_best, float(chosen.max()))
    return sum_best, makespan_best


def assignment_quality_snapshot(
    *,
    env_id: int,
    step: int,
    agent_positions: torch.Tensor,
    target_positions: torch.Tensor,
    pre_active: torch.Tensor,
    pre_assigned_target: torch.Tensor,
    post_assigned_target: torch.Tensor,
    detected_unvisited: torch.Tensor,
) -> dict[str, Any] | None:
    """Build an offline-only quality snapshot when a new commitment forms."""
    newly_committed = (
        (pre_assigned_target < 0)
        & (post_assigned_target >= 0)
        & pre_active
    )
    if not bool(newly_committed.any()):
        return None

    pre_claimed = torch.zeros_like(detected_unvisited)
    valid_pre_targets = pre_assigned_target[pre_assigned_target >= 0]
    if valid_pre_targets.numel():
        pre_claimed[valid_pre_targets] = True
    eligible_agents = torch.nonzero(
        pre_active & (pre_assigned_target < 0), as_tuple=False
    ).flatten()
    available_targets = torch.nonzero(
        detected_unvisited & ~pre_claimed, as_tuple=False
    ).flatten()
    eligibility_cost = torch.cdist(
        agent_positions[eligible_agents].unsqueeze(0),
        target_positions[available_targets].unsqueeze(0),
    ).squeeze(0)

    committed_agents = torch.nonzero(
        pre_active & (post_assigned_target >= 0), as_tuple=False
    ).flatten()
    committed_targets = post_assigned_target[committed_agents]
    committed_cost = torch.cdist(
        agent_positions[committed_agents].unsqueeze(0),
        target_positions[committed_targets].unsqueeze(0),
    ).squeeze(0)
    actual_costs = committed_cost.diagonal()
    actual_sum = float(actual_costs.sum())
    actual_makespan = float(actual_costs.max()) if actual_costs.numel() else 0.0
    optimal_sum, optimal_makespan = optimal_assignment_costs(committed_cost)

    crossing_pairs: list[dict[str, Any]] = []
    for left in range(committed_agents.numel()):
        for right in range(left + 1, committed_agents.numel()):
            actual_pair = committed_cost[left, left] + committed_cost[right, right]
            swapped_pair = committed_cost[left, right] + committed_cost[right, left]
            actual_pair_max = torch.maximum(
                committed_cost[left, left], committed_cost[right, right]
            )
            swapped_pair_max = torch.maximum(
                committed_cost[left, right], committed_cost[right, left]
            )
            if float(swapped_pair) + 1e-6 < float(actual_pair) or (
                float(swapped_pair_max) + 1e-6 < float(actual_pair_max)
            ):
                crossing_pairs.append(
                    {
                        "agent_a": int(committed_agents[left]),
                        "target_a": int(committed_targets[left]),
                        "agent_b": int(committed_agents[right]),
                        "target_b": int(committed_targets[right]),
                        "actual_sum": float(actual_pair),
                        "swapped_sum": float(swapped_pair),
                        "actual_makespan": float(actual_pair_max),
                        "swapped_makespan": float(swapped_pair_max),
                    }
                )

    comparative_violations: list[dict[str, Any]] = []
    for agent_id in torch.nonzero(newly_committed, as_tuple=False).flatten().tolist():
        target_id = int(post_assigned_target[agent_id])
        target_matches = torch.nonzero(
            available_targets == target_id, as_tuple=False
        ).flatten()
        agent_matches = torch.nonzero(
            eligible_agents == agent_id, as_tuple=False
        ).flatten()
        if target_matches.numel() == 0 or agent_matches.numel() == 0:
            continue
        column = int(target_matches[0])
        row = int(agent_matches[0])
        assigned_cost = float(eligibility_cost[row, column])
        best_cost = float(eligibility_cost[:, column].min())
        if assigned_cost > best_cost + 1e-6:
            comparative_violations.append(
                {
                    "agent_id": agent_id,
                    "target_id": target_id,
                    "assigned_cost": assigned_cost,
                    "best_eligible_cost": best_cost,
                }
            )

    return {
        "env_id": env_id,
        "step": step,
        "new_agent_ids": torch.nonzero(newly_committed, as_tuple=False)
        .flatten()
        .cpu()
        .tolist(),
        "new_target_ids": post_assigned_target[newly_committed].cpu().tolist(),
        "eligible_agent_ids": eligible_agents.cpu().tolist(),
        "available_target_ids": available_targets.cpu().tolist(),
        "eligibility_cost_matrix": eligibility_cost.cpu().tolist(),
        "committed_agent_ids": committed_agents.cpu().tolist(),
        "committed_target_ids": committed_targets.cpu().tolist(),
        "committed_cost_matrix": committed_cost.cpu().tolist(),
        "actual_sum_cost": actual_sum,
        "actual_makespan": actual_makespan,
        "optimal_sum_cost": optimal_sum,
        "optimal_makespan": optimal_makespan,
        "sum_regret": max(actual_sum - optimal_sum, 0.0),
        "makespan_regret": max(actual_makespan - optimal_makespan, 0.0),
        "crossing_pair_count": len(crossing_pairs),
        "crossing_pairs": crossing_pairs,
        "comparative_advantage_violation_count": len(comparative_violations),
        "comparative_advantage_violations": comparative_violations,
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    all_detected_ablation = (
        args.enable_module_ablation
        and args.ablation_timing_module == "all_detected"
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
        sensor_radius=args.sensor_radius,
        max_steps=args.max_steps,
        retire_on_rescue=args.retire_on_rescue,
        high_level_progress_features=args.progress_features,
        target_assignment_features=args.target_assignment_features,
        staged_rescue=args.staged_rescue or all_detected_ablation,
        rescue_detected_threshold=(
            3 if all_detected_ablation else args.rescue_detected_threshold
        ),
        rescue_entropy_threshold=args.rescue_entropy_threshold,
        dynamic_rescue_release=args.dynamic_rescue_release,
        dynamic_release_min_searchers=args.dynamic_release_min_searchers,
        dynamic_release_entropy_ratio_threshold=(
            args.dynamic_release_entropy_ratio_threshold
        ),
        dynamic_release_entropy_rate_threshold=(
            args.dynamic_release_entropy_rate_threshold
        ),
        dynamic_release_min_stagnation_step=(
            args.dynamic_release_min_stagnation_step
        ),
        dynamic_release_search_steps_per_target=(
            args.dynamic_release_search_steps_per_target
        ),
        dynamic_release_speed_per_step=args.dynamic_release_speed_per_step,
        dynamic_release_time_margin=args.dynamic_release_time_margin,
        dynamic_release_max_new_agents_per_event=(
            args.dynamic_release_max_new_agents_per_event
        ),
        dynamic_rescue_only_after_all_detected=(
            args.dynamic_rescue_only_after_all_detected
        ),
        redecide_on_detection_change=args.redecide_on_detection_change,
        redecide_on_assignment_change=args.redecide_on_assignment_change,
        enable_finder_first_cascade=args.enable_finder_first_cascade,
        finder_cascade_mode=args.finder_cascade_mode,
        early_rescue_penalty=args.early_rescue_penalty,
        early_rescue_detected_threshold=args.early_rescue_detected_threshold,
        search_capacity_discovery_bonus=args.search_capacity_discovery_bonus,
        discovery_active_agents_threshold=args.discovery_active_agents_threshold,
        all_targets_detected_bonus=args.all_targets_detected_bonus,
        all_targets_detected_bonus_requires_no_rescue=args.all_targets_detected_bonus_requires_no_rescue,
        rescue_phase_explore_penalty=args.rescue_phase_explore_penalty,
        rescue_phase_detected_threshold=args.rescue_phase_detected_threshold,
        rescue_phase_min_search_agents=args.rescue_phase_min_search_agents,
        unique_rescue_assignment_bonus=args.unique_rescue_assignment_bonus,
        duplicate_rescue_assignment_penalty=args.duplicate_rescue_assignment_penalty,
        detected_unassigned_target_penalty=args.detected_unassigned_target_penalty,
    )
    scenario = env.scenario
    low_policy = (
        BenchMARLLowLevelPolicy(
            args.low_level_checkpoint,
            device=args.device,
            seed=args.seed,
            deterministic=True,
            max_steps=scenario.max_steps,
            enable_goal_reaching_fallback=args.enable_low_level_goal_fallback,
            goal_near_threshold=args.low_level_goal_near_threshold,
            fallback_proportional_gain=args.low_level_fallback_gain,
            fallback_stagnation_steps=args.low_level_fallback_stagnation_steps,
            fallback_progress_epsilon=args.low_level_fallback_progress_epsilon,
        )
        if args.low_level_controller == "checkpoint"
        else None
    )
    high_policy = build_policy(args, scenario)
    collector = AsyncSMDPCollector(
        env,
        high_level_policy=high_policy,
        low_level_policy=low_policy,
        coordinated_target_selection=args.coordinated_target_selection,
    )
    coverage_tracker = (
        CoverageTrajectoryTracker(scenario)
        if args.enable_coverage_metrics else None
    )

    n_envs = args.num_envs
    device = scenario.world.device
    discovered_first_step = torch.full((n_envs, scenario.n_targets), -1, dtype=torch.long, device=device)
    visited_first_step = torch.full((n_envs, scenario.n_targets), -1, dtype=torch.long, device=device)
    first_visit_retired_agents = torch.full((n_envs, scenario.n_targets), -1, dtype=torch.long, device=device)
    per_env_explore_actions = torch.zeros(n_envs, device=device)
    per_env_target_actions = torch.zeros(n_envs, device=device)
    per_env_goal_hits = torch.zeros(n_envs, device=device)
    per_env_active_steps = torch.zeros(n_envs, device=device)
    per_env_goal_dist_sum = torch.zeros(n_envs, device=device)
    per_env_goal_dist_count = torch.zeros(n_envs, device=device)
    per_env_duplicate_assignment_steps = torch.zeros(n_envs, device=device)
    per_env_duplicate_assignment_excess = torch.zeros(n_envs, device=device)
    per_env_detected_unassigned_steps = torch.zeros(n_envs, device=device)
    per_env_switch_away_actions = torch.zeros(n_envs, device=device)
    first_assignment_step = torch.full((n_envs, scenario.n_targets), -1, dtype=torch.long, device=device)
    per_env_steps = torch.zeros(n_envs, dtype=torch.long, device=device)
    first_detection_step = torch.full((n_envs,), -1, dtype=torch.long, device=device)
    second_detection_step = torch.full_like(first_detection_step, -1)
    all_detected_step = torch.full_like(first_detection_step, -1)
    first_target_action_step = torch.full_like(first_detection_step, -1)
    first_target_action_after_all_detected_step = torch.full_like(first_detection_step, -1)
    first_rescue_commitment_after_second_detection_step = torch.full_like(
        first_detection_step, -1
    )
    searchers_at_first_detection = torch.full_like(first_detection_step, -1)
    searchers_at_second_detection = torch.full_like(first_detection_step, -1)
    all_detected_remaining_steps = torch.full_like(first_detection_step, -1)
    rescue_workload_steps_at_all_detected = torch.full(
        (n_envs,), torch.nan, device=device
    )
    nearest_target_distance_at_all_detected = torch.full(
        (n_envs,), torch.nan, device=device
    )
    assignment_distance = torch.full(
        (n_envs, scenario.n_targets), torch.nan, device=device
    )
    pre_all_detected_unassigned_steps = torch.zeros(n_envs, device=device)
    post_all_detected_unassigned_steps = torch.zeros(n_envs, device=device)
    pre_all_entropy_reduction = torch.zeros(n_envs, device=device)
    per_agent_pre_all_entropy_reduction = torch.zeros(
        n_envs, scenario.n_agents, device=device
    )
    premature_retirement = torch.zeros(n_envs, dtype=torch.bool, device=device)
    post_all_explore_actions = torch.zeros(n_envs, device=device)
    post_all_decisions = torch.zeros(n_envs, device=device)
    active_searcher_trace = torch.full(
        (n_envs, args.steps), -1, dtype=torch.long, device=device
    )
    committed_rescuer_trace = torch.full_like(active_searcher_trace, -1)
    entropy_trace = torch.full(
        (n_envs, args.steps), torch.nan, device=device
    )
    gate_open_trace = torch.full_like(active_searcher_trace, -1)
    gate_events: list[list[dict[str, Any]]] = [[] for _ in range(n_envs)]
    assignment_quality_events: list[list[dict[str, Any]]] = [
        [] for _ in range(n_envs)
    ]
    previous_gate_open = torch.zeros(n_envs, dtype=torch.bool, device=device)
    seen_decisions: set[tuple[int, int, int]] = set()
    nonfinite_state_count = torch.zeros(n_envs, dtype=torch.long, device=device)
    invalid_assignment_count = torch.zeros(n_envs, dtype=torch.long, device=device)

    try:
        # Policy/checkpoint construction consumes global PyTorch RNG. Reseed
        # immediately before the first environment reset so paired controller
        # and gating evaluations receive exactly the same initial layouts.
        torch.manual_seed(args.seed)
        collector.reset()
        if coverage_tracker is not None:
            coverage_tracker.start()
        initial_agents = stack_agent_positions(scenario).detach().clone().cpu().contiguous()
        initial_targets = stack_target_positions(scenario).detach().clone().cpu().contiguous()
        initial_layout = torch.cat([initial_agents, initial_targets], dim=1)
        layout_sha256 = hashlib.sha256(initial_layout.numpy().tobytes()).hexdigest()
        per_env_layout_sha256 = [
            hashlib.sha256(initial_layout[env_id].numpy().tobytes()).hexdigest()
            for env_id in range(n_envs)
        ]
        entropy_initial = scenario._compute_entropy(scenario.belief_maps).sum(dim=(-1, -2, -3)).detach().clone()
        for step in range(args.steps):
            pre_detected = scenario.target_detected.any(dim=1).detach().clone()
            pre_visited = scenario.target_visited.detach().clone()
            pre_goal_done = scenario.goal_done.detach().clone()
            pre_active = scenario.active_agents.detach().clone()
            pre_positions = stack_agent_positions(scenario).detach().clone()
            pre_search_mask = pre_active & ~scenario.assigned_tasks[..., 0].bool()

            pre_goals = scenario.assigned_goals.detach().clone()
            pre_assigned_target = assigned_target_indices(scenario).detach().clone()
            pre_entropy_by_agent = scenario._compute_entropy(
                scenario.belief_maps
            ).sum(dim=(-1, -2)).detach().clone()
            dones = collector.step()

            # Attribute finalized high-level actions to envs since last check.
            for transition in collector.transitions:
                if getattr(transition, "_counted", False):
                    continue
                env_id = int(transition.env_id)
                if int(transition.action.item()) >= scenario.rrt_top_k:
                    per_env_target_actions[env_id] += 1
                else:
                    per_env_explore_actions[env_id] += 1
                setattr(transition, "_counted", True)

            post_pos = torch.stack([agent.state.pos for agent in scenario.world.agents], dim=1)
            nonfinite_state_count += (
                ~torch.isfinite(post_pos).all(dim=-1)
            ).sum(dim=-1)
            if coverage_tracker is not None:
                coverage_tracker.update(
                    pre_positions=pre_positions,
                    post_positions=post_pos,
                    pre_search_mask=pre_search_mask,
                )

            pre_pending = pre_active & ~pre_goal_done
            dist = torch.linalg.vector_norm(post_pos - pre_goals, dim=-1)
            hit = (dist <= scenario.goal_radius) & pre_pending
            per_env_goal_hits += hit.float().sum(dim=1)
            per_env_active_steps += pre_pending.float().sum(dim=1)
            per_env_goal_dist_sum += torch.where(pre_pending, dist, torch.zeros_like(dist)).sum(dim=1)
            per_env_goal_dist_count += pre_pending.float().sum(dim=1)

            detected_now = scenario.target_detected.any(dim=1)
            newly_detected = detected_now & ~pre_detected
            newly_visited = scenario.target_visited & ~pre_visited
            detected_count = detected_now.sum(dim=-1)
            active_collect = scenario.active_agents & scenario.assigned_tasks[..., 0].bool()
            active_search = scenario.active_agents & ~active_collect
            active_searcher_trace[:, step] = active_search.sum(dim=-1)
            committed_rescuer_trace[:, step] = active_collect.sum(dim=-1)
            entropy_now_by_agent = scenario._compute_entropy(
                scenario.belief_maps
            ).sum(dim=(-1, -2))
            entropy_trace[:, step] = entropy_now_by_agent.sum(dim=-1)

            first_now = (detected_count >= 1) & (first_detection_step < 0)
            second_now = (detected_count >= 2) & (second_detection_step < 0)
            all_now = (detected_count >= scenario.n_targets) & (all_detected_step < 0)
            step_value = torch.full_like(first_detection_step, step + 1)
            first_detection_step = torch.where(first_now, step_value, first_detection_step)
            second_detection_step = torch.where(second_now, step_value, second_detection_step)
            all_detected_step = torch.where(all_now, step_value, all_detected_step)
            searchers_at_first_detection = torch.where(
                first_now, active_search.sum(dim=-1), searchers_at_first_detection
            )
            searchers_at_second_detection = torch.where(
                second_now, active_search.sum(dim=-1), searchers_at_second_detection
            )

            pre_all_mask = ~pre_detected.all(dim=-1)
            entropy_delta_by_agent = (pre_entropy_by_agent - entropy_now_by_agent).clamp_min(0)
            pre_all_entropy_reduction += (
                entropy_delta_by_agent.sum(dim=-1) * pre_all_mask.float()
            )
            per_agent_pre_all_entropy_reduction += (
                entropy_delta_by_agent * pre_all_mask.unsqueeze(-1).float()
            )
            newly_retired = pre_active & ~scenario.active_agents
            premature_retirement |= newly_retired.any(dim=-1) & ~detected_now.all(dim=-1)

            if args.dynamic_rescue_release:
                dynamic_allowed = getattr(
                    scenario,
                    "dynamic_release_agent_allowed",
                    torch.zeros_like(scenario.active_agents),
                )
                gate_open = dynamic_allowed.any(dim=-1)
                reason_codes = getattr(
                    scenario,
                    "dynamic_release_reason_code",
                    torch.zeros(n_envs, dtype=torch.long, device=device),
                )
                reason_names = {
                    0: "no_known_target",
                    1: "preserve_searchers",
                    2: "time_pressure",
                    3: "low_unknown",
                    4: "coverage_stagnation",
                    5: "all_detected",
                }
                gate_reason = [
                    reason_names.get(int(reason_codes[env_id]), "unknown")
                    for env_id in range(n_envs)
                ]
            elif args.staged_rescue:
                gate_open = detected_count >= args.rescue_detected_threshold
                if args.rescue_entropy_threshold is not None:
                    entropy_mean = scenario._compute_entropy(
                        scenario.belief_maps
                    ).mean(dim=(-1, -2, -3))
                    gate_open |= entropy_mean <= args.rescue_entropy_threshold
                gate_reason = [
                    "detected_threshold" if bool(gate_open[env_id]) else "staged_closed"
                    for env_id in range(n_envs)
                ]
            else:
                gate_open = torch.ones(n_envs, dtype=torch.bool, device=device)
                gate_reason = ["no_gate"] * n_envs
            gate_open_trace[:, step] = gate_open.long()
            changed_gate = gate_open != previous_gate_open
            for env_id in torch.nonzero(changed_gate, as_tuple=False).flatten().tolist():
                gate_events[env_id].append(
                    {
                        "step": step + 1,
                        "open": bool(gate_open[env_id]),
                        "reason": gate_reason[env_id],
                        "detected_count": int(detected_count[env_id]),
                        "active_searchers": int(active_search[env_id].sum()),
                        "committed_rescuers": int(active_collect[env_id].sum()),
                    }
                )
            previous_gate_open = gate_open

            if bool(all_now.any()):
                current_agent_pos = stack_agent_positions(scenario)
                current_target_pos = stack_target_positions(scenario)
                for env_id in torch.nonzero(all_now, as_tuple=False).flatten().tolist():
                    remaining = scenario.max_steps - (step + 1)
                    all_detected_remaining_steps[env_id] = remaining
                    unvisited = ~scenario.target_visited[env_id]
                    rescue_workload_steps_at_all_detected[env_id] = (
                        estimated_parallel_rescue_steps(
                            current_agent_pos[env_id],
                            scenario.active_agents[env_id],
                            current_target_pos[env_id],
                            unvisited,
                            args.diagnostic_speed_per_step,
                        )
                    )
                    if bool(unvisited.any()) and bool(scenario.active_agents[env_id].any()):
                        distances = torch.cdist(
                            current_agent_pos[env_id, scenario.active_agents[env_id]].unsqueeze(0),
                            current_target_pos[env_id, unvisited].unsqueeze(0),
                        ).squeeze(0)
                        nearest_target_distance_at_all_detected[env_id] = distances.min()

            post_assigned_target = assigned_target_indices(scenario)
            invalid_assignment_count += (
                scenario.assigned_tasks[..., 0].bool()
                & (post_assigned_target < 0)
            ).sum(dim=-1)
            current_agent_pos = stack_agent_positions(scenario)
            current_target_pos = stack_target_positions(scenario)
            detected_unvisited_now = detected_now & ~scenario.target_visited
            for env_id in range(n_envs):
                event = assignment_quality_snapshot(
                    env_id=env_id,
                    step=step + 1,
                    agent_positions=current_agent_pos[env_id],
                    target_positions=current_target_pos[env_id],
                    pre_active=pre_active[env_id],
                    pre_assigned_target=pre_assigned_target[env_id],
                    post_assigned_target=post_assigned_target[env_id],
                    detected_unvisited=detected_unvisited_now[env_id],
                )
                if event is not None:
                    assignment_quality_events[env_id].append(event)
            assigned_one_hot = torch.nn.functional.one_hot(
                post_assigned_target.clamp_min(0),
                num_classes=scenario.n_targets,
            ).bool()
            assigned_one_hot = assigned_one_hot & (post_assigned_target >= 0).unsqueeze(-1)
            claim_counts = assigned_one_hot.sum(dim=1)
            duplicate_excess = (claim_counts - 1).clamp_min(0).float().sum(dim=-1)
            per_env_duplicate_assignment_excess += duplicate_excess
            per_env_duplicate_assignment_steps += (duplicate_excess > 0).float()

            known_unvisited = detected_now & ~scenario.target_visited
            unassigned_count = (known_unvisited & (claim_counts == 0)).float().sum(dim=-1)
            per_env_detected_unassigned_steps += unassigned_count
            detected_all_now = detected_now.all(dim=-1)
            pre_all_detected_unassigned_steps += unassigned_count * (~detected_all_now).float()
            post_all_detected_unassigned_steps += unassigned_count * detected_all_now.float()
            newly_assigned_targets = (claim_counts > 0) & detected_now & (first_assignment_step < 0)
            first_assignment_step = torch.where(
                newly_assigned_targets,
                torch.full_like(first_assignment_step, step + 1),
                first_assignment_step,
            )
            if bool(newly_assigned_targets.any()):
                current_agent_pos = stack_agent_positions(scenario)
                current_target_pos = stack_target_positions(scenario)
                agent_target_dist = torch.cdist(current_agent_pos, current_target_pos)
                claimed_dist = agent_target_dist.masked_fill(~assigned_one_hot, torch.inf)
                minimum_claimed_dist = claimed_dist.min(dim=1).values
                assignment_distance = torch.where(
                    newly_assigned_targets,
                    minimum_claimed_dist,
                    assignment_distance,
                )

            for (env_id, agent_id), pending in collector._pending.items():
                decision_start = int(pending["start_t"])
                signature = (env_id, agent_id, decision_start)
                if signature in seen_decisions:
                    continue
                seen_decisions.add(signature)
                action = int(pending["action"].item())
                is_target_action = action >= scenario.rrt_top_k
                if is_target_action:
                    if first_target_action_step[env_id] < 0:
                        first_target_action_step[env_id] = decision_start
                    if (
                        all_detected_step[env_id] >= 0
                        and decision_start >= int(all_detected_step[env_id])
                        and first_target_action_after_all_detected_step[env_id] < 0
                    ):
                        first_target_action_after_all_detected_step[env_id] = decision_start
                    if (
                        second_detection_step[env_id] >= 0
                        and decision_start >= int(second_detection_step[env_id])
                        and first_rescue_commitment_after_second_detection_step[env_id] < 0
                    ):
                        first_rescue_commitment_after_second_detection_step[env_id] = decision_start
                if (
                    all_detected_step[env_id] >= 0
                    and decision_start >= int(all_detected_step[env_id])
                ):
                    post_all_decisions[env_id] += 1
                    if not is_target_action:
                        post_all_explore_actions[env_id] += 1

            changed_assignment = (
                (pre_assigned_target >= 0)
                & (post_assigned_target != pre_assigned_target)
                & pre_active
            )
            pre_target_one_hot = torch.nn.functional.one_hot(
                pre_assigned_target.clamp_min(0),
                num_classes=scenario.n_targets,
            ).bool()
            pre_target_one_hot = pre_target_one_hot & (pre_assigned_target >= 0).unsqueeze(-1)
            old_target_still_unvisited = (pre_target_one_hot & ~scenario.target_visited.unsqueeze(1)).any(dim=-1)
            per_env_switch_away_actions += (changed_assignment & old_target_still_unvisited).float().sum(dim=-1)
            if newly_detected.any():
                discovered_first_step = torch.where(
                    newly_detected & (discovered_first_step < 0),
                    torch.full_like(discovered_first_step, step + 1),
                    discovered_first_step,
                )
            if newly_visited.any():
                retired_count = (~scenario.active_agents).float().sum(dim=-1).long().unsqueeze(-1).expand_as(visited_first_step)
                visited_first_step = torch.where(
                    newly_visited & (visited_first_step < 0),
                    torch.full_like(visited_first_step, step + 1),
                    visited_first_step,
                )
                first_visit_retired_agents = torch.where(
                    newly_visited & (first_visit_retired_agents < 0),
                    retired_count,
                    first_visit_retired_agents,
                )
            per_env_steps += (~dones).long()
            if bool(dones.all()):
                break

        entropy_final = scenario._compute_entropy(scenario.belief_maps).sum(dim=(-1, -2, -3))
        coverage_rows = (
            coverage_tracker.rows() if coverage_tracker is not None else None
        )
        rows = build_rows(
            args=args,
            scenario=scenario,
            entropy_initial=entropy_initial,
            entropy_final=entropy_final,
            discovered_first_step=discovered_first_step,
            visited_first_step=visited_first_step,
            first_visit_retired_agents=first_visit_retired_agents,
            per_env_explore_actions=per_env_explore_actions,
            per_env_target_actions=per_env_target_actions,
            per_env_goal_hits=per_env_goal_hits,
            per_env_active_steps=per_env_active_steps,
            per_env_goal_dist_sum=per_env_goal_dist_sum,
            per_env_goal_dist_count=per_env_goal_dist_count,
            per_env_duplicate_assignment_steps=per_env_duplicate_assignment_steps,
            per_env_duplicate_assignment_excess=per_env_duplicate_assignment_excess,
            per_env_detected_unassigned_steps=per_env_detected_unassigned_steps,
            per_env_switch_away_actions=per_env_switch_away_actions,
            first_assignment_step=first_assignment_step,
            per_env_steps=per_env_steps,
            first_detection_step=first_detection_step,
            second_detection_step=second_detection_step,
            all_detected_step=all_detected_step,
            first_target_action_step=first_target_action_step,
            first_target_action_after_all_detected_step=(
                first_target_action_after_all_detected_step
            ),
            first_rescue_commitment_after_second_detection_step=(
                first_rescue_commitment_after_second_detection_step
            ),
            searchers_at_first_detection=searchers_at_first_detection,
            searchers_at_second_detection=searchers_at_second_detection,
            all_detected_remaining_steps=all_detected_remaining_steps,
            rescue_workload_steps_at_all_detected=rescue_workload_steps_at_all_detected,
            nearest_target_distance_at_all_detected=nearest_target_distance_at_all_detected,
            assignment_distance=assignment_distance,
            pre_all_detected_unassigned_steps=pre_all_detected_unassigned_steps,
            post_all_detected_unassigned_steps=post_all_detected_unassigned_steps,
            pre_all_entropy_reduction=pre_all_entropy_reduction,
            per_agent_pre_all_entropy_reduction=per_agent_pre_all_entropy_reduction,
            premature_retirement=premature_retirement,
            post_all_explore_actions=post_all_explore_actions,
            post_all_decisions=post_all_decisions,
            active_searcher_trace=active_searcher_trace,
            committed_rescuer_trace=committed_rescuer_trace,
            entropy_trace=entropy_trace,
            gate_open_trace=gate_open_trace,
            gate_events=gate_events,
            assignment_quality_events=assignment_quality_events,
            finder_cascade_events=collector.cascade_events,
            layout_sha256=layout_sha256,
            per_env_layout_sha256=per_env_layout_sha256,
            initial_agents=initial_agents,
            initial_targets=initial_targets,
            coverage_rows=coverage_rows,
            nonfinite_state_count=nonfinite_state_count,
            invalid_assignment_count=invalid_assignment_count,
        )
        summary = summarize(rows, scenario.n_targets)
        summary["layout_sha256"] = layout_sha256
        policy_metrics = getattr(high_policy, "metrics", None)
        if policy_metrics is not None:
            summary.update(policy_metrics())
        return rows, summary
    finally:
        if low_policy is not None:
            low_policy.close()
        if hasattr(env, "close"):
            env.close()


def assigned_target_indices(scenario: Any) -> torch.Tensor:
    target_pos = torch.stack([target.state.pos for target in scenario.targets], dim=1)
    collect_task = scenario.assigned_tasks[..., 0] > 0.5
    dists = torch.cdist(scenario.assigned_goals, target_pos)
    min_dist, index = dists.min(dim=-1)
    return torch.where(
        collect_task & (min_dist <= scenario.goal_radius),
        index.long(),
        torch.full_like(index.long(), -1),
    )


def build_rows(**kwargs: Any) -> list[dict[str, float | int | str]]:
    args = kwargs["args"]
    scenario = kwargs["scenario"]
    detected_any = scenario.target_detected.any(dim=1)
    visited = scenario.target_visited
    retired = (~scenario.active_agents).float().sum(dim=-1)
    active = scenario.active_agents.float().sum(dim=-1)
    target_pos = torch.stack([target.state.pos for target in scenario.targets], dim=1)
    agent_pos = torch.stack([agent.state.pos for agent in scenario.world.agents], dim=1)
    min_active_target_dist = torch.cdist(agent_pos, target_pos).masked_fill(~scenario.active_agents.unsqueeze(-1), torch.inf).min(dim=1).values

    rows = []
    for env_id in range(args.num_envs):
        detected_count = int(detected_any[env_id].sum().item())
        visited_count = int(visited[env_id].sum().item())
        success = int(visited_count == scenario.n_targets)
        detected_unvisited = int((detected_any[env_id] & ~visited[env_id]).sum().item())
        undiscovered = int((~detected_any[env_id]).sum().item())
        active_count = int(active[env_id].item())
        retired_count = int(retired[env_id].item())
        all_step = int(kwargs["all_detected_step"][env_id].item())
        remaining_at_all = int(kwargs["all_detected_remaining_steps"][env_id].item())
        workload_at_all = float(
            kwargs["rescue_workload_steps_at_all_detected"][env_id].cpu()
        )
        post_all_decisions = float(kwargs["post_all_decisions"][env_id].cpu())
        post_all_explore = float(kwargs["post_all_explore_actions"][env_id].cpu())
        post_all_explore_ratio = post_all_explore / max(post_all_decisions, 1.0)
        first_after_all = int(
            kwargs["first_target_action_after_all_detected_step"][env_id].item()
        )
        all_to_target_delay = (
            first_after_all - all_step
            if first_after_all >= 0 and all_step >= 0
            else float("nan")
        )
        second_step = int(kwargs["second_detection_step"][env_id].item())
        first_after_second = int(
            kwargs["first_rescue_commitment_after_second_detection_step"][env_id].item()
        )
        second_to_rescue_delay = (
            first_after_second - second_step
            if first_after_second >= 0 and second_step >= 0
            else float("nan")
        )
        theoretical_late = (
            all_step >= 0
            and math.isfinite(workload_at_all)
            and workload_at_all > remaining_at_all
        )
        if success:
            reason = "success"
        elif undiscovered > 0 and detected_unvisited == 0:
            reason = "undiscovered_remaining"
        elif detected_unvisited > 0 and active_count > 0:
            reason = "detected_unvisited_remaining"
        elif detected_unvisited > 0 and active_count == 0:
            reason = "no_active_agents_for_detected_target"
        elif undiscovered > 0 and active_count == 0:
            reason = "no_active_agents_for_search"
        else:
            reason = "other_timeout"
        if success:
            failure_category = "success"
        elif all_step < 0:
            failure_category = "1_not_all_targets_detected"
        elif theoretical_late:
            failure_category = "2_all_detected_too_late"
        elif (
            post_all_explore_ratio >= 0.5
            or (all_to_target_delay == all_to_target_delay and all_to_target_delay > 10)
        ):
            failure_category = "3_post_detection_over_exploration"
        elif float(kwargs["per_env_duplicate_assignment_excess"][env_id].cpu()) > 0:
            failure_category = "5_target_assignment_conflict"
        elif args.low_level_controller == "checkpoint":
            failure_category = "6_residual_low_level_failure"
        else:
            failure_category = "4_assigned_but_execution_timeout"

        trace_length = int(
            (kwargs["active_searcher_trace"][env_id] >= 0).sum().item()
        )
        assignment_distances = kwargs["assignment_distance"][env_id].cpu()
        estimated_arrival = assignment_distances / max(
            args.diagnostic_speed_per_step, 1e-6
        )
        actual_arrival = (
            kwargs["visited_first_step"][env_id].float()
            - kwargs["first_assignment_step"][env_id].float()
        )
        valid_actual = (
            (kwargs["visited_first_step"][env_id] >= 0)
            & (kwargs["first_assignment_step"][env_id] >= 0)
        )
        actual_arrival = torch.where(
            valid_actual,
            actual_arrival,
            torch.full_like(actual_arrival, torch.nan),
        )
        quality_events = kwargs["assignment_quality_events"][env_id]
        cascade_events = kwargs["finder_cascade_events"][env_id]
        finder_opens = [
            event for event in cascade_events if event["event"] == "finder_open"
        ]
        finder_accepts = [
            event for event in cascade_events if event["event"] == "accept"
        ]
        finder_rejects = [
            event for event in cascade_events if event["event"] == "reject"
        ]
        finder_unavailable = [
            event
            for event in cascade_events
            if event["event"] == "finder_unavailable"
        ]
        cascade_diffusions = [
            event for event in cascade_events if event["event"] == "diffuse"
        ]
        open_step_by_target = {
            int(event["target_id"]): int(event["step"])
            for event in finder_opens
        }
        commitment_delays = [
            int(event["step"]) - open_step_by_target[int(event["target_id"])]
            for event in finder_accepts
            if int(event["target_id"]) in open_step_by_target
        ]
        finder_commitment_delays = [
            int(event["step"]) - open_step_by_target[int(event["target_id"])]
            for event in finder_accepts
            if bool(event["committer_is_finder"])
            and int(event["target_id"]) in open_step_by_target
        ]
        rejected_target_waits: list[int] = []
        rejected_target_later_claimed_count = 0
        for reject_index, reject_event in enumerate(cascade_events):
            if reject_event["event"] != "reject":
                continue
            target_id = int(reject_event["target_id"])
            later_accept = next(
                (
                    event
                    for event in cascade_events[reject_index + 1 :]
                    if event["event"] == "accept"
                    and int(event["target_id"]) == target_id
                ),
                None,
            )
            if later_accept is not None:
                rejected_target_later_claimed_count += 1
                rejected_target_waits.append(
                    int(later_accept["step"]) - int(reject_event["step"])
                )
        crossing_pair_count = sum(
            int(event["crossing_pair_count"]) for event in quality_events
        )
        comparative_violation_count = sum(
            int(event["comparative_advantage_violation_count"])
            for event in quality_events
        )
        rows.append(
            {
                "env_id": env_id,
                "layout_sha256": kwargs["per_env_layout_sha256"][env_id],
                "initial_agent_positions": kwargs["initial_agents"][env_id].tolist(),
                "initial_target_positions": kwargs["initial_targets"][env_id].tolist(),
                "success": success,
                "failure_reason": reason,
                "failure_category": failure_category,
                "visited_targets": visited_count,
                "detected_targets": detected_count,
                "undiscovered_targets": undiscovered,
                "detected_unvisited_targets": detected_unvisited,
                "retired_agents": retired_count,
                "active_agents": active_count,
                "nonfinite_state_count": int(
                    kwargs["nonfinite_state_count"][env_id].cpu()
                ),
                "invalid_assignment_count": int(
                    kwargs["invalid_assignment_count"][env_id].cpu()
                ),
                "explore_actions": float(kwargs["per_env_explore_actions"][env_id].cpu()),
                "target_actions": float(kwargs["per_env_target_actions"][env_id].cpu()),
                "goal_hits": float(kwargs["per_env_goal_hits"][env_id].cpu()),
                "active_step_hit_rate": float((kwargs["per_env_goal_hits"][env_id] / kwargs["per_env_active_steps"][env_id].clamp_min(1)).cpu()),
                "mean_goal_distance": float((kwargs["per_env_goal_dist_sum"][env_id] / kwargs["per_env_goal_dist_count"][env_id].clamp_min(1)).cpu()),
                "duplicate_assignment_steps": float(kwargs["per_env_duplicate_assignment_steps"][env_id].cpu()),
                "duplicate_assignment_excess": float(kwargs["per_env_duplicate_assignment_excess"][env_id].cpu()),
                "detected_unassigned_steps": float(kwargs["per_env_detected_unassigned_steps"][env_id].cpu()),
                "pre_all_detected_unassigned_steps": float(
                    kwargs["pre_all_detected_unassigned_steps"][env_id].cpu()
                ),
                "post_all_detected_unassigned_steps": float(
                    kwargs["post_all_detected_unassigned_steps"][env_id].cpu()
                ),
                "switch_away_actions": float(kwargs["per_env_switch_away_actions"][env_id].cpu()),
                "first_assignment_step_mean": mean_positive(kwargs["first_assignment_step"][env_id]),
                "discovery_to_assignment_delay_mean": mean_pair_delay(
                    kwargs["discovered_first_step"][env_id],
                    kwargs["first_assignment_step"][env_id],
                ),
                "assignment_to_visit_delay_mean": mean_pair_delay(
                    kwargs["first_assignment_step"][env_id],
                    kwargs["visited_first_step"][env_id],
                ),
                "assignment_quality_event_count": len(quality_events),
                "crossing_event_count": sum(
                    int(int(event["crossing_pair_count"]) > 0)
                    for event in quality_events
                ),
                "crossing_pair_count": crossing_pair_count,
                "assignment_sum_regret": sum(
                    float(event["sum_regret"]) for event in quality_events
                ),
                "assignment_makespan_regret": sum(
                    float(event["makespan_regret"]) for event in quality_events
                ),
                "comparative_advantage_violation_count": comparative_violation_count,
                "assignment_quality_events": quality_events,
                "finder_cascade_events": cascade_events,
                "finder_event_count": len(finder_opens),
                "finder_nearest_count": sum(
                    int(bool(event["finder_is_nearest"]))
                    for event in finder_opens
                ),
                "finder_accept_count": sum(
                    int(bool(event["committer_is_finder"]))
                    for event in finder_accepts
                ),
                "finder_reject_count": len(finder_rejects),
                "finder_unavailable_count": len(finder_unavailable),
                "cascade_diffusion_count": len(cascade_diffusions),
                "rejected_target_later_claimed_count": (
                    rejected_target_later_claimed_count
                ),
                "rejected_target_wait_steps": rejected_target_waits,
                "rejected_target_wait_mean": (
                    sum(rejected_target_waits) / len(rejected_target_waits)
                    if rejected_target_waits
                    else float("nan")
                ),
                "final_committer_is_finder_count": sum(
                    int(bool(event["committer_is_finder"]))
                    for event in finder_accepts
                ),
                "cascade_open_to_commitment_delay_mean": (
                    sum(commitment_delays) / len(commitment_delays)
                    if commitment_delays
                    else float("nan")
                ),
                "finder_to_commitment_delay_mean": (
                    sum(finder_commitment_delays) / len(finder_commitment_delays)
                    if finder_commitment_delays
                    else float("nan")
                ),
                "entropy_reduction": float((kwargs["entropy_initial"][env_id] - kwargs["entropy_final"][env_id]).cpu()),
                "pre_all_detected_entropy_reduction": float(
                    kwargs["pre_all_entropy_reduction"][env_id].cpu()
                ),
                "per_agent_pre_all_entropy_reduction": kwargs[
                    "per_agent_pre_all_entropy_reduction"
                ][env_id].cpu().tolist(),
                "steps": int(kwargs["per_env_steps"][env_id].cpu()),
                "first_detection_step": int(kwargs["first_detection_step"][env_id].cpu()),
                "second_detection_step": second_step,
                "all_detected_step": all_step,
                "all_detected_remaining_steps": remaining_at_all,
                "searchers_at_first_detection": int(
                    kwargs["searchers_at_first_detection"][env_id].cpu()
                ),
                "searchers_at_second_detection": int(
                    kwargs["searchers_at_second_detection"][env_id].cpu()
                ),
                "premature_retirement": int(
                    kwargs["premature_retirement"][env_id].cpu()
                ),
                "first_target_action_step": int(
                    kwargs["first_target_action_step"][env_id].cpu()
                ),
                "first_target_action_after_all_detected_step": first_after_all,
                "all_detected_to_first_target_action_delay": all_to_target_delay,
                "second_detection_to_first_rescue_commitment_delay": second_to_rescue_delay,
                "post_all_detected_explore_actions": post_all_explore,
                "post_all_detected_decisions": post_all_decisions,
                "post_all_detected_explore_ratio": post_all_explore_ratio,
                "rescue_workload_steps_at_all_detected": workload_at_all,
                "nearest_target_distance_at_all_detected": float(
                    kwargs["nearest_target_distance_at_all_detected"][env_id].cpu()
                ),
                "assignment_distance_by_target": assignment_distances.tolist(),
                "estimated_arrival_steps_by_target": estimated_arrival.tolist(),
                "actual_arrival_steps_by_target": actual_arrival.cpu().tolist(),
                "remaining_unvisited_target_distances": min_active_target_dist[
                    env_id
                ][~visited[env_id]].cpu().tolist(),
                "active_searcher_trace": kwargs["active_searcher_trace"][
                    env_id, :trace_length
                ].cpu().tolist(),
                "committed_rescuer_trace": kwargs["committed_rescuer_trace"][
                    env_id, :trace_length
                ].cpu().tolist(),
                "entropy_trace": kwargs["entropy_trace"][
                    env_id, :trace_length
                ].cpu().tolist(),
                "gate_open_trace": kwargs["gate_open_trace"][
                    env_id, :trace_length
                ].cpu().tolist(),
                "gate_events": kwargs["gate_events"][env_id],
                "first_discovery_step_mean": mean_positive(kwargs["discovered_first_step"][env_id]),
                "first_visit_step_mean": mean_positive(kwargs["visited_first_step"][env_id]),
                "last_visit_step": int(kwargs["visited_first_step"][env_id].max().cpu()),
                "success_step": int(kwargs["visited_first_step"][env_id].max().cpu()) if success else -1,
                "min_active_distance_to_unvisited_target": finite_mean(min_active_target_dist[env_id][~visited[env_id]]),
            }
        )
        if kwargs["coverage_rows"] is not None:
            rows[-1].update(kwargs["coverage_rows"][env_id])
    return rows


def mean_positive(values: torch.Tensor) -> float:
    valid = values[values >= 0].float()
    return float(valid.mean().cpu()) if valid.numel() else float("nan")


def finite_mean(values: torch.Tensor) -> float:
    valid = values[torch.isfinite(values)].float()
    return float(valid.mean().cpu()) if valid.numel() else float("nan")


def mean_pair_delay(start: torch.Tensor, end: torch.Tensor) -> float:
    valid = (start >= 0) & (end >= 0)
    if not bool(valid.any()):
        return float("nan")
    return float((end[valid].float() - start[valid].float()).mean().cpu())


def summarize(rows: list[dict[str, Any]], n_targets: int) -> dict[str, Any]:
    total = len(rows)
    failures = [r for r in rows if not int(r["success"])]
    partial = [r for r in failures if int(r["visited_targets"]) > 0]
    summary: dict[str, float | int | str] = {
        "episodes": total,
        "success_rate": sum(int(r["success"]) for r in rows) / max(total, 1),
        "success_step_mean": avg([r for r in rows if int(r["success"])], "success_step"),
        "failure_rate": len(failures) / max(total, 1),
        "partial_failure_rate": len(partial) / max(total, 1),
        "visited_targets_mean": avg(rows, "visited_targets"),
        "detected_targets_mean": avg(rows, "detected_targets"),
        "retired_agents_mean": avg(rows, "retired_agents"),
        "explore_actions_mean": avg(rows, "explore_actions"),
        "target_actions_mean": avg(rows, "target_actions"),
        "goal_hits_mean": avg(rows, "goal_hits"),
        "active_step_hit_rate_mean": avg(rows, "active_step_hit_rate"),
        "nonfinite_state_count": int(
            sum(int(row["nonfinite_state_count"]) for row in rows)
        ),
        "invalid_assignment_count": int(
            sum(int(row["invalid_assignment_count"]) for row in rows)
        ),
        "duplicate_assignment_steps_mean": avg(rows, "duplicate_assignment_steps"),
        "duplicate_assignment_excess_mean": avg(rows, "duplicate_assignment_excess"),
        "detected_unassigned_steps_mean": avg(rows, "detected_unassigned_steps"),
        "pre_all_detected_unassigned_steps_mean": avg(
            rows, "pre_all_detected_unassigned_steps"
        ),
        "post_all_detected_unassigned_steps_mean": avg(
            rows, "post_all_detected_unassigned_steps"
        ),
        "switch_away_actions_mean": avg(rows, "switch_away_actions"),
        "discovery_to_assignment_delay_mean": avg(rows, "discovery_to_assignment_delay_mean"),
        "assignment_to_visit_delay_mean": avg(rows, "assignment_to_visit_delay_mean"),
        "assignment_quality_events": int(
            sum(int(r["assignment_quality_event_count"]) for r in rows)
        ),
        "crossing_event_rate": (
            sum(int(r["crossing_event_count"]) for r in rows)
            / max(sum(int(r["assignment_quality_event_count"]) for r in rows), 1)
        ),
        "crossing_pair_count": int(
            sum(int(r["crossing_pair_count"]) for r in rows)
        ),
        "assignment_sum_regret_mean": avg(rows, "assignment_sum_regret"),
        "assignment_makespan_regret_mean": avg(
            rows, "assignment_makespan_regret"
        ),
        "comparative_advantage_violation_count": int(
            sum(int(r["comparative_advantage_violation_count"]) for r in rows)
        ),
        "finder_event_count": int(sum(int(r["finder_event_count"]) for r in rows)),
        "finder_is_nearest_rate": (
            sum(int(r["finder_nearest_count"]) for r in rows)
            / max(sum(int(r["finder_event_count"]) for r in rows), 1)
        ),
        "finder_accept_rate": (
            sum(int(r["finder_accept_count"]) for r in rows)
            / max(sum(int(r["finder_event_count"]) for r in rows), 1)
        ),
        "finder_reject_count": int(
            sum(int(r["finder_reject_count"]) for r in rows)
        ),
        "finder_unavailable_count": int(
            sum(int(r["finder_unavailable_count"]) for r in rows)
        ),
        "rejected_target_later_claimed_count": int(
            sum(int(r["rejected_target_later_claimed_count"]) for r in rows)
        ),
        "rejected_target_wait_mean": avg(rows, "rejected_target_wait_mean"),
        "cascade_diffusion_count": int(
            sum(int(r["cascade_diffusion_count"]) for r in rows)
        ),
        "final_committer_is_finder_rate": (
            sum(int(r["final_committer_is_finder_count"]) for r in rows)
            / max(
                sum(
                    len(
                        [
                            event
                            for event in r["finder_cascade_events"]
                            if event["event"] == "accept"
                        ]
                    )
                    for r in rows
                ),
                1,
            )
        ),
        "cascade_open_to_commitment_delay_mean": avg(
            rows, "cascade_open_to_commitment_delay_mean"
        ),
        "finder_to_commitment_delay_mean": avg(
            rows, "finder_to_commitment_delay_mean"
        ),
        "successful_assignment_sum_regret_mean": avg(
            [r for r in rows if int(r["success"])], "assignment_sum_regret"
        ),
        "failed_assignment_sum_regret_mean": avg(
            [r for r in rows if not int(r["success"])], "assignment_sum_regret"
        ),
        "timeout_assignment_sum_regret_mean": avg(
            [r for r in rows if str(r["failure_category"]) != "success"],
            "assignment_sum_regret",
        ),
        "entropy_reduction_mean": avg(rows, "entropy_reduction"),
        "pre_all_detected_entropy_reduction_mean": avg(
            rows, "pre_all_detected_entropy_reduction"
        ),
        "first_detection_step_mean": avg_nonnegative(rows, "first_detection_step"),
        "second_detection_step_mean": avg_nonnegative(rows, "second_detection_step"),
        "all_detected_step_mean": avg_nonnegative(rows, "all_detected_step"),
        "all_detected_remaining_steps_mean": avg_nonnegative(
            rows, "all_detected_remaining_steps"
        ),
        "all_detected_to_first_target_action_delay_mean": avg(
            rows, "all_detected_to_first_target_action_delay"
        ),
        "second_detection_to_first_rescue_commitment_delay_mean": avg(
            rows, "second_detection_to_first_rescue_commitment_delay"
        ),
        "post_all_detected_explore_ratio_mean": avg(
            rows, "post_all_detected_explore_ratio"
        ),
        "premature_retirement_rate": avg(rows, "premature_retirement"),
        "searchers_at_first_detection_mean": avg(
            rows, "searchers_at_first_detection"
        ),
        "searchers_at_second_detection_mean": avg(
            rows, "searchers_at_second_detection"
        ),
        "rescue_workload_steps_at_all_detected_mean": avg(
            rows, "rescue_workload_steps_at_all_detected"
        ),
    }
    for reason in sorted({str(r["failure_reason"]) for r in rows}):
        count = sum(1 for r in rows if r["failure_reason"] == reason)
    detected_all_rows = [r for r in rows if int(r["all_detected_step"]) >= 0]
    success_rows = [r for r in rows if int(r["success"])]
    rejected_waits = [
        int(wait)
        for row in rows
        for wait in row["rejected_target_wait_steps"]
    ]
    summary.update(
        {
            "detect_all_count": len(detected_all_rows),
            "detect_all_rate": len(detected_all_rows) / max(total, 1),
            "success_given_detect_all": len(success_rows)
            / max(len(detected_all_rows), 1),
            "terminal_not_all_detected_count": total - len(detected_all_rows),
            "detected_all_but_incomplete_count": sum(
                int(not int(row["success"])) for row in detected_all_rows
            ),
            "success_step_p95": percentile(
                [float(row["success_step"]) for row in success_rows], 0.95
            ),
            "rejected_target_wait_p95": percentile(
                [float(wait) for wait in rejected_waits], 0.95
            ),
        }
    )
    if rows and "team_coverage_rate" in rows[0]:
        for key in (
            "team_explored_area",
            "team_coverage_rate",
            "multi_uav_overlap_area",
            "overlap_ratio",
            "explored_area_revisit_ratio",
            "search_travel_distance",
            "coverage_efficiency",
            "search_option_switches",
            "search_heading_continuity",
        ):
            summary[f"{key}_mean"] = avg(rows, key)
        summary["per_agent_explored_area_mean"] = mean_lists(
            rows, "per_agent_explored_area"
        )
        summary["per_agent_search_travel_distance_mean"] = mean_lists(
            rows, "per_agent_search_travel_distance"
        )
        summary["per_agent_search_option_switches_mean"] = mean_lists(
            rows, "per_agent_search_option_switches"
        )
        summary[f"reason_{reason}"] = count
        summary[f"reason_{reason}_rate"] = count / max(total, 1)
    for category in sorted({str(r["failure_category"]) for r in rows}):
        count = sum(1 for r in rows if r["failure_category"] == category)
        summary[f"category_{category}"] = count
        summary[f"category_{category}_rate"] = count / max(total, 1)
    if partial:
        summary.update(
            {
                "partial_visited_mean": avg(partial, "visited_targets"),
                "partial_detected_mean": avg(partial, "detected_targets"),
                "partial_undiscovered_mean": avg(partial, "undiscovered_targets"),
                "partial_detected_unvisited_mean": avg(partial, "detected_unvisited_targets"),
                "partial_retired_mean": avg(partial, "retired_agents"),
                "partial_explore_actions_mean": avg(partial, "explore_actions"),
                "partial_target_actions_mean": avg(partial, "target_actions"),
                "partial_duplicate_assignment_excess_mean": avg(partial, "duplicate_assignment_excess"),
                "partial_detected_unassigned_steps_mean": avg(partial, "detected_unassigned_steps"),
                "partial_switch_away_actions_mean": avg(partial, "switch_away_actions"),
                "partial_discovery_to_assignment_delay_mean": avg(partial, "discovery_to_assignment_delay_mean"),
                "partial_assignment_to_visit_delay_mean": avg(partial, "assignment_to_visit_delay_mean"),
                "partial_last_visit_step_mean": avg(partial, "last_visit_step"),
            }
        )
    return summary


def avg(rows: list[dict[str, float | int | str]], key: str) -> float:
    vals = []
    for row in rows:
        value = row[key]
        if isinstance(value, str):
            continue
        value = float(value)
        if math.isfinite(value):
            vals.append(value)
    return sum(vals) / max(len(vals), 1)


def avg_nonnegative(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row[key], (float, int))
        and math.isfinite(float(row[key]))
        and float(row[key]) >= 0
    ]
    return sum(values) / max(len(values), 1)


def percentile(values: list[float], quantile: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    rank = (len(finite) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return finite[lower]
    weight = rank - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def mean_lists(rows: list[dict[str, Any]], key: str) -> list[float]:
    width = len(rows[0][key])
    return [
        sum(float(row[key][index]) for row in rows) / max(len(rows), 1)
        for index in range(width)
    ]


def reproducibility_metadata(args: argparse.Namespace) -> dict[str, Any]:
    workspace_root = PROJECT_ROOT.parents[1]
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=workspace_root, text=True
        ).strip()
        git_diff = subprocess.check_output(
            ["git", "diff", "--binary"], cwd=workspace_root
        )
        git_diff_sha256 = hashlib.sha256(git_diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        git_head = "unknown"
        git_diff_sha256 = "unknown"
    return {
        "command": [sys.executable, *sys.argv],
        "git_head": git_head,
        "git_diff_sha256": git_diff_sha256,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def write_outputs(args: argparse.Namespace, rows: list[dict[str, float | int | str]], summary: dict[str, float | int | str]) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "metadata": reproducibility_metadata(args),
                "summary": summary,
                "episodes": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with args.csv_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if args.coverage_metrics_output is not None:
        coverage_columns = [
            "env_id",
            "layout_sha256",
            "success",
            "per_agent_explored_area",
            "per_agent_search_travel_distance",
            "team_explored_area",
            "team_coverage_rate",
            "multi_uav_overlap_area",
            "overlap_ratio",
            "explored_area_revisit_ratio",
            "search_travel_distance",
            "coverage_efficiency",
            "search_option_switches",
            "per_agent_search_option_switches",
            "search_heading_continuity",
        ]
        args.coverage_metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with args.coverage_metrics_output.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=coverage_columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {key: row.get(key) for key in coverage_columns}
                )
    assignment_events_path = args.assignment_events_output or args.output.with_name(
        f"{args.output.stem}_assignment_events.csv"
    )
    cost_matrices_path = args.assignment_cost_matrices_output or args.output.with_name(
        f"{args.output.stem}_assignment_cost_matrices.json"
    )
    crossing_log_path = args.crossing_log_output or args.output.with_name(
        f"{args.output.stem}_crossing_events.csv"
    )
    assignment_events = [
        event
        for row in rows
        for event in row.get("assignment_quality_events", [])
    ]
    event_columns = [
        "env_id", "step", "new_agent_ids", "new_target_ids",
        "eligible_agent_ids", "available_target_ids", "actual_sum_cost",
        "actual_makespan", "optimal_sum_cost", "optimal_makespan",
        "sum_regret", "makespan_regret", "crossing_pair_count",
        "comparative_advantage_violation_count",
    ]
    assignment_events_path.parent.mkdir(parents=True, exist_ok=True)
    with assignment_events_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=event_columns)
        writer.writeheader()
        for event in assignment_events:
            writer.writerow({key: event.get(key) for key in event_columns})
    cost_matrices_path.write_text(
        json.dumps(assignment_events, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    crossing_rows = [
        {"env_id": event["env_id"], "step": event["step"], **pair}
        for event in assignment_events
        for pair in event["crossing_pairs"]
    ]
    crossing_columns = [
        "env_id", "step", "agent_a", "target_a", "agent_b", "target_b",
        "actual_sum", "swapped_sum", "actual_makespan", "swapped_makespan",
    ]
    with crossing_log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=crossing_columns)
        writer.writeheader()
        writer.writerows(crossing_rows)
    lines = [
        "# SAR Failure Mode Analysis",
        "",
        f"policy: `{args.policy}`",
        f"target_threshold: `{args.target_threshold}`",
        f"sensor_radius: `{args.sensor_radius}`",
        f"max_steps: `{args.max_steps}`",
        f"retire_on_rescue: `{args.retire_on_rescue}`",
        f"low_level_controller: `{args.low_level_controller}`",
        f"enable_module_ablation: `{args.enable_module_ablation}`",
        f"ablation_search_module: `{args.ablation_search_module}`",
        f"ablation_timing_module: `{args.ablation_timing_module}`",
        f"ablation_assignment_module: `{args.ablation_assignment_module}`",
        f"progress_features: `{args.progress_features}`",
        f"enable_phase_policy: `{args.enable_phase_policy}`",
        f"phase_initial_rescue_logit: `{args.phase_initial_rescue_logit}`",
        f"rescue_distance_logit_scale: `{args.rescue_distance_logit_scale}`",
        f"staged_rescue: `{args.staged_rescue}`",
        f"rescue_detected_threshold: `{args.rescue_detected_threshold}`",
        f"rescue_entropy_threshold: `{args.rescue_entropy_threshold}`",
        f"dynamic_rescue_release: `{args.dynamic_rescue_release}`",
        f"dynamic_release_min_searchers: `{args.dynamic_release_min_searchers}`",
        f"dynamic_release_entropy_ratio_threshold: `{args.dynamic_release_entropy_ratio_threshold}`",
        f"dynamic_release_entropy_rate_threshold: `{args.dynamic_release_entropy_rate_threshold}`",
        f"dynamic_release_min_stagnation_step: `{args.dynamic_release_min_stagnation_step}`",
        f"dynamic_release_search_steps_per_target: `{args.dynamic_release_search_steps_per_target}`",
        f"dynamic_release_speed_per_step: `{args.dynamic_release_speed_per_step}`",
        f"dynamic_release_time_margin: `{args.dynamic_release_time_margin}`",
        f"dynamic_rescue_only_after_all_detected: `{args.dynamic_rescue_only_after_all_detected}`",
        f"redecide_on_detection_change: `{args.redecide_on_detection_change}`",
        f"redecide_on_assignment_change: `{args.redecide_on_assignment_change}`",
        f"enable_finder_first_cascade: `{args.enable_finder_first_cascade}`",
        f"finder_cascade_mode: `{args.finder_cascade_mode}`",
        f"dynamic_release_max_new_agents_per_event: `{args.dynamic_release_max_new_agents_per_event}`",
        f"early_rescue_penalty: `{args.early_rescue_penalty}`",
        f"early_rescue_detected_threshold: `{args.early_rescue_detected_threshold}`",
        f"search_capacity_discovery_bonus: `{args.search_capacity_discovery_bonus}`",
        f"discovery_active_agents_threshold: `{args.discovery_active_agents_threshold}`",
        f"all_targets_detected_bonus: `{args.all_targets_detected_bonus}`",
        f"all_targets_detected_bonus_requires_no_rescue: `{args.all_targets_detected_bonus_requires_no_rescue}`",
        f"rescue_phase_explore_penalty: `{args.rescue_phase_explore_penalty}`",
        f"rescue_phase_detected_threshold: `{args.rescue_phase_detected_threshold}`",
        f"rescue_phase_min_search_agents: `{args.rescue_phase_min_search_agents}`",
        f"unique_rescue_assignment_bonus: `{args.unique_rescue_assignment_bonus}`",
        f"duplicate_rescue_assignment_penalty: `{args.duplicate_rescue_assignment_penalty}`",
        f"coordinated_target_selection: `{args.coordinated_target_selection}`",
        f"enable_commitment_aware_actor: `{args.enable_commitment_aware_actor}`",
        f"enable_teammate_intention_coordination: `{args.enable_teammate_intention_coordination}`",
        f"intent_loss_coef: `{args.intent_loss_coef}`",
        f"coordination_beta: `{args.coordination_beta}`",
        f"coordination_margin: `{args.coordination_margin}`",
        f"coordination_temperature: `{args.coordination_temperature}`",
        f"detected_unassigned_target_penalty: `{args.detected_unassigned_target_penalty}`",
        f"episodes: `{summary['episodes']}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"- `{key}`: {value:.6g}")
        else:
            lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Interpretation", ""])
    lines.append("- `undiscovered_remaining` means the episode timed out without detecting every target.")
    lines.append("- `detected_unvisited_remaining` means at least one target was known but not rescued before timeout.")
    lines.append("- `no_active_agents_for_search` or `no_active_agents_for_detected_target` indicates retirement removed all remaining search/rescue capacity.")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.enable_phase_policy and not args.progress_features:
        raise ValueError("phase policy requires --progress-features")
    if args.low_level_controller == "proportional" and args.enable_low_level_goal_fallback:
        raise ValueError(
            "low-level goal fallback applies only to the checkpoint controller"
        )
    if args.coverage_metrics_output is not None and not args.enable_coverage_metrics:
        raise ValueError(
            "--coverage-metrics-output requires --enable-coverage-metrics"
        )
    if args.enable_module_ablation:
        if args.policy != "hgsar":
            raise ValueError("module ablation requires --policy hgsar")
        if not args.enable_finder_first_cascade:
            raise ValueError(
                "module ablation requires --enable-finder-first-cascade"
            )
        if args.finder_cascade_mode != "finder_only":
            raise ValueError(
                "module ablation requires --finder-cascade-mode finder_only"
            )
        if not args.dynamic_rescue_release:
            raise ValueError(
                "module ablation requires --dynamic-rescue-release"
            )
        if not (
            args.redecide_on_detection_change
            and args.redecide_on_assignment_change
        ):
            raise ValueError(
                "module ablation requires detection and assignment redecision"
            )
    if args.dynamic_rescue_release and args.staged_rescue:
        raise ValueError("dynamic rescue release cannot be combined with staged rescue")
    if args.dynamic_rescue_release and args.coordinated_target_selection:
        raise ValueError(
            "dynamic rescue release cannot be combined with sequential hard masking"
        )
    learned_coordination_modes = int(args.enable_commitment_aware_actor) + int(
        args.enable_teammate_intention_coordination
    )
    if learned_coordination_modes > 1:
        raise ValueError(
            "commitment-aware actor and teammate intention coordination "
            "cannot be enabled together"
        )
    if args.coordinated_target_selection and learned_coordination_modes:
        raise ValueError(
            "sequential hard mask cannot be combined with a learned "
            "coordination experiment"
        )
    torch.manual_seed(args.seed)
    rows, summary = run(args)
    write_outputs(args, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(f"wrote {args.csv_output}")
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
