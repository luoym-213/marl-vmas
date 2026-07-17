"""Measure attainable SAR ceilings under progressively relaxed information/control."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from comm_spread.async_smdp import AsyncSMDPCollector
from comm_spread.env_factory import make_sar_env
from comm_spread.low_level_policy import BenchMARLLowLevelPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOW_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/sar_low_full/low_safe_0.06/checkpoints/checkpoint_50040000.pt"
)
DISTANCE_BINS = (0.0, 0.5, 1.0, 1.5, 2.0, math.inf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-level-checkpoint", type=Path, default=DEFAULT_LOW_CHECKPOINT)
    parser.add_argument("--enable-low-level-goal-fallback", action="store_true")
    parser.add_argument("--low-level-goal-near-threshold", type=float, default=0.25)
    parser.add_argument("--low-level-fallback-gain", type=float, default=2.0)
    parser.add_argument("--low-level-fallback-stagnation-steps", type=int, default=5)
    parser.add_argument("--low-level-fallback-progress-epsilon", type=float, default=0.005)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--sensor-radius", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "full_info_fixed_low",
            "coverage_sensor_fixed_low",
            "coverage_sensor_proportional",
            "coverage_immediate_proportional",
            "coverage_threshold2_proportional",
            "coverage_all_detected_proportional",
            "coverage_reserve_one_proportional",
            "coverage_reserve_two_proportional",
            "coverage_reserve_undiscovered_proportional",
            "coverage_dynamic_proportional",
            "full_info_proportional",
            "normal_sensor_fixed_low",
            "normal_sensor_proportional",
        ),
        default=(
            "full_info_fixed_low",
            "full_info_proportional",
            "normal_sensor_fixed_low",
            "normal_sensor_proportional",
        ),
    )
    parser.add_argument("--dynamic-min-searchers", type=int, default=2)
    parser.add_argument("--dynamic-entropy-ratio-threshold", type=float, default=0.58)
    parser.add_argument("--dynamic-entropy-rate-threshold", type=float, default=0.0015)
    parser.add_argument("--dynamic-min-stagnation-step", type=int, default=25)
    parser.add_argument("--dynamic-search-steps-per-target", type=float, default=22.0)
    parser.add_argument("--dynamic-speed-per-step", type=float, default=0.035)
    parser.add_argument("--dynamic-time-margin", type=float, default=8.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "projects/CommSpread/outputs/hierarchical_sar_diagnostics/"
            "sar_ceiling_sensor_0.60.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("projects/CommSpread/sar_ceiling_sensor_0.60.md"),
    )
    return parser.parse_args()


def stack_agent_positions(scenario: Any) -> Tensor:
    return torch.stack([agent.state.pos for agent in scenario.world.agents], dim=1)


def stack_target_positions(scenario: Any) -> Tensor:
    return torch.stack([target.state.pos for target in scenario.targets], dim=1)


def minimax_full_matching(agent_positions: Tensor, target_positions: Tensor) -> Tensor:
    """Return exact min-makespan bijections for small equal-size teams."""
    batch_size, n_agents, _ = agent_positions.shape
    n_targets = target_positions.shape[1]
    if n_agents != n_targets:
        raise ValueError("exact full matching requires n_agents == n_targets")
    distances = torch.cdist(agent_positions, target_positions)
    permutations = torch.tensor(
        list(itertools.permutations(range(n_targets))),
        dtype=torch.long,
        device=agent_positions.device,
    )
    agent_ids = torch.arange(n_agents, device=agent_positions.device)
    candidate_costs = distances[:, agent_ids, permutations]
    makespan = candidate_costs.max(dim=-1).values
    total = candidate_costs.sum(dim=-1)
    # Makespan is primary; total distance gives deterministic, quality-preserving ties.
    scores = makespan + total * 1e-4
    best = scores.argmin(dim=-1)
    return permutations[best]


def optimal_partial_matching(
    row_ids: list[int],
    agent_ids: list[int],
    target_ids: list[int],
    distances: Tensor,
    max_matches: int | None = None,
) -> dict[int, int]:
    """Exact min-makespan matching between deciding rows and available targets."""
    match_count = min(len(agent_ids), len(target_ids))
    if max_matches is not None:
        match_count = min(match_count, max(0, max_matches))
    if match_count == 0:
        return {}
    best_score: tuple[float, float] | None = None
    best: dict[int, int] = {}
    for chosen_indices in itertools.combinations(range(len(agent_ids)), match_count):
        for target_order in itertools.permutations(target_ids, match_count):
            pairs = [
                (
                    row_ids[agent_index],
                    agent_ids[agent_index],
                    target_id,
                )
                for agent_index, target_id in zip(chosen_indices, target_order)
            ]
            costs = [float(distances[agent_id, target_id]) for _, agent_id, target_id in pairs]
            score = (max(costs), sum(costs))
            if best_score is None or score < best_score:
                best_score = score
                best = {row_id: target_id for row_id, _, target_id in pairs}
    return best


class FullInformationOracle:
    """Clairvoyant exact matching; target information is revealed at reset."""

    def __init__(self, scenario: Any):
        self.scenario = scenario
        self.assignments: Tensor | None = None
        self.initial_distances: Tensor | None = None

    def __call__(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        scenario = self.scenario
        if self.assignments is None:
            agent_positions = stack_agent_positions(scenario)
            target_positions = stack_target_positions(scenario)
            self.assignments = minimax_full_matching(agent_positions, target_positions)
            distances = torch.cdist(agent_positions, target_positions)
            self.initial_distances = distances.gather(
                2, self.assignments.unsqueeze(-1)
            ).squeeze(-1)
        # This mutation is the explicit information relaxation used by this oracle.
        scenario.target_detected[:] = True
        scenario.target_detected_step[:] = 0
        env_ids = observation["env_id"].long()
        agent_ids = observation["agent_id"].long()
        target_ids = self.assignments[env_ids, agent_ids]
        actions = scenario.rrt_top_k + target_ids
        zeros = torch.zeros(len(actions), 1, device=actions.device)
        return actions.long(), zeros, zeros


class CentralizedSensorOracle:
    """Share fresh detections, uniquely assign known targets, and otherwise search."""

    def __init__(self, scenario: Any):
        self.scenario = scenario

    def share_detections(self) -> Tensor:
        globally_detected = self.scenario.target_detected.any(dim=1)
        self.scenario.target_detected |= globally_detected.unsqueeze(1)
        return globally_detected

    def new_rescue_budget(
        self, *, env_id: int, deciding_agent_ids: list[int],
        committed_rescue_count: int, globally_detected: Tensor,
    ) -> int | None:
        del env_id, deciding_agent_ids, committed_rescue_count, globally_detected
        return None

    def __call__(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        scenario = self.scenario
        globally_detected = self.share_detections()
        env_ids = observation["env_id"].long()
        deciding_agent_ids = observation["agent_id"].long()
        actions = observation["explore_nodes"][..., 2].argmax(dim=-1).long()
        agent_positions = stack_agent_positions(scenario)
        target_positions = stack_target_positions(scenario)

        for env_id_tensor in env_ids.unique(sorted=True):
            env_id = int(env_id_tensor)
            rows_tensor = torch.nonzero(env_ids == env_id_tensor, as_tuple=False).flatten()
            row_ids = [int(row) for row in rows_tensor]
            agent_ids = [int(deciding_agent_ids[row]) for row in rows_tensor]
            deciding_set = set(agent_ids)

            claimed: set[int] = set()
            goal_distances = torch.cdist(
                scenario.assigned_goals[env_id].unsqueeze(0),
                target_positions[env_id].unsqueeze(0),
            ).squeeze(0)
            commitment_ids = goal_distances.argmin(dim=-1)
            commitment_valid = (
                scenario.assigned_tasks[env_id, :, 0].bool()
                & scenario.active_agents[env_id]
                & (goal_distances.min(dim=-1).values <= scenario.goal_radius)
            )
            for agent_id in range(scenario.n_agents):
                if agent_id not in deciding_set and bool(commitment_valid[agent_id]):
                    claimed.add(int(commitment_ids[agent_id]))

            available = [
                target_id
                for target_id in range(scenario.n_targets)
                if bool(globally_detected[env_id, target_id])
                and not bool(scenario.target_visited[env_id, target_id])
                and target_id not in claimed
            ]
            distances = torch.cdist(
                agent_positions[env_id].unsqueeze(0),
                target_positions[env_id].unsqueeze(0),
            ).squeeze(0)
            rescue_budget = self.new_rescue_budget(
                env_id=env_id, deciding_agent_ids=agent_ids,
                committed_rescue_count=len(claimed), globally_detected=globally_detected,
            )
            matches = optimal_partial_matching(
                row_ids, agent_ids, available, distances, max_matches=rescue_budget
            )
            for row_id, target_id in matches.items():
                actions[row_id] = scenario.rrt_top_k + target_id

        zeros = torch.zeros(len(actions), 1, device=actions.device)
        return actions, zeros, zeros


class CoverageSensorOracle(CentralizedSensorOracle):
    """Use deterministic sweep lanes with a configurable rescue timing rule."""

    TIMINGS = {
        "immediate",
        "threshold2",
        "all_detected",
        "reserve_one",
        "reserve_two",
        "reserve_undiscovered",
    }

    def __init__(self, scenario: Any, rescue_timing: str = "all_detected"):
        super().__init__(scenario)
        if scenario.n_agents != 3:
            raise ValueError("coverage sweep is defined for the formal three-agent task")
        if rescue_timing not in self.TIMINGS:
            raise ValueError(f"unsupported coverage rescue timing: {rescue_timing}")
        self.rescue_timing = rescue_timing
        self.route_index = torch.full(
            (scenario.world.batch_dim, scenario.n_agents),
            -1,
            dtype=torch.long,
            device=scenario.world.device,
        )
        lane_y = torch.tensor([-0.75, 0.0, 0.75], device=scenario.world.device)
        route_x = torch.tensor(
            [
                [-0.75, 0.0, 0.75],
                [0.75, 0.0, -0.75],
                [-0.75, 0.0, 0.75],
            ],
            device=scenario.world.device,
        )
        self.routes = torch.stack(
            [route_x, lane_y.view(-1, 1).expand(-1, 3)],
            dim=-1,
        )

    def new_rescue_budget(
        self, *, env_id: int, deciding_agent_ids: list[int],
        committed_rescue_count: int, globally_detected: Tensor,
    ) -> int | None:
        del deciding_agent_ids
        if self.rescue_timing == "immediate": return None
        detected_count = int(globally_detected[env_id].sum())
        if self.rescue_timing == "threshold2":
            return None if detected_count >= 2 else 0
        undiscovered = int((~globally_detected[env_id]).sum())
        if undiscovered == 0: return None
        if self.rescue_timing == "all_detected": return 0
        active_count = int(self.scenario.active_agents[env_id].sum())
        if self.rescue_timing == "reserve_one":
            reserve = 1
        elif self.rescue_timing == "reserve_two":
            reserve = 2
        else:
            reserve = undiscovered
        max_total_rescue = max(0, active_count - min(reserve, active_count))
        return max(0, max_total_rescue - committed_rescue_count)

    def __call__(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        actions, log_probs, values = super().__call__(observation)
        scenario = self.scenario
        env_ids = observation["env_id"].long()
        agent_ids = observation["agent_id"].long()
        exploring_rows = torch.nonzero(
            actions < scenario.rrt_top_k,
            as_tuple=False,
        ).flatten()
        agent_positions = stack_agent_positions(scenario)
        for row_tensor in exploring_rows:
            row = int(row_tensor)
            env_id = int(env_ids[row])
            agent_id = int(agent_ids[row])
            route_index = int(self.route_index[env_id, agent_id])
            if route_index < 0:
                endpoints = self.routes[agent_id, [0, 2]]
                nearest_endpoint = torch.linalg.vector_norm(
                    endpoints - agent_positions[env_id, agent_id], dim=-1
                ).argmin()
                route_index = 0 if int(nearest_endpoint) == 0 else 2
            else:
                current = self.routes[agent_id, route_index]
                reached = torch.linalg.vector_norm(
                    current - agent_positions[env_id, agent_id]
                ) <= scenario.goal_radius
                if bool(reached):
                    route_index = (route_index + 1) % self.routes.shape[1]
            self.route_index[env_id, agent_id] = route_index
            waypoint = self.routes[agent_id, route_index]
            observation["explore_nodes"][row, 0, 0:2] = (
                waypoint - agent_positions[env_id, agent_id]
            )
            actions[row] = 0
        return actions, log_probs, values


class DynamicCoverageSensorOracle(CoverageSensorOracle):
    """Observable dynamic rescue budget for deterministic coverage diagnostics.

    The rule never reads undiscovered target positions. It balances the time
    needed to rescue the nearest known target against an entropy-based estimate
    of remaining search work. Before all targets are detected it normally keeps
    ``min_searchers`` active searchers, releasing additional UAVs only when the
    map is sufficiently covered, coverage progress has stalled, or the remaining
    horizon is no longer compatible with delaying known rescues.
    """

    def __init__(self, scenario: Any, args: argparse.Namespace):
        super().__init__(scenario, rescue_timing="reserve_two")
        self.rescue_timing = "dynamic"
        self.min_searchers = args.dynamic_min_searchers
        self.entropy_ratio_threshold = args.dynamic_entropy_ratio_threshold
        self.entropy_rate_threshold = args.dynamic_entropy_rate_threshold
        self.min_stagnation_step = args.dynamic_min_stagnation_step
        self.search_steps_per_target = args.dynamic_search_steps_per_target
        self.speed_per_step = args.dynamic_speed_per_step
        self.time_margin = args.dynamic_time_margin
        batch = scenario.world.batch_dim
        device = scenario.world.device
        self.initial_entropy = torch.full((batch,), torch.nan, device=device)
        self.last_entropy = torch.full((batch,), torch.nan, device=device)
        self.last_entropy_step = torch.full((batch,), -1, dtype=torch.long, device=device)
        self.decision_log: list[list[dict[str, Any]]] = [[] for _ in range(batch)]

    def new_rescue_budget(
        self, *, env_id: int, deciding_agent_ids: list[int],
        committed_rescue_count: int, globally_detected: Tensor,
    ) -> int | None:
        del deciding_agent_ids
        scenario = self.scenario
        detected = globally_detected[env_id] & ~scenario.target_visited[env_id]
        detected_count = int(detected.sum())
        undiscovered = int((~globally_detected[env_id]).sum())
        active_count = int(scenario.active_agents[env_id].sum())
        step = int(scenario.world_steps[env_id])
        remaining = max(int(scenario.max_steps) - step, 0)

        entropy = float(
            scenario._compute_entropy(scenario.belief_maps[env_id : env_id + 1])
            .mean()
            .item()
        )
        if not math.isfinite(float(self.initial_entropy[env_id])):
            self.initial_entropy[env_id] = entropy
        initial_entropy = max(float(self.initial_entropy[env_id]), 1e-8)
        entropy_ratio = entropy / initial_entropy
        previous_entropy = float(self.last_entropy[env_id])
        previous_step = int(self.last_entropy_step[env_id])
        entropy_rate = math.inf
        if math.isfinite(previous_entropy) and step > previous_step:
            entropy_rate = (previous_entropy - entropy) / (step - previous_step)
        self.last_entropy[env_id] = entropy
        self.last_entropy_step[env_id] = step

        if detected_count:
            agent_positions = stack_agent_positions(scenario)[env_id]
            target_positions = stack_target_positions(scenario)[env_id, detected]
            distances = torch.cdist(
                agent_positions[scenario.active_agents[env_id]].unsqueeze(0),
                target_positions.unsqueeze(0),
            ).squeeze(0)
            nearest_distance = float(distances.min().item())
            rescue_eta = nearest_distance / max(self.speed_per_step, 1e-6)
        else:
            nearest_distance = math.inf
            rescue_eta = 0.0
        search_eta = (
            self.search_steps_per_target
            * undiscovered
            * max(entropy_ratio, 0.1)
        )
        time_pressure = (
            detected_count > 0
            and remaining <= rescue_eta + search_eta + self.time_margin
        )
        low_unknown = entropy_ratio <= self.entropy_ratio_threshold
        stagnated = (
            step >= self.min_stagnation_step
            and math.isfinite(entropy_rate)
            and entropy_rate <= self.entropy_rate_threshold
        )

        if undiscovered == 0:
            max_total_rescue = active_count
            reason = "all_detected"
        elif detected_count == 0:
            max_total_rescue = 0
            reason = "no_known_target"
        else:
            normal_capacity = max(0, active_count - self.min_searchers)
            release_extra = time_pressure or low_unknown or stagnated
            max_total_rescue = (
                min(detected_count, active_count)
                if release_extra
                else min(detected_count, normal_capacity)
            )
            reasons = []
            if time_pressure:
                reasons.append("time_pressure")
            if low_unknown:
                reasons.append("low_unknown")
            if stagnated:
                reasons.append("coverage_stagnation")
            reason = "+".join(reasons) if reasons else "preserve_searchers"

        budget = max(0, max_total_rescue - committed_rescue_count)
        self.decision_log[env_id].append(
            {
                "step": step,
                "detected_count": int(globally_detected[env_id].sum()),
                "unvisited_detected_count": detected_count,
                "active_count": active_count,
                "committed_rescue_count": committed_rescue_count,
                "new_rescue_budget": budget,
                "remaining_steps": remaining,
                "entropy_ratio": entropy_ratio,
                "entropy_rate": entropy_rate if math.isfinite(entropy_rate) else None,
                "nearest_known_target_distance": (
                    nearest_distance if math.isfinite(nearest_distance) else None
                ),
                "estimated_rescue_steps": rescue_eta,
                "estimated_search_steps": search_eta,
                "reason": reason,
            }
        )
        return budget


@dataclass
class ModeResult:
    summary: dict[str, Any]
    per_env: dict[str, list[Any]]


def percentile(values: Tensor, q: float) -> float | None:
    if values.numel() == 0:
        return None
    return float(torch.quantile(values.float(), q).cpu())


def assigned_distance_bins(
    distances: Tensor,
    assignments: Tensor,
    visited_step: Tensor,
) -> dict[str, dict[str, float | int]]:
    assigned_visit = visited_step.gather(1, assignments)
    result: dict[str, dict[str, float | int]] = {}
    for low, high in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
        mask = (distances >= low) & (distances < high)
        label = f"{low:.1f}-{high:.1f}" if math.isfinite(high) else f">={low:.1f}"
        count = int(mask.sum())
        success = (assigned_visit >= 0) & mask
        result[label] = {
            "count": count,
            "success_rate": float(success.sum() / max(count, 1)),
            "mean_visit_step": float(assigned_visit[success].float().mean())
            if success.any()
            else 0.0,
        }
    return result


def run_mode(args: argparse.Namespace, mode: str) -> ModeResult:
    full_info = mode.startswith("full_info")
    fixed_low = mode.endswith("fixed_low")
    env = make_sar_env(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        mode="high",
        emit_info=False,
        enable_high_level_state=True,
        enable_rrt_candidates=not full_info,
        auto_resample_goals=False,
        sensor_radius=args.sensor_radius,
        max_steps=args.max_steps,
        retire_on_rescue=True,
    )
    scenario = env.scenario
    if full_info:
        oracle: FullInformationOracle | CentralizedSensorOracle = FullInformationOracle(scenario)
    elif mode.startswith("coverage"):
        timing_by_mode = {
            "coverage_sensor_fixed_low": "all_detected",
            "coverage_sensor_proportional": "all_detected",
            "coverage_immediate_proportional": "immediate",
            "coverage_threshold2_proportional": "threshold2",
            "coverage_all_detected_proportional": "all_detected",
            "coverage_reserve_one_proportional": "reserve_one",
            "coverage_reserve_two_proportional": "reserve_two",
            "coverage_reserve_undiscovered_proportional": "reserve_undiscovered",
        }
        oracle = (
            DynamicCoverageSensorOracle(scenario, args)
            if mode == "coverage_dynamic_proportional"
            else CoverageSensorOracle(scenario, rescue_timing=timing_by_mode[mode])
        )
    else:
        oracle = CentralizedSensorOracle(scenario)
    low_policy = (
        BenchMARLLowLevelPolicy(
            args.low_level_checkpoint,
            device=args.device,
            seed=args.seed,
            deterministic=True,
            max_steps=args.max_steps,
            enable_goal_reaching_fallback=args.enable_low_level_goal_fallback,
            goal_near_threshold=args.low_level_goal_near_threshold,
            fallback_proportional_gain=args.low_level_fallback_gain,
            fallback_stagnation_steps=args.low_level_fallback_stagnation_steps,
            fallback_progress_epsilon=args.low_level_fallback_progress_epsilon,
        )
        if fixed_low
        else None
    )
    collector = AsyncSMDPCollector(
        env,
        high_level_policy=oracle,
        low_level_policy=low_policy,
    )

    torch.manual_seed(args.seed)
    collector.reset()
    initial_agents = stack_agent_positions(scenario).detach().clone()
    initial_targets = stack_target_positions(scenario).detach().clone()
    initial_layout = torch.cat([initial_agents, initial_targets], dim=1).cpu().contiguous()
    initial_checksum = float(initial_layout.sum())
    layout_sha256 = hashlib.sha256(initial_layout.numpy().tobytes()).hexdigest()
    per_env_layout_sha256 = [
        hashlib.sha256(initial_layout[env_id].numpy().tobytes()).hexdigest()
        for env_id in range(args.num_envs)
    ]
    detected_step = torch.full(
        (args.num_envs, scenario.n_targets),
        0 if full_info else -1,
        dtype=torch.long,
        device=scenario.world.device,
    )
    visited_step = torch.full_like(detected_step, -1)
    success_step = torch.full((args.num_envs,), -1, dtype=torch.long, device=scenario.world.device)
    first_rescue_step = torch.full_like(success_step, -1)
    min_searchers_before_all_detected = torch.full(
        (args.num_envs,), scenario.n_agents, dtype=torch.long, device=scenario.world.device
    )

    try:
        for step in range(args.max_steps):
            detected_before = scenario.target_detected.any(dim=1).clone()
            visited_before = scenario.target_visited.clone()
            dones = collector.step()
            detected_after = scenario.target_detected.any(dim=1)
            visited_after = scenario.target_visited
            newly_detected = detected_after & ~detected_before
            newly_visited = visited_after & ~visited_before
            first_rescue_step = torch.where(
                newly_visited.any(dim=-1) & (first_rescue_step < 0),
                torch.full_like(first_rescue_step, step + 1), first_rescue_step,
            )
            detected_step = torch.where(
                newly_detected & (detected_step < 0),
                torch.full_like(detected_step, step + 1),
                detected_step,
            )
            visited_step = torch.where(
                newly_visited & (visited_step < 0),
                torch.full_like(visited_step, step + 1),
                visited_step,
            )
            new_success = scenario.success & (success_step < 0)
            success_step = torch.where(
                new_success, torch.full_like(success_step, step + 1), success_step,
            )
            not_all_detected = ~detected_after.all(dim=-1)
            searchers = (scenario.active_agents & ~scenario.assigned_tasks[..., 0].bool()).sum(dim=-1)
            min_searchers_before_all_detected = torch.where(
                not_all_detected, torch.minimum(min_searchers_before_all_detected, searchers),
                min_searchers_before_all_detected,
            )

            if not full_info and newly_detected.any():
                oracle.share_detections()
                scenario._refresh_high_level_state()
                exploration = ~scenario.assigned_tasks[..., 0].bool()
                replan = scenario.active_agents & exploration & newly_detected.any(dim=-1, keepdim=True)
                if replan.any():
                    collector._finalize(replan, dones)
                    collector._decide(replan)
            if bool(dones.all()):
                break
    finally:
        if low_policy is not None:
            low_policy.close()

    success = scenario.success.detach()
    detected_all = scenario.target_detected.any(dim=1).all(dim=-1).detach()
    successful_steps = success_step[success_step >= 0]
    last_detection = detected_step.max(dim=-1).values
    complete_detection_steps = last_detection[detected_all]
    visited_count = scenario.target_visited.sum(dim=-1)
    premature_rescue = (first_rescue_step >= 0) & ((last_detection < 0) | (first_rescue_step < last_detection))
    failed = ~success
    failed_not_all_detected = failed & ~detected_all
    failed_after_all_detected = failed & detected_all
    exhausted_search_before_detection = (min_searchers_before_all_detected == 0) & ~detected_all
    successful_after_detection = success[detected_all]
    coverage_timing = oracle.rescue_timing if isinstance(oracle, CoverageSensorOracle) else None
    summary: dict[str, Any] = {
        "mode": mode,
        "episodes": args.num_envs,
        "max_steps": args.max_steps,
        "sensor_radius": args.sensor_radius,
        "information": "all_targets_at_reset" if full_info else "normal_sensor_shared_fresh",
        "search": "coverage_sweep" if mode.startswith("coverage") else "rrt_voronoi",
        "rescue_timing": coverage_timing,
        "low_level": "fixed_mappo" if fixed_low else "proportional_same_physics",
        "success_rate": float(success.float().mean().cpu()),
        "success_count": int(success.sum().cpu()),
        "detected_all_rate": float(detected_all.float().mean().cpu()),
        "detected_targets_mean": float(
            scenario.target_detected.any(dim=1).float().sum(dim=-1).mean().cpu()
        ),
        "visited_targets_mean": float(scenario.target_visited.float().sum(dim=-1).mean().cpu()),
        "mean_success_step": float(successful_steps.float().mean().cpu())
        if successful_steps.numel()
        else None,
        "p95_success_step": percentile(successful_steps, 0.95),
        "mean_all_detected_step": float(complete_detection_steps.float().mean().cpu()) if complete_detection_steps.numel() else None,
        "mean_first_rescue_step": float(first_rescue_step[first_rescue_step >= 0].float().mean().cpu()) if (first_rescue_step >= 0).any() else None,
        "premature_rescue_rate": float(premature_rescue.float().mean().cpu()),
        "success_given_all_detected": float(successful_after_detection.float().mean().cpu()) if successful_after_detection.numel() else None,
        "failed_not_all_detected_count": int(failed_not_all_detected.sum().cpu()),
        "failed_after_all_detected_count": int(failed_after_all_detected.sum().cpu()),
        "search_exhausted_before_all_detected_count": int(exhausted_search_before_detection.sum().cpu()),
        "failed_visited_count_0": int((failed & (visited_count == 0)).sum().cpu()),
        "failed_visited_count_1": int((failed & (visited_count == 1)).sum().cpu()),
        "failed_visited_count_2": int((failed & (visited_count == 2)).sum().cpu()),
        "layout_checksum": initial_checksum,
        "layout_sha256": layout_sha256,
    }
    if full_info:
        assert oracle.assignments is not None
        assert oracle.initial_distances is not None
        summary.update(
            {
                "initial_assigned_distance_mean": float(oracle.initial_distances.mean().cpu()),
                "initial_assigned_max_distance_mean": float(
                    oracle.initial_distances.max(dim=-1).values.mean().cpu()
                ),
                "initial_assigned_max_distance_p95": percentile(
                    oracle.initial_distances.max(dim=-1).values, 0.95
                ),
                "distance_bins": assigned_distance_bins(
                    oracle.initial_distances,
                    oracle.assignments,
                    visited_step,
                ),
            }
        )
    per_env = {
        "success": success.cpu().tolist(),
        "success_step": success_step.cpu().tolist(),
        "detected_count": scenario.target_detected.any(dim=1).sum(dim=-1).cpu().tolist(),
        "visited_count": scenario.target_visited.sum(dim=-1).cpu().tolist(),
        "last_detection_step": last_detection.cpu().tolist(),
        "first_rescue_step": first_rescue_step.cpu().tolist(),
        "min_searchers_before_all_detected": min_searchers_before_all_detected.cpu().tolist(),
        "layout_sha256": per_env_layout_sha256,
        "initial_agent_positions": initial_agents.cpu().tolist(),
        "initial_target_positions": initial_targets.cpu().tolist(),
    }
    if isinstance(oracle, DynamicCoverageSensorOracle):
        per_env["gate_decisions"] = oracle.decision_log
    close_env = getattr(env, "close", None)
    if close_env is not None:
        close_env()
    return ModeResult(summary=summary, per_env=per_env)


def write_markdown(args: argparse.Namespace, results: list[ModeResult]) -> None:
    lines = [
        "# SAR ceiling diagnostics",
        "",
        "These are progressively relaxed, attainable oracle references. A heuristic",
        "oracle score is not a proof of the exact POMDP optimum.",
        "",
        f"- episodes per mode: `{args.num_envs}`",
        f"- seed: `{args.seed}`",
        f"- max steps: `{args.max_steps}`",
        f"- sensor radius: `{args.sensor_radius}`",
        f"- retire on rescue: `True`",
        "",
        "| Mode | Rescue timing | Success | Detect all | Success given detected | First rescue | All detected | Completion |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        item = result.summary
        mean_step = item["mean_success_step"]
        mean_step_text = f"{mean_step:.3f}" if mean_step is not None else "n/a"
        first_rescue = item["mean_first_rescue_step"]
        first_rescue_text = f"{first_rescue:.3f}" if first_rescue is not None else "n/a"
        detected_step = item["mean_all_detected_step"]
        detected_step_text = f"{detected_step:.3f}" if detected_step is not None else "n/a"
        conditional = item["success_given_all_detected"]
        conditional_text = f"{conditional:.6f}" if conditional is not None else "n/a"
        lines.append(
            f"| `{item['mode']}` | {item['rescue_timing'] or 'n/a'} | "
            f"{item['success_rate']:.6f} | {item['detected_all_rate']:.6f} | "
            f"{conditional_text} | {first_rescue_text} | {detected_step_text} | {mean_step_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Full-information modes remove discovery uncertainty but retain the configured horizon, physics, collisions, goal radius, and retirement semantics.",
            "- Full-information assignment exactly minimizes geometric makespan over all 3! bijections; it does not claim to know the fixed controller's counterfactual travel time.",
            "- Fixed-low versus proportional isolates sensitivity to the trained low-level checkpoint; proportional control is a diagnostic controller, not a policy used by the formal method.",
            "- Normal-sensor modes share only newly detected targets and use exact unique matching among current decision agents.",
            "- Coverage timing modes use the same deterministic three-lane routes and exact unique matching. They differ only in how many fresh rescue commitments are allowed while targets remain undiscovered.",
            "- The gap from full information to normal sensing estimates the combined discovery/search bottleneck. The gap from the normal-sensor oracle to HGSAR estimates high-level policy and coordination headroom.",
        ]
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def paired_success_comparisons(results: list[ModeResult]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for left, right in itertools.combinations(results, 2):
        if left.summary["layout_sha256"] != right.summary["layout_sha256"]:
            raise RuntimeError(
                f"layout mismatch: {left.summary['mode']} vs {right.summary['mode']}"
            )
        left_success = torch.tensor(left.per_env["success"], dtype=torch.bool)
        right_success = torch.tensor(right.per_env["success"], dtype=torch.bool)
        comparisons.append(
            {
                "left": left.summary["mode"],
                "right": right.summary["mode"],
                "both_success": int((left_success & right_success).sum()),
                "left_only_success": int((left_success & ~right_success).sum()),
                "right_only_success": int((~left_success & right_success).sum()),
                "both_fail": int((~left_success & ~right_success).sum()),
                "paired_success_difference": float(
                    right_success.float().mean() - left_success.float().mean()
                ),
            }
        )
    return comparisons


def main() -> None:
    args = parse_args()
    results = []
    for mode in args.modes:
        print(f"[ceiling] running {mode}", flush=True)
        result = run_mode(args, mode)
        results.append(result)
        print(json.dumps(result.summary, indent=2, sort_keys=True), flush=True)
    payload = {
        "config": {
            "num_envs": args.num_envs,
            "max_steps": args.max_steps,
            "sensor_radius": args.sensor_radius,
            "seed": args.seed,
            "low_level_checkpoint": str(args.low_level_checkpoint),
            "enable_low_level_goal_fallback": args.enable_low_level_goal_fallback,
            "low_level_goal_near_threshold": args.low_level_goal_near_threshold,
            "low_level_fallback_gain": args.low_level_fallback_gain,
            "low_level_fallback_stagnation_steps": args.low_level_fallback_stagnation_steps,
            "low_level_fallback_progress_epsilon": args.low_level_fallback_progress_epsilon,
            "dynamic_min_searchers": args.dynamic_min_searchers,
            "dynamic_entropy_ratio_threshold": args.dynamic_entropy_ratio_threshold,
            "dynamic_entropy_rate_threshold": args.dynamic_entropy_rate_threshold,
            "dynamic_min_stagnation_step": args.dynamic_min_stagnation_step,
            "dynamic_search_steps_per_target": args.dynamic_search_steps_per_target,
            "dynamic_speed_per_step": args.dynamic_speed_per_step,
            "dynamic_time_margin": args.dynamic_time_margin,
        },
        "results": [
            {"summary": result.summary, "per_env": result.per_env}
            for result in results
        ],
        "paired_success_comparisons": paired_success_comparisons(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args, results)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
