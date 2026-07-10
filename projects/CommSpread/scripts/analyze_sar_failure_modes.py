"""Analyze SAR failure modes for trained high-level policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from comm_spread.async_smdp import AsyncSMDPCollector, TargetFirstPolicy
from comm_spread.env_factory import make_sar_env
from comm_spread.high_level_policy import HGSARActorCriticPolicy
from comm_spread.low_level_policy import BenchMARLLowLevelPolicy


DEFAULT_LOW_CHECKPOINT = Path(
    "projects/CommSpread/outputs/sar_low_full/low_safe_0.06/checkpoints/checkpoint_50040000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-level-checkpoint", type=Path, default=DEFAULT_LOW_CHECKPOINT)
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
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("projects/CommSpread/outputs/hierarchical_sar_diagnostics/failure_modes.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("projects/CommSpread/outputs/hierarchical_sar_diagnostics/failure_modes.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("projects/CommSpread/sar_failure_modes.md"))
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
    )
    checkpoint = torch.load(args.high_level_checkpoint, map_location=args.device, weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy


@torch.no_grad()
def run(args: argparse.Namespace) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
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
        staged_rescue=args.staged_rescue,
        rescue_detected_threshold=args.rescue_detected_threshold,
        rescue_entropy_threshold=args.rescue_entropy_threshold,
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
    low_policy = BenchMARLLowLevelPolicy(
        args.low_level_checkpoint,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        max_steps=scenario.max_steps,
    )
    high_policy = build_policy(args, scenario)
    collector = AsyncSMDPCollector(env, high_level_policy=high_policy, low_level_policy=low_policy)

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

    try:
        collector.reset()
        entropy_initial = scenario._compute_entropy(scenario.belief_maps).sum(dim=(-1, -2, -3)).detach().clone()
        for step in range(args.steps):
            pre_detected = scenario.target_detected.any(dim=1).detach().clone()
            pre_visited = scenario.target_visited.detach().clone()
            pre_goal_done = scenario.goal_done.detach().clone()
            pre_active = scenario.active_agents.detach().clone()
            pre_goals = scenario.assigned_goals.detach().clone()
            pre_assigned_target = assigned_target_indices(scenario).detach().clone()
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

            post_assigned_target = assigned_target_indices(scenario)
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
            per_env_detected_unassigned_steps += (known_unvisited & (claim_counts == 0)).float().sum(dim=-1)
            newly_assigned_targets = (claim_counts > 0) & detected_now & (first_assignment_step < 0)
            first_assignment_step = torch.where(
                newly_assigned_targets,
                torch.full_like(first_assignment_step, step + 1),
                first_assignment_step,
            )

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
        )
        summary = summarize(rows, scenario.n_targets)
        return rows, summary
    finally:
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
        rows.append(
            {
                "env_id": env_id,
                "success": success,
                "failure_reason": reason,
                "visited_targets": visited_count,
                "detected_targets": detected_count,
                "undiscovered_targets": undiscovered,
                "detected_unvisited_targets": detected_unvisited,
                "retired_agents": retired_count,
                "active_agents": active_count,
                "explore_actions": float(kwargs["per_env_explore_actions"][env_id].cpu()),
                "target_actions": float(kwargs["per_env_target_actions"][env_id].cpu()),
                "goal_hits": float(kwargs["per_env_goal_hits"][env_id].cpu()),
                "active_step_hit_rate": float((kwargs["per_env_goal_hits"][env_id] / kwargs["per_env_active_steps"][env_id].clamp_min(1)).cpu()),
                "mean_goal_distance": float((kwargs["per_env_goal_dist_sum"][env_id] / kwargs["per_env_goal_dist_count"][env_id].clamp_min(1)).cpu()),
                "duplicate_assignment_steps": float(kwargs["per_env_duplicate_assignment_steps"][env_id].cpu()),
                "duplicate_assignment_excess": float(kwargs["per_env_duplicate_assignment_excess"][env_id].cpu()),
                "detected_unassigned_steps": float(kwargs["per_env_detected_unassigned_steps"][env_id].cpu()),
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
                "entropy_reduction": float((kwargs["entropy_initial"][env_id] - kwargs["entropy_final"][env_id]).cpu()),
                "steps": int(kwargs["per_env_steps"][env_id].cpu()),
                "first_discovery_step_mean": mean_positive(kwargs["discovered_first_step"][env_id]),
                "first_visit_step_mean": mean_positive(kwargs["visited_first_step"][env_id]),
                "last_visit_step": int(kwargs["visited_first_step"][env_id].max().cpu()),
                "success_step": int(kwargs["visited_first_step"][env_id].max().cpu()) if success else -1,
                "min_active_distance_to_unvisited_target": finite_mean(min_active_target_dist[env_id][~visited[env_id]]),
            }
        )
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


def summarize(rows: list[dict[str, float | int | str]], n_targets: int) -> dict[str, float | int | str]:
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
        "duplicate_assignment_steps_mean": avg(rows, "duplicate_assignment_steps"),
        "duplicate_assignment_excess_mean": avg(rows, "duplicate_assignment_excess"),
        "detected_unassigned_steps_mean": avg(rows, "detected_unassigned_steps"),
        "switch_away_actions_mean": avg(rows, "switch_away_actions"),
        "discovery_to_assignment_delay_mean": avg(rows, "discovery_to_assignment_delay_mean"),
        "assignment_to_visit_delay_mean": avg(rows, "assignment_to_visit_delay_mean"),
        "entropy_reduction_mean": avg(rows, "entropy_reduction"),
    }
    for reason in sorted({str(r["failure_reason"]) for r in rows}):
        count = sum(1 for r in rows if r["failure_reason"] == reason)
        summary[f"reason_{reason}"] = count
        summary[f"reason_{reason}_rate"] = count / max(total, 1)
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
        if value == value:
            vals.append(value)
    return sum(vals) / max(len(vals), 1)


def write_outputs(args: argparse.Namespace, rows: list[dict[str, float | int | str]], summary: dict[str, float | int | str]) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.csv_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# SAR Failure Mode Analysis",
        "",
        f"policy: `{args.policy}`",
        f"target_threshold: `{args.target_threshold}`",
        f"sensor_radius: `{args.sensor_radius}`",
        f"max_steps: `{args.max_steps}`",
        f"retire_on_rescue: `{args.retire_on_rescue}`",
        f"progress_features: `{args.progress_features}`",
        f"staged_rescue: `{args.staged_rescue}`",
        f"rescue_detected_threshold: `{args.rescue_detected_threshold}`",
        f"rescue_entropy_threshold: `{args.rescue_entropy_threshold}`",
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
    rows, summary = run(args)
    write_outputs(args, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(f"wrote {args.csv_output}")
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
