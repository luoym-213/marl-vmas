"""Default-off high-level module replacements for SAR evaluation ablations."""

from __future__ import annotations

import itertools
from typing import Any

import torch
from torch import Tensor


def _stack_agent_positions(scenario: Any) -> Tensor:
    return torch.stack([agent.state.pos for agent in scenario.world.agents], dim=1)


def _stack_target_positions(scenario: Any) -> Tensor:
    return torch.stack([target.state.pos for target in scenario.targets], dim=1)


class DeterministicThreeLaneSearch:
    """Replace an explore action with the formal deterministic three-lane sweep."""

    def __init__(self, scenario: Any):
        if scenario.n_agents != 3:
            raise ValueError("three-lane search requires exactly three agents")
        self.scenario = scenario
        device = scenario.world.device
        self.route_index = torch.full(
            (scenario.world.batch_dim, scenario.n_agents),
            -1,
            dtype=torch.long,
            device=device,
        )
        lane_y = torch.tensor([-0.75, 0.0, 0.75], device=device)
        route_x = torch.tensor(
            [
                [-0.75, 0.0, 0.75],
                [0.75, 0.0, -0.75],
                [-0.75, 0.0, 0.75],
            ],
            device=device,
        )
        self.routes = torch.stack(
            [route_x, lane_y.view(-1, 1).expand(-1, 3)],
            dim=-1,
        )

    def override(
        self,
        observation: dict[str, Tensor],
        actions: Tensor,
    ) -> Tensor:
        scenario = self.scenario
        env_ids = observation["env_id"].long()
        agent_ids = observation["agent_id"].long()
        agent_positions = _stack_agent_positions(scenario)
        explore_rows = torch.nonzero(
            actions < scenario.rrt_top_k,
            as_tuple=False,
        ).flatten()
        for row_tensor in explore_rows:
            row = int(row_tensor)
            env_id = int(env_ids[row])
            agent_id = int(agent_ids[row])
            route_index = int(self.route_index[env_id, agent_id])
            if route_index < 0:
                endpoints = self.routes[agent_id, [0, 2]]
                nearest = torch.linalg.vector_norm(
                    endpoints - agent_positions[env_id, agent_id],
                    dim=-1,
                ).argmin()
                route_index = 0 if int(nearest) == 0 else 2
            else:
                waypoint = self.routes[agent_id, route_index]
                reached = torch.linalg.vector_norm(
                    waypoint - agent_positions[env_id, agent_id]
                ) <= scenario.goal_radius
                if bool(reached):
                    route_index = (route_index + 1) % self.routes.shape[1]
            self.route_index[env_id, agent_id] = route_index
            waypoint = self.routes[agent_id, route_index]
            observation["explore_nodes"][row, 0, 0:2] = (
                waypoint - agent_positions[env_id, agent_id]
            )
            actions[row] = 0
        return actions


def exact_visible_matching(
    *,
    scenario: Any,
    observation: dict[str, Tensor],
    actions: Tensor,
    force_all_rows_after_all_detected: bool,
) -> Tensor:
    """Assign rescue rows by exact makespan matching over visible targets only."""

    env_ids = observation["env_id"].long()
    row_agent_ids = observation["agent_id"].long()
    agent_positions = _stack_agent_positions(scenario)
    target_positions = _stack_target_positions(scenario)
    globally_detected = scenario.target_detected.any(dim=1)
    target_claimed = scenario._target_claimed()[:, 0].bool()
    all_detected = globally_detected.all(dim=-1)

    for env_tensor in env_ids.unique(sorted=True):
        env_id = int(env_tensor)
        env_rows = torch.nonzero(env_ids == env_tensor, as_tuple=False).flatten()
        actor_rescue_budget = int(
            (actions[env_rows] >= scenario.rrt_top_k).sum()
        )
        actor_rescue_rows = actions[env_rows] >= scenario.rrt_top_k
        actions[env_rows[actor_rescue_rows]] = observation[
            "explore_nodes"
        ][env_rows[actor_rescue_rows], :, 2].argmax(dim=-1).long()
        eligible_rows: list[int] = []
        for row_tensor in env_rows:
            row = int(row_tensor)
            agent_id = int(row_agent_ids[row])
            uncommitted = not bool(scenario.assigned_tasks[env_id, agent_id, 0] > 0.5)
            if (
                bool(scenario.active_agents[env_id, agent_id])
                and uncommitted
            ):
                eligible_rows.append(row)

        available_targets = [
            target_id
            for target_id in range(scenario.n_targets)
            if bool(globally_detected[env_id, target_id])
            and not bool(scenario.target_visited[env_id, target_id])
            and not bool(target_claimed[env_id, target_id])
        ]
        rescue_budget = (
            len(eligible_rows)
            if force_all_rows_after_all_detected and bool(all_detected[env_id])
            else actor_rescue_budget
        )
        match_count = min(
            rescue_budget, len(eligible_rows), len(available_targets)
        )
        if match_count == 0:
            continue

        distances = torch.cdist(
            agent_positions[env_id].unsqueeze(0),
            target_positions[env_id].unsqueeze(0),
        ).squeeze(0)
        best_score: tuple[float, float, tuple[tuple[int, int], ...]] | None = None
        best_pairs: tuple[tuple[int, int], ...] = ()
        for chosen_rows in itertools.combinations(eligible_rows, match_count):
            for target_order in itertools.permutations(
                available_targets,
                match_count,
            ):
                pairs = tuple(zip(chosen_rows, target_order))
                costs = [
                    float(distances[int(row_agent_ids[row]), target_id])
                    for row, target_id in pairs
                ]
                tie_break = tuple(
                    (int(row_agent_ids[row]), int(target_id))
                    for row, target_id in pairs
                )
                score = (max(costs), sum(costs), tie_break)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pairs = pairs
        for row, target_id in best_pairs:
            actions[row] = scenario.rrt_top_k + target_id
    return actions


class ModularAblationPolicy:
    """Evaluation-only wrapper that replaces search and/or assignment modules."""

    def __init__(
        self,
        base_policy: Any,
        scenario: Any,
        *,
        search_module: str,
        timing_module: str,
        assignment_module: str,
    ) -> None:
        if search_module not in {"hgsar", "three_lane"}:
            raise ValueError(f"unknown search module: {search_module}")
        if timing_module not in {"finder", "all_detected"}:
            raise ValueError(f"unknown timing module: {timing_module}")
        if assignment_module not in {"actor", "exact"}:
            raise ValueError(f"unknown assignment module: {assignment_module}")
        self.base_policy = base_policy
        self.scenario = scenario
        self.search_module = search_module
        self.timing_module = timing_module
        self.assignment_module = assignment_module
        self.coverage = (
            DeterministicThreeLaneSearch(scenario)
            if search_module == "three_lane"
            else None
        )

    def __call__(
        self,
        observation: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        actions, log_probs, values = self.base_policy(observation)
        actions = actions.clone()
        if self.assignment_module == "exact":
            actions = exact_visible_matching(
                scenario=self.scenario,
                observation=observation,
                actions=actions,
                force_all_rows_after_all_detected=(
                    self.timing_module == "all_detected"
                ),
            )
        if self.coverage is not None:
            actions = self.coverage.override(observation, actions)
        return actions, log_probs, values

    def metrics(self) -> dict[str, int]:
        metrics = getattr(self.base_policy, "metrics", None)
        return metrics() if metrics is not None else {}


class CoverageTrajectoryTracker:
    """Track sensor-footprint coverage and search-path geometry per episode."""

    def __init__(self, scenario: Any):
        self.scenario = scenario
        batch = scenario.world.batch_dim
        agents = scenario.n_agents
        cells = scenario.map_dim
        device = scenario.world.device
        self.covered_by_agent = torch.zeros(
            batch,
            agents,
            cells,
            cells,
            dtype=torch.bool,
            device=device,
        )
        self.search_travel = torch.zeros(batch, agents, device=device)
        self.total_footprint_visits = torch.zeros(batch, device=device)
        self.search_option_switches = torch.zeros(
            batch, agents, dtype=torch.long, device=device
        )
        self.previous_search_goal = torch.zeros(batch, agents, 2, device=device)
        self.previous_goal_valid = torch.zeros(
            batch, agents, dtype=torch.bool, device=device
        )
        self.previous_motion = torch.zeros(batch, agents, 2, device=device)
        self.previous_motion_valid = torch.zeros(
            batch, agents, dtype=torch.bool, device=device
        )
        self.heading_cosine_sum = torch.zeros(batch, agents, device=device)
        self.heading_cosine_count = torch.zeros(batch, agents, device=device)

    def start(self) -> None:
        scenario = self.scenario
        search = scenario.active_agents & ~scenario.assigned_tasks[..., 0].bool()
        self.previous_search_goal.copy_(scenario.assigned_goals)
        self.previous_goal_valid.copy_(search)

    def update(
        self,
        *,
        pre_positions: Tensor,
        post_positions: Tensor,
        pre_search_mask: Tensor,
    ) -> None:
        scenario = self.scenario
        movement = post_positions - pre_positions
        distance = torch.linalg.vector_norm(movement, dim=-1)
        moving_search = pre_search_mask & (distance > 1e-8)
        self.search_travel += distance * pre_search_mask.float()

        pair_valid = moving_search & self.previous_motion_valid
        cosine = torch.nn.functional.cosine_similarity(
            movement,
            self.previous_motion,
            dim=-1,
            eps=1e-8,
        )
        self.heading_cosine_sum += torch.where(
            pair_valid, cosine, torch.zeros_like(cosine)
        )
        self.heading_cosine_count += pair_valid.float()
        self.previous_motion = torch.where(
            moving_search.unsqueeze(-1), movement, self.previous_motion
        )
        self.previous_motion_valid = moving_search

        dx = scenario.cell_world_x.view(1, 1, scenario.map_dim, scenario.map_dim) - (
            post_positions[..., 0].view(-1, scenario.n_agents, 1, 1)
        )
        dy = scenario.cell_world_y.view(1, 1, scenario.map_dim, scenario.map_dim) - (
            post_positions[..., 1].view(-1, scenario.n_agents, 1, 1)
        )
        footprint = (dx.square() + dy.square()) <= scenario.sensor_radius**2
        footprint &= pre_search_mask.view(-1, scenario.n_agents, 1, 1)
        self.covered_by_agent |= footprint
        self.total_footprint_visits += footprint.float().sum(dim=(1, 2, 3))

        post_search = scenario.active_agents & ~scenario.assigned_tasks[..., 0].bool()
        goal_changed = (
            torch.linalg.vector_norm(
                scenario.assigned_goals - self.previous_search_goal,
                dim=-1,
            )
            > 1e-6
        )
        self.search_option_switches += (
            post_search & self.previous_goal_valid & goal_changed
        ).long()
        self.previous_search_goal = torch.where(
            post_search.unsqueeze(-1),
            scenario.assigned_goals,
            self.previous_search_goal,
        )
        self.previous_goal_valid = post_search

    def rows(self) -> list[dict[str, Any]]:
        scenario = self.scenario
        cell_area = float(scenario.cell_size**2)
        map_cells = float(scenario.map_dim**2)
        individual_cells = self.covered_by_agent.sum(dim=(-1, -2)).float()
        coverage_count = self.covered_by_agent.sum(dim=1)
        team_cells = (coverage_count > 0).sum(dim=(-1, -2)).float()
        overlap_cells = (coverage_count > 1).sum(dim=(-1, -2)).float()
        team_area = team_cells * cell_area
        travel = self.search_travel.sum(dim=-1)
        revisit_ratio = 1.0 - team_cells / self.total_footprint_visits.clamp_min(1.0)
        heading = self.heading_cosine_sum.sum(dim=-1) / (
            self.heading_cosine_count.sum(dim=-1).clamp_min(1.0)
        )
        result: list[dict[str, Any]] = []
        for env_id in range(scenario.world.batch_dim):
            result.append(
                {
                    "per_agent_explored_area": (
                        individual_cells[env_id] * cell_area
                    ).cpu().tolist(),
                    "per_agent_search_travel_distance": self.search_travel[
                        env_id
                    ].cpu().tolist(),
                    "team_explored_area": float(team_area[env_id].cpu()),
                    "team_coverage_rate": float(
                        (team_cells[env_id] / map_cells).cpu()
                    ),
                    "multi_uav_overlap_area": float(
                        (overlap_cells[env_id] * cell_area).cpu()
                    ),
                    "overlap_ratio": float(
                        (overlap_cells[env_id] / team_cells[env_id].clamp_min(1.0)).cpu()
                    ),
                    "explored_area_revisit_ratio": float(revisit_ratio[env_id].cpu()),
                    "search_travel_distance": float(travel[env_id].cpu()),
                    "coverage_efficiency": float(
                        (team_area[env_id] / travel[env_id].clamp_min(1e-8)).cpu()
                    ),
                    "search_option_switches": int(
                        self.search_option_switches[env_id].sum().cpu()
                    ),
                    "per_agent_search_option_switches": self.search_option_switches[
                        env_id
                    ].cpu().tolist(),
                    "search_heading_continuity": float(heading[env_id].cpu()),
                }
            )
        return result
