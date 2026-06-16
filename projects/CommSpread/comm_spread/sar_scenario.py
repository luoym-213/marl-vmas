"""VMAS search-and-rescue scenario with goal-conditioned low-level control."""

from __future__ import annotations

import typing
from typing import Dict, List

import torch
from torch import Tensor
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import (
    ANGULAR_FRICTION,
    DRAG,
    LINEAR_FRICTION,
    Color,
    ScenarioUtils,
)

from comm_spread.rrt import RRTConfig, plan_batch

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class SarScenario(BaseScenario):
    """Hierarchical SAR environment state for VMAS and BenchMARL experiments.

    The base scenario exposes a low-level goal-conditioned control problem. It
    also maintains the high-level belief-map state and RRT candidates needed by
    the fixed-interval high-level wrapper/model.
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.n_agents = kwargs.pop("n_agents", 3)
        self.n_targets = kwargs.pop("n_targets", self.n_agents)
        self.max_steps = kwargs.pop("max_steps", 100)

        self.world_spawning_x = kwargs.pop("world_spawning_x", 1.0)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 1.0)
        self.world_size = kwargs.pop("belief_world_size", self.world_spawning_x * 2.0)
        self.cell_size = kwargs.pop("belief_cell_size", 0.02)
        self.map_dim = int(round(self.world_size / self.cell_size))

        self.agent_radius = kwargs.pop("agent_radius", 0.05)
        self.target_radius = kwargs.pop("target_radius", 0.05)
        self.goal_radius = kwargs.pop("goal_radius", 0.1)
        self.sensor_radius = kwargs.pop("sensor_radius", 0.3)
        self.sensor_fidelity = kwargs.pop("sensor_fidelity", 0.8)
        self.initial_belief = kwargs.pop("initial_belief", 0.5)

        self.goal_reward = kwargs.pop("goal_reward", 2.0)
        self.rescue_reward = kwargs.pop("rescue_reward", 10.0)
        self.discovery_reward = kwargs.pop("discovery_reward", 1.0)
        self.distance_reward_scale = kwargs.pop("distance_reward_scale", 1.0)
        self.collision_penalty = kwargs.pop("collision_penalty", -20.0)
        self.boundary_penalty = kwargs.pop("boundary_penalty", -5.0)
        self.time_penalty = kwargs.pop("time_penalty", 0.0)
        self.retire_on_rescue = kwargs.pop("retire_on_rescue", True)
        self.auto_resample_goals = kwargs.pop("auto_resample_goals", True)
        self.done_when_all_targets_visited = kwargs.pop(
            "done_when_all_targets_visited",
            False,
        )

        self.high_level_interval = kwargs.pop("high_level_interval", 5)
        self.rrt_top_k = kwargs.pop("rrt_top_k", 5)
        self.rrt_max_iter = kwargs.pop("rrt_max_iter", 40)
        self.rrt_seed = kwargs.pop("rrt_seed", 0)
        self.enable_rrt_candidates = kwargs.pop("enable_rrt_candidates", True)
        self.comms_rendering_range = kwargs.pop("comms_rendering_range", 0.0)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        world = World(
            batch_dim,
            device,
            x_semidim=self.world_spawning_x,
            y_semidim=self.world_spawning_y,
            substeps=5,
            collision_force=500,
            dt=0.1,
            gravity=(0.0, 0.0),
            drag=DRAG,
            linear_friction=LINEAR_FRICTION,
            angular_friction=ANGULAR_FRICTION,
        )

        colors = [
            Color.BLUE,
            Color.ORANGE,
            Color.GREEN,
            Color.PINK,
            Color.PURPLE,
            Color.YELLOW,
            Color.RED,
        ]
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                color=colors[i] if i < len(colors) else torch.rand(3, device=device),
                render_action=True,
                shape=Sphere(radius=self.agent_radius),
                u_range=[1.0, 1.0],
                u_multiplier=[1.0, 1.0],
                dynamics=Holonomic(),
            )
            agent.sar_index = i
            world.add_agent(agent)

        self.targets: List[Landmark] = []
        for i in range(self.n_targets):
            target = Landmark(
                name=f"target_{i}",
                collide=False,
                movable=False,
                color=Color.GRAY,
                shape=Sphere(radius=self.target_radius),
            )
            target.sar_index = i
            world.add_landmark(target)
            self.targets.append(target)

        self._init_map_geometry(device)
        self._init_state_tensors(batch_dim, device)
        return world

    def _init_map_geometry(self, device: torch.device) -> None:
        world_min = -self.world_size / 2.0
        coords = world_min + (
            torch.arange(self.map_dim, device=device, dtype=torch.float32) + 0.5
        ) * self.cell_size
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
        self.cell_world_x = grid_x
        self.cell_world_y = grid_y

    def _init_state_tensors(self, batch_dim: int, device: torch.device) -> None:
        self.assigned_goals = torch.zeros(batch_dim, self.n_agents, 2, device=device)
        self.assigned_tasks = torch.zeros(batch_dim, self.n_agents, 1, device=device)
        self.active_agents = torch.ones(batch_dim, self.n_agents, dtype=torch.bool, device=device)
        self.goal_done = torch.ones(batch_dim, self.n_agents, dtype=torch.bool, device=device)
        self.previous_goal_dist = torch.zeros(batch_dim, self.n_agents, device=device)

        self.belief_map = torch.full(
            (batch_dim, self.map_dim, self.map_dim),
            self.initial_belief,
            dtype=torch.float32,
            device=device,
        )
        self.entropy_map = self._compute_entropy(self.belief_map)
        self.voronoi_masks = torch.zeros(
            batch_dim,
            self.n_agents,
            self.map_dim,
            self.map_dim,
            dtype=torch.bool,
            device=device,
        )
        self.agent_heatmap = torch.zeros(batch_dim, self.map_dim, self.map_dim, device=device)
        self.landmark_heatmap = torch.zeros_like(self.agent_heatmap)
        self.target_detected = torch.zeros(
            batch_dim,
            self.n_targets,
            dtype=torch.bool,
            device=device,
        )
        self.target_visited = torch.zeros_like(self.target_detected)
        self.detected_targets = torch.zeros(batch_dim, self.n_targets, 4, device=device)
        self.explore_candidates = torch.zeros(
            batch_dim,
            self.n_agents,
            self.rrt_top_k,
            4,
            device=device,
        )
        self.agent_rewards = torch.zeros(batch_dim, self.n_agents, device=device)
        self.high_rewards = torch.zeros_like(self.agent_rewards)
        self.success = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.world_steps = torch.zeros(batch_dim, dtype=torch.long, device=device)

    def reset_world_at(self, env_index: int | None = None) -> None:
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + self.targets,
            self.world,
            env_index,
            max(self.agent_radius, self.target_radius) * 2 + 0.05,
            x_bounds=(-self.world_spawning_x, self.world_spawning_x),
            y_bounds=(-self.world_spawning_y, self.world_spawning_y),
        )

        batch_slice = slice(None) if env_index is None else env_index
        self.active_agents[batch_slice] = True
        self.goal_done[batch_slice] = True
        self.assigned_tasks[batch_slice] = 0.0
        self.target_detected[batch_slice] = False
        self.target_visited[batch_slice] = False
        self.success[batch_slice] = False
        self.world_steps[batch_slice] = 0
        self.belief_map[batch_slice] = self.initial_belief

        self._refresh_maps(env_index)
        if env_index is None:
            self._sample_goals(torch.ones_like(self.goal_done))
        else:
            mask = torch.zeros_like(self.goal_done)
            mask[env_index] = True
            self._sample_goals(mask)
        self.previous_goal_dist[batch_slice] = self._goal_distances()[batch_slice]
        self._refresh_high_level_state(env_index)

    def process_action(self, agent: Agent) -> None:
        if not hasattr(agent.action, "u") or agent.action.u is None:
            return
        index = agent.sar_index
        agent.action.u = agent.action.u * self.active_agents[:, index].unsqueeze(-1)

    def reward(self, agent: Agent) -> Tensor:
        if agent == self.world.agents[0]:
            self._compute_step_rewards()
        return self.agent_rewards[:, agent.sar_index]

    def observation(self, agent: Agent) -> Dict[str, Tensor]:
        index = agent.sar_index
        other_positions = [
            other.state.pos - agent.state.pos
            for other in self.world.agents
            if other is not agent
        ]
        if other_positions:
            other_rel = torch.cat(other_positions, dim=-1)
        else:
            other_rel = torch.zeros(self.world.batch_dim, 0, device=self.world.device)

        goal = self.assigned_goals[:, index]
        obs = torch.cat([agent.state.vel, goal - agent.state.pos, other_rel], dim=-1)
        return {
            "obs": obs,
            "pos": agent.state.pos,
            "vel": agent.state.vel,
            "goal": goal,
            "goal_done": self.goal_done[:, index].unsqueeze(-1).float(),
            "active_mask": self.active_agents[:, index].unsqueeze(-1).float(),
        }

    def done(self) -> Tensor:
        timeout = self.world_steps >= self.max_steps
        if self.done_when_all_targets_visited:
            return timeout | self.success
        return timeout

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "belief_map": self.belief_map,
            "entropy_map": self.entropy_map,
            "voronoi_masks": self.voronoi_masks.float(),
            "heatmap": self.agent_heatmap,
            "landmark_heatmap": self.landmark_heatmap,
            "detected_targets": self.detected_targets,
            "visited_targets": self.target_visited.float(),
            "retired_agents": (~self.active_agents).float(),
            "success": self.success.float(),
            "goal_done": self.goal_done.float(),
            "assigned_goals": self.assigned_goals,
            "assigned_tasks": self.assigned_tasks,
            "explore_candidates": self.explore_candidates,
            "high_rewards": self.high_rewards,
            "world_steps": self.world_steps.float(),
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        geoms: List[Geom] = []
        if self.comms_rendering_range <= 0:
            return geoms
        from vmas.simulator import rendering

        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                dist = torch.linalg.vector_norm(agent1.state.pos - agent2.state.pos, dim=-1)
                if dist[env_index] <= self.comms_rendering_range:
                    line = rendering.Line(
                        agent1.state.pos[env_index],
                        agent2.state.pos[env_index],
                        width=1,
                    )
                    line.set_color(*Color.BLACK.value)
                    geoms.append(line)
        return geoms

    def set_high_level_assignments(
        self,
        goals: Tensor,
        tasks: Tensor,
        mask: Tensor | None = None,
    ) -> None:
        """Set macro goals/tasks from a fixed-interval high-level controller."""

        if mask is None:
            mask = torch.ones_like(self.goal_done)
        task_values = tasks.float()
        if task_values.ndim == 2:
            task_values = task_values.unsqueeze(-1)
        self.assigned_goals = torch.where(mask.unsqueeze(-1), goals, self.assigned_goals)
        self.assigned_tasks = torch.where(mask.unsqueeze(-1), task_values, self.assigned_tasks)
        self.goal_done = torch.where(mask, torch.zeros_like(self.goal_done), self.goal_done)
        self.previous_goal_dist = self._goal_distances()

    def _compute_step_rewards(self) -> None:
        goal_dist = self._goal_distances()
        self.goal_done = (goal_dist <= self.goal_radius) & self.active_agents

        distance_reward = self.distance_reward_scale * (self.previous_goal_dist - goal_dist)
        distance_reward = torch.where(self.active_agents, distance_reward, torch.zeros_like(distance_reward))
        active_float = self.active_agents.float()
        rewards = distance_reward - self.time_penalty * active_float

        collision_penalty = self._collision_penalties()
        boundary_penalty = self._boundary_penalties()
        rewards = rewards + collision_penalty + boundary_penalty

        rescue_reward = self._rescue_rewards()
        discovery_reward = self._update_beliefs_and_discoveries()
        rewards = rewards + rescue_reward + self.goal_reward * self.goal_done.float()

        self.high_rewards = (discovery_reward + rescue_reward - self.time_penalty) * active_float
        self.agent_rewards = rewards

        if self.auto_resample_goals:
            resample_mask = self.goal_done & self.active_agents
            if resample_mask.any():
                self._sample_goals(resample_mask)

        self.previous_goal_dist = self._goal_distances()
        self.world_steps += 1
        self.success = torch.all(self.target_visited, dim=-1)
        self._refresh_high_level_state()

    def _goal_distances(self) -> Tensor:
        agent_pos = torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        return torch.linalg.vector_norm(agent_pos - self.assigned_goals, dim=-1)

    def _sample_goals(self, mask: Tensor) -> None:
        random_goals = torch.empty_like(self.assigned_goals).uniform_(-1.0, 1.0)
        random_goals[..., 0] *= self.world_spawning_x
        random_goals[..., 1] *= self.world_spawning_y
        self.assigned_goals = torch.where(mask.unsqueeze(-1), random_goals, self.assigned_goals)
        self.assigned_tasks = torch.where(
            mask.unsqueeze(-1),
            torch.zeros_like(self.assigned_tasks),
            self.assigned_tasks,
        )
        self.goal_done = torch.where(mask, torch.zeros_like(self.goal_done), self.goal_done)

    def _collision_penalties(self) -> Tensor:
        penalties = torch.zeros_like(self.agent_rewards)
        for i, agent in enumerate(self.world.agents):
            for j, other in enumerate(self.world.agents):
                if i >= j:
                    continue
                dist = self.world.get_distance(agent, other)
                collision = dist <= (self.agent_radius * 2)
                active_pair = self.active_agents[:, i] & self.active_agents[:, j]
                penalty_mask = collision & active_pair
                penalties[:, i] += self.collision_penalty * penalty_mask.float()
                penalties[:, j] += self.collision_penalty * penalty_mask.float()
        return penalties

    def _boundary_penalties(self) -> Tensor:
        penalties = torch.zeros_like(self.agent_rewards)
        for i, agent in enumerate(self.world.agents):
            pos = agent.state.pos
            out = (
                (pos[..., 0].abs() + self.agent_radius > self.world_spawning_x)
                | (pos[..., 1].abs() + self.agent_radius > self.world_spawning_y)
            )
            penalties[:, i] = self.boundary_penalty * (out & self.active_agents[:, i]).float()
        return penalties

    def _rescue_rewards(self) -> Tensor:
        rewards = torch.zeros_like(self.agent_rewards)
        target_pos = torch.stack([target.state.pos for target in self.targets], dim=1)
        for agent_index in range(self.n_agents):
            collect_task = self.assigned_tasks[:, agent_index, 0] > 0.5
            eligible = self.goal_done[:, agent_index] & collect_task & self.active_agents[:, agent_index]
            if not eligible.any():
                continue
            goal = self.assigned_goals[:, agent_index]
            target_dists = torch.linalg.vector_norm(goal.unsqueeze(1) - target_pos, dim=-1)
            matched_dist, matched_index = target_dists.min(dim=-1)
            for target_index in range(self.n_targets):
                matched = matched_index == target_index
                new_visit = (
                    eligible
                    & matched
                    & (matched_dist < self.goal_radius)
                    & self.target_detected[:, target_index]
                    & ~self.target_visited[:, target_index]
                )
                if new_visit.any():
                    rescue_order = self.target_visited.float().sum(dim=-1) + 1.0
                    rewards[:, agent_index] += self.rescue_reward * rescue_order * new_visit.float()
                    self.target_visited[:, target_index] |= new_visit
                    if self.retire_on_rescue:
                        self.active_agents[:, agent_index] &= ~new_visit
        return rewards

    def _update_beliefs_and_discoveries(self) -> Tensor:
        before_entropy = self.entropy_map.sum(dim=(-1, -2))
        agent_pos = torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        for agent_index in range(self.n_agents):
            pos = agent_pos[:, agent_index]
            dx = self.cell_world_x.unsqueeze(0) - pos[:, 0].view(-1, 1, 1)
            dy = self.cell_world_y.unsqueeze(0) - pos[:, 1].view(-1, 1, 1)
            in_fov = (dx.square() + dy.square()) <= self.sensor_radius**2
            in_fov = in_fov & self.active_agents[:, agent_index].view(-1, 1, 1)
            updated = self._bayes_positive_update(self.belief_map)
            self.belief_map = torch.where(in_fov, updated, self.belief_map)

        self.entropy_map = self._compute_entropy(self.belief_map)
        after_entropy = self.entropy_map.sum(dim=(-1, -2))
        entropy_gain = (before_entropy - after_entropy).clamp(min=0.0)

        target_pos = torch.stack([target.state.pos for target in self.targets], dim=1)
        agent_target_dist = torch.cdist(agent_pos, target_pos)
        active_mask = self.active_agents.unsqueeze(-1)
        newly_detected = ((agent_target_dist <= self.sensor_radius) & active_mask).any(dim=1)
        newly_detected = newly_detected & ~self.target_detected
        self.target_detected |= newly_detected

        discover_counts = newly_detected.float().sum(dim=-1)
        per_agent_discovery = (
            self.discovery_reward
            * discover_counts.unsqueeze(-1)
            / max(float(self.n_agents), 1.0)
        ).expand(-1, self.n_agents)
        return per_agent_discovery + entropy_gain.unsqueeze(-1) / max(float(self.map_dim**2), 1.0)

    def _refresh_maps(self, env_index: int | None = None) -> None:
        self.entropy_map = self._compute_entropy(self.belief_map)
        self._refresh_high_level_state(env_index)

    def _refresh_high_level_state(self, env_index: int | None = None) -> None:
        agent_pos = torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        target_pos = torch.stack([target.state.pos for target in self.targets], dim=1)
        self.voronoi_masks = self._compute_voronoi_masks(agent_pos)
        self.agent_heatmap = self._heatmap(agent_pos, self.sensor_radius * 0.2)
        self.landmark_heatmap = self._heatmap(target_pos, self.target_radius)
        self.detected_targets = torch.cat(
            [
                target_pos,
                self._target_utility().unsqueeze(-1),
                self._target_claimed().unsqueeze(-1),
            ],
            dim=-1,
        )
        self.detected_targets = torch.where(
            self.target_detected.unsqueeze(-1),
            self.detected_targets,
            torch.zeros_like(self.detected_targets),
        )
        if self.enable_rrt_candidates:
            self.explore_candidates = self._compute_rrt_candidates(agent_pos)

    def _compute_voronoi_masks(self, agent_pos: Tensor) -> Tensor:
        dx = self.cell_world_x.view(1, 1, self.map_dim, self.map_dim) - agent_pos[..., 0].view(
            -1,
            self.n_agents,
            1,
            1,
        )
        dy = self.cell_world_y.view(1, 1, self.map_dim, self.map_dim) - agent_pos[..., 1].view(
            -1,
            self.n_agents,
            1,
            1,
        )
        dists = dx.square() + dy.square()
        dists = torch.where(
            self.active_agents.view(-1, self.n_agents, 1, 1),
            dists,
            torch.full_like(dists, torch.inf),
        )
        nearest = dists.argmin(dim=1)
        has_active = self.active_agents.any(dim=1).view(-1, 1, 1)
        masks = torch.stack([nearest == i for i in range(self.n_agents)], dim=1)
        return masks & has_active.unsqueeze(1)

    def _heatmap(self, positions: Tensor, radius: float) -> Tensor:
        sigma = max(radius / 2.0, self.cell_size)
        maps = []
        for entity_index in range(positions.shape[1]):
            pos = positions[:, entity_index]
            dx = self.cell_world_x.unsqueeze(0) - pos[:, 0].view(-1, 1, 1)
            dy = self.cell_world_y.unsqueeze(0) - pos[:, 1].view(-1, 1, 1)
            dist_sq = dx.square() + dy.square()
            heat = torch.exp(-dist_sq / (2 * sigma**2))
            heat = torch.where(dist_sq <= radius**2, heat, torch.zeros_like(heat))
            maps.append(heat)
        return torch.stack(maps, dim=0).max(dim=0).values if maps else torch.zeros_like(self.belief_map)

    def _target_utility(self) -> Tensor:
        retired = (~self.active_agents).float().sum(dim=-1, keepdim=True)
        return torch.where(
            self.target_detected & ~self.target_visited,
            retired + 2.0,
            torch.zeros_like(self.target_detected, dtype=torch.float32),
        )

    def _target_claimed(self) -> Tensor:
        target_pos = torch.stack([target.state.pos for target in self.targets], dim=1)
        collect_goals = self.assigned_tasks[..., 0] > 0.5
        dists = torch.cdist(self.assigned_goals, target_pos)
        claimed = ((dists <= self.goal_radius) & collect_goals.unsqueeze(-1)).any(dim=1)
        return claimed.float()

    def _compute_rrt_candidates(self, agent_pos: Tensor) -> Tensor:
        cfg = RRTConfig(
            top_k=self.rrt_top_k,
            max_iterations=self.rrt_max_iter,
            world_semidim=self.world_size / 2.0,
        )
        candidates = plan_batch(
            agent_pos.detach().cpu().numpy(),
            self.assigned_goals.detach().cpu().numpy(),
            self.voronoi_masks.detach().cpu().numpy(),
            self.entropy_map.detach().cpu().numpy(),
            config=cfg,
            seed=self.rrt_seed,
        )
        return torch.as_tensor(candidates, dtype=torch.float32, device=self.world.device)

    def _bayes_positive_update(self, belief: Tensor) -> Tensor:
        ps = self.sensor_fidelity
        numerator = ps * belief
        denominator = numerator + (1.0 - ps) * (1.0 - belief)
        return numerator / denominator.clamp_min(1e-8)

    @staticmethod
    def _compute_entropy(belief: Tensor) -> Tensor:
        p = belief.clamp(1e-6, 1 - 1e-6)
        return -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))
