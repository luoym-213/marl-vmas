"""VMAS search-and-rescue scenario with goal-conditioned low-level control."""

from __future__ import annotations

import typing
from itertools import combinations, permutations
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

from comm_spread.mpe_physics import MPEParityWorld
from comm_spread.rrt import RRTConfig, plan_batch
from comm_spread.team_belief import (
    bayesian_update,
    detected_target_centroids,
    individual_fov_masks,
    target_occupancy_map,
)

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class SarScenario(BaseScenario):
    """Hierarchical SAR environment state for VMAS and BenchMARL experiments.

    The base scenario exposes a low-level goal-conditioned control problem. It
    also maintains the high-level belief-map state and RRT candidates needed by
    the fixed-interval high-level wrapper/model.
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.mode = kwargs.pop("mode", "debug")
        self.emit_info = kwargs.pop("emit_info", self.mode != "low")
        self.enable_high_level_state = kwargs.pop(
            "enable_high_level_state",
            self.mode != "low",
        )
        self.high_level_progress_features = kwargs.pop("high_level_progress_features", False)
        self.target_assignment_features = kwargs.pop("target_assignment_features", False)
        self.n_agents = kwargs.pop("n_agents", 3)
        self.n_targets = kwargs.pop("n_targets", self.n_agents)
        self.physics_profile = kwargs.pop("physics_profile", "legacy")
        if self.physics_profile not in {"legacy", "mpe_strict"}:
            raise ValueError(f"invalid physics profile: {self.physics_profile}")
        strict_physics = self.physics_profile == "mpe_strict"
        self.mpe_action_force_scale = kwargs.pop("mpe_action_force_scale", 5.0)
        self.world_substeps = kwargs.pop("world_substeps", 1 if strict_physics else 5)
        self.world_collision_force = kwargs.pop(
            "world_collision_force", 100 if strict_physics else 500
        )
        self.world_contact_margin = kwargs.pop("world_contact_margin", 0.001)
        self.world_dt = kwargs.pop("world_dt", 0.1)
        self.world_drag = kwargs.pop("world_drag", 0.25)
        self.world_linear_friction = kwargs.pop("world_linear_friction", 0.0)
        self.world_angular_friction = kwargs.pop("world_angular_friction", 0.0)
        self.world_hard_bounds = kwargs.pop("world_hard_bounds", not strict_physics)
        self.staged_rescue = kwargs.pop("staged_rescue", False)
        self.rescue_detected_threshold = kwargs.pop("rescue_detected_threshold", self.n_targets)
        self.rescue_entropy_threshold = kwargs.pop("rescue_entropy_threshold", None)
        self.dynamic_rescue_release = kwargs.pop("dynamic_rescue_release", False)
        self.dynamic_release_min_searchers = kwargs.pop(
            "dynamic_release_min_searchers", 2
        )
        self.dynamic_release_entropy_ratio_threshold = kwargs.pop(
            "dynamic_release_entropy_ratio_threshold", 0.58
        )
        self.dynamic_release_entropy_rate_threshold = kwargs.pop(
            "dynamic_release_entropy_rate_threshold", 0.0015
        )
        self.dynamic_release_min_stagnation_step = kwargs.pop(
            "dynamic_release_min_stagnation_step", 25
        )
        self.dynamic_release_search_steps_per_target = kwargs.pop(
            "dynamic_release_search_steps_per_target", 22.0
        )
        self.dynamic_release_speed_per_step = kwargs.pop(
            "dynamic_release_speed_per_step", 0.035
        )
        self.dynamic_release_time_margin = kwargs.pop(
            "dynamic_release_time_margin", 8.0
        )
        self.dynamic_release_max_new_agents_per_event = kwargs.pop(
            "dynamic_release_max_new_agents_per_event", 1
        )
        self.dynamic_rescue_only_after_all_detected = kwargs.pop(
            "dynamic_rescue_only_after_all_detected", False
        )
        self.redecide_on_detection_change = kwargs.pop(
            "redecide_on_detection_change", False
        )
        self.redecide_on_assignment_change = kwargs.pop(
            "redecide_on_assignment_change", False
        )
        self.enable_finder_first_cascade = kwargs.pop(
            "enable_finder_first_cascade", False
        )
        self.finder_cascade_mode = kwargs.pop(
            "finder_cascade_mode", "immediate"
        )
        if self.finder_cascade_mode not in {
            "finder_only",
            "immediate",
            "immediate_open",
            "one_event",
        }:
            raise ValueError(f"invalid finder cascade mode: {self.finder_cascade_mode}")
        self.max_steps = kwargs.pop("scenario_max_steps", kwargs.pop("max_steps", 100))

        self.world_spawning_x = kwargs.pop("world_spawning_x", 1.0)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 1.0)
        self.world_size = kwargs.pop("belief_world_size", self.world_spawning_x * 2.0)
        self.cell_size = kwargs.pop("belief_cell_size", 0.02)
        self.map_dim = int(round(self.world_size / self.cell_size))

        self.agent_radius = kwargs.pop("agent_radius", 0.05)
        self.target_radius = kwargs.pop("target_radius", 0.05)
        self.goal_radius = kwargs.pop("goal_radius", self.target_radius)
        self.sensor_radius = kwargs.pop("sensor_radius", 0.3)
        self.sensor_fidelity = kwargs.pop("sensor_fidelity", 0.8)
        self.initial_belief = kwargs.pop("initial_belief", 0.5)
        self.belief_detection_threshold = kwargs.pop(
            "belief_detection_threshold", 0.95
        )
        self.belief_include_inactive_agents = kwargs.pop(
            "belief_include_inactive_agents", True
        )

        self.goal_reward = kwargs.pop("goal_reward", 2.0)
        self.rescue_reward = kwargs.pop("rescue_reward", 10.0)
        self.discovery_reward = kwargs.pop("discovery_reward", 1.0)
        self.early_rescue_penalty = kwargs.pop("early_rescue_penalty", 0.0)
        self.early_rescue_detected_threshold = kwargs.pop(
            "early_rescue_detected_threshold",
            self.n_targets,
        )
        self.search_capacity_discovery_bonus = kwargs.pop(
            "search_capacity_discovery_bonus",
            0.0,
        )
        self.discovery_active_agents_threshold = kwargs.pop(
            "discovery_active_agents_threshold",
            2,
        )
        self.all_targets_detected_bonus = kwargs.pop("all_targets_detected_bonus", 0.0)
        self.all_targets_detected_bonus_requires_no_rescue = kwargs.pop(
            "all_targets_detected_bonus_requires_no_rescue",
            True,
        )
        self.rescue_phase_explore_penalty = kwargs.pop("rescue_phase_explore_penalty", 0.0)
        self.rescue_phase_detected_threshold = kwargs.pop(
            "rescue_phase_detected_threshold",
            2,
        )
        self.rescue_phase_min_search_agents = kwargs.pop(
            "rescue_phase_min_search_agents",
            1,
        )
        self.unique_rescue_assignment_bonus = kwargs.pop(
            "unique_rescue_assignment_bonus",
            0.0,
        )
        self.duplicate_rescue_assignment_penalty = kwargs.pop(
            "duplicate_rescue_assignment_penalty",
            0.0,
        )
        self.detected_unassigned_target_penalty = kwargs.pop(
            "detected_unassigned_target_penalty",
            0.0,
        )
        self.distance_reward_scale = kwargs.pop("distance_reward_scale", 1.0)
        self.collision_penalty = kwargs.pop("collision_penalty", -20.0)
        self.collision_distance = kwargs.pop("collision_distance", 0.0)
        self.collision_safe_distance = kwargs.pop("collision_safe_distance", 0.0)
        self.max_collision_penalty = kwargs.pop(
            "max_collision_penalty",
            self.collision_penalty,
        )
        self.boundary_penalty = kwargs.pop("boundary_penalty", -5.0)
        self.time_penalty = kwargs.pop("time_penalty", 0.0)
        self.retire_on_rescue = kwargs.pop("retire_on_rescue", True)
        # End-to-end baselines do not own a high-level assignment interface.
        # In this mode a detected target is rescued by physical proximity,
        # without manufacturing an assigned goal or collect option.
        self.direct_rescue = kwargs.pop("direct_rescue", False)
        self.direct_reward_profile = kwargs.pop(
            "direct_reward_profile", "sparse_v1"
        )
        if self.direct_reward_profile not in {
            "sparse_v1",
            "observable_dense_v2",
            "higsar_aligned_assignment_v1",
        }:
            raise ValueError(
                f"invalid direct reward profile: {self.direct_reward_profile}"
            )
        self.direct_discovery_reward_scale = kwargs.pop(
            "direct_discovery_reward_scale", 1.0
        )
        self.direct_target_progress_scale = kwargs.pop(
            "direct_target_progress_scale", 0.0
        )
        self.direct_assignment_mode = kwargs.pop(
            "direct_assignment_mode", "none"
        )
        if self.direct_assignment_mode not in {
            "none",
            "minimum_distance_one_to_one",
        }:
            raise ValueError(
                f"invalid direct assignment mode: {self.direct_assignment_mode}"
            )
        self.direct_assignment_progress_scale = float(
            kwargs.pop("direct_assignment_progress_scale", 0.0)
        )
        if self.direct_assignment_progress_scale < 0.0:
            raise ValueError("direct assignment progress scale must be nonnegative")
        self.auto_resample_goals = kwargs.pop("auto_resample_goals", True)
        self.done_when_all_targets_visited = kwargs.pop(
            "done_when_all_targets_visited",
            False,
        )

        self.high_level_interval = kwargs.pop("high_level_interval", 5)
        self.rrt_top_k = kwargs.pop("rrt_top_k", 5)
        self.rrt_max_iter = kwargs.pop("rrt_max_iter", 40)
        self.rrt_seed = kwargs.pop("rrt_seed", 0)
        self.enable_rrt_candidates = kwargs.pop(
            "enable_rrt_candidates",
            self.mode != "low",
        )
        self.comms_rendering_range = kwargs.pop("comms_rendering_range", 0.0)
        self.render_sensor_range = kwargs.pop("render_sensor_range", True)
        self.render_assigned_goals = kwargs.pop("render_assigned_goals", True)
        self.render_entropy_map = kwargs.pop("render_entropy_map", True)
        self.render_entropy_grid = kwargs.pop("render_entropy_grid", True)
        self.render_recent_rrt_candidates = kwargs.pop(
            "render_recent_rrt_candidates",
            True,
        )
        self.sensor_range_alpha = kwargs.pop("sensor_range_alpha", 0.12)
        self.goal_marker_alpha = kwargs.pop("goal_marker_alpha", 0.9)
        self.goal_marker_radius = kwargs.pop("goal_marker_radius", None)
        self.entropy_map_alpha = kwargs.pop("entropy_map_alpha", 0.55)
        self.entropy_grid_alpha = kwargs.pop("entropy_grid_alpha", 0.12)
        self.rrt_candidate_alpha = kwargs.pop("rrt_candidate_alpha", 0.95)
        self.rrt_candidate_radius = kwargs.pop("rrt_candidate_radius", None)
        self.recent_decision_render_steps = kwargs.pop("recent_decision_render_steps", 5)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        world_class = MPEParityWorld if strict_physics else World
        world = world_class(
            batch_dim,
            device,
            x_semidim=self.world_spawning_x if self.world_hard_bounds else None,
            y_semidim=self.world_spawning_y if self.world_hard_bounds else None,
            substeps=self.world_substeps,
            collision_force=self.world_collision_force,
            contact_margin=self.world_contact_margin,
            dt=self.world_dt,
            gravity=(0.0, 0.0),
            drag=self.world_drag,
            linear_friction=self.world_linear_friction,
            angular_friction=self.world_angular_friction,
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
            action_kwargs = (
                {
                    "action_size": 1,
                    "discrete_action_nvec": [5],
                    "u_range": [1.0],
                    "u_multiplier": [1.0],
                }
                if strict_physics
                else {
                    "u_range": [1.0, 1.0],
                    "u_multiplier": [1.0, 1.0],
                }
            )
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                color=colors[i] if i < len(colors) else torch.rand(3, device=device),
                render_action=True,
                shape=Sphere(radius=self.agent_radius),
                dynamics=Holonomic(),
                **action_kwargs,
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

        self.belief_maps = torch.full(
            (batch_dim, self.n_agents, self.map_dim, self.map_dim),
            self.initial_belief,
            dtype=torch.float32,
            device=device,
        )
        self.target_detected = torch.zeros(
            batch_dim,
            self.n_agents,
            self.n_targets,
            dtype=torch.bool,
            device=device,
        )
        self.detected_target_positions = torch.zeros(
            batch_dim, self.n_targets, 2, dtype=torch.float32, device=device
        )
        self.last_individual_fov = torch.zeros(
            batch_dim, self.n_agents, self.map_dim, self.map_dim,
            dtype=torch.bool, device=device
        )
        self.last_joint_fov = torch.zeros(
            batch_dim, self.map_dim, self.map_dim, dtype=torch.bool, device=device
        )
        self.target_visited = torch.zeros(
            batch_dim,
            self.n_targets,
            dtype=torch.bool,
            device=device,
        )
        self.target_detected_step = torch.full(
            (batch_dim, self.n_targets),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.last_new_target_finders = torch.zeros(
            batch_dim,
            self.n_agents,
            self.n_targets,
            dtype=torch.bool,
            device=device,
        )
        self.target_first_finder_mask = torch.zeros_like(
            self.last_new_target_finders
        )
        self.detected_targets = torch.zeros(batch_dim, self.n_agents, self.n_targets, 4, device=device)
        self.explore_candidates = torch.zeros(
            batch_dim,
            self.n_agents,
            self.rrt_top_k,
            4,
            device=device,
        )
        self.recent_decision_ttl = torch.zeros(
            batch_dim,
            self.n_agents,
            dtype=torch.long,
            device=device,
        )
        self.recent_rrt_candidate_world = torch.zeros(
            batch_dim,
            self.n_agents,
            self.rrt_top_k,
            2,
            device=device,
        )
        self.agent_rewards = torch.zeros(batch_dim, self.n_agents, device=device)
        self.high_rewards = torch.zeros_like(self.agent_rewards)
        self.success = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.world_steps = torch.zeros(batch_dim, dtype=torch.long, device=device)
        self.previous_direct_agent_positions = torch.zeros(
            batch_dim, self.n_agents, 2, device=device
        )
        self.direct_reward_components = {
            name: torch.zeros_like(self.agent_rewards)
            for name in (
                "distance_progress",
                "discovery_entropy",
                "assignment_progress",
                "rescue",
                "collision",
                "boundary",
                "time",
            )
        }

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
        self.detected_target_positions[batch_slice] = 0.0
        self.last_individual_fov[batch_slice] = False
        self.last_joint_fov[batch_slice] = False
        self.target_visited[batch_slice] = False
        self.target_detected_step[batch_slice] = -1
        self.last_new_target_finders[batch_slice] = False
        self.target_first_finder_mask[batch_slice] = False
        self.recent_decision_ttl[batch_slice] = 0
        self.recent_rrt_candidate_world[batch_slice] = 0.0
        self.success[batch_slice] = False
        self.world_steps[batch_slice] = 0
        self.belief_maps[batch_slice] = self.initial_belief
        current_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        self.previous_direct_agent_positions[batch_slice] = current_positions[
            batch_slice
        ]
        for component in self.direct_reward_components.values():
            component[batch_slice] = 0.0

        # Match MPE reset semantics: the shared team map receives one sensor
        # update before the first high-level decision. One positive update is
        # still below the 0.95 detection threshold.
        self._update_beliefs_and_discoveries(
            record_discovery_events=False, env_index=env_index
        )
        self._refresh_maps(env_index)
        if env_index is None:
            self._sample_goals(torch.ones_like(self.goal_done))
        else:
            mask = torch.zeros_like(self.goal_done)
            mask[env_index] = True
            self._sample_goals(mask)
        self.previous_goal_dist[batch_slice] = self._goal_distances()[batch_slice]
        if self.enable_high_level_state:
            self._refresh_high_level_state(env_index)

    def process_action(self, agent: Agent) -> None:
        if not hasattr(agent.action, "u") or agent.action.u is None:
            return
        index = agent.sar_index
        if self.physics_profile == "mpe_strict":
            raw = agent.action.u[:, 0]
            mapped = torch.zeros(
                self.world.batch_dim,
                2,
                dtype=torch.float32,
                device=self.world.device,
            )
            mapped[raw <= -0.75, 0] = -1.0
            mapped[(raw > -0.75) & (raw < -1e-6), 0] = 1.0
            mapped[(raw > 1e-6) & (raw < 0.75), 1] = -1.0
            mapped[raw >= 0.75, 1] = 1.0
            agent.action.u = mapped * self.mpe_action_force_scale
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
        if self.mode == "low":
            return {"obs": obs}
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
        if not self.emit_info:
            return {}

        index = agent.sar_index
        entropy_maps = self._compute_entropy(self.belief_maps)
        agent_heatmap = self._agent_heatmaps()
        landmark_heatmap = self._target_heatmaps()
        voronoi_masks = self._compute_voronoi_masks(
            torch.stack([a.state.pos for a in self.world.agents], dim=1)
        )
        return {
            "belief_map": self.belief_maps[:, index],
            "team_belief_map": self.belief_maps[:, 0],
            "joint_fov_mask": self.last_joint_fov,
            "individual_fov_masks": self.last_individual_fov,
            "detected_target_positions": self.detected_target_positions,
            "entropy_map": entropy_maps[:, index],
            "voronoi_masks": voronoi_masks[:, index].float(),
            "heatmap": agent_heatmap[:, index],
            "landmark_heatmap": landmark_heatmap[:, index],
            "detected_targets": self.detected_targets[:, index],
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
        from vmas.simulator import rendering

        if self.render_entropy_map:
            geoms.append(self._make_entropy_render_image(env_index, rendering))

        if self.render_entropy_grid:
            geoms.extend(self._make_entropy_grid_lines(rendering))

        for agent_index, agent in enumerate(self.world.agents):
            if not bool(self.active_agents[env_index, agent_index]):
                continue

            color = self._render_color(agent.color, env_index)
            if self.render_sensor_range:
                sensor_circle = rendering.make_circle(
                    float(self.sensor_radius),
                    filled=True,
                )
                sensor_xform = rendering.Transform()
                sensor_circle.add_attr(sensor_xform)
                sensor_pos = agent.state.pos[env_index].detach().cpu()
                sensor_xform.set_translation(float(sensor_pos[0]), float(sensor_pos[1]))
                sensor_circle.set_color(
                    color[0],
                    color[1],
                    color[2],
                    alpha=float(self.sensor_range_alpha),
                )
                geoms.append(sensor_circle)

            if (
                self.render_recent_rrt_candidates
                and bool(self.recent_decision_ttl[env_index, agent_index] > 0)
            ):
                geoms.extend(
                    self._make_rrt_candidate_geoms(
                        env_index,
                        agent_index,
                        agent,
                        color,
                        rendering,
                    )
                )

            if self.render_assigned_goals:
                marker_radius = (
                    max(float(self.goal_radius) * 0.5, float(self.agent_radius) * 0.5)
                    if self.goal_marker_radius is None
                    else float(self.goal_marker_radius)
                )
                goal_circle = rendering.make_circle(marker_radius, filled=False)
                goal_xform = rendering.Transform()
                goal_circle.add_attr(goal_xform)
                goal = self.assigned_goals[env_index, agent_index].detach().cpu()
                goal_xform.set_translation(float(goal[0]), float(goal[1]))
                goal_circle.set_color(
                    color[0],
                    color[1],
                    color[2],
                    alpha=float(self.goal_marker_alpha),
                )
                geoms.append(goal_circle)

        if self.comms_rendering_range <= 0:
            return geoms

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

    def _make_entropy_render_image(self, env_index: int, rendering):
        entropy = self._render_entropy_map(env_index).detach().cpu().numpy()
        entropy = np.clip(entropy, 0.0, 1.0)
        low = np.asarray([255.0, 240.0, 190.0], dtype=np.float32)
        high = np.asarray([247.0, 130.0, 142.0], dtype=np.float32)
        rgb = low[None, None, :] * (1.0 - entropy[..., None]) + high[None, None, :] * entropy[..., None]
        alpha = np.full((*entropy.shape, 1), 255.0 * float(self.entropy_map_alpha), dtype=np.float32)
        image = np.concatenate([rgb, alpha], axis=-1).astype(np.uint8)
        image = np.transpose(image, (1, 0, 2))
        return rendering.Image(
            image,
            -self.world_size / 2.0,
            -self.world_size / 2.0,
            self.cell_size,
        )

    def _render_entropy_map(self, env_index: int) -> Tensor:
        entropy = self._compute_entropy(self.belief_maps[env_index])
        return entropy.min(dim=0).values

    def _make_entropy_grid_lines(self, rendering) -> "List[Geom]":
        geoms: List[Geom] = []
        world_min = -self.world_size / 2.0
        world_max = self.world_size / 2.0
        color = (0.35, 0.35, 0.35)
        for index in range(self.map_dim + 1):
            coord = world_min + index * self.cell_size
            vertical = rendering.Line((coord, world_min), (coord, world_max), width=0.25)
            vertical.set_color(*color, alpha=float(self.entropy_grid_alpha))
            geoms.append(vertical)
            horizontal = rendering.Line((world_min, coord), (world_max, coord), width=0.25)
            horizontal.set_color(*color, alpha=float(self.entropy_grid_alpha))
            geoms.append(horizontal)
        return geoms

    def _make_rrt_candidate_geoms(
        self,
        env_index: int,
        agent_index: int,
        agent: Agent,
        color: tuple[float, float, float],
        rendering,
    ) -> "List[Geom]":
        geoms: List[Geom] = []
        candidates = self.recent_rrt_candidate_world[env_index, agent_index].detach().cpu()
        for candidate_index, candidate_pos in enumerate(candidates):
            base_radius = (
                self.agent_radius * 0.35
                if self.rrt_candidate_radius is None
                else float(self.rrt_candidate_radius)
            )
            radius = float(base_radius) * (1.45 if candidate_index == 0 else 1.0)
            marker = rendering.make_circle(radius, filled=candidate_index == 0)
            marker_xform = rendering.Transform()
            marker.add_attr(marker_xform)
            marker_xform.set_translation(float(candidate_pos[0]), float(candidate_pos[1]))
            marker.set_color(
                color[0],
                color[1],
                color[2],
                alpha=float(self.rrt_candidate_alpha),
            )
            geoms.append(marker)
        return geoms

    @staticmethod
    def _render_color(color, env_index: int) -> tuple[float, float, float]:
        if hasattr(color, "value"):
            color = color.value
        if isinstance(color, torch.Tensor):
            if color.ndim > 1:
                color = color[env_index]
            color = color.detach().cpu().tolist()
        values = list(color)
        if len(values) < 3:
            return (0.0, 0.0, 0.0)
        return (float(values[0]), float(values[1]), float(values[2]))

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

    def mark_recent_high_level_decisions(
        self,
        mask: Tensor,
        candidate_world: Tensor | None = None,
    ) -> None:
        ttl = torch.full_like(
            self.recent_decision_ttl,
            int(self.recent_decision_render_steps),
        )
        self.recent_decision_ttl = torch.where(mask, ttl, self.recent_decision_ttl)
        if candidate_world is not None:
            self.recent_rrt_candidate_world = torch.where(
                mask.unsqueeze(-1).unsqueeze(-1),
                candidate_world,
                self.recent_rrt_candidate_world,
            )

    def _compute_step_rewards(self) -> None:
        self.recent_decision_ttl = torch.clamp(self.recent_decision_ttl - 1, min=0)
        goal_dist = self._goal_distances()
        if self.direct_rescue:
            self.goal_done.zero_()
        else:
            self.goal_done = (goal_dist <= self.goal_radius) & self.active_agents

        distance_reward = self.distance_reward_scale * (self.previous_goal_dist - goal_dist)
        distance_reward = torch.where(self.active_agents, distance_reward, torch.zeros_like(distance_reward))
        if self.direct_rescue:
            distance_reward.zero_()
        direct_progress_reward = torch.zeros_like(distance_reward)
        if self.direct_rescue:
            direct_progress_reward = self._direct_target_progress_rewards()
        direct_assignment_reward = torch.zeros_like(distance_reward)
        if self.direct_rescue:
            direct_assignment_reward = self._direct_assignment_progress_rewards()
        active_float = self.active_agents.float()
        time_reward = -self.time_penalty * active_float
        rewards = (
            distance_reward
            + direct_progress_reward
            + direct_assignment_reward
            + time_reward
        )

        collision_penalty = self._collision_penalties()
        boundary_penalty = self._boundary_penalties()
        rewards = rewards + collision_penalty + boundary_penalty

        active_count_before = self.active_agents.float().sum(dim=-1)
        visited_count_before = self.target_visited.float().sum(dim=-1)
        rescue_phase_reward = self._rescue_phase_rewards(active_count_before)
        assignment_shaping_reward = self._assignment_shaping_rewards()
        rescue_reward = self._rescue_rewards()
        discovery_reward = self._update_beliefs_and_discoveries(
            active_count_before=active_count_before,
            visited_count_before=visited_count_before,
        )
        rewards = rewards + rescue_reward
        if self.direct_rescue:
            scaled_discovery_reward = (
                self.direct_discovery_reward_scale * discovery_reward
            )
            rewards = rewards + scaled_discovery_reward
        if not self.direct_rescue:
            rewards = rewards + self.goal_reward * self.goal_done.float()

        self.high_rewards = (
            discovery_reward
            + rescue_reward
            + rescue_phase_reward
            + assignment_shaping_reward
            - self.time_penalty
        ) * active_float
        self.agent_rewards = rewards
        if self.direct_rescue:
            self.direct_reward_components["distance_progress"].copy_(
                direct_progress_reward
            )
            self.direct_reward_components["discovery_entropy"].copy_(
                scaled_discovery_reward
            )
            self.direct_reward_components["assignment_progress"].copy_(
                direct_assignment_reward
            )
            self.direct_reward_components["rescue"].copy_(rescue_reward)
            self.direct_reward_components["collision"].copy_(collision_penalty)
            self.direct_reward_components["boundary"].copy_(boundary_penalty)
            self.direct_reward_components["time"].copy_(time_reward)

        if self.auto_resample_goals:
            resample_mask = self.goal_done & self.active_agents
            if resample_mask.any():
                self._sample_goals(resample_mask)

        self.previous_goal_dist = self._goal_distances()
        self.previous_direct_agent_positions.copy_(
            torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        )
        self.world_steps += 1
        self.success = torch.all(self.target_visited, dim=-1)
        self._refresh_high_level_state()

    def _direct_target_progress_rewards(self) -> Tensor:
        """Potential progress to the nearest observable, unvisited target."""

        rewards = torch.zeros_like(self.agent_rewards)
        if self.direct_target_progress_scale == 0:
            return rewards
        known = self.target_detected.any(dim=1) & ~self.target_visited
        if not known.any():
            return rewards
        current_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        target_positions = self.detected_target_positions
        previous_distances = torch.cdist(
            self.previous_direct_agent_positions, target_positions
        )
        current_distances = torch.cdist(current_positions, target_positions)
        mask = known.unsqueeze(1)
        previous_min = previous_distances.masked_fill(~mask, torch.inf).min(
            dim=-1
        ).values
        current_min = current_distances.masked_fill(~mask, torch.inf).min(
            dim=-1
        ).values
        valid = (
            known.any(dim=-1).unsqueeze(-1)
            & self.active_agents
            & torch.isfinite(previous_min)
            & torch.isfinite(current_min)
        )
        progress = previous_min - current_min
        return torch.where(
            valid,
            self.direct_target_progress_scale * progress,
            rewards,
        )

    def _direct_assignment_progress_rewards(self) -> Tensor:
        """Progress under a deterministic observable one-to-one assignment."""

        rewards = torch.zeros_like(self.agent_rewards)
        if (
            self.direct_assignment_mode == "none"
            or self.direct_assignment_progress_scale == 0.0
        ):
            return rewards

        # This snapshot is deliberately taken before rescue and discovery updates
        # for the current transition. Hidden or newly discovered targets cannot
        # influence either the matching or its reward.
        known = self.target_detected.any(dim=1) & ~self.target_visited
        if not known.any():
            return rewards

        current_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        target_positions = self.detected_target_positions
        previous_distances = torch.cdist(
            self.previous_direct_agent_positions, target_positions
        )
        current_distances = torch.cdist(current_positions, target_positions)

        for env_index in range(self.world.batch_dim):
            agent_ids = torch.nonzero(
                self.active_agents[env_index], as_tuple=False
            ).flatten().tolist()
            target_ids = torch.nonzero(
                known[env_index], as_tuple=False
            ).flatten().tolist()
            pair_count = min(len(agent_ids), len(target_ids))
            if pair_count == 0:
                continue

            best_pairs = None
            best_cost = float("inf")
            for selected_agents in combinations(agent_ids, pair_count):
                for selected_targets in permutations(target_ids, pair_count):
                    pairs = tuple(zip(selected_agents, selected_targets))
                    cost = sum(
                        float(previous_distances[env_index, agent_id, target_id])
                        for agent_id, target_id in pairs
                    )
                    # combinations/permutations traverse sorted ids, so keeping
                    # the first minimum gives a stable lexicographic tie break.
                    if cost < best_cost - 1e-12:
                        best_cost = cost
                        best_pairs = pairs

            assert best_pairs is not None
            for agent_id, target_id in best_pairs:
                progress = (
                    previous_distances[env_index, agent_id, target_id]
                    - current_distances[env_index, agent_id, target_id]
                )
                rewards[env_index, agent_id] = (
                    self.direct_assignment_progress_scale * progress
                )
        return rewards

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
                active_pair = self.active_agents[:, i] & self.active_agents[:, j]
                if self.collision_safe_distance > 0:
                    violation = (
                        1.0 - dist / self.collision_safe_distance
                    ).clamp(min=0.0)
                    pair_penalty = self.collision_penalty * violation.square()
                    pair_penalty = pair_penalty.clamp(min=self.max_collision_penalty)
                    pair_penalty = torch.where(
                        active_pair,
                        pair_penalty,
                        torch.zeros_like(pair_penalty),
                    )
                else:
                    collision = dist <= self.collision_distance
                    pair_penalty = self.collision_penalty * (collision & active_pair).float()
                penalties[:, i] += pair_penalty
                penalties[:, j] += pair_penalty
        return penalties

    def _boundary_penalties(self) -> Tensor:
        penalties = torch.zeros_like(self.agent_rewards)
        for i, agent in enumerate(self.world.agents):
            pos = agent.state.pos
            if self.physics_profile == "mpe_strict":
                out = (
                    (pos[..., 0].abs() >= self.world_spawning_x)
                    | (pos[..., 1].abs() >= self.world_spawning_y)
                )
            else:
                out = (
                    (pos[..., 0].abs() + self.agent_radius > self.world_spawning_x)
                    | (pos[..., 1].abs() + self.agent_radius > self.world_spawning_y)
                )
            penalties[:, i] = self.boundary_penalty * (out & self.active_agents[:, i]).float()
        return penalties

    def _rescue_phase_rewards(self, active_count_before: Tensor) -> Tensor:
        rewards = torch.zeros_like(self.agent_rewards)
        if self.rescue_phase_explore_penalty <= 0:
            return rewards
        detected_any = self.target_detected.any(dim=1)
        detected_count = detected_any.float().sum(dim=-1)
        known_unvisited_count = (detected_any & ~self.target_visited).float().sum(dim=-1)
        collect_task = self.assigned_tasks[..., 0] > 0.5
        active_collect_count = (collect_task & self.active_agents).float().sum(dim=-1)
        max_rescuers = (
            active_count_before - float(self.rescue_phase_min_search_agents)
        ).clamp(min=1.0, max=float(self.n_agents))
        desired_rescuers = torch.minimum(known_unvisited_count, max_rescuers)
        need_more_rescuers = active_collect_count < desired_rescuers
        rescue_phase = (
            (detected_count >= float(self.rescue_phase_detected_threshold))
            & (known_unvisited_count > 0)
            & need_more_rescuers
        )
        explore_task = ~collect_task
        rewards = rewards - (
            self.rescue_phase_explore_penalty
            * rescue_phase.float().unsqueeze(-1)
            * explore_task.float()
            * self.active_agents.float()
        )
        return rewards

    def _assignment_shaping_rewards(self) -> Tensor:
        rewards = torch.zeros_like(self.agent_rewards)
        if (
            self.unique_rescue_assignment_bonus <= 0
            and self.duplicate_rescue_assignment_penalty <= 0
            and self.detected_unassigned_target_penalty <= 0
        ):
            return rewards
        target_pos = self.detected_target_positions
        collect_task = self.assigned_tasks[..., 0] > 0.5
        dists = torch.cdist(self.assigned_goals, target_pos)
        min_dist, assigned_index = dists.min(dim=-1)
        assigned_valid = collect_task & (min_dist <= self.goal_radius) & self.active_agents
        assigned_one_hot = torch.nn.functional.one_hot(
            assigned_index.clamp_min(0),
            num_classes=self.n_targets,
        ).bool()
        assigned_one_hot = assigned_one_hot & assigned_valid.unsqueeze(-1)
        detected_any = self.target_detected.any(dim=1)
        known_unvisited = detected_any & ~self.target_visited
        claim_counts = assigned_one_hot.sum(dim=1)
        if self.unique_rescue_assignment_bonus > 0:
            unique_claim = (
                assigned_one_hot
                & known_unvisited.unsqueeze(1)
                & (claim_counts == 1).unsqueeze(1)
            )
            rewards = rewards + self.unique_rescue_assignment_bonus * unique_claim.any(dim=-1).float()
        if self.duplicate_rescue_assignment_penalty > 0:
            duplicate_claim = (
                assigned_one_hot
                & known_unvisited.unsqueeze(1)
                & (claim_counts > 1).unsqueeze(1)
            )
            rewards = rewards - self.duplicate_rescue_assignment_penalty * duplicate_claim.any(dim=-1).float()
        if self.detected_unassigned_target_penalty > 0:
            unassigned_count = (known_unvisited & (claim_counts == 0)).float().sum(dim=-1)
            active_count = self.active_agents.float().sum(dim=-1).clamp_min(1.0)
            rewards = rewards - (
                self.detected_unassigned_target_penalty
                * unassigned_count.unsqueeze(-1)
                * self.active_agents.float()
                / active_count.unsqueeze(-1)
            )
        return rewards

    def _rescue_rewards(self) -> Tensor:
        if self.direct_rescue:
            return self._direct_rescue_rewards()

        rewards = torch.zeros_like(self.agent_rewards)
        target_pos = torch.stack([target.state.pos for target in self.targets], dim=1)
        detected_count = self.target_detected.any(dim=1).float().sum(dim=-1)
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
                    & self.target_detected[:, agent_index, target_index]
                    & ~self.target_visited[:, target_index]
                )
                if new_visit.any():
                    rescue_order = self.target_visited.float().sum(dim=-1) + 1.0
                    rescue_value = self.rescue_reward * rescue_order
                    if self.early_rescue_penalty > 0:
                        early_rescue = detected_count < float(self.early_rescue_detected_threshold)
                        rescue_value = rescue_value - self.early_rescue_penalty * early_rescue.float()
                    rewards[:, agent_index] += rescue_value * new_visit.float()
                    self.target_visited[:, target_index] |= new_visit
                    if self.retire_on_rescue:
                        self.active_agents[:, agent_index] &= ~new_visit
        return rewards

    def _direct_rescue_rewards(self) -> Tensor:
        """Rescue detected targets for a non-hierarchical physical policy.

        The policy is never given target ground truth. Eligibility uses the
        persistent, team-shared detection state and physical proximity. Agent
        and target iteration order only resolves the measure-zero case where
        multiple agents enter the same rescue radius in one simulator step.
        """

        rewards = torch.zeros_like(self.agent_rewards)
        agent_pos = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        target_pos = torch.stack(
            [target.state.pos for target in self.targets], dim=1
        )
        distances = torch.cdist(agent_pos, target_pos)
        known = self.target_detected.any(dim=1)
        detected_count = known.float().sum(dim=-1)

        for agent_index in range(self.n_agents):
            for target_index in range(self.n_targets):
                new_visit = (
                    self.active_agents[:, agent_index]
                    & known[:, target_index]
                    & ~self.target_visited[:, target_index]
                    & (distances[:, agent_index, target_index] <= self.goal_radius)
                )
                if not new_visit.any():
                    continue
                rescue_order = self.target_visited.float().sum(dim=-1) + 1.0
                rescue_value = self.rescue_reward * rescue_order
                if self.early_rescue_penalty > 0:
                    early_rescue = detected_count < float(
                        self.early_rescue_detected_threshold
                    )
                    rescue_value = (
                        rescue_value
                        - self.early_rescue_penalty * early_rescue.float()
                    )
                rewards[:, agent_index] += rescue_value * new_visit.float()
                self.target_visited[:, target_index] |= new_visit
                if self.retire_on_rescue:
                    self.active_agents[:, agent_index] &= ~new_visit
        return rewards

    def _update_beliefs_and_discoveries(
        self,
        *,
        active_count_before: Tensor | None = None,
        visited_count_before: Tensor | None = None,
        record_discovery_events: bool = True,
        env_index: int | None = None,
    ) -> Tensor:
        """Update one team-shared Bayesian map and threshold detections.

        Every agent receives an identical copy in ``belief_maps`` for backward
        compatibility with existing actor/critic and RRT tensor shapes. Sensors
        from inactive agents remain enabled by default to match audited MPE.
        """

        shared_prior = self.belief_maps[:, 0]
        before_entropy = self._compute_entropy(shared_prior).sum(dim=(-1, -2))
        agent_pos = torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        target_pos = torch.stack([target.state.pos for target in self.targets], dim=1)
        occupied = target_occupancy_map(
            self.cell_world_x, self.cell_world_y, target_pos, self.target_radius
        )
        sensor_enabled = (
            None if self.belief_include_inactive_agents else self.active_agents
        )
        individual_fov = individual_fov_masks(
            self.cell_world_x,
            self.cell_world_y,
            agent_pos,
            self.sensor_radius,
            sensor_enabled,
        )
        update_env = torch.ones(
            self.world.batch_dim, dtype=torch.bool, device=self.world.device
        )
        if env_index is not None:
            update_env.zero_()
            update_env[env_index] = True
            individual_fov &= update_env.view(-1, 1, 1, 1)
        joint_fov = individual_fov.any(dim=1)
        self.last_individual_fov = torch.where(
            update_env.view(-1, 1, 1, 1),
            individual_fov,
            self.last_individual_fov,
        )
        self.last_joint_fov = torch.where(
            update_env.view(-1, 1, 1), joint_fov, self.last_joint_fov
        )

        # Preserve MPE-style per-agent counterfactual exploration rewards while
        # using the union FOV for the actual shared map state.
        individual_posteriors = bayesian_update(
            shared_prior.unsqueeze(1),
            individual_fov,
            occupied.unsqueeze(1),
            self.sensor_fidelity,
        )
        individual_entropy = self._compute_entropy(individual_posteriors).sum(
            dim=(-1, -2)
        )
        entropy_gain = (before_entropy.unsqueeze(-1) - individual_entropy).clamp_min(0.0)

        shared_posterior = bayesian_update(
            shared_prior, joint_fov, occupied, self.sensor_fidelity
        )
        self.belief_maps.copy_(
            shared_posterior.unsqueeze(1).expand_as(self.belief_maps)
        )

        belief_detected, belief_centroids = detected_target_centroids(
            shared_posterior,
            self.belief_detection_threshold,
            self.cell_world_x,
            self.cell_world_y,
            target_pos,
            self.target_radius,
        )
        detected_any_before = self.target_detected.any(dim=1)
        globally_new = belief_detected & ~detected_any_before
        self.detected_target_positions = torch.where(
            belief_detected.unsqueeze(-1),
            belief_centroids,
            self.detected_target_positions,
        )
        shared_detected = belief_detected.unsqueeze(1).expand(
            -1, self.n_agents, -1
        )
        self.target_detected |= shared_detected
        self.target_detected_step = torch.where(
            globally_new & (self.target_detected_step < 0),
            self.world_steps.view(-1, 1).expand_as(self.target_detected_step),
            self.target_detected_step,
        )

        self.last_new_target_finders.zero_()
        newly_detected = torch.zeros_like(self.target_detected)
        if record_discovery_events and globally_new.any():
            agent_target_dist = torch.cdist(agent_pos, target_pos)
            candidates = (agent_target_dist <= self.sensor_radius)
            candidates &= self.active_agents.unsqueeze(-1)
            candidates &= globally_new.unsqueeze(1)
            candidate_distances = agent_target_dist.masked_fill(~candidates, torch.inf)
            minimum_finder_distance = candidate_distances.min(dim=1).values
            self.last_new_target_finders = (
                candidates
                & (candidate_distances <= minimum_finder_distance.unsqueeze(1) + 1e-6)
            )
            self.target_first_finder_mask |= self.last_new_target_finders
            newly_detected = candidates

        discover_counts = newly_detected.float().sum(dim=-1)
        per_agent_discovery = self.discovery_reward * discover_counts
        if self.search_capacity_discovery_bonus > 0:
            if active_count_before is None:
                active_count_before = self.active_agents.float().sum(dim=-1)
            enough_search_capacity = active_count_before >= float(
                self.discovery_active_agents_threshold
            )
            per_agent_discovery = per_agent_discovery + (
                self.search_capacity_discovery_bonus
                * discover_counts
                * enough_search_capacity.float().unsqueeze(-1)
            )
        if self.all_targets_detected_bonus > 0:
            detected_any_after = self.target_detected.any(dim=1)
            crossed_all_detected = (
                detected_any_after.all(dim=-1)
                & ~detected_any_before.all(dim=-1)
                & record_discovery_events
            )
            if self.all_targets_detected_bonus_requires_no_rescue:
                if visited_count_before is None:
                    visited_count_before = self.target_visited.float().sum(dim=-1)
                crossed_all_detected &= visited_count_before <= 0
            detector_count = newly_detected.float().sum(dim=(1, 2)).clamp_min(1.0)
            detector_share = newly_detected.float().sum(dim=-1) / detector_count.unsqueeze(-1)
            per_agent_discovery = per_agent_discovery + (
                self.all_targets_detected_bonus
                * crossed_all_detected.float().unsqueeze(-1)
                * detector_share
            )
        return per_agent_discovery + entropy_gain / max(float(self.map_dim**2), 1.0)

    def _refresh_maps(self, env_index: int | None = None) -> None:
        if self.enable_high_level_state:
            self._refresh_high_level_state(env_index)

    def _refresh_high_level_state(self, env_index: int | None = None) -> None:
        agent_pos = torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        voronoi_masks = self._compute_voronoi_masks(agent_pos)
        target_pos_per_agent = self.detected_target_positions.unsqueeze(1).expand(
            -1,
            self.n_agents,
            -1,
            -1,
        )
        self.detected_targets = torch.cat(
            [
                target_pos_per_agent,
                self._target_utility().unsqueeze(-1),
                self._target_claimed().unsqueeze(-1),
            ],
            dim=-1,
        )
        self.detected_targets = torch.where(
            (self.target_detected & ~self.target_visited.unsqueeze(1)).unsqueeze(-1),
            self.detected_targets,
            torch.zeros_like(self.detected_targets),
        )
        if self.enable_rrt_candidates:
            entropy_maps = self._compute_entropy(self.belief_maps)
            self.explore_candidates = self._compute_rrt_candidates(
                agent_pos,
                voronoi_masks,
                entropy_maps,
            )

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
        return torch.stack(maps, dim=0).max(dim=0).values if maps else torch.zeros_like(self.belief_maps[:, 0])

    def _agent_heatmaps(self) -> Tensor:
        agent_pos = torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        heatmaps = []
        sigma = max((self.sensor_radius * 0.2) / 2.0, self.cell_size)
        for agent_index in range(self.n_agents):
            pos = agent_pos[:, agent_index]
            dx = self.cell_world_x.unsqueeze(0) - pos[:, 0].view(-1, 1, 1)
            dy = self.cell_world_y.unsqueeze(0) - pos[:, 1].view(-1, 1, 1)
            dist_sq = dx.square() + dy.square()
            heat = torch.exp(-dist_sq / (2 * sigma**2))
            heat = torch.where(
                dist_sq <= (self.sensor_radius * 0.2) ** 2,
                heat,
                torch.zeros_like(heat),
            )
            heatmaps.append(heat)
        return torch.stack(heatmaps, dim=1)

    def _target_heatmaps(self) -> Tensor:
        target_pos = self.detected_target_positions
        sigma = max(self.target_radius / 2.0, self.cell_size)
        heatmaps = torch.zeros_like(self.belief_maps)
        for target_index in range(self.n_targets):
            pos = target_pos[:, target_index]
            dx = self.cell_world_x.unsqueeze(0) - pos[:, 0].view(-1, 1, 1)
            dy = self.cell_world_y.unsqueeze(0) - pos[:, 1].view(-1, 1, 1)
            dist_sq = dx.square() + dy.square()
            heat = torch.exp(-dist_sq / (2 * sigma**2))
            heat = torch.where(dist_sq <= self.target_radius**2, heat, torch.zeros_like(heat))
            visible = (
                self.target_detected[:, :, target_index]
                & ~self.target_visited[:, target_index].unsqueeze(-1)
            ).unsqueeze(-1).unsqueeze(-1)
            heatmaps = torch.maximum(heatmaps, heat.unsqueeze(1) * visible.float())
        return heatmaps

    def _target_utility(self) -> Tensor:
        retired = (~self.active_agents).float().sum(dim=-1).view(-1, 1, 1)
        return torch.where(
            self.target_detected & ~self.target_visited.unsqueeze(1),
            retired + 2.0,
            torch.zeros_like(self.target_detected, dtype=torch.float32),
        )

    def _target_claimed(self) -> Tensor:
        target_pos = self.detected_target_positions
        collect_goals = self.assigned_tasks[..., 0] > 0.5
        dists = torch.cdist(self.assigned_goals, target_pos)
        known = self.target_detected.any(dim=1) & ~self.target_visited
        claimed = (
            (dists <= self.goal_radius)
            & collect_goals.unsqueeze(-1)
            & known.unsqueeze(1)
        ).any(dim=1)
        return claimed.unsqueeze(1).expand(-1, self.n_agents, -1).float()

    def _compute_rrt_candidates(
        self,
        agent_pos: Tensor,
        voronoi_masks: Tensor,
        entropy_maps: Tensor,
    ) -> Tensor:
        cfg = RRTConfig(
            top_k=self.rrt_top_k,
            max_iterations=self.rrt_max_iter,
            world_semidim=self.world_size / 2.0,
        )
        candidates = plan_batch(
            agent_pos.detach().cpu().numpy(),
            self.assigned_goals.detach().cpu().numpy(),
            voronoi_masks.detach().cpu().numpy(),
            entropy_maps.detach().cpu().numpy(),
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
