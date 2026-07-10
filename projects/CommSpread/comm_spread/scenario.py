"""VMAS simple_spread-style cooperative coverage scenario."""

from __future__ import annotations

import itertools
import typing
from typing import Dict, List

import numpy as np
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

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - dependency is declared in pyproject.
    linear_sum_assignment = None


def _exhaustive_min_assignment(dists: np.ndarray) -> np.ndarray:
    """Small exact Hungarian fallback for environments without scipy."""

    n_rows, n_cols = dists.shape
    if max(n_rows, n_cols) > 8:
        raise ImportError(
            "spread_mpe_parity with more than 8 agents/landmarks requires scipy. "
            "Install project dependencies with `python -m pip install -e .`."
        )

    if n_rows <= n_cols:
        best_cols: tuple[int, ...] | None = None
        best_cost = float("inf")
        row_indices = np.arange(n_rows)
        for cols in itertools.permutations(range(n_cols), n_rows):
            cost = float(dists[row_indices, cols].sum())
            if cost < best_cost:
                best_cost = cost
                best_cols = cols
        assert best_cols is not None
        return dists[row_indices, best_cols]

    best_rows: tuple[int, ...] | None = None
    best_cost = float("inf")
    col_indices = np.arange(n_cols)
    for rows in itertools.permutations(range(n_rows), n_cols):
        cost = float(dists[rows, col_indices].sum())
        if cost < best_cost:
            best_cost = cost
            best_rows = rows
    assert best_rows is not None
    return dists[best_rows, col_indices]


class CommSpreadScenario(BaseScenario):
    """Cooperative landmark coverage task inspired by MPE simple_spread.

    Agents are homogeneous holonomic discs. Landmarks are shared team targets,
    not agent-specific goals. The dense reward is the negative sum of each
    landmark's distance to its nearest agent, plus local collision penalties.
    The dense spread reward uses the one-step decrease in that distance sum, so
    agents are rewarded when they move the team closer to landmark coverage.
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.n_agents = kwargs.pop("n_agents", 3)
        self.n_landmarks = kwargs.pop("n_landmarks", self.n_agents)
        self.reward_mode = kwargs.pop("reward_mode", "delta")
        self.parity_include_other_agents = kwargs.pop("parity_include_other_agents", False)
        self.emit_global_state = kwargs.pop("emit_global_state", False)
        self.global_state_include_pairwise = kwargs.pop("global_state_include_pairwise", True)

        self.world_spawning_x = kwargs.pop("world_spawning_x", 1.0)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 1.0)

        self.agent_radius = kwargs.pop("agent_radius", 0.05)
        self.landmark_radius = kwargs.pop("landmark_radius", 0.04)
        self.coverage_radius = kwargs.pop("coverage_radius", 0.1)
        self.dist_threshold = kwargs.pop("dist_threshold", self.coverage_radius)
        self.min_distance_between_entities = kwargs.pop(
            "min_distance_between_entities",
            max(self.agent_radius, self.landmark_radius) * 2 + 0.05,
        )
        self.min_collision_distance = kwargs.pop(
            "min_collision_distance",
            self.agent_radius * 2,
        )

        self.collision_penalty = kwargs.pop("collision_penalty", -1.0)
        self.out_of_bounds_penalty = kwargs.pop("out_of_bounds_penalty", -1.0)
        self.distance_reward_scale = kwargs.pop("distance_reward_scale", 1.0)
        self.coverage_reward = kwargs.pop("coverage_reward", 1.0)
        self.success_reward = kwargs.pop("success_reward", 20.0)
        self.shared_rew = kwargs.pop("shared_rew", False)
        self.done_when_all_covered = kwargs.pop("done_when_all_covered", False)
        self.observe_other_velocities = kwargs.pop("observe_other_velocities", False)
        self.comms_rendering_range = kwargs.pop("comms_rendering_range", 0.0)

        self.mpe_action_force_scale = kwargs.pop("mpe_action_force_scale", 1.0)
        self.world_substeps = kwargs.pop("world_substeps", 5)
        self.world_collision_force = kwargs.pop("world_collision_force", 500)
        self.world_contact_margin = kwargs.pop("world_contact_margin", 0.001)
        self.world_dt = kwargs.pop("world_dt", 0.1)
        self.world_drag = kwargs.pop("world_drag", DRAG)
        self.world_linear_friction = kwargs.pop("world_linear_friction", LINEAR_FRICTION)
        self.world_angular_friction = kwargs.pop("world_angular_friction", ANGULAR_FRICTION)
        self.mpe_independent_spawn = kwargs.pop("mpe_independent_spawn", False)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        world = World(
            batch_dim,
            device,
            x_semidim=self.world_spawning_x,
            y_semidim=self.world_spawning_y,
            substeps=self.world_substeps,
            collision_force=self.world_collision_force,
            contact_margin=self.world_contact_margin,
            dt=self.world_dt,
            gravity=(0.0, 0.0),
            drag=self.world_drag,
            linear_friction=self.world_linear_friction,
            angular_friction=self.world_angular_friction,
        )

        known_colors = [
            Color.BLUE,
            Color.ORANGE,
            Color.GREEN,
            Color.PINK,
            Color.PURPLE,
            Color.YELLOW,
            Color.RED,
        ]

        for i in range(self.n_agents):
            color = known_colors[i] if i < len(known_colors) else torch.rand(3, device=device)
            action_kwargs = (
                {
                    "action_size": 1,
                    "discrete_action_nvec": [5],
                    "u_range": [1.0],
                    "u_multiplier": [1.0],
                }
                if self.reward_mode == "mpe_parity"
                else {
                    "u_range": [1.0, 1.0],
                    "u_multiplier": [1.0, 1.0],
                }
            )
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                color=color,
                render_action=True,
                shape=Sphere(radius=self.agent_radius),
                dynamics=Holonomic(),
                **action_kwargs,
            )
            agent.collision_rew = torch.zeros(batch_dim, device=device)
            agent.out_of_bounds_rew = torch.zeros(batch_dim, device=device)
            world.add_agent(agent)

        self.landmarks: List[Landmark] = []
        for i in range(self.n_landmarks):
            landmark = Landmark(
                name=f"landmark_{i}",
                collide=False,
                movable=False,
                color=Color.GRAY,
                shape=Sphere(radius=self.landmark_radius),
            )
            world.add_landmark(landmark)
            self.landmarks.append(landmark)

        self.spread_rew = torch.zeros(batch_dim, device=device)
        self.coverage_rew = torch.zeros(batch_dim, device=device)
        self.success_rew = torch.zeros(batch_dim, device=device)
        self.all_landmarks_covered = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.landmark_min_dists = torch.zeros(batch_dim, self.n_landmarks, device=device)
        self.matched_landmark_dists = torch.zeros(
            batch_dim,
            min(self.n_agents, self.n_landmarks),
            device=device,
        )
        self.matched_mean_dist = torch.zeros(batch_dim, device=device)
        self.previous_landmark_dist_sum = torch.zeros(batch_dim, device=device)
        self.world_steps = torch.zeros(batch_dim, dtype=torch.long, device=device)

        return world

    def global_state(self) -> Tensor:
        agent_pos = torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        agent_vel = torch.stack([agent.state.vel for agent in self.world.agents], dim=1)
        landmark_pos = torch.stack([landmark.state.pos for landmark in self.landmarks], dim=1)
        parts = [
            agent_pos.reshape(self.world.batch_dim, -1),
            agent_vel.reshape(self.world.batch_dim, -1),
            landmark_pos.reshape(self.world.batch_dim, -1),
        ]
        if self.global_state_include_pairwise:
            pairwise = torch.cdist(agent_pos, landmark_pos)
            parts.append(pairwise.reshape(self.world.batch_dim, -1))
        return torch.cat(parts, dim=-1)

    def _compute_landmark_min_dists(self) -> Tensor:
        agents_pos = torch.stack(
            [current_agent.state.pos for current_agent in self.world.agents],
            dim=1,
        )
        landmarks_pos = torch.stack(
            [landmark.state.pos for landmark in self.landmarks],
            dim=1,
        )
        return torch.cdist(agents_pos, landmarks_pos).min(dim=1).values

    def _compute_mpe_matched_dists(self) -> Tensor:
        """Match old MPE simple_spread's scipy Hungarian distance reward."""

        agents_pos = torch.stack(
            [current_agent.state.pos for current_agent in self.world.agents],
            dim=1,
        )
        landmarks_pos = torch.stack(
            [landmark.state.pos for landmark in self.landmarks],
            dim=1,
        )
        dists = torch.cdist(agents_pos, landmarks_pos)
        matched_dists = []
        for env_dists in dists.detach().cpu().numpy():
            if linear_sum_assignment is not None:
                row_index, col_index = linear_sum_assignment(env_dists)
                matched_dists.append(env_dists[row_index, col_index])
            else:
                matched_dists.append(_exhaustive_min_assignment(env_dists))
        matched_np = np.stack(matched_dists, axis=0).astype(np.float32)
        return torch.as_tensor(matched_np, device=dists.device, dtype=dists.dtype)

    def reset_world_at(self, env_index: int | None = None) -> None:
        if self.mpe_independent_spawn:
            entities = self.world.agents + self.landmarks
            for entity in entities:
                if env_index is None:
                    pos = torch.empty(
                        self.world.batch_dim,
                        self.world.dim_p,
                        device=self.world.device,
                    )
                    pos[:, 0].uniform_(-self.world_spawning_x, self.world_spawning_x)
                    pos[:, 1].uniform_(-self.world_spawning_y, self.world_spawning_y)
                    entity.set_pos(pos, batch_index=None)
                    if entity.movable:
                        entity.set_vel(torch.zeros_like(pos), batch_index=None)
                else:
                    pos = torch.empty(self.world.dim_p, device=self.world.device)
                    pos[0].uniform_(-self.world_spawning_x, self.world_spawning_x)
                    pos[1].uniform_(-self.world_spawning_y, self.world_spawning_y)
                    entity.set_pos(pos, batch_index=env_index)
                    if entity.movable:
                        entity.set_vel(torch.zeros_like(pos), batch_index=env_index)
        else:
            ScenarioUtils.spawn_entities_randomly(
                self.world.agents + self.landmarks,
                self.world,
                env_index,
                self.min_distance_between_entities,
                x_bounds=(-self.world_spawning_x, self.world_spawning_x),
                y_bounds=(-self.world_spawning_y, self.world_spawning_y),
            )

        landmark_min_dists = self._compute_landmark_min_dists()
        matched_landmark_dists = self._compute_mpe_matched_dists()
        if env_index is None:
            self.all_landmarks_covered[:] = False
            self.landmark_min_dists[:] = landmark_min_dists
            self.matched_landmark_dists[:] = matched_landmark_dists
            self.matched_mean_dist[:] = matched_landmark_dists.mean(dim=-1)
            self.previous_landmark_dist_sum[:] = landmark_min_dists.sum(dim=-1)
            self.world_steps[:] = 0
        else:
            self.all_landmarks_covered[env_index] = False
            self.landmark_min_dists[env_index] = landmark_min_dists[env_index]
            self.matched_landmark_dists[env_index] = matched_landmark_dists[env_index]
            self.matched_mean_dist[env_index] = matched_landmark_dists[env_index].mean()
            self.previous_landmark_dist_sum[env_index] = landmark_min_dists[
                env_index
            ].sum()
            self.world_steps[env_index] = 0

    def process_action(self, agent: Agent) -> None:
        if self.reward_mode != "mpe_parity":
            return
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

    def reward(self, agent: Agent) -> Tensor:
        is_first = agent == self.world.agents[0]

        if is_first:
            self.world_steps += 1
            if self.reward_mode == "mpe_parity":
                self.matched_landmark_dists = self._compute_mpe_matched_dists()
                self.matched_mean_dist = self.matched_landmark_dists.mean(dim=-1)
                self.landmark_min_dists = self._compute_landmark_min_dists()
                self.all_landmarks_covered = torch.all(
                    self.matched_landmark_dists < self.dist_threshold,
                    dim=-1,
                )
                self.spread_rew = torch.clamp(-self.matched_mean_dist, -15.0, 15.0)
                self.coverage_rew.zero_()
                self.success_rew.zero_()
                for current_agent in self.world.agents:
                    current_agent.collision_rew[:] = 0
                    current_agent.out_of_bounds_rew[:] = 0
                return self.spread_rew

            self.landmark_min_dists = self._compute_landmark_min_dists()
            landmark_dist_sum = self.landmark_min_dists.sum(dim=-1)
            self.spread_rew = self.distance_reward_scale * (
                self.previous_landmark_dist_sum - landmark_dist_sum
            )
            self.previous_landmark_dist_sum[:] = landmark_dist_sum
            covered_landmarks = self.landmark_min_dists <= self.coverage_radius
            self.coverage_rew = self.coverage_reward * covered_landmarks.float().sum(dim=-1)
            self.all_landmarks_covered = torch.all(
                covered_landmarks,
                dim=-1,
            )
            self.success_rew = self.success_reward * self.all_landmarks_covered.float()

            for current_agent in self.world.agents:
                current_agent.collision_rew[:] = 0
                current_agent.out_of_bounds_rew[:] = 0

            for current_agent in self.world.agents:
                pos = current_agent.state.pos
                out_of_bounds = (
                    (pos[..., 0].abs() + self.agent_radius > self.world_spawning_x)
                    | (pos[..., 1].abs() + self.agent_radius > self.world_spawning_y)
                )
                current_agent.out_of_bounds_rew[
                    out_of_bounds
                ] += self.out_of_bounds_penalty

            for i, current_agent in enumerate(self.world.agents):
                for j, other_agent in enumerate(self.world.agents):
                    if i >= j:
                        continue
                    collision = (
                        self.world.get_distance(current_agent, other_agent)
                        <= self.min_collision_distance
                    )
                    current_agent.collision_rew[collision] += self.collision_penalty
                    other_agent.collision_rew[collision] += self.collision_penalty

        if self.reward_mode == "mpe_parity":
            return self.spread_rew

        if self.shared_rew:
            collision_rew = sum(a.collision_rew for a in self.world.agents)
            out_of_bounds_rew = sum(a.out_of_bounds_rew for a in self.world.agents)
        else:
            collision_rew = agent.collision_rew
            out_of_bounds_rew = agent.out_of_bounds_rew
        return (
            self.spread_rew
            + self.coverage_rew
            + self.success_rew
            + collision_rew
            + out_of_bounds_rew
        )

    def observation(self, agent: Agent) -> Dict[str, Tensor]:
        landmark_rel_pos = [
            landmark.state.pos - agent.state.pos for landmark in self.landmarks
        ]
        if self.reward_mode == "mpe_parity":
            obs_parts = [agent.state.vel, agent.state.pos] + landmark_rel_pos
            if self.parity_include_other_agents:
                obs_parts += [
                    other_agent.state.pos - agent.state.pos
                    for other_agent in self.world.agents
                    if other_agent is not agent
                ]
            return {
                "obs": torch.cat(obs_parts, dim=-1),
                "pos": agent.state.pos,
                "vel": agent.state.vel,
            }

        other_rel_pos = [
            other_agent.state.pos - agent.state.pos
            for other_agent in self.world.agents
            if other_agent is not agent
        ]
        obs_parts = [agent.state.vel, agent.state.pos] + landmark_rel_pos + other_rel_pos

        if self.observe_other_velocities:
            obs_parts += [
                other_agent.state.vel
                for other_agent in self.world.agents
                if other_agent is not agent
            ]

        return {
            "obs": torch.cat(obs_parts, dim=-1),
            "pos": agent.state.pos,
            "vel": agent.state.vel,
        }

    def done(self) -> Tensor:
        if self.reward_mode == "mpe_parity":
            return self.all_landmarks_covered
        if self.done_when_all_covered:
            return self.all_landmarks_covered
        return torch.zeros(
            self.world.batch_dim,
            dtype=torch.bool,
            device=self.world.device,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        if self.reward_mode == "mpe_parity":
            info = {
                "spread_rew": self.spread_rew,
                "landmark_min_dists": self.landmark_min_dists,
                "matched_dists": self.matched_landmark_dists,
                "matched_mean_dist": self.matched_mean_dist,
                "all_landmarks_covered": self.all_landmarks_covered,
                "is_success": self.all_landmarks_covered.float(),
                "world_steps": self.world_steps.float(),
            }
            if self.emit_global_state:
                info["global_state"] = self.global_state()
            return info

        return {
            "spread_rew": self.spread_rew,
            "coverage_rew": self.coverage_rew,
            "success_rew": self.success_rew,
            "collision_rew": (
                sum(a.collision_rew for a in self.world.agents)
                if self.shared_rew
                else agent.collision_rew
            ),
            "out_of_bounds_rew": (
                sum(a.out_of_bounds_rew for a in self.world.agents)
                if self.shared_rew
                else agent.out_of_bounds_rew
            ),
            "landmark_min_dists": self.landmark_min_dists,
            "all_landmarks_covered": self.all_landmarks_covered,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        if self.comms_rendering_range <= 0:
            return []

        from vmas.simulator import rendering

        geoms: List[rendering.Geom] = []
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos,
                    dim=-1,
                )
                if agent_dist[env_index] <= self.comms_rendering_range:
                    line = rendering.Line(
                        agent1.state.pos[env_index],
                        agent2.state.pos[env_index],
                        width=1,
                    )
                    line.set_color(*Color.BLACK.value)
                    geoms.append(line)
        return geoms
