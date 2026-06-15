"""VMAS simple_spread-style cooperative coverage scenario."""

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

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class CommSpreadScenario(BaseScenario):
    """Cooperative landmark coverage task inspired by MPE simple_spread.

    Agents are homogeneous holonomic discs. Landmarks are shared team targets,
    not agent-specific goals. The dense reward is the negative sum of each
    landmark's distance to its nearest agent, plus local collision penalties.
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.n_agents = kwargs.pop("n_agents", 3)
        self.n_landmarks = kwargs.pop("n_landmarks", self.n_agents)

        self.world_spawning_x = kwargs.pop("world_spawning_x", 1.0)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 1.0)

        self.agent_radius = kwargs.pop("agent_radius", 0.05)
        self.landmark_radius = kwargs.pop("landmark_radius", 0.04)
        self.coverage_radius = kwargs.pop("coverage_radius", 0.1)
        self.min_distance_between_entities = kwargs.pop(
            "min_distance_between_entities",
            max(self.agent_radius, self.landmark_radius) * 2 + 0.05,
        )
        self.min_collision_distance = kwargs.pop(
            "min_collision_distance",
            self.agent_radius * 2,
        )

        self.collision_penalty = kwargs.pop("collision_penalty", -1.0)
        self.distance_reward_scale = kwargs.pop("distance_reward_scale", 1.0)
        self.shared_rew = kwargs.pop("shared_rew", False)
        self.done_when_all_covered = kwargs.pop("done_when_all_covered", False)
        self.observe_other_velocities = kwargs.pop("observe_other_velocities", False)
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
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                color=color,
                render_action=True,
                shape=Sphere(radius=self.agent_radius),
                u_range=[1.0, 1.0],
                u_multiplier=[1.0, 1.0],
                dynamics=Holonomic(),
            )
            agent.collision_rew = torch.zeros(batch_dim, device=device)
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
        self.all_landmarks_covered = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.landmark_min_dists = torch.zeros(batch_dim, self.n_landmarks, device=device)

        return world

    def reset_world_at(self, env_index: int | None = None) -> None:
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + self.landmarks,
            self.world,
            env_index,
            self.min_distance_between_entities,
            x_bounds=(-self.world_spawning_x, self.world_spawning_x),
            y_bounds=(-self.world_spawning_y, self.world_spawning_y),
        )

        if env_index is None:
            self.all_landmarks_covered[:] = False
            self.landmark_min_dists[:] = 0
        else:
            self.all_landmarks_covered[env_index] = False
            self.landmark_min_dists[env_index] = 0

    def reward(self, agent: Agent) -> Tensor:
        is_first = agent == self.world.agents[0]

        if is_first:
            agents_pos = torch.stack(
                [current_agent.state.pos for current_agent in self.world.agents],
                dim=1,
            )
            landmarks_pos = torch.stack(
                [landmark.state.pos for landmark in self.landmarks],
                dim=1,
            )

            agent_landmark_dists = torch.cdist(agents_pos, landmarks_pos)
            self.landmark_min_dists = agent_landmark_dists.min(dim=1).values
            self.spread_rew = -self.distance_reward_scale * self.landmark_min_dists.sum(dim=-1)
            self.all_landmarks_covered = torch.all(
                self.landmark_min_dists <= self.coverage_radius,
                dim=-1,
            )

            for current_agent in self.world.agents:
                current_agent.collision_rew[:] = 0

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

        if self.shared_rew:
            collision_rew = sum(a.collision_rew for a in self.world.agents)
        else:
            collision_rew = agent.collision_rew
        return self.spread_rew + collision_rew

    def observation(self, agent: Agent) -> Dict[str, Tensor]:
        landmark_rel_pos = [
            landmark.state.pos - agent.state.pos for landmark in self.landmarks
        ]
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
        if self.done_when_all_covered:
            return self.all_landmarks_covered
        return torch.zeros(
            self.world.batch_dim,
            dtype=torch.bool,
            device=self.world.device,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "spread_rew": self.spread_rew,
            "collision_rew": (
                sum(a.collision_rew for a in self.world.agents)
                if self.shared_rew
                else agent.collision_rew
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
