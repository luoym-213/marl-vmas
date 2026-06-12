"""VMAS scenario from the Colab tutorial, written as a reusable module."""

from __future__ import annotations

import typing
from typing import Dict, List

import torch
from torch import Tensor
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import (
    ANGULAR_FRICTION,
    DRAG,
    LINEAR_FRICTION,
    Color,
    ScenarioUtils,
)

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class HeterogeneousNavigationScenario(BaseScenario):
    """Navigation task with holonomic, differential-drive, and car agents.

    Agents must reach color-matched goals while avoiding other agents and fixed
    obstacles. The implementation is intentionally close to the tutorial, but
    packaged as a normal VMAS scenario class.
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        ################
        # Scenario configuration
        ################
        self.plot_grid = kwargs.pop("plot_grid", False)

        self.n_agents_holonomic = kwargs.pop("n_agents_holonomic", 2)
        self.n_agents_diff_drive = kwargs.pop("n_agents_diff_drive", 1)
        self.n_agents_car = kwargs.pop("n_agents_car", 1)
        self.n_agents = (
            self.n_agents_holonomic
            + self.n_agents_diff_drive
            + self.n_agents_car
        )
        self.n_obstacles = kwargs.pop("n_obstacles", 2)

        self.world_spawning_x = kwargs.pop("world_spawning_x", 1)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 1)

        self.comms_rendering_range = kwargs.pop("comms_rendering_range", 0) # Used for rendering communication lines between agents (just visual)
        self.lidar_range = kwargs.pop("lidar_range", 0.3)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 12) # Number of LIDAR rays around the agent, each ray gives an observation between 0 and lidar_range

        self.shared_rew = kwargs.pop("shared_rew", False)
        self.final_reward = kwargs.pop("final_reward", 0.01)
        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)

        self.agent_radius = kwargs.pop("agent_radius", 0.1)
        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = kwargs.pop("min_collision_distance", 0.005)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        ################
        # Make world
        ################
        world = World(
            batch_dim, # Number of environments simulated
            device,
            substeps=5, # Number of physical substeps (more yields more accurate but more expensive physics)
            collision_force=500,
            dt=0.1, # Simulation timestep
            gravity=(0.0, 0.0),
            drag=DRAG, # Physics parameters
            linear_friction=LINEAR_FRICTION,
            angular_friction=ANGULAR_FRICTION,
        )

        ################
        # Add agents
        ################
        known_colors = [
            Color.BLUE,
            Color.ORANGE,
            Color.GREEN,
            Color.PINK,
            Color.PURPLE,
            Color.YELLOW,
            Color.RED,
        ]
        extra_colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )

        self.goals: List[Landmark] = [] # We will store our agent goal entities here for easy access
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else extra_colors[i - len(known_colors)]
            )

            sensors = [
                Lidar(
                    world,
                    n_rays=self.n_lidar_rays,
                    max_range=self.lidar_range,
                    entity_filter=lambda entity: isinstance(entity, Agent),
                    angle_start=0.0,
                    angle_end=2 * torch.pi,
                )
            ]

            if i < self.n_agents_holonomic:
                agent = Agent(
                    name=f"holonomic_{i}",
                    collide=True,
                    color=color,
                    render_action=True,
                    sensors=sensors,
                    shape=Sphere(radius=self.agent_radius),
                    u_range=[1, 1],
                    u_multiplier=[1, 1],
                    dynamics=Holonomic(),
                )
            elif i < self.n_agents_holonomic + self.n_agents_diff_drive:
                agent = Agent(
                    name=f"diff_drive_{i - self.n_agents_holonomic}",
                    collide=True,
                    color=color,
                    render_action=True,
                    sensors=sensors,
                    shape=Sphere(radius=self.agent_radius),
                    u_range=[1, 1],
                    u_multiplier=[0.5, 1],
                    dynamics=DiffDrive(world),
                )
            else:
                max_steering_angle = torch.pi / 4
                width = self.agent_radius
                agent = Agent(
                    name=(
                        "car_"
                        f"{i - self.n_agents_holonomic - self.n_agents_diff_drive}"
                    ),
                    collide=True,
                    color=color,
                    render_action=True,
                    sensors=sensors,
                    shape=Box(length=self.agent_radius * 2, width=width),
                    u_range=[1, max_steering_angle],
                    u_multiplier=[0.5, 1],
                    dynamics=KinematicBicycle(
                        world,
                        width=width,
                        l_f=self.agent_radius,
                        l_r=self.agent_radius,
                        max_steering_angle=max_steering_angle,
                    ),
                )

            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

            ################
            # Add goals
            ################
            goal = Landmark(name=f"goal_{i}", collide=False, color=color)
            world.add_landmark(goal)
            agent.goal = goal
            self.goals.append(goal)

        ################
        # Add obstacles
        ################
        self.obstacles: List[Landmark] = []
        for i in range(self.n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                color=Color.BLACK,
                shape=Sphere(radius=self.agent_radius * 2 / 3),
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()
        self.all_goal_reached = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int | None = None) -> None:
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + self.obstacles + self.goals,
            self.world,
            env_index,
            self.min_distance_between_entities,
            x_bounds=(-self.world_spawning_x, self.world_spawning_x),
            y_bounds=(-self.world_spawning_y, self.world_spawning_y),
        )

        for agent in self.world.agents:
            if env_index is None:
                agent.goal_dist = torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=-1,
                )
            else:
                agent.goal_dist[env_index] = torch.linalg.vector_norm(
                    agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                )

    def reward(self, agent: Agent) -> Tensor:
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for current_agent in self.world.agents:
                current_agent.agent_collision_rew[:] = 0
                distance_to_goal = torch.linalg.vector_norm(
                    current_agent.state.pos - current_agent.goal.state.pos,
                    dim=-1,
                )
                current_agent.on_goal = (
                    distance_to_goal < current_agent.shape.circumscribed_radius()
                )
                current_agent.pos_rew = current_agent.goal_dist - distance_to_goal
                current_agent.goal_dist = distance_to_goal
                self.pos_rew += current_agent.pos_rew

            self.all_goal_reached = torch.all(
                torch.stack(
                    [current_agent.on_goal for current_agent in self.world.agents],
                    dim=-1,
                ),
                dim=-1,
            )
            self.final_rew[self.all_goal_reached] = self.final_reward

            for i, current_agent in enumerate(self.world.agents):
                for j, other_agent in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(current_agent, other_agent):
                        distance = self.world.get_distance(current_agent, other_agent)
                        collision = distance <= self.min_collision_distance
                        current_agent.agent_collision_rew[
                            collision
                        ] += self.agent_collision_penalty
                        other_agent.agent_collision_rew[
                            collision
                        ] += self.agent_collision_penalty

                for obstacle in self.obstacles:
                    if self.world.collides(current_agent, obstacle):
                        distance = self.world.get_distance(current_agent, obstacle)
                        current_agent.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew

    def observation(self, agent: Agent) -> Dict[str, Tensor]:
        observation = {
            "obs": torch.cat(
                [agent.state.pos - agent.goal.state.pos]
                + [
                    agent.state.pos - obstacle.state.pos
                    for obstacle in self.obstacles
                ]
                + [sensor._max_range - sensor.measure() for sensor in agent.sensors],
                dim=-1,
            ),
            "pos": agent.state.pos,
            "vel": agent.state.vel,
        }
        if not isinstance(agent.dynamics, Holonomic):
            observation.update(
                {
                    "rot": agent.state.rot,
                    "ang_vel": agent.state.ang_vel,
                }
            )
        return observation

    def done(self) -> Tensor:
        return self.all_goal_reached

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collision_rew": agent.agent_collision_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms = [
            ScenarioUtils.plot_entity_rotation(agent, env_index)
            for agent in self.world.agents
            if not isinstance(agent.dynamics, Holonomic)
        ]

        if self.comms_rendering_range > 0:
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
