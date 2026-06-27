"""Hidden-goal leader-follower navigation with communication limits."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

from comm_limited_vmas.communication import CommunicationManager


class Scenario(BaseScenario):
    """Custom VMAS scenario where only the leader observes the goal."""

    def __init__(
        self,
        comm_config: dict | None = None,
        estimator_config: dict | None = None,
    ):
        super().__init__()
        self.comm_config = comm_config or {}
        self.estimator_config = estimator_config or {}
        self.comm_manager: CommunicationManager | None = None

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        kwargs.pop("scenario", None)
        kwargs.pop("continuous_actions", None)
        kwargs.pop("shared_reward", None)

        self.n_agents = int(kwargs.pop("n_agents", 3))
        if self.n_agents != 3:
            raise ValueError("comm_hidden_goal_navigation requires n_agents=3")

        self.agent_radius = float(kwargs.pop("agent_radius", 0.06))
        self.u_range = float(kwargs.pop("u_range", 1.0))
        self.world_spawning_x = float(kwargs.pop("world_spawning_x", 1.0))
        self.world_spawning_y = float(kwargs.pop("world_spawning_y", 1.0))
        self.agent_spawn_radius = float(kwargs.pop("agent_spawn_radius", 0.2))
        self.target_spawn_radius = float(kwargs.pop("target_spawn_radius", 0.9))
        self.min_leader_target_dist = float(kwargs.pop("min_leader_target_dist", 0.7))

        self.w_goal = float(kwargs.pop("w_goal", 1.0))
        self.w_leader = float(kwargs.pop("w_leader", 0.5))
        self.w_follow = float(kwargs.pop("w_follow", 0.3))
        self.w_time = float(kwargs.pop("w_time", 0.0))
        self.w_progress = float(kwargs.pop("w_progress", 2.0))
        self.w_leader_progress = float(kwargs.pop("w_leader_progress", 0.0))
        self.w_follow_progress = float(kwargs.pop("w_follow_progress", 0.0))
        self.success_reward = float(kwargs.pop("success_reward", 5.0))
        self.epsilon_goal = float(kwargs.pop("epsilon_goal", 0.12))
        self.d_follow = float(kwargs.pop("d_follow", 0.0))
        self.aoi_normalizer = float(kwargs.pop("aoi_normalizer", 100.0))
        if kwargs:
            raise ValueError(f"Unknown scenario kwargs: {sorted(kwargs)}")

        self.plot_grid = True
        self.grid_spacing = 0.2
        self.viewer_zoom = 1.4

        world = World(
            batch_dim,
            device,
            substeps=2,
            x_semidim=self.world_spawning_x,
            y_semidim=self.world_spawning_y,
        )

        role_colors = [Color.BLUE, Color.GREEN, Color.GREEN]
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=False,
                shape=Sphere(radius=self.agent_radius),
                color=role_colors[i],
                u_range=self.u_range,
                render_action=True,
            )
            world.add_agent(agent)

        target = Landmark(
            name="target",
            collide=False,
            shape=Sphere(radius=self.agent_radius),
            color=Color.RED,
        )
        world.add_landmark(target)
        self.target = target

        self.prev_team_distance = torch.zeros(batch_dim, device=device)
        self.prev_leader_distance = torch.zeros(batch_dim, device=device)
        self.prev_follow_error = torch.zeros(batch_dim, device=device)
        self.last_team_distance = torch.zeros(batch_dim, device=device)
        self.last_leader_distance = torch.zeros(batch_dim, device=device)
        self.last_follow_error = torch.zeros(batch_dim, device=device)
        self.last_team_progress = torch.zeros(batch_dim, device=device)
        self.last_leader_progress = torch.zeros(batch_dim, device=device)
        self.last_follow_progress = torch.zeros(batch_dim, device=device)
        self.last_success = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.team_reward = torch.zeros(batch_dim, device=device)

        self.comm_manager = CommunicationManager(
            config=self.comm_config,
            num_envs=batch_dim,
            n_agents=self.n_agents,
            device=device,
            estimator_config=self.estimator_config,
            dt=world.dt,
        )
        return world

    def reset_world_at(self, env_index: int | None = None) -> None:
        batch_size = self.world.batch_dim if env_index is None else 1
        device = self.world.device

        agent_pos = torch.empty(
            batch_size,
            self.n_agents,
            2,
            dtype=torch.float32,
            device=device,
        ).uniform_(-self.agent_spawn_radius, self.agent_spawn_radius)
        zero_vel = torch.zeros(batch_size, 2, dtype=torch.float32, device=device)

        for i, agent in enumerate(self.world.agents):
            agent.set_pos(agent_pos[:, i], batch_index=env_index)
            agent.set_vel(zero_vel, batch_index=env_index)

        target_pos = self._sample_target_pos(agent_pos[:, 0])
        self.target.set_pos(target_pos, batch_index=env_index)
        self.target.set_vel(zero_vel, batch_index=env_index)

        (
            reset_team_distance,
            reset_leader_distance,
            reset_follow_error,
            reset_success,
        ) = self._compute_metrics(agent_pos, target_pos)
        if env_index is None:
            self.prev_team_distance.copy_(reset_team_distance)
            self.prev_leader_distance.copy_(reset_leader_distance)
            self.prev_follow_error.copy_(reset_follow_error)
            self.last_team_distance.copy_(reset_team_distance)
            self.last_leader_distance.copy_(reset_leader_distance)
            self.last_follow_error.copy_(reset_follow_error)
            self.last_team_progress.zero_()
            self.last_leader_progress.zero_()
            self.last_follow_progress.zero_()
            self.last_success.copy_(reset_success)
            self.team_reward.zero_()
        else:
            if reset_team_distance.numel() == 1:
                reset_team_distance = reset_team_distance.squeeze(0)
                reset_leader_distance = reset_leader_distance.squeeze(0)
                reset_follow_error = reset_follow_error.squeeze(0)
                reset_success = reset_success.squeeze(0)
            self.prev_team_distance[env_index] = reset_team_distance
            self.prev_leader_distance[env_index] = reset_leader_distance
            self.prev_follow_error[env_index] = reset_follow_error
            self.last_team_distance[env_index] = reset_team_distance
            self.last_leader_distance[env_index] = reset_leader_distance
            self.last_follow_error[env_index] = reset_follow_error
            self.last_team_progress[env_index] = 0.0
            self.last_leader_progress[env_index] = 0.0
            self.last_follow_progress[env_index] = 0.0
            self.last_success[env_index] = reset_success
            self.team_reward[env_index] = 0.0

        self._comm_manager.reset(self._current_agent_states(), env_index)

    def post_step(self) -> None:
        self._comm_manager.step(
            states=self._current_agent_states(),
            positions=self._current_agent_positions(),
        )

    def observation(self, agent: Agent) -> Tensor:
        ego_index = self.world.agents.index(agent)
        ego_role = self._role_one_hot(ego_index)

        if ego_index == 0:
            goal_rel = self.target.state.pos - agent.state.pos
            goal_visible = torch.ones(
                self.world.batch_dim,
                1,
                dtype=torch.float32,
                device=self.world.device,
            )
        else:
            goal_rel = torch.zeros(
                self.world.batch_dim,
                2,
                dtype=torch.float32,
                device=self.world.device,
            )
            goal_visible = torch.zeros(
                self.world.batch_dim,
                1,
                dtype=torch.float32,
                device=self.world.device,
            )

        aoi = self._comm_manager.get_aoi(ego_index)
        comm_mask = self._comm_manager.get_current_comm_mask(ego_index)

        teammate_blocks = []
        for other_index, _ in enumerate(self.world.agents):
            if other_index == ego_index:
                continue
            cached_state = self._comm_manager.get_state(
                receiver=ego_index,
                sender=other_index,
            )
            teammate_blocks.append(
                torch.cat(
                    [
                        cached_state[:, :2] - agent.state.pos,
                        cached_state[:, 2:] - agent.state.vel,
                        self._normalized_aoi(aoi[:, other_index]).unsqueeze(-1),
                        comm_mask[:, other_index].unsqueeze(-1),
                        self._role_one_hot(other_index),
                    ],
                    dim=-1,
                )
            )

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                goal_rel,
                goal_visible,
                ego_role,
                *teammate_blocks,
            ],
            dim=-1,
        )

    def reward(self, agent: Agent) -> Tensor:
        if agent is self.world.agents[0]:
            self._update_reward_cache()

        return self.team_reward

    def done(self) -> Tensor:
        positions = self._current_agent_positions()
        _, _, _, success = self._compute_metrics(positions, self.target.state.pos)
        return success

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        ego_index = self.world.agents.index(agent)
        aoi = self._comm_manager.get_aoi(ego_index)
        comm_mask = self._comm_manager.get_current_comm_mask(ego_index)

        return {
            "aoi": aoi,
            "comm_mask": comm_mask,
            "cached_state": self._comm_manager.get_receiver_states(ego_index),
            "mean_aoi": aoi.mean(dim=-1, keepdim=True),
            "mean_comm_mask": comm_mask.mean(dim=-1, keepdim=True),
            "team_distance": self.last_team_distance.unsqueeze(-1),
            "leader_distance": self.last_leader_distance.unsqueeze(-1),
            "follow_error": self.last_follow_error.unsqueeze(-1),
            "team_progress": self.last_team_progress.unsqueeze(-1),
            "leader_progress": self.last_leader_progress.unsqueeze(-1),
            "follow_progress": self.last_follow_progress.unsqueeze(-1),
            "all_goals_reached": self.last_success.float().unsqueeze(-1),
        }

    @property
    def _comm_manager(self) -> CommunicationManager:
        if self.comm_manager is None:
            raise RuntimeError("CommunicationManager is not initialized")
        return self.comm_manager

    def _sample_target_pos(self, leader_pos: Tensor) -> Tensor:
        batch_size = leader_pos.shape[0]
        target_pos = torch.empty(
            batch_size,
            2,
            dtype=torch.float32,
            device=self.world.device,
        ).uniform_(-self.target_spawn_radius, self.target_spawn_radius)

        for _ in range(10_000):
            too_close = (
                torch.linalg.vector_norm(target_pos - leader_pos, dim=-1)
                <= self.min_leader_target_dist
            )
            if not too_close.any():
                break
            target_pos[too_close] = torch.empty(
                int(too_close.sum()),
                2,
                dtype=torch.float32,
                device=self.world.device,
            ).uniform_(-self.target_spawn_radius, self.target_spawn_radius)

        return target_pos

    def _role_one_hot(self, agent_index: int) -> Tensor:
        role = torch.zeros(
            self.world.batch_dim,
            2,
            dtype=torch.float32,
            device=self.world.device,
        )
        role[:, 0 if agent_index == 0 else 1] = 1.0
        return role

    def _normalized_aoi(self, aoi: Tensor) -> Tensor:
        return (aoi.float() / max(self.aoi_normalizer, 1.0)).clamp(0.0, 1.0)

    def _current_agent_states(self) -> Tensor:
        return torch.cat(
            [
                self._current_agent_positions(),
                torch.stack([agent.state.vel for agent in self.world.agents], dim=1),
            ],
            dim=-1,
        )

    def _current_agent_positions(self) -> Tensor:
        return torch.stack([agent.state.pos for agent in self.world.agents], dim=1)

    def _compute_metrics(
        self,
        positions: Tensor,
        target_pos: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        target_distances = torch.linalg.vector_norm(
            positions - target_pos.unsqueeze(1),
            dim=-1,
        )
        team_distance = target_distances.mean(dim=-1)
        leader_distance = target_distances[:, 0]
        follow_error = self._follow_error_from_positions(positions)
        success = (target_distances <= self.epsilon_goal).all(dim=-1)
        return team_distance, leader_distance, follow_error, success

    def _update_reward_cache(self) -> None:
        positions = self._current_agent_positions()
        team_distance, leader_distance, follow_error, success = (
            self._compute_metrics(positions, self.target.state.pos)
        )
        team_progress = self.prev_team_distance - team_distance
        leader_progress = self.prev_leader_distance - leader_distance
        follow_progress = self.prev_follow_error - follow_error

        self.last_team_distance.copy_(team_distance)
        self.last_leader_distance.copy_(leader_distance)
        self.last_follow_error.copy_(follow_error)
        self.last_team_progress.copy_(team_progress)
        self.last_leader_progress.copy_(leader_progress)
        self.last_follow_progress.copy_(follow_progress)
        self.last_success.copy_(success)

        self.team_reward = (
            self.w_progress * team_progress
            + self.w_leader_progress * leader_progress
            + self.w_follow_progress * follow_progress
            - self.w_goal * team_distance
            - self.w_leader * leader_distance
            - self.w_follow * follow_error
            + self.success_reward * success.float()
        )

        self.prev_team_distance.copy_(team_distance.detach())
        self.prev_leader_distance.copy_(leader_distance.detach())
        self.prev_follow_error.copy_(follow_error.detach())

    def _follow_error(self) -> Tensor:
        return self._follow_error_from_positions(self._current_agent_positions())

    def _follow_error_from_positions(self, positions: Tensor) -> Tensor:
        leader_pos = positions[:, 0]
        follower_distances = torch.linalg.vector_norm(
            positions[:, 1:] - leader_pos.unsqueeze(1),
            dim=-1,
        )
        return torch.abs(follower_distances - self.d_follow).sum(dim=-1)
