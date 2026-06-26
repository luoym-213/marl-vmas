"""Communication-limited wrapper around VMAS navigation."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
from vmas.scenarios.navigation import Scenario as VmasNavigationScenario
from vmas.simulator.core import Agent, World

from comm_limited_vmas.communication import CommunicationManager


class Scenario(VmasNavigationScenario):
    """VMAS navigation with communication-limited peer observations."""

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

        world = super().make_world(batch_dim, device, **kwargs)
        self.comm_manager = CommunicationManager(
            config=self.comm_config,
            num_envs=batch_dim,
            n_agents=len(world.agents),
            device=device,
        )
        return world

    def reset_world_at(self, env_index: int | None = None) -> None:
        super().reset_world_at(env_index)
        self._comm_manager.reset(self._current_agent_states(), env_index)

    def post_step(self) -> None:
        self._comm_manager.step(
            states=self._current_agent_states(),
            positions=self._current_agent_positions(),
        )

    def observation(self, agent: Agent) -> Tensor:
        goal_poses = []
        if self.observe_all_goals:
            for other in self.world.agents:
                goal_poses.append(agent.state.pos - other.goal.state.pos)
        else:
            goal_poses.append(agent.state.pos - agent.goal.state.pos)

        ego_index = self.world.agents.index(agent)
        other_pos = []
        for other_index, other in enumerate(self.world.agents):
            if other is agent:
                continue
            cached_state = self._comm_manager.get_state(
                receiver=ego_index,
                sender=other_index,
            )
            other_pos.append(cached_state[:, :2] - agent.state.pos)

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                *goal_poses,
                *other_pos,
            ],
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = dict(super().info(agent))
        ego_index = self.world.agents.index(agent)
        aoi = self._comm_manager.get_aoi(ego_index)
        comm_mask = self._comm_manager.get_current_comm_mask(ego_index)

        info["aoi"] = aoi
        info["comm_mask"] = comm_mask
        info["cached_state"] = self._comm_manager.get_receiver_states(ego_index)
        info["mean_aoi"] = aoi.mean(dim=-1, keepdim=True)
        info["mean_comm_mask"] = comm_mask.mean(dim=-1, keepdim=True)
        info["all_goals_reached"] = self._all_goals_reached()
        info["pair_collision_count"] = self._pair_collision_count()
        return info

    @property
    def _comm_manager(self) -> CommunicationManager:
        if self.comm_manager is None:
            raise RuntimeError("CommunicationManager is not initialized")
        return self.comm_manager

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

    def _all_goals_reached(self) -> Tensor:
        reached = torch.stack(
            [
                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=-1,
                )
                < agent.shape.radius
                for agent in self.world.agents
            ],
            dim=-1,
        ).all(dim=-1, keepdim=True)
        return reached.float()

    def _pair_collision_count(self) -> Tensor:
        count = torch.zeros(
            self.world.batch_dim,
            1,
            dtype=torch.float32,
            device=self.world.device,
        )
        for i, agent in enumerate(self.world.agents):
            for other in self.world.agents[i + 1 :]:
                distance = self.world.get_distance(agent, other)
                count += (distance <= self.min_collision_distance).float().unsqueeze(
                    -1
                )
        return count
