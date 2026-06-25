"""Communication-limited wrapper around VMAS simple_spread."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
from vmas.scenarios.mpe.simple_spread import Scenario as VmasSimpleSpreadScenario
from vmas.simulator.core import Agent, World

from comm_limited_vmas.communication import CommunicationManager


class Scenario(VmasSimpleSpreadScenario):
    """VMAS simple_spread with communication limits applied to peer observations."""

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
        kwargs.pop("shared_reward", None)
        kwargs.pop("collisions", None)

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
        landmark_pos = [
            landmark.state.pos - agent.state.pos for landmark in self.world.landmarks
        ]
        observation_parts = [agent.state.pos, agent.state.vel, *landmark_pos]

        if self.obs_agents:
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
            observation_parts.extend(other_pos)

        return torch.cat(observation_parts, dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = dict(super().info(agent))
        ego_index = self.world.agents.index(agent)
        aoi = self._comm_manager.get_aoi(ego_index)
        comm_mask = self._comm_manager.get_current_comm_mask(ego_index)
        landmark_min_dists = self._landmark_min_dists()

        info["aoi"] = aoi
        info["comm_mask"] = comm_mask
        info["cached_state"] = self._comm_manager.get_receiver_states(ego_index)
        info["landmark_min_dists"] = landmark_min_dists
        info["all_landmarks_covered"] = (
            landmark_min_dists <= self._coverage_radius()
        ).all(dim=-1, keepdim=True)
        info["mean_aoi"] = aoi.mean(dim=-1, keepdim=True)
        info["mean_comm_mask"] = comm_mask.mean(dim=-1, keepdim=True)
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

    def _landmark_min_dists(self) -> Tensor:
        agent_positions = self._current_agent_positions()
        landmark_positions = torch.stack(
            [landmark.state.pos for landmark in self.world.landmarks],
            dim=1,
        )
        distances = torch.linalg.vector_norm(
            landmark_positions.unsqueeze(2) - agent_positions.unsqueeze(1),
            dim=-1,
        )
        return distances.min(dim=-1).values

    def _coverage_radius(self) -> float:
        agent_radius = self.world.agents[0].shape.circumscribed_radius()
        return float(getattr(self, "coverage_radius", agent_radius))
