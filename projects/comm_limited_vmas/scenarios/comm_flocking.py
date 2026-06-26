"""Communication-limited wrapper around VMAS flocking."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
from vmas.scenarios.flocking import Scenario as VmasFlockingScenario
from vmas.simulator.core import Agent, World

from comm_limited_vmas.communication import CommunicationManager


class Scenario(VmasFlockingScenario):
    """VMAS flocking with cached peer states appended to observations."""

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
            n_agents=len(world.policy_agents),
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
        observation_parts = [super().observation(agent)]

        ego_index = self.world.policy_agents.index(agent)
        for other_index, other in enumerate(self.world.policy_agents):
            if other is agent:
                continue
            cached_state = self._comm_manager.get_state(
                receiver=ego_index,
                sender=other_index,
            )
            observation_parts.append(cached_state[:, :2] - agent.state.pos)
            observation_parts.append(cached_state[:, 2:])

        return torch.cat(observation_parts, dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = dict(super().info(agent))
        ego_index = self.world.policy_agents.index(agent)
        aoi = self._comm_manager.get_aoi(ego_index)
        comm_mask = self._comm_manager.get_current_comm_mask(ego_index)

        info["aoi"] = aoi
        info["comm_mask"] = comm_mask
        info["cached_state"] = self._comm_manager.get_receiver_states(ego_index)
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
                torch.stack(
                    [agent.state.vel for agent in self.world.policy_agents],
                    dim=1,
                ),
            ],
            dim=-1,
        )

    def _current_agent_positions(self) -> Tensor:
        return torch.stack(
            [agent.state.pos for agent in self.world.policy_agents],
            dim=1,
        )
