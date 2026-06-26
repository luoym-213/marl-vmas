"""Communication-limited wrapper around VMAS dispersion."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
from vmas.scenarios.dispersion import Scenario as VmasDispersionScenario
from vmas.simulator.core import Agent, World

from comm_limited_vmas.communication import CommunicationManager


class Scenario(VmasDispersionScenario):
    """VMAS dispersion with communication-limited peer observations."""

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
        food_obs = []
        for landmark in self.world.landmarks:
            food_obs.append(
                torch.cat(
                    [
                        landmark.state.pos - agent.state.pos,
                        landmark.eaten.to(torch.int).unsqueeze(-1),
                    ],
                    dim=-1,
                )
            )

        ego_index = self.world.agents.index(agent)
        peer_pos = []
        for other_index, other in enumerate(self.world.agents):
            if other is agent:
                continue
            cached_state = self._comm_manager.get_state(
                receiver=ego_index,
                sender=other_index,
            )
            peer_pos.append(cached_state[:, :2] - agent.state.pos)

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                *food_obs,
                *peer_pos,
            ],
            dim=-1,
        )

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
            "food_eaten": self._food_eaten_count(),
            "all_food_eaten": self.done().float().unsqueeze(-1),
        }

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

    def _food_eaten_count(self) -> Tensor:
        eaten = torch.stack(
            [landmark.eaten.float() for landmark in self.world.landmarks],
            dim=-1,
        )
        return eaten.sum(dim=-1, keepdim=True)
