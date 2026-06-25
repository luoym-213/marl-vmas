"""High-level communication manager for VMAS scenarios."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from comm_limited_vmas.communication.cache import AgentStateCache
from comm_limited_vmas.communication.channel import CommunicationChannel
from comm_limited_vmas.communication.delay_queue import DelayQueue


class CommunicationManager:
    """Coordinates channel decisions, delayed delivery, and receiver caches."""

    def __init__(
        self,
        config: dict[str, Any] | None,
        num_envs: int,
        n_agents: int,
        device: torch.device,
        state_dim: int = 4,
    ):
        self.channel = CommunicationChannel(config)
        self.num_envs = num_envs
        self.n_agents = n_agents
        self.device = device
        self.state_dim = state_dim
        self.current_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cache = AgentStateCache(num_envs, n_agents, state_dim, device)
        self.delay_queue = DelayQueue()
        self.current_comm_mask = torch.zeros(
            num_envs,
            n_agents,
            n_agents,
            dtype=torch.float32,
            device=device,
        )

    def reset(self, states: Tensor, env_index: int | None = None) -> None:
        if env_index is None:
            self.current_steps.zero_()
        else:
            self.current_steps[env_index] = 0

        self.delay_queue.reset(env_index)
        self.cache.reset(states, self.current_steps, env_index)
        self.current_comm_mask.zero_()
        if self.channel.is_full:
            self._mark_all_peer_messages_current(env_index)

    def step(self, states: Tensor, positions: Tensor) -> None:
        self.current_steps += 1
        self.current_comm_mask.zero_()

        for receiver in range(self.n_agents):
            for sender in range(self.n_agents):
                if sender == receiver:
                    continue

                can_send = self.channel.can_send(sender, receiver, positions)
                delay = self.channel.sample_delay(
                    (self.num_envs,),
                    device=states.device,
                )
                self.delay_queue.push(
                    receiver=receiver,
                    sender=sender,
                    env_mask=can_send,
                    send_steps=self.current_steps,
                    arrival_steps=self.current_steps + delay,
                    states=states[:, sender],
                )

        for message in self.delay_queue.pop_arrived(self.current_steps):
            accepted_envs = self.cache.update(
                receiver=message.receiver,
                sender=message.sender,
                env_indices=message.env_indices,
                states=message.states,
                send_steps=message.send_steps,
            )
            if accepted_envs.numel() > 0:
                self.current_comm_mask[
                    accepted_envs,
                    message.receiver,
                    message.sender,
                ] = 1.0

        self.cache.update_self(states, self.current_steps)

    def get_state(self, receiver: int, sender: int) -> Tensor:
        return self.cache.get_state(receiver, sender)

    def get_receiver_states(self, receiver: int) -> Tensor:
        return self.cache.get_receiver_states(receiver)

    def get_aoi(self, receiver: int) -> Tensor:
        return self.cache.get_aoi(receiver, self.current_steps)

    def get_current_comm_mask(self, receiver: int) -> Tensor:
        return self.current_comm_mask[:, receiver]

    def _mark_all_peer_messages_current(self, env_index: int | None = None) -> None:
        for receiver in range(self.n_agents):
            for sender in range(self.n_agents):
                if sender == receiver:
                    continue
                if env_index is None:
                    self.current_comm_mask[:, receiver, sender] = 1.0
                else:
                    self.current_comm_mask[env_index, receiver, sender] = 1.0
