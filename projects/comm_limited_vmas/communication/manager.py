"""High-level communication manager for VMAS scenarios."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from comm_limited_vmas.communication.cache import AgentStateCache
from comm_limited_vmas.communication.channel import CommunicationChannel
from comm_limited_vmas.communication.delay_buffer import TensorDelayBuffer
from comm_limited_vmas.estimators import build_estimator


class CommunicationManager:
    """Coordinates channel decisions, delayed delivery, and receiver caches."""

    def __init__(
        self,
        config: dict[str, Any] | None,
        num_envs: int,
        n_agents: int,
        device: torch.device,
        state_dim: int = 4,
        estimator_config: dict[str, Any] | None = None,
        dt: float = 0.1,
    ):
        self.channel = CommunicationChannel(config)
        self.num_envs = num_envs
        self.n_agents = n_agents
        self.device = device
        self.state_dim = state_dim
        self.dt = float(dt)
        self.current_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cache = AgentStateCache(num_envs, n_agents, state_dim, device)
        self.estimator = build_estimator(
            estimator_config,
            state_dim=state_dim,
            device=device,
            dt=self.dt,
        )
        self.delay_buffer = (
            TensorDelayBuffer(
                max_delay=self.channel.max_delay,
                num_envs=num_envs,
                n_agents=n_agents,
                state_dim=state_dim,
                device=device,
            )
            if self.channel.max_delay > 0
            else None
        )
        self.current_comm_mask = torch.zeros(
            num_envs,
            n_agents,
            n_agents,
            dtype=torch.float32,
            device=device,
        )
        agent_indices = torch.arange(n_agents, device=device)
        self.peer_mask = agent_indices.view(-1, 1) != agent_indices.view(1, -1)
        self._full_fast_path = self.channel.is_full and self.channel.dropout_prob == 0

    def reset(self, states: Tensor, env_index: int | None = None) -> None:
        if env_index is None:
            self.current_steps.zero_()
        else:
            self.current_steps[env_index] = 0

        if self.delay_buffer is not None:
            self.delay_buffer.reset(env_index)
        self.cache.reset(states, self.current_steps, env_index)
        self.current_comm_mask.zero_()
        if self.channel.is_full:
            self._mark_all_peer_messages_current(env_index)

    def step(self, states: Tensor, positions: Tensor) -> None:
        self.current_steps += 1
        self.current_comm_mask.zero_()

        states_by_sender = self._states_by_sender(states)
        send_steps = self._send_steps_by_pair()

        if self._full_fast_path:
            self.cache.x_last.copy_(states_by_sender)
            self.cache.t_last.copy_(send_steps)
            self.current_comm_mask.copy_(self.peer_mask.float().unsqueeze(0))
            return

        send_mask = self._current_send_mask(positions)

        if self.channel.max_delay <= 0:
            accepted = self.cache.update_many(
                states_by_sender=states_by_sender,
                send_steps=send_steps,
                valid_mask=send_mask,
            )
            self.current_comm_mask.copy_(accepted.float())
            self.cache.update_self(states, self.current_steps)
            return

        delay = self.channel.sample_delay(
            (self.num_envs, self.n_agents, self.n_agents),
            device=states.device,
        )
        arrival_steps = send_steps + delay
        assert self.delay_buffer is not None
        self.delay_buffer.push(
            arrival_steps=arrival_steps,
            send_steps=send_steps,
            states_by_sender=states_by_sender,
            valid_mask=send_mask,
        )

        arrived_mask, arrived_send_steps, arrived_states = (
            self.delay_buffer.pop_current(self.current_steps)
        )
        accepted = self.cache.update_many(
            states_by_sender=arrived_states,
            send_steps=arrived_send_steps,
            valid_mask=arrived_mask,
        )
        self.current_comm_mask.copy_(accepted.float())

        self.cache.update_self(states, self.current_steps)

    def get_state(self, receiver: int, sender: int) -> Tensor:
        cached_state = self.cache.get_state(receiver, sender)
        aoi_steps = self.cache.get_aoi(receiver, self.current_steps)[:, sender]
        comm_mask = self.current_comm_mask[:, receiver, sender]
        return self.estimator.estimate_state(cached_state, aoi_steps, comm_mask)

    def get_receiver_states(self, receiver: int) -> Tensor:
        return self.cache.get_receiver_states(receiver)

    def get_estimated_receiver_states(self, receiver: int) -> Tensor:
        cached_states = self.cache.get_receiver_states(receiver)
        aoi_steps = self.cache.get_aoi(receiver, self.current_steps)
        comm_mask = self.current_comm_mask[:, receiver]
        return self.estimator.estimate_state(cached_states, aoi_steps, comm_mask)

    def get_estimator_covariance_diag(self, receiver: int) -> Tensor:
        aoi_steps = self.cache.get_aoi(receiver, self.current_steps)
        return self.estimator.covariance_diag(aoi_steps)

    def get_aoi(self, receiver: int) -> Tensor:
        return self.cache.get_aoi(receiver, self.current_steps)

    def get_current_comm_mask(self, receiver: int) -> Tensor:
        return self.current_comm_mask[:, receiver]

    def _states_by_sender(self, states: Tensor) -> Tensor:
        return states.unsqueeze(1).expand(
            self.num_envs,
            self.n_agents,
            self.n_agents,
            self.state_dim,
        )

    def _send_steps_by_pair(self) -> Tensor:
        return self.current_steps.view(-1, 1, 1).expand(
            self.num_envs,
            self.n_agents,
            self.n_agents,
        )

    def _current_send_mask(self, positions: Tensor) -> Tensor:
        send_mask = self.peer_mask.unsqueeze(0).expand(
            self.num_envs,
            self.n_agents,
            self.n_agents,
        )

        if (
            self.channel.mode in self.channel.RADIUS_DELAY_MODES
            and self.channel.comm_radius is not None
        ):
            distances = torch.linalg.vector_norm(
                positions.unsqueeze(1) - positions.unsqueeze(2),
                dim=-1,
            )
            send_mask = send_mask & (distances <= float(self.channel.comm_radius))

        if self.channel.dropout_prob > 0:
            keep = torch.rand(
                self.num_envs,
                self.n_agents,
                self.n_agents,
                device=positions.device,
            )
            send_mask = send_mask & (keep >= self.channel.dropout_prob)

        return send_mask

    def _mark_all_peer_messages_current(self, env_index: int | None = None) -> None:
        for receiver in range(self.n_agents):
            for sender in range(self.n_agents):
                if sender == receiver:
                    continue
                if env_index is None:
                    self.current_comm_mask[:, receiver, sender] = 1.0
                else:
                    self.current_comm_mask[env_index, receiver, sender] = 1.0
