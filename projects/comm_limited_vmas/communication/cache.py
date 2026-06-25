"""Pairwise state cache for communication-limited observations."""

from __future__ import annotations

import torch
from torch import Tensor


class AgentStateCache:
    """Caches last received sender state for every receiver-sender pair."""

    def __init__(
        self,
        num_envs: int,
        n_agents: int,
        state_dim: int,
        device: torch.device,
    ):
        self.num_envs = num_envs
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.device = device
        self.x_last = torch.zeros(
            num_envs,
            n_agents,
            n_agents,
            state_dim,
            dtype=torch.float32,
            device=device,
        )
        self.t_last = torch.zeros(
            num_envs,
            n_agents,
            n_agents,
            dtype=torch.long,
            device=device,
        )

    def reset(
        self,
        states: Tensor,
        current_steps: Tensor,
        env_index: int | None = None,
    ) -> None:
        expanded_states = states.unsqueeze(1).expand(
            states.shape[0],
            self.n_agents,
            self.n_agents,
            self.state_dim,
        )
        expanded_steps = current_steps.view(-1, 1, 1).expand(
            states.shape[0],
            self.n_agents,
            self.n_agents,
        )

        if env_index is None:
            self.x_last.copy_(expanded_states)
            self.t_last.copy_(expanded_steps)
        else:
            self.x_last[env_index].copy_(expanded_states[env_index])
            self.t_last[env_index].copy_(expanded_steps[env_index])

    def update_self(self, states: Tensor, current_steps: Tensor) -> None:
        agent_indices = torch.arange(self.n_agents, device=self.device)
        self.x_last[:, agent_indices, agent_indices] = states
        self.t_last[:, agent_indices, agent_indices] = current_steps.unsqueeze(
            -1
        ).expand(-1, self.n_agents)

    def update(
        self,
        receiver: int,
        sender: int,
        env_indices: Tensor,
        states: Tensor,
        send_steps: Tensor,
    ) -> Tensor:
        previous_steps = self.t_last[env_indices, receiver, sender]
        accepted = send_steps > previous_steps
        if accepted.any():
            accepted_envs = env_indices[accepted]
            self.x_last[accepted_envs, receiver, sender] = states[accepted]
            self.t_last[accepted_envs, receiver, sender] = send_steps[accepted]
            return accepted_envs
        return env_indices.new_empty((0,))

    def get_state(self, receiver: int, sender: int) -> Tensor:
        return self.x_last[:, receiver, sender]

    def get_receiver_states(self, receiver: int) -> Tensor:
        return self.x_last[:, receiver]

    def get_aoi(self, receiver: int, current_steps: Tensor) -> Tensor:
        return (current_steps.unsqueeze(-1) - self.t_last[:, receiver]).float()
