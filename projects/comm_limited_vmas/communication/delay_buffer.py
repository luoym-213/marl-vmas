"""Tensorized bounded-delay buffer for pairwise agent messages."""

from __future__ import annotations

import torch
from torch import Tensor


class TensorDelayBuffer:
    """Stores delayed messages in fixed circular arrival slots."""

    def __init__(
        self,
        max_delay: int,
        num_envs: int,
        n_agents: int,
        state_dim: int,
        device: torch.device,
    ):
        self.max_delay = max(0, int(max_delay))
        self.window = self.max_delay + 1
        self.num_envs = num_envs
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.device = device

        self.states = torch.zeros(
            self.window,
            num_envs,
            n_agents,
            n_agents,
            state_dim,
            dtype=torch.float32,
            device=device,
        )
        self.send_steps = torch.zeros(
            self.window,
            num_envs,
            n_agents,
            n_agents,
            dtype=torch.long,
            device=device,
        )
        self.valid = torch.zeros(
            self.window,
            num_envs,
            n_agents,
            n_agents,
            dtype=torch.bool,
            device=device,
        )

        self.env_indices = torch.arange(num_envs, device=device)
        self.receiver_indices = torch.arange(n_agents, device=device).view(
            1,
            n_agents,
            1,
        )
        self.sender_indices = torch.arange(n_agents, device=device).view(
            1,
            1,
            n_agents,
        )

    def reset(self, env_index: int | None = None) -> None:
        if env_index is None:
            self.valid.zero_()
        else:
            self.valid[:, env_index].zero_()

    def push(
        self,
        arrival_steps: Tensor,
        send_steps: Tensor,
        states_by_sender: Tensor,
        valid_mask: Tensor,
    ) -> None:
        slots = arrival_steps.remainder(self.window)
        env_indices = self.env_indices.view(-1, 1, 1).expand_as(valid_mask)
        receiver_indices = self.receiver_indices.expand_as(valid_mask)
        sender_indices = self.sender_indices.expand_as(valid_mask)

        index = (
            slots[valid_mask],
            env_indices[valid_mask],
            receiver_indices[valid_mask],
            sender_indices[valid_mask],
        )
        self.states[index] = states_by_sender[valid_mask]
        self.send_steps[index] = send_steps[valid_mask]
        self.valid[index] = True

    def pop_current(
        self,
        current_steps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        slots = current_steps.remainder(self.window)
        valid = self.valid[slots, self.env_indices].clone()
        send_steps = self.send_steps[slots, self.env_indices].clone()
        states = self.states[slots, self.env_indices].clone()
        self.valid[slots, self.env_indices] = False
        return valid, send_steps, states

    def pending_count(self) -> int:
        return int(self.valid.sum().detach().cpu())
