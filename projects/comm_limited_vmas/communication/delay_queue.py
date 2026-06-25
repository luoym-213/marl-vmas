"""Delay queue for pairwise agent state messages."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class QueuedMessage:
    receiver: int
    sender: int
    env_indices: Tensor
    send_steps: Tensor
    arrival_steps: Tensor
    states: Tensor


@dataclass
class ArrivedMessage:
    receiver: int
    sender: int
    env_indices: Tensor
    send_steps: Tensor
    states: Tensor


class DelayQueue:
    """Stores messages until their per-env arrival step is reached."""

    def __init__(self):
        self._messages: list[QueuedMessage] = []

    def reset(self, env_index: int | None = None) -> None:
        if env_index is None:
            self._messages = []
            return

        kept_messages = []
        for message in self._messages:
            keep = message.env_indices != env_index
            if keep.any():
                kept_messages.append(self._subset(message, keep))
        self._messages = kept_messages

    def push(
        self,
        receiver: int,
        sender: int,
        env_mask: Tensor,
        send_steps: Tensor,
        arrival_steps: Tensor,
        states: Tensor,
    ) -> None:
        env_indices = env_mask.nonzero(as_tuple=False).squeeze(-1)
        if env_indices.numel() == 0:
            return

        self._messages.append(
            QueuedMessage(
                receiver=receiver,
                sender=sender,
                env_indices=env_indices,
                send_steps=send_steps[env_indices].clone(),
                arrival_steps=arrival_steps[env_indices].clone(),
                states=states[env_indices].clone(),
            )
        )

    def pop_arrived(self, current_steps: Tensor) -> list[ArrivedMessage]:
        arrived_messages = []
        pending_messages = []

        for message in self._messages:
            arrived = message.arrival_steps <= current_steps[message.env_indices]
            if arrived.any():
                arrived_messages.append(
                    ArrivedMessage(
                        receiver=message.receiver,
                        sender=message.sender,
                        env_indices=message.env_indices[arrived],
                        send_steps=message.send_steps[arrived],
                        states=message.states[arrived],
                    )
                )

            pending = ~arrived
            if pending.any():
                pending_messages.append(self._subset(message, pending))

        self._messages = pending_messages
        return arrived_messages

    @staticmethod
    def _subset(message: QueuedMessage, mask: Tensor) -> QueuedMessage:
        return QueuedMessage(
            receiver=message.receiver,
            sender=message.sender,
            env_indices=message.env_indices[mask],
            send_steps=message.send_steps[mask],
            arrival_steps=message.arrival_steps[mask],
            states=message.states[mask],
        )
