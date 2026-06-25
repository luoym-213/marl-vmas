"""Communication channel policies for communication-limited VMAS tasks."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class CommunicationChannel:
    """Decides whether messages are sent and how much delay they incur."""

    FULL_MODES = {"full", "comm_full"}
    DELAY_MODES = {"delay"}
    RADIUS_DELAY_MODES = {"radius_delay", "radius_delay_dropout"}

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.mode = str(self.config.get("mode", "comm_full"))
        self.comm_radius = self.config.get("comm_radius")
        self.dropout_prob = float(self.config.get("dropout_prob") or 0.0)
        self.max_delay = self._parse_max_delay(self.config)

    @property
    def is_full(self) -> bool:
        return self.mode in self.FULL_MODES and self.max_delay == 0

    def can_send(self, sender: int, receiver: int, positions: Tensor) -> Tensor:
        """Return a boolean mask with shape [num_envs]."""
        can_send = torch.ones(
            positions.shape[0],
            dtype=torch.bool,
            device=positions.device,
        )

        if self.mode in self.RADIUS_DELAY_MODES and self.comm_radius is not None:
            distance = torch.linalg.vector_norm(
                positions[:, sender] - positions[:, receiver],
                dim=-1,
            )
            can_send = can_send & (distance <= float(self.comm_radius))

        if self.dropout_prob > 0:
            keep = torch.rand(positions.shape[0], device=positions.device)
            can_send = can_send & (keep >= self.dropout_prob)

        return can_send

    def sample_delay(self, shape: torch.Size | tuple[int, ...], device: torch.device):
        if self.max_delay <= 0:
            return torch.zeros(shape, dtype=torch.long, device=device)
        return torch.randint(
            low=0,
            high=self.max_delay + 1,
            size=shape,
            dtype=torch.long,
            device=device,
        )

    @staticmethod
    def _parse_max_delay(config: dict[str, Any]) -> int:
        delay_config = config.get("delay")
        if isinstance(delay_config, dict):
            delay_type = delay_config.get("type", "constant")
            if delay_type == "uniform_int":
                return int(delay_config.get("max", 0))
            return int(delay_config.get("value", 0))
        return int(config.get("delay_steps") or 0)
