"""State estimators for communication-limited observations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseEstimator(ABC):
    """Maps cached sender states and AoI to observation states."""

    def __init__(self, state_dim: int, device: torch.device):
        if state_dim != 4:
            raise ValueError(f"Only [px, py, vx, vy] states are supported, got {state_dim}")
        self.state_dim = state_dim
        self.device = device

    @abstractmethod
    def estimate_state(
        self,
        cached_state: Tensor,
        aoi_steps: Tensor,
        comm_mask: Tensor | None = None,
    ) -> Tensor:
        """Return estimated state with the same shape as cached_state."""

    @abstractmethod
    def covariance_diag(self, aoi_steps: Tensor) -> Tensor:
        """Return covariance diagonal with shape ``aoi_steps.shape + (state_dim,)``."""

    def estimate_distribution(
        self,
        cached_state: Tensor,
        aoi_steps: Tensor,
        comm_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return estimated state and covariance diagonal.

        Estimators that only model AoI-dependent uncertainty can rely on this default.
        Richer estimators may override it when covariance depends on cached state or mask.
        """
        mean = self.estimate_state(cached_state, aoi_steps, comm_mask)
        covariance_diag = self.covariance_diag(aoi_steps).to(
            device=mean.device,
            dtype=mean.dtype,
        )
        return mean, covariance_diag

    def estimate_receiver_states(
        self,
        cached_states: Tensor,
        aoi_steps: Tensor,
        comm_mask: Tensor | None = None,
        receiver: int | None = None,
    ) -> Tensor:
        """Return all sender estimates for one receiver.

        The default is element-wise and works for non-graph estimators. Graph
        estimators can override this to use all sender nodes jointly.
        """
        return self.estimate_state(cached_states, aoi_steps, comm_mask)

    def estimate_receiver_distribution(
        self,
        cached_states: Tensor,
        aoi_steps: Tensor,
        comm_mask: Tensor | None = None,
        receiver: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return all sender means and covariance diagonals for one receiver."""
        mean = self.estimate_receiver_states(cached_states, aoi_steps, comm_mask, receiver)
        covariance_diag = self.covariance_diag(aoi_steps).to(
            device=mean.device,
            dtype=mean.dtype,
        )
        return mean, covariance_diag
