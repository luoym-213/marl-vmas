"""Stale communication estimator."""

from __future__ import annotations

import torch
from torch import Tensor

from comm_limited_vmas.estimators.base import BaseEstimator


class StaleEstimator(BaseEstimator):
    """Returns the last received state unchanged."""

    def __init__(
        self,
        state_dim: int,
        device: torch.device,
        sigma_p: float = 0.01,
        sigma_v: float = 0.01,
        **_: object,
    ):
        super().__init__(state_dim=state_dim, device=device)
        self.sigma_p = float(sigma_p)
        self.sigma_v = float(sigma_v)

    def estimate_state(self, cached_state: Tensor, aoi_steps: Tensor) -> Tensor:
        return cached_state

    def covariance_diag(self, aoi_steps: Tensor) -> Tensor:
        shape = (*aoi_steps.shape, self.state_dim)
        cov = torch.empty(shape, dtype=torch.float32, device=self.device)
        cov[..., 0:2] = self.sigma_p**2
        cov[..., 2:4] = self.sigma_v**2
        return cov
