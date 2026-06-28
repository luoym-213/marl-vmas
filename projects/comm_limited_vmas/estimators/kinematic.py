"""Constant-velocity kinematic estimator."""

from __future__ import annotations

import torch
from torch import Tensor

from comm_limited_vmas.estimators.base import BaseEstimator


class KinematicEstimator(BaseEstimator):
    """Extrapolates stale [px, py, vx, vy] states with a CV model."""

    def __init__(
        self,
        state_dim: int,
        device: torch.device,
        dt: float = 0.1,
        sigma_p: float = 0.01,
        sigma_v: float = 0.01,
        process_noise_q: float = 1.0,
        max_extrapolation_steps: int | None = 15,
        **_: object,
    ):
        super().__init__(state_dim=state_dim, device=device)
        self.dt = float(dt)
        self.sigma_p = float(sigma_p)
        self.sigma_v = float(sigma_v)
        self.process_noise_q = float(process_noise_q)
        self.max_extrapolation_steps = (
            None if max_extrapolation_steps is None else int(max_extrapolation_steps)
        )

    def estimate_state(
        self,
        cached_state: Tensor,
        aoi_steps: Tensor,
        comm_mask: Tensor | None = None,
    ) -> Tensor:
        delta_t = self._delta_t(aoi_steps, clamp=True).unsqueeze(-1)
        estimated = cached_state.clone()
        estimated[..., :2] = cached_state[..., :2] + cached_state[..., 2:] * delta_t
        return estimated

    def covariance_diag(self, aoi_steps: Tensor) -> Tensor:
        delta_t = self._delta_t(aoi_steps, clamp=False)
        dt2 = delta_t.square()
        dt3 = dt2 * delta_t

        pos_var = (
            self.sigma_p**2
            + (delta_t.square() * self.sigma_v**2)
            + self.process_noise_q * dt3 / 3.0
        )
        vel_var = self.sigma_v**2 + self.process_noise_q * delta_t

        return torch.stack([pos_var, pos_var, vel_var, vel_var], dim=-1)

    def _delta_t(self, aoi_steps: Tensor, clamp: bool) -> Tensor:
        steps = aoi_steps.float()
        if clamp and self.max_extrapolation_steps is not None:
            steps = steps.clamp(max=float(self.max_extrapolation_steps))
        return steps * self.dt
