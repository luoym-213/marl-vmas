"""Communication state estimators."""

from __future__ import annotations

from typing import Any

import torch

from comm_limited_vmas.estimators.base import BaseEstimator
from comm_limited_vmas.estimators.kinematic import KinematicEstimator
from comm_limited_vmas.estimators.stale import StaleEstimator


def build_estimator(
    config: dict[str, Any] | None,
    state_dim: int,
    device: torch.device,
    dt: float = 0.1,
) -> BaseEstimator:
    config = dict(config or {})
    name = str(config.pop("name", "stale"))

    if name == "stale":
        return StaleEstimator(state_dim=state_dim, device=device, **config)
    if name == "kinematic":
        return KinematicEstimator(state_dim=state_dim, device=device, dt=dt, **config)

    raise ValueError(f"Unknown estimator: {name}")


__all__ = [
    "BaseEstimator",
    "KinematicEstimator",
    "StaleEstimator",
    "build_estimator",
]
