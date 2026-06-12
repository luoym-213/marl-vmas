"""Convenience constructors for VMAS examples."""

from __future__ import annotations

from typing import Any

from vmas import make_env

from custom_vmas_benchmarl.scenario import HeterogeneousNavigationScenario


DEFAULT_SCENARIO_CONFIG: dict[str, Any] = {
    "n_agents_holonomic": 2,
    "n_agents_diff_drive": 1,
    "n_agents_car": 1,
    "n_obstacles": 2,
    "lidar_range": 0.3,
    "n_lidar_rays": 12,
    "comms_rendering_range": 0,
    "shared_rew": False,
}


def make_navigation_env(
    *,
    num_envs: int = 8,
    device: str = "cpu",
    seed: int = 0,
    max_steps: int | None = None,
    continuous_actions: bool = True,
    **scenario_config: Any,
):
    """Create the tutorial VMAS environment."""

    config = DEFAULT_SCENARIO_CONFIG | scenario_config
    return make_env(
        scenario=HeterogeneousNavigationScenario(),
        num_envs=num_envs,
        device=device,
        seed=seed,
        continuous_actions=continuous_actions,
        max_steps=max_steps,
        dict_spaces=False,
        multidiscrete_actions=False,
        grad_enabled=False,
        terminated_truncated=False,
        **config,
    )
