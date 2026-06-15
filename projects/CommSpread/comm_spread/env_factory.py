"""Convenience constructors for CommSpread VMAS environments."""

from __future__ import annotations

from typing import Any

from vmas import make_env

from comm_spread.scenario import CommSpreadScenario


DEFAULT_SCENARIO_CONFIG: dict[str, Any] = {
    "n_agents": 3,
    "n_landmarks": 3,
    "shared_rew": False,
    "done_when_all_covered": False,
    "coverage_radius": 0.1,
    "collision_penalty": -1.0,
    "out_of_bounds_penalty": -1.0,
    "distance_reward_scale": 1.0,
    "coverage_reward": 1.0,
    "success_reward": 20.0,
    "comms_rendering_range": 0.0,
}


def make_comm_spread_env(
    *,
    num_envs: int = 8,
    device: str = "cpu",
    seed: int = 0,
    max_steps: int | None = None,
    continuous_actions: bool = True,
    **scenario_config: Any,
):
    """Create a VMAS CommSpread environment."""

    config = DEFAULT_SCENARIO_CONFIG | scenario_config
    return make_env(
        scenario=CommSpreadScenario(),
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
