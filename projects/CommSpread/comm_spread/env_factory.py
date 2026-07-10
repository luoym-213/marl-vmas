"""Convenience constructors for CommSpread VMAS environments."""

from __future__ import annotations

from typing import Any

from vmas import make_env

from comm_spread.sar_scenario import SarScenario
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

DEFAULT_SAR_CONFIG: dict[str, Any] = {
    "mode": "debug",
    "emit_info": True,
    "enable_high_level_state": True,
    "n_agents": 3,
    "n_targets": 3,
    "max_steps": 100,
    "belief_world_size": 2.0,
    "belief_cell_size": 0.02,
    "sensor_radius": 0.3,
    "sensor_fidelity": 0.8,
    "goal_radius": 0.05,
    "goal_reward": 3.0,
    "rescue_reward": 10.0,
    "discovery_reward": 1.0,
    "early_rescue_penalty": 0.0,
    "early_rescue_detected_threshold": 3,
    "search_capacity_discovery_bonus": 0.0,
    "discovery_active_agents_threshold": 2,
    "all_targets_detected_bonus": 0.0,
    "all_targets_detected_bonus_requires_no_rescue": True,
    "rescue_phase_explore_penalty": 0.0,
    "rescue_phase_detected_threshold": 2,
    "rescue_phase_min_search_agents": 1,
    "unique_rescue_assignment_bonus": 0.0,
    "duplicate_rescue_assignment_penalty": 0.0,
    "detected_unassigned_target_penalty": 0.0,
    "distance_reward_scale": 1.0,
    "collision_penalty": -10.0,
    "collision_distance": 0.0,
    "collision_safe_distance": 0.06,
    "max_collision_penalty": -20.0,
    "boundary_penalty": -5.0,
    "time_penalty": 0.0,
    "retire_on_rescue": True,
    "auto_resample_goals": True,
    "done_when_all_targets_visited": False,
    "high_level_interval": 5,
    "rrt_top_k": 5,
    "rrt_max_iter": 40,
    "enable_rrt_candidates": True,
    "comms_rendering_range": 0.0,
    "render_sensor_range": True,
    "render_assigned_goals": True,
    "render_entropy_map": True,
    "render_entropy_grid": True,
    "render_recent_rrt_candidates": True,
    "sensor_range_alpha": 0.12,
    "goal_marker_alpha": 0.9,
    "goal_marker_radius": None,
    "entropy_map_alpha": 0.55,
    "entropy_grid_alpha": 0.12,
    "rrt_candidate_alpha": 0.95,
    "rrt_candidate_radius": None,
    "recent_decision_render_steps": 5,
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


def make_sar_env(
    *,
    num_envs: int = 8,
    device: str = "cpu",
    seed: int = 0,
    max_steps: int | None = None,
    continuous_actions: bool = True,
    **scenario_config: Any,
):
    """Create the VMAS SAR environment."""

    config = DEFAULT_SAR_CONFIG | scenario_config
    if max_steps is not None:
        config["max_steps"] = max_steps
    wrapper_max_steps = config.pop("max_steps", None)
    config["scenario_max_steps"] = wrapper_max_steps
    return make_env(
        scenario=SarScenario(),
        num_envs=num_envs,
        device=device,
        seed=seed,
        continuous_actions=continuous_actions,
        max_steps=wrapper_max_steps,
        dict_spaces=False,
        multidiscrete_actions=False,
        grad_enabled=False,
        terminated_truncated=False,
        **config,
    )
