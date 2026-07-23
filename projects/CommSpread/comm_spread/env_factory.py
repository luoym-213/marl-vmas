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
    "physics_profile": "legacy",
    "emit_info": True,
    "enable_high_level_state": True,
    "n_agents": 3,
    "n_targets": 3,
    "max_steps": 100,
    "belief_world_size": 2.0,
    "belief_cell_size": 0.02,
    "sensor_radius": 0.3,
    "sensor_fidelity": 0.8,
    "initial_belief": 0.5,
    "belief_detection_threshold": 0.95,
    "belief_include_inactive_agents": True,
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
    "dynamic_rescue_release": False,
    "dynamic_release_min_searchers": 2,
    "dynamic_release_entropy_ratio_threshold": 0.58,
    "dynamic_release_entropy_rate_threshold": 0.0015,
    "dynamic_release_min_stagnation_step": 25,
    "dynamic_release_search_steps_per_target": 22.0,
    "dynamic_release_speed_per_step": 0.035,
    "dynamic_release_time_margin": 8.0,
    "dynamic_release_max_new_agents_per_event": 1,
    "dynamic_rescue_only_after_all_detected": False,
    "redecide_on_detection_change": False,
    "redecide_on_assignment_change": False,
    "enable_finder_first_cascade": False,
    "finder_cascade_mode": "immediate",
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

MPE_STRICT_SAR_OVERRIDES: dict[str, Any] = {
    "physics_profile": "mpe_strict",
    "mpe_action_force_scale": 5.0,
    "world_substeps": 1,
    "world_collision_force": 100,
    "world_contact_margin": 0.001,
    "world_dt": 0.1,
    "world_drag": 0.25,
    "world_linear_friction": 0.0,
    "world_angular_friction": 0.0,
    "world_hard_bounds": False,
    "collision_penalty": -20.0,
    "collision_safe_distance": 0.15,
    "max_collision_penalty": -20.0,
    "boundary_penalty": -2.0,
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
    continuous_actions: bool | None = None,
    **scenario_config: Any,
):
    """Create the VMAS SAR environment with an explicit physics profile."""

    physics_profile = scenario_config.get("physics_profile", "legacy")
    if physics_profile not in {"legacy", "mpe_strict"}:
        raise ValueError(f"invalid physics profile: {physics_profile}")
    profile_config = (
        MPE_STRICT_SAR_OVERRIDES if physics_profile == "mpe_strict" else {}
    )
    config = DEFAULT_SAR_CONFIG | profile_config | scenario_config
    expected_continuous = physics_profile == "legacy"
    if continuous_actions is None:
        continuous_actions = expected_continuous
    elif continuous_actions != expected_continuous:
        raise ValueError(
            f"physics_profile={physics_profile!r} requires "
            f"continuous_actions={expected_continuous}"
        )
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
