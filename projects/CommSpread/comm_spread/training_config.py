"""Training configuration builders for CommSpread."""

from __future__ import annotations

from typing import Any

import torch


TASK_VARIANTS: dict[str, dict[str, Any]] = {
    "spread": {
        "max_steps": 100,
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
    },
    "spread_early_done": {
        "max_steps": 100,
        "n_agents": 3,
        "n_landmarks": 3,
        "shared_rew": False,
        "done_when_all_covered": True,
        "coverage_radius": 0.1,
        "collision_penalty": -1.0,
        "out_of_bounds_penalty": -1.0,
        "distance_reward_scale": 1.0,
        "coverage_reward": 1.0,
        "success_reward": 20.0,
        "comms_rendering_range": 0.0,
    },
    "spread_mpe_parity": {
        "max_steps": 50,
        "n_agents": 3,
        "n_landmarks": 3,
        "continuous_actions": False,
        "reward_mode": "mpe_parity",
        "coverage_radius": 0.1,
        "dist_threshold": 0.1,
        "collision_penalty": 0.0,
        "out_of_bounds_penalty": 0.0,
        "distance_reward_scale": 1.0,
        "coverage_reward": 0.0,
        "success_reward": 0.0,
        "shared_rew": True,
        "done_when_all_covered": True,
        "comms_rendering_range": 0.0,
    },
    "spread_mpe_physics_parity": {
        "max_steps": 50,
        "n_agents": 3,
        "n_landmarks": 3,
        "continuous_actions": False,
        "reward_mode": "mpe_parity",
        "mpe_action_force_scale": 5.0,
        "world_substeps": 1,
        "world_collision_force": 100,
        "world_contact_margin": 0.001,
        "world_dt": 0.1,
        "world_drag": 0.25,
        "world_linear_friction": 0.0,
        "world_angular_friction": 0.0,
        "coverage_radius": 0.1,
        "dist_threshold": 0.1,
        "collision_penalty": 0.0,
        "out_of_bounds_penalty": 0.0,
        "distance_reward_scale": 1.0,
        "coverage_reward": 0.0,
        "success_reward": 0.0,
        "shared_rew": True,
        "done_when_all_covered": True,
        "comms_rendering_range": 0.0,
    },
    "spread_mpe_physics_spawn_parity": {
        "max_steps": 50,
        "n_agents": 3,
        "n_landmarks": 3,
        "continuous_actions": False,
        "reward_mode": "mpe_parity",
        "mpe_action_force_scale": 5.0,
        "world_substeps": 1,
        "world_collision_force": 100,
        "world_contact_margin": 0.001,
        "world_dt": 0.1,
        "world_drag": 0.25,
        "world_linear_friction": 0.0,
        "world_angular_friction": 0.0,
        "mpe_independent_spawn": True,
        "coverage_radius": 0.1,
        "dist_threshold": 0.1,
        "collision_penalty": 0.0,
        "out_of_bounds_penalty": 0.0,
        "distance_reward_scale": 1.0,
        "coverage_reward": 0.0,
        "success_reward": 0.0,
        "shared_rew": True,
        "done_when_all_covered": True,
        "comms_rendering_range": 0.0,
    },
    "spread_mpe_physics_spawn_oldppo": {
        "max_steps": 50,
        "n_agents": 3,
        "n_landmarks": 3,
        "continuous_actions": False,
        "reward_mode": "mpe_parity",
        "mpe_action_force_scale": 5.0,
        "world_substeps": 1,
        "world_collision_force": 100,
        "world_contact_margin": 0.001,
        "world_dt": 0.1,
        "world_drag": 0.25,
        "world_linear_friction": 0.0,
        "world_angular_friction": 0.0,
        "mpe_independent_spawn": True,
        "reward_return_norm": True,
        "reward_return_clip": 10.0,
        "reward_return_gamma": 0.99,
        "coverage_radius": 0.1,
        "dist_threshold": 0.1,
        "collision_penalty": 0.0,
        "out_of_bounds_penalty": 0.0,
        "distance_reward_scale": 1.0,
        "coverage_reward": 0.0,
        "success_reward": 0.0,
        "shared_rew": True,
        "done_when_all_covered": True,
        "comms_rendering_range": 0.0,
    },
    "spread_mpe_global_critic": {
        "max_steps": 50,
        "n_agents": 3,
        "n_landmarks": 3,
        "continuous_actions": False,
        "reward_mode": "mpe_parity",
        "emit_global_state": True,
        "global_state_include_pairwise": True,
        "coverage_radius": 0.1,
        "dist_threshold": 0.1,
        "collision_penalty": 0.0,
        "out_of_bounds_penalty": 0.0,
        "distance_reward_scale": 1.0,
        "coverage_reward": 0.0,
        "success_reward": 0.0,
        "shared_rew": True,
        "done_when_all_covered": True,
        "comms_rendering_range": 0.0,
    },
    "spread_mpe_actor_rel": {
        "max_steps": 50,
        "n_agents": 3,
        "n_landmarks": 3,
        "continuous_actions": False,
        "reward_mode": "mpe_parity",
        "parity_include_other_agents": True,
        "coverage_radius": 0.1,
        "dist_threshold": 0.1,
        "collision_penalty": 0.0,
        "out_of_bounds_penalty": 0.0,
        "distance_reward_scale": 1.0,
        "coverage_reward": 0.0,
        "success_reward": 0.0,
        "shared_rew": True,
        "done_when_all_covered": True,
        "comms_rendering_range": 0.0,
    },
    "spread_mpe_richer_both": {
        "max_steps": 50,
        "n_agents": 3,
        "n_landmarks": 3,
        "continuous_actions": False,
        "reward_mode": "mpe_parity",
        "parity_include_other_agents": True,
        "emit_global_state": True,
        "global_state_include_pairwise": True,
        "coverage_radius": 0.1,
        "dist_threshold": 0.1,
        "collision_penalty": 0.0,
        "out_of_bounds_penalty": 0.0,
        "distance_reward_scale": 1.0,
        "coverage_reward": 0.0,
        "success_reward": 0.0,
        "shared_rew": True,
        "done_when_all_covered": True,
        "comms_rendering_range": 0.0,
    },
    "spread_gnn": {
        "max_steps": 100,
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
        "comms_rendering_range": 1.0,
    },
    "sar_low": {
        "scenario_type": "sar_low",
        "mode": "low",
        "emit_info": False,
        "enable_high_level_state": False,
        "max_steps": 100,
        "n_agents": 3,
        "n_targets": 3,
        "belief_world_size": 2.0,
        "belief_cell_size": 0.02,
        "sensor_radius": 0.3,
        "sensor_fidelity": 0.8,
        "goal_radius": 0.05,
        "goal_reward": 3.0,
        "rescue_reward": 10.0,
        "discovery_reward": 1.0,
        "distance_reward_scale": 1.0,
        "collision_penalty": -10.0,
        "collision_distance": 0.0,
        "collision_safe_distance": 0.02,
        "max_collision_penalty": -20.0,
        "boundary_penalty": -5.0,
        "time_penalty": 0.0,
        "retire_on_rescue": True,
        "auto_resample_goals": True,
        "done_when_all_targets_visited": False,
        "high_level_interval": 5,
        "rrt_top_k": 5,
        "rrt_max_iter": 40,
        "enable_rrt_candidates": False,
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
    },
    "sar_high_fixed": {
        "scenario_type": "sar_high_fixed",
        "mode": "high",
        "emit_info": True,
        "enable_high_level_state": True,
        "max_steps": 100,
        "n_agents": 3,
        "n_targets": 3,
        "belief_world_size": 2.0,
        "belief_cell_size": 0.02,
        "sensor_radius": 0.3,
        "sensor_fidelity": 0.8,
        "goal_radius": 0.05,
        "goal_reward": 3.0,
        "rescue_reward": 10.0,
        "discovery_reward": 1.0,
        "distance_reward_scale": 1.0,
        "collision_penalty": -10.0,
        "collision_distance": 0.0,
        "collision_safe_distance": 0.06,
        "max_collision_penalty": -20.0,
        "boundary_penalty": -5.0,
        "time_penalty": 0.2,
        "retire_on_rescue": True,
        "auto_resample_goals": False,
        "done_when_all_targets_visited": True,
        "high_level_interval": 5,
        "rrt_top_k": 5,
        "rrt_max_iter": 40,
        "enable_rrt_candidates": True,
        "comms_rendering_range": 1.0,
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
    },
}

TASK_VARIANTS["sar_low_mpe_physics"] = TASK_VARIANTS["sar_low"] | {
    "physics_profile": "mpe_strict",
    "continuous_actions": False,
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


def build_experiment_config(
    *,
    train_device: str,
    sampling_device: str,
    quick: bool,
    save_folder: str = "outputs",
    restore_file: str | None = None,
    restore_map_location: str | None = None,
    max_n_frames: int | None = None,
    render: bool = True,
    old_ppo_profile: bool = False,
):
    from benchmarl.experiment import ExperimentConfig

    config = ExperimentConfig.get_from_yaml()
    config.save_folder = save_folder
    config.sampling_device = sampling_device
    config.train_device = train_device
    config.gamma = 0.99
    config.evaluation = True
    config.render = render
    config.share_policy_params = True
    config.checkpoint_interval = 300_000
    config.checkpoint_at_end = True
    config.keep_checkpoints_num = 5
    config.loggers = ["csv"]

    if restore_file is not None:
        config.restore_file = restore_file
    if restore_map_location is not None:
        config.restore_map_location = restore_map_location

    if quick:
        config.max_n_frames = 6_000
        config.on_policy_collected_frames_per_batch = 6_000
        config.on_policy_n_envs_per_worker = 60
        config.on_policy_n_minibatch_iters = 1
        config.on_policy_minibatch_size = 4096
        config.evaluation_interval = 6_000
        config.evaluation_episodes = 16
    else:
        config.max_n_frames = 50_000_000
        config.on_policy_collected_frames_per_batch = 60_000
        config.on_policy_n_envs_per_worker = 600
        config.on_policy_n_minibatch_iters = 45
        config.on_policy_minibatch_size = 4096
        config.evaluation_interval = 120_000
        config.evaluation_episodes = 200

    if old_ppo_profile:
        config.lr = 1e-4
        config.on_policy_collected_frames_per_batch = 4_096
        config.on_policy_n_envs_per_worker = 32
        config.on_policy_n_minibatch_iters = 4
        config.on_policy_minibatch_size = 128
        config.evaluation_interval = 40_960
        config.checkpoint_interval = 409_600
        config.evaluation_episodes = 200

    if max_n_frames is not None:
        config.max_n_frames = max_n_frames
        if quick:
            config.on_policy_collected_frames_per_batch = min(
                config.on_policy_collected_frames_per_batch,
                max_n_frames,
                1_000,
            )
            config.on_policy_n_envs_per_worker = min(
                config.on_policy_n_envs_per_worker,
                max(1, min(4, max_n_frames // 100)),
            )
            config.on_policy_minibatch_size = min(
                config.on_policy_minibatch_size,
                1024,
            )
            config.evaluation_interval = min(config.evaluation_interval, max_n_frames)
            config.evaluation_episodes = min(config.evaluation_episodes, 4)

    return config


def build_mappo_config(*, old_ppo_profile: bool = False):
    from benchmarl.algorithms import MappoConfig

    return MappoConfig(
        share_param_critic=True,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        critic_coef=0.5 if old_ppo_profile else 1,
        loss_critic_type="l2",
        lmbda=0.95,
        scale_mapping="biased_softplus_1.0",
        use_tanh_normal=True,
        minibatch_advantage=False,
    )


def build_mlp_configs():
    from benchmarl.models.mlp import MlpConfig

    return MlpConfig.get_from_yaml(), MlpConfig.get_from_yaml()


def build_gnn_actor_config(*, comms_radius: float):
    import torch_geometric
    from benchmarl.models import GnnConfig, SequenceModelConfig
    from benchmarl.models.mlp import MlpConfig

    gnn_config = GnnConfig(
        topology="from_pos",
        edge_radius=comms_radius,
        self_loops=False,
        gnn_class=torch_geometric.nn.conv.GATv2Conv,
        gnn_kwargs={"add_self_loops": False, "residual": True},
        position_key="pos",
        pos_features=2,
        velocity_key="vel",
        vel_features=2,
        exclude_pos_from_node_features=True,
    )
    mlp_config = MlpConfig.get_from_yaml()
    return SequenceModelConfig(
        model_configs=[gnn_config, mlp_config],
        intermediate_sizes=[256],
    )


def build_demo_mlp_config():
    from benchmarl.models.mlp import MlpConfig

    return MlpConfig(
        num_cells=[256, 256],
        layer_class=torch.nn.Linear,
        activation_class=torch.nn.Tanh,
    )
