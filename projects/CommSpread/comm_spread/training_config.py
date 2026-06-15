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
        "comms_rendering_range": 1.0,
    },
}


def build_experiment_config(
    *,
    train_device: str,
    sampling_device: str,
    quick: bool,
    save_folder: str = "outputs",
):
    from benchmarl.experiment import ExperimentConfig

    config = ExperimentConfig.get_from_yaml()
    config.save_folder = save_folder
    config.sampling_device = sampling_device
    config.train_device = train_device
    config.gamma = 0.99
    config.evaluation = True
    config.render = True
    config.share_policy_params = True
    config.checkpoint_interval = 300_000
    config.checkpoint_at_end = True
    config.keep_checkpoints_num = 5
    config.loggers = ["csv"]

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

    return config


def build_mappo_config():
    from benchmarl.algorithms import MappoConfig

    return MappoConfig(
        share_param_critic=True,
        clip_epsilon=0.2,
        entropy_coef=0.001,
        critic_coef=1,
        loss_critic_type="l2",
        lmbda=0.9,
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
