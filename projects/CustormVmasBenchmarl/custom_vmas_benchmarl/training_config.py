"""Training configuration builders used by the scripts."""

from __future__ import annotations

from typing import Any

import torch


TASK_VARIANTS: dict[str, dict[str, Any]] = {
    "heterogeneous": {
        "max_steps": 100,
        "n_agents_holonomic": 2,
        "n_agents_diff_drive": 1,
        "n_agents_car": 1,
        "n_obstacles": 2,
        "lidar_range": 0.35,
        "comms_rendering_range": 0,
        "shared_rew": False,
    },
    "homogeneous": {
        "max_steps": 100,
        "n_agents_holonomic": 4,
        "n_agents_diff_drive": 0,
        "n_agents_car": 0,
        "n_obstacles": 2,
        "lidar_range": 0.35,
        "comms_rendering_range": 0,
        "shared_rew": False,
    },
    "no_lidar": {
        "max_steps": 100,
        "n_agents_holonomic": 4,
        "n_agents_diff_drive": 0,
        "n_agents_car": 0,
        "n_obstacles": 2,
        "lidar_range": 0,
        "comms_rendering_range": 0,
        "shared_rew": False,
    },
    "no_lidar_gnn": {
        "max_steps": 100,
        "n_agents_holonomic": 4,
        "n_agents_diff_drive": 0,
        "n_agents_car": 0,
        "n_obstacles": 2,
        "lidar_range": 0,
        "comms_rendering_range": 1,
        "shared_rew": False,
    },
}


def build_experiment_config(
    *,
    train_device: str,
    sampling_device: str,
    quick: bool,
):
    from benchmarl.experiment import ExperimentConfig

    config = ExperimentConfig.get_from_yaml()
    config.sampling_device = sampling_device
    config.train_device = train_device
    config.gamma = 0.99
    config.evaluation = True
    config.render = True
    config.share_policy_params = True
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
