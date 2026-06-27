"""Train communication-limited VMAS scenarios with BenchMARL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

CONFIG_ROOT = PROJECT_ROOT / "configs"

from comm_limited_vmas.benchmarl_task import build_comm_vmas_task


ALGORITHM_REGISTRY = {
    "mappo": MappoConfig,
    "ippo": IppoConfig,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train comm-limited VMAS tasks with BenchMARL."
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="mappo",
        choices=list(ALGORITHM_REGISTRY.keys()),
    )

    parser.add_argument(
        "--task",
        type=str,
        default="comm_spread",
        help="Task config name under configs/task, e.g. comm_spread.",
    )

    parser.add_argument(
        "--comm",
        type=str,
        default="comm_full",
        help="Communication config name under configs/comm, e.g. comm_full, delay, radius_delay_dropout.",
    )

    parser.add_argument(
        "--estimator",
        type=str,
        default="stale",
        help="Estimator config name under configs/estimator, e.g. stale.",
    )

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train-device", type=str, default="cpu")
    parser.add_argument("--sampling-device", type=str, default="cpu")
    parser.add_argument("--buffer-device", type=str, default=None)

    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["debug", "server", "full"],
        help=(
            "Training scale profile. debug is a smoke test, server is the "
            "first formal single-GPU run, full is a long final run."
        ),
    )

    parser.add_argument(
        "--max-n-frames",
        type=int,
        default=None,
        help="Override BenchMARL max_n_frames. If omitted, use default config.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override experiment learning rate.",
    )

    parser.add_argument(
        "--ppo-iters",
        type=int,
        default=None,
        help="Override on-policy minibatch optimization iterations.",
    )

    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=None,
        help="Override on-policy minibatch size.",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small debug training setting.",
    )

    return parser.parse_args()


def load_yaml(group: str, name: str) -> dict[str, Any]:
    path = CONFIG_ROOT / group / f"{name}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def apply_overrides(config_obj: Any, overrides: dict[str, Any]) -> Any:
    """Apply YAML values to BenchMARL config objects if the field exists."""
    for key, value in overrides.items():
        if hasattr(config_obj, key):
            setattr(config_obj, key, value)
        else:
            print(
                f"[Warning] Skip unknown config field: {key} "
                f"for {type(config_obj).__name__}"
            )
    return config_obj


def build_algorithm_config(name: str) -> Any:
    algorithm_cls = ALGORITHM_REGISTRY[name]
    algorithm_config = algorithm_cls.get_from_yaml()

    yaml_overrides = load_yaml("algorithm", name)
    apply_overrides(algorithm_config, yaml_overrides)

    return algorithm_config


def build_model_configs() -> tuple[Any, Any]:
    """First version: use MLP actor and MLP critic."""
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()
    return model_config, critic_model_config


def apply_training_profile(
    experiment_config: ExperimentConfig,
    profile: str,
) -> None:
    if profile == "debug":
        experiment_config.max_n_frames = 20_000
        experiment_config.on_policy_n_envs_per_worker = 8
        experiment_config.on_policy_collected_frames_per_batch = 1_000
        experiment_config.on_policy_n_minibatch_iters = 1
        experiment_config.on_policy_minibatch_size = 256
        experiment_config.evaluation_interval = 1_000
        experiment_config.evaluation_episodes = 2
        experiment_config.render = False
        experiment_config.checkpoint_interval = 0
        experiment_config.checkpoint_at_end = False
        return

    experiment_config.on_policy_collected_frames_per_batch = 60_000
    experiment_config.on_policy_n_envs_per_worker = 600
    experiment_config.on_policy_n_minibatch_iters = 20
    experiment_config.on_policy_minibatch_size = 4096
    experiment_config.lr = 3e-5
    experiment_config.evaluation_interval = 120_000
    experiment_config.render = False
    experiment_config.checkpoint_interval = 600_000
    experiment_config.checkpoint_at_end = True
    experiment_config.keep_checkpoints_num = 5

    if profile == "server":
        experiment_config.max_n_frames = 20_000_000
        experiment_config.evaluation_episodes = 100
    elif profile == "full":
        experiment_config.max_n_frames = 50_000_000
        experiment_config.evaluation_episodes = 200
    else:
        raise ValueError(f"Unknown training profile: {profile}")


def build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    experiment_config = ExperimentConfig.get_from_yaml()

    experiment_config.train_device = args.train_device
    experiment_config.sampling_device = args.sampling_device
    experiment_config.buffer_device = (
        args.buffer_device if args.buffer_device is not None else args.train_device
    )

    profile = "debug" if args.quick else args.profile
    if profile is not None:
        apply_training_profile(experiment_config, profile)

    if args.max_n_frames is not None:
        experiment_config.max_n_frames = args.max_n_frames

    if args.lr is not None:
        experiment_config.lr = args.lr

    if args.ppo_iters is not None:
        experiment_config.on_policy_n_minibatch_iters = args.ppo_iters

    if args.minibatch_size is not None:
        experiment_config.on_policy_minibatch_size = args.minibatch_size

    save_folder = (
        PROJECT_ROOT
        / "outputs"
        / f"{args.task}_{args.algorithm}_{args.comm}_{args.estimator}_seed{args.seed}"
    )
    save_folder.mkdir(parents=True, exist_ok=True)

    if hasattr(experiment_config, "save_folder"):
        experiment_config.save_folder = str(save_folder)

    return experiment_config


def main() -> None:
    args = parse_args()

    task_config = load_yaml("task", args.task)
    comm_config = load_yaml("comm", args.comm)
    estimator_config = load_yaml("estimator", args.estimator)

    task = build_comm_vmas_task(
        task_name=args.task,
        task_config=task_config,
        comm_config=comm_config,
        estimator_config=estimator_config,
    )

    algorithm_config = build_algorithm_config(args.algorithm)
    model_config, critic_model_config = build_model_configs()
    experiment_config = build_experiment_config(args)

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=experiment_config,
    )

    experiment.run()


if __name__ == "__main__":
    main()
