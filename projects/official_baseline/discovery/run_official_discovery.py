"""Run the official BenchMARL MAPPO baseline on VMAS discovery."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train official BenchMARL MAPPO on VMAS discovery."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--buffer-device", default=None)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs")
    parser.add_argument(
        "--max-n-frames",
        type=int,
        default=None,
        help="Optional smoke-test override. Omit for official default.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Optional checkpoint interval override. Omit for official default.",
    )
    parser.add_argument(
        "--checkpoint-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a final checkpoint for evaluation. Does not affect training.",
    )
    return parser.parse_args()


def build_experiment(args: argparse.Namespace) -> Experiment:
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.train_device = args.device
    experiment_config.sampling_device = args.device
    experiment_config.buffer_device = args.buffer_device or args.device
    experiment_config.save_folder = str(args.output_dir.expanduser().resolve())
    experiment_config.checkpoint_at_end = args.checkpoint_at_end

    if args.max_n_frames is not None:
        experiment_config.max_n_frames = args.max_n_frames
    if args.checkpoint_interval is not None:
        experiment_config.checkpoint_interval = args.checkpoint_interval

    return Experiment(
        task=VmasTask.DISCOVERY.get_from_yaml(),
        algorithm_config=MappoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=args.seed,
        config=experiment_config,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment = build_experiment(args)
    experiment.run()


if __name__ == "__main__":
    main()
