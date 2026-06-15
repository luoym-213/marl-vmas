"""Train CommSpread with BenchMARL MAPPO."""

from __future__ import annotations

import argparse

from benchmarl.experiment import Experiment

from comm_spread.benchmarl_task import CommSpreadTask
from comm_spread.training_config import (
    TASK_VARIANTS,
    build_experiment_config,
    build_gnn_actor_config,
    build_mappo_config,
    build_mlp_configs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=sorted(TASK_VARIANTS),
        default="spread",
        help="CommSpread task variant.",
    )
    parser.add_argument("--model", choices=["mlp", "gnn"], default="mlp")
    parser.add_argument("--comms-radius", type=float, default=1.0)
    parser.add_argument("--train-device", default="cpu")
    parser.add_argument("--sampling-device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the long training setup instead of the quick smoke run.",
    )
    parser.add_argument("--save-folder", default="outputs", help="Directory to save the experiment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_config = dict(TASK_VARIANTS[args.variant])

    if args.model == "gnn":
        task_config["comms_rendering_range"] = args.comms_radius
        model_config = build_gnn_actor_config(comms_radius=args.comms_radius)
        _, critic_model_config = build_mlp_configs()
    else:
        model_config, critic_model_config = build_mlp_configs()

    task = CommSpreadTask.SPREAD.update_config(task_config)
    experiment = Experiment(
        task=task,
        algorithm_config=build_mappo_config(),
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=build_experiment_config(
            train_device=args.train_device,
            sampling_device=args.sampling_device,
            quick=not args.full,
            save_folder=args.save_folder,
        ),
    )
    experiment.run()


if __name__ == "__main__":
    main()
