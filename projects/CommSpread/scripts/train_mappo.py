"""Train CommSpread with BenchMARL MAPPO."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarl.experiment import Experiment

from comm_spread.benchmarl_task import CommSpreadTask
from comm_spread.chapter1_config import (
    chapter1_sar_kwargs,
    fingerprint_lines,
    resolve_chapter1_config,
    save_resolved_config,
    write_checkpoint_metadata_sidecars,
)
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
    parser.add_argument("--restore-file", default=None)
    parser.add_argument("--restore-map-location", default=None)
    parser.add_argument(
        "--max-n-frames",
        type=int,
        default=None,
        help="Override the total frame target, useful when resuming from a checkpoint.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the long training setup instead of the quick smoke run.",
    )
    parser.add_argument("--save-folder", default="outputs", help="Directory to save the experiment.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable evaluation rendering, useful on headless training machines.",
    )
    parser.add_argument(
        "--chapter1-config",
        type=Path,
        default=None,
        help="Resolve and enforce a versioned Chapter 1 low-training config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chapter1 = None
    if args.chapter1_config is not None:
        chapter1 = resolve_chapter1_config(args.chapter1_config)
        if chapter1.run_type != "low_train":
            raise ValueError("--chapter1-config must reference a low_train config")
        run = chapter1.run["training"]
        if args.variant != "spread" and args.variant != run["variant"]:
            raise ValueError("Chapter 1 low training forbids changing --variant")
        if args.model != "mlp" and args.model != run["model"]:
            raise ValueError("Chapter 1 low training forbids changing --model")
        args.variant = run["variant"]
        args.model = run["model"]
        args.seed = run["seed"]
        args.train_device = run["train_device"]
        args.sampling_device = run["sampling_device"]
        args.save_folder = run["output_dir"]
        args.max_n_frames = run["max_frames"]
        args.full = True
        args.no_render = True
        args.task_version = chapter1.task["task_version"]
        args.task_config_sha256 = chapter1.task_config_sha256
    if args.variant == "sar_high_fixed":
        raise SystemExit(
            "sar_high_fixed uses comm_spread.macro_env.SarFixedIntervalMacroEnv "
            "and needs a custom BenchMARL collector before train_mappo.py can train it."
        )
    Path(args.save_folder).mkdir(parents=True, exist_ok=True)

    task_config = (
        {"scenario_type": "sar_low"}
        | chapter1_sar_kwargs(chapter1.task, mode="low")
        if chapter1 is not None
        else dict(TASK_VARIANTS[args.variant])
    )
    if chapter1 is not None:
        save_resolved_config(chapter1, args.save_folder)
        print("\n".join(fingerprint_lines(chapter1)), flush=True)

    if args.model == "gnn":
        task_config["comms_rendering_range"] = args.comms_radius
        model_config = build_gnn_actor_config(comms_radius=args.comms_radius)
        _, critic_model_config = build_mlp_configs()
    else:
        model_config, critic_model_config = build_mlp_configs()

    task_member = (
        CommSpreadTask.SAR_LOW
        if args.variant.startswith("sar_low")
        else CommSpreadTask.SAR_HIGH_FIXED
        if args.variant == "sar_high_fixed"
        else CommSpreadTask.SPREAD
    )
    task = task_member.update_config(task_config)
    experiment = Experiment(
        task=task,
        algorithm_config=build_mappo_config(old_ppo_profile=args.variant.endswith("oldppo")),
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=args.seed,
        config=build_experiment_config(
            train_device=args.train_device,
            sampling_device=args.sampling_device,
            quick=not args.full,
            save_folder=args.save_folder,
            restore_file=args.restore_file,
            restore_map_location=args.restore_map_location,
            max_n_frames=args.max_n_frames,
            render=not args.no_render,
            old_ppo_profile=args.variant.endswith("oldppo"),
        ),
    )
    experiment.run()
    if chapter1 is not None:
        for sidecar in write_checkpoint_metadata_sidecars(chapter1, args.save_folder):
            print(f"wrote checkpoint metadata: {sidecar}")


if __name__ == "__main__":
    main()
