"""Run the existing SAR diagnostic under strict MPE physics.

This wrapper deliberately leaves the large, pre-existing analysis script
untouched. It injects the strict environment and matching discrete low-level
adapter, while preserving every existing CLI flag and output schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from functools import partial

import analyze_sar_failure_modes as analysis
from comm_spread.chapter1_config import (
    PROJECT_ROOT,
    chapter1_sar_kwargs,
    fingerprint_lines,
    resolve_chapter1_config,
    save_resolved_config,
    validate_runtime_task_values,
    verify_checkpoint_sha256,
)
from comm_spread.env_factory import make_sar_env
from comm_spread.low_level_policy import BenchMARLLowLevelPolicy


class StrictMPELowLevelPolicy(BenchMARLLowLevelPolicy):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("task_variant", "sar_low_mpe_physics")
        super().__init__(*args, **kwargs)


def main() -> None:
    original_parse_args = analysis.parse_args
    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument("--chapter1-config", type=Path, default=None)
    wrapper_args, remaining = wrapper_parser.parse_known_args()
    config_path = wrapper_args.chapter1_config or (
        PROJECT_ROOT / "configs/chapter1/eval_v1.yaml"
    )
    sys.argv = [sys.argv[0], *remaining]

    def parse_args():
        args = original_parse_args()
        resolved = resolve_chapter1_config(
            config_path,
            cli_overrides={
                "run.evaluation.output_dir": str(args.output.parent),
                "run.evaluation.current_seed": args.seed,
                "run.evaluation.episodes_this_process": args.num_envs,
            },
        )
        validate_runtime_task_values(
            resolved,
            {
                "physics_profile": "mpe_strict",
                "max_steps": args.max_steps,
                "sensor_radius": args.sensor_radius,
                "retire_on_rescue": args.retire_on_rescue,
            },
        )
        protocol = resolved.run["evaluation"]
        if args.seed not in protocol["seeds"]:
            raise ValueError(f"final eval seed must be one of {protocol['seeds']}")
        if args.num_envs != protocol["episodes_per_seed"]:
            raise ValueError(
                "final eval requires "
                f"num_envs={protocol['episodes_per_seed']}, got {args.num_envs}"
            )
        task = resolved.task
        hierarchy = task["hierarchy"]
        release = hierarchy["rescue_release"]
        low = protocol["low_level_checkpoint"]
        high = protocol["high_level_checkpoint"]
        args.low_level_checkpoint = verify_checkpoint_sha256(
            PROJECT_ROOT / low["path"], low["sha256"]
        )
        args.high_level_checkpoint = verify_checkpoint_sha256(
            PROJECT_ROOT / high["path"], high["sha256"]
        )
        args.physics_profile = task["physics"]["profile"]
        args.physics_dt = task["physics"]["dt"]
        args.low_level_task_variant = low["variant"]
        args.max_steps = task["scenario"]["horizon"]
        args.steps = task["scenario"]["horizon"]
        args.sensor_radius = task["perception"]["sensor_radius"]
        args.retire_on_rescue = task["task_semantics"]["retire_on_rescue"]
        args.dynamic_rescue_release = release["dynamic_staggered"]
        args.dynamic_rescue_only_after_all_detected = release[
            "only_after_all_targets_detected"
        ]
        args.dynamic_release_max_new_agents_per_event = release[
            "max_new_agents_per_event"
        ]
        args.redecide_on_detection_change = hierarchy["replanning"][
            "on_detection_change"
        ]
        args.redecide_on_assignment_change = hierarchy["replanning"][
            "on_assignment_change"
        ]
        args.enable_finder_first_cascade = hierarchy["finder_first"]["enabled"]
        args.finder_cascade_mode = hierarchy["finder_first"]["cascade_mode"]
        fallback = task["low_level_interface"]["execution"]["stagnation_fallback"]
        args.enable_low_level_goal_fallback = fallback["enabled"]
        args.low_level_goal_near_threshold = fallback["near_threshold"]
        args.low_level_fallback_stagnation_steps = fallback["stagnation_steps"]
        args.low_level_fallback_progress_epsilon = fallback["progress_epsilon"]
        save_resolved_config(resolved, args.output.parent)
        print("\n".join(fingerprint_lines(resolved)), flush=True)
        return args

    analysis.parse_args = parse_args
    frozen = resolve_chapter1_config(config_path)
    analysis.make_sar_env = partial(
        make_sar_env,
        **chapter1_sar_kwargs(frozen.task, mode="high"),
    )
    analysis.BenchMARLLowLevelPolicy = StrictMPELowLevelPolicy
    analysis.main()


if __name__ == "__main__":
    main()
