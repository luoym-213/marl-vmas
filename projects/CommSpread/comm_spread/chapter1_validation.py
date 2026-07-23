"""Deterministic, training-loop-independent Chapter 1 validation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from comm_spread.chapter1_config import (
    PROJECT_ROOT,
    ResolvedChapter1Config,
    chapter1_sar_kwargs,
    verify_checkpoint_sha256,
)


DEFAULT_VALIDATION_SEED = 10000
DEFAULT_VALIDATION_EPISODES = 128


def validation_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the stable Chapter 1 validation schema from episode rows."""
    episodes = len(rows)
    success_rows = [row for row in rows if bool(row["success"])]
    detect_all_rows = [row for row in rows if int(row["all_detected_step"]) >= 0]
    reasons = Counter(str(row["failure_reason"]) for row in rows)
    return {
        "episodes": episodes,
        "success_count": len(success_rows),
        "success_rate": len(success_rows) / max(episodes, 1),
        "detect_all_count": len(detect_all_rows),
        "detect_all_rate": len(detect_all_rows) / max(episodes, 1),
        "success_step": _step_summary(success_rows, "success_step"),
        "all_detected_step": _step_summary(detect_all_rows, "all_detected_step"),
        "timeout_count": episodes - len(success_rows),
        "failure_classification": dict(sorted(reasons.items())),
        "invalid_count": sum(int(row.get("invalid_assignment_count", 0)) for row in rows),
        "nonfinite_count": sum(int(row.get("nonfinite_state_count", 0)) for row in rows),
    }


def write_validation_outputs(
    output_dir: str | Path,
    rows: list[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    episode_path = destination / "per_episode.jsonl"
    summary_path = destination / "summary.json"
    episode_path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return episode_path, summary_path


def run_chapter1_validation(
    *,
    resolved: ResolvedChapter1Config,
    low_level_checkpoint: str | Path,
    low_level_sha256: str,
    output_dir: str | Path,
    high_level_policy: Any | None = None,
    high_level_checkpoint: str | Path | None = None,
    seed: int = DEFAULT_VALIDATION_SEED,
    episodes: int = DEFAULT_VALIDATION_EPISODES,
    device: str = "cuda:0",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run frozen Chapter 1 validation with a policy object or checkpoint."""
    if (high_level_policy is None) == (high_level_checkpoint is None):
        raise ValueError("provide exactly one high-level policy or checkpoint")
    if episodes <= 0:
        raise ValueError("validation episodes must be positive")
    low_path = verify_checkpoint_sha256(low_level_checkpoint, low_level_sha256)

    from scripts import analyze_sar_failure_modes as analysis

    args: argparse.Namespace = analysis.build_parser().parse_args([])
    task = resolved.task
    hierarchy = task["hierarchy"]
    release = hierarchy["rescue_release"]
    fallback = task["low_level_interface"]["execution"]["stagnation_fallback"]
    args.low_level_checkpoint = low_path
    args.high_level_checkpoint = Path(high_level_checkpoint) if high_level_checkpoint else None
    args.num_envs = episodes
    args.seed = seed
    args.device = device
    args.steps = task["scenario"]["horizon"]
    args.max_steps = task["scenario"]["horizon"]
    args.sensor_radius = task["perception"]["sensor_radius"]
    args.retire_on_rescue = task["task_semantics"]["retire_on_rescue"]
    args.dynamic_rescue_release = release["dynamic_staggered"]
    args.dynamic_rescue_only_after_all_detected = release["only_after_all_targets_detected"]
    args.dynamic_release_max_new_agents_per_event = release["max_new_agents_per_event"]
    args.low_level_task_variant = task["low_level_interface"]["variant"]
    args.redecide_on_detection_change = hierarchy["replanning"]["on_detection_change"]
    args.redecide_on_assignment_change = hierarchy["replanning"]["on_assignment_change"]
    args.enable_finder_first_cascade = hierarchy["finder_first"]["enabled"]
    args.finder_cascade_mode = hierarchy["finder_first"]["cascade_mode"]
    args.enable_low_level_goal_fallback = fallback["enabled"]
    args.low_level_goal_near_threshold = fallback["near_threshold"]
    args.low_level_fallback_stagnation_steps = fallback["stagnation_steps"]
    args.low_level_fallback_progress_epsilon = fallback["progress_epsilon"]

    original_factory = analysis.make_sar_env
    previous_deterministic = getattr(high_level_policy, "deterministic", None)
    analysis.make_sar_env = lambda **kwargs: original_factory(
        **(kwargs | chapter1_sar_kwargs(task, mode="high"))
    )
    if high_level_policy is not None and previous_deterministic is not None:
        high_level_policy.deterministic = True
    try:
        torch.manual_seed(seed)
        rows, _ = analysis.run(args, high_level_policy=high_level_policy)
    finally:
        analysis.make_sar_env = original_factory
        if high_level_policy is not None and previous_deterministic is not None:
            high_level_policy.deterministic = previous_deterministic
    summary = validation_summary(rows)
    summary.update(
        {
            "deterministic": True,
            "seed": seed,
            "task_config_sha256": resolved.task_config_sha256,
            "low_level_checkpoint_sha256": low_level_sha256,
        }
    )
    write_validation_outputs(output_dir, rows, summary)
    return rows, summary


def _step_summary(rows: list[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key])) and float(row[key]) >= 0]
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }
