"""Fail-closed Chapter 1 health-gate-v2 retrospective checkpoint selection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from comm_spread.chapter1_config import PROJECT_ROOT, resolve_chapter1_config


SCHEMA_VERSION = 2
SELECTION_VERSION = "chapter1_health_gate_v2_retrospective"
DEFAULT_BOOTSTRAP_SAMPLES = 100_000
DEFAULT_BOOTSTRAP_SEED = 20260722


def select_retrospective_checkpoint(
    run_dir: str | Path,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    target_update: int = 200,
    minimum_improvement: float = 0.05,
    maximum_kl: float = 0.02,
) -> dict[str, Any]:
    """Select a stable, paired-validation checkpoint from a completed strict run."""

    run = Path(run_dir).resolve()
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    manifest = _read_json(run / "run_manifest.json")
    resolved = _read_json(run / "resolved_config.json")
    _validate_run_identity(run, manifest, resolved)
    health_decisions = [
        json.loads(line)
        for line in (run / "health_decisions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    if not health_decisions or health_decisions[-1] != {
        "reason": "success_improvement_below_0.05",
        "status": "completed_but_gate_failed",
        "update": target_update,
    }:
        raise ValueError(
            "retrospective v2 requires the preserved v1 gate-failed terminal decision"
        )
    diagnostics = _read_training_diagnostics(
        run / "scalars/train.csv",
        target_update=target_update,
        maximum_kl=maximum_kl,
    )
    windows = _read_validation_windows(run, manifest)
    if set(windows) != set(range(0, target_update + 1, 10)):
        raise ValueError("validation windows must cover updates 0..200 every 10 updates")
    baseline = windows[0]
    baseline_layouts = set(baseline["success_by_layout"])
    if any(set(item["success_by_layout"]) != baseline_layouts for item in windows.values()):
        raise ValueError("validation layout sets differ across checkpoints")

    rng = np.random.default_rng(bootstrap_seed)
    ordered_updates = sorted(windows)
    candidates: list[dict[str, Any]] = []
    for index, update in enumerate(ordered_updates):
        if update == 0:
            continue
        window = windows[update]
        paired = _paired_success(
            baseline,
            window,
            rng=rng,
            bootstrap_samples=bootstrap_samples,
        )
        previous_update = ordered_updates[index - 1] if index > 0 else None
        next_update = ordered_updates[index + 1] if index + 1 < len(ordered_updates) else None
        stable_neighbors = (
            previous_update is not None
            and previous_update != 0
            and next_update is not None
            and windows[previous_update]["success_rate"] - baseline["success_rate"]
            >= minimum_improvement
            and windows[next_update]["success_rate"] - baseline["success_rate"]
            >= minimum_improvement
        )
        diagnostic = diagnostics.get(update)
        eligible = (
            diagnostic is not None
            and diagnostic["effective_options"] > 0
            and diagnostic["approx_kl"] <= maximum_kl
            and window["success_rate"] - baseline["success_rate"] >= minimum_improvement
            and paired["ci95"][0] >= minimum_improvement
            and stable_neighbors
        )
        candidates.append(
            {
                "update": update,
                "checkpoint": window["checkpoint"],
                "checkpoint_sha256": window["checkpoint_sha256"],
                "success_rate": window["success_rate"],
                "detect_all_rate": window["detect_all_rate"],
                "high_reward_mean": diagnostic["high_reward_mean"] if diagnostic else None,
                "improvement_vs_update_0": (
                    window["success_rate"] - baseline["success_rate"]
                ),
                "paired_vs_update_0": paired,
                "stable_neighbors": stable_neighbors,
                "eligible": eligible,
            }
        )
    eligible = [item for item in candidates if item["eligible"]]
    if not eligible:
        raise ValueError("no checkpoint satisfies retrospective health gate v2")
    selected = max(
        eligible,
        key=lambda item: (
            item["success_rate"],
            item["detect_all_rate"],
            item["high_reward_mean"],
            -item["update"],
        ),
    )
    latest = windows[target_update]
    late = _paired_success(
        windows[selected["update"]],
        latest,
        rng=np.random.default_rng(bootstrap_seed),
        bootstrap_samples=bootstrap_samples,
    )
    late_regression = late["ci95"][1] < 0.0
    status = (
        "passed_for_final_evaluation_with_late_regression"
        if late_regression
        else "passed_for_final_evaluation"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_version": SELECTION_VERSION,
        "status": status,
        "run_integrity_passed": True,
        "checkpoint_candidate_passed": True,
        "late_training_regression": late_regression,
        "v1_health_gate_preserved": True,
        "parameters": {
            "target_update": target_update,
            "minimum_improvement": minimum_improvement,
            "maximum_kl": maximum_kl,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "inputs": {
            "run_dir": str(run),
            "task_config_sha256": manifest["task_config_sha256"],
            "config_sha256": manifest["config_sha256"],
            "low_level_checkpoint_sha256": manifest["low_level_checkpoint"]["sha256"],
            "validation_layout_count": len(baseline_layouts),
            "artifact_sha256": {
                name: _sha256_file(run / name)
                for name in (
                    "run_manifest.json",
                    "resolved_config.json",
                    "health_decisions.jsonl",
                    "scalars/train.csv",
                )
            },
            "v1_terminal_decision": health_decisions[-1],
        },
        "baseline": {
            key: baseline[key]
            for key in ("update", "success_rate", "detect_all_rate")
        },
        "selected_checkpoint": selected,
        "late_regression_vs_selected": late,
        "candidates": candidates,
    }


def write_retrospective_outputs(
    run_dir: str | Path,
    decision: dict[str, Any],
    *,
    base_eval_config: str | Path,
    episodes_per_seed: int = 500,
    decision_name: str = "health_gate_v2_retrospective.json",
    config_name: str = "derived_eval_v2.yaml",
) -> tuple[Path, Path]:
    """Write versioned decision evidence and a frozen 2-seed eval config."""

    run = Path(run_dir).resolve()
    if episodes_per_seed <= 0:
        raise ValueError("episodes_per_seed must be positive")
    decision_path = run / decision_name
    config_path = run / config_name
    if decision_path.exists() or config_path.exists():
        raise FileExistsError("retrospective outputs already exist")
    selected = decision["selected_checkpoint"]
    checkpoint = run / selected["checkpoint"]
    if _sha256_file(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("selected checkpoint SHA256 changed before output write")

    source = Path(base_eval_config).resolve()
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    task_source = (source.parent / document["task_ref"]).resolve()
    document["task_ref"] = os.path.relpath(task_source, config_path.parent)
    protocol = document["evaluation"]
    protocol["seeds"] = [0, 1]
    protocol["episodes_per_seed"] = episodes_per_seed
    protocol["total_episodes"] = episodes_per_seed * 2
    protocol["output_dir"] = str(
        (run / "evaluation_v2_1000").relative_to(PROJECT_ROOT)
    )
    protocol["high_level_checkpoint"] = {
        "path": str(checkpoint.relative_to(PROJECT_ROOT)),
        "sha256": selected["checkpoint_sha256"],
        "provenance": "strict_smdp_random_init_health_gate_v2_retrospective",
    }
    protocol["selection"] = {
        "version": decision["selection_version"],
        "selected_update": selected["update"],
        "decision_path": str(decision_path.relative_to(PROJECT_ROOT)),
        "late_training_regression": decision["late_training_regression"],
    }

    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    resolved_eval = resolve_chapter1_config(config_path)
    if resolved_eval.task_config_sha256 != decision["inputs"]["task_config_sha256"]:
        raise ValueError("derived eval task hash differs from retrospective run")
    return decision_path, config_path


def _validate_run_identity(
    run: Path,
    manifest: dict[str, Any],
    resolved: dict[str, Any],
) -> None:
    if manifest.get("high_level_initialization") != "random":
        raise ValueError("strict retrospective requires random high-level initialization")
    if manifest.get("resume_checkpoint") is not None:
        raise ValueError("strict retrospective rejects resumed runs")
    if not manifest.get("git_worktree_clean"):
        raise ValueError("strict retrospective requires a clean training worktree")
    fingerprints = resolved.get("fingerprints", {})
    if fingerprints.get("task_config_sha256") != manifest.get("task_config_sha256"):
        raise ValueError("task_config_sha256 mismatch")
    if fingerprints.get("config_sha256") != manifest.get("config_sha256"):
        raise ValueError("config_sha256 mismatch")
    training = resolved.get("resolved", {}).get("run", {}).get("training", {})
    high = training.get("high_level", {})
    required = {
        "return_mode": "strict_smdp",
        "training_success_terminal": True,
        "environment_continue_after_success": True,
        "exclude_post_success_steps_from_training": True,
        "time_limit_bootstrap": True,
    }
    for key, expected in required.items():
        if high.get(key) != expected:
            raise ValueError(
                f"strict retrospective requires high_level.{key}={expected!r}"
            )
    low = manifest.get("low_level_checkpoint", {})
    low_path = Path(low.get("path", ""))
    if not low_path.is_file() or _sha256_file(low_path) != low.get("sha256"):
        raise ValueError("low-level checkpoint SHA256 mismatch")
    summary = (run / "texts/run_summary.txt").read_text(encoding="utf-8")
    if "status: complete" not in summary or "final_update: 200" not in summary:
        raise ValueError("run did not complete update 200")


def _read_training_diagnostics(
    path: Path,
    *,
    target_update: int,
    maximum_kl: float,
) -> dict[int, dict[str, float]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    updates = [int(row["update"]) for row in rows]
    if updates != list(range(1, target_update + 1)):
        raise ValueError("training diagnostics must contain exactly updates 1..200")
    required = (
        "effective_options",
        "approx_kl",
        "actor_invalid_actions",
        "actor_nonfinite_logits",
        "duration_mean",
        "duration_p50",
        "duration_p90",
        "duration_max",
        "high_reward_mean",
    )
    result: dict[int, dict[str, float]] = {}
    for row in rows:
        values = {key: float(row[key]) for key in required}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite training diagnostic at update {row['update']}")
        if values["effective_options"] <= 0:
            raise ValueError(f"no effective options at update {row['update']}")
        if values["approx_kl"] > maximum_kl:
            raise ValueError(f"KL exceeds health limit at update {row['update']}")
        if values["actor_invalid_actions"] or values["actor_nonfinite_logits"]:
            raise ValueError(f"invalid/nonfinite actor output at update {row['update']}")
        if not (
            0 < values["duration_mean"]
            <= values["duration_p90"]
            <= values["duration_max"]
            <= 100
        ):
            raise ValueError(f"abnormal option duration at update {row['update']}")
        result[int(row["update"])] = values
    return result


def _read_validation_windows(
    run: Path,
    manifest: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for directory in sorted((run / "validation").glob("update_*")):
        summary = _read_json(directory / "summary.json")
        binding = _read_json(directory / "checkpoint_binding.json")
        update = int(binding["update"])
        rows = [
            json.loads(line)
            for line in (directory / "per_episode.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        if len(rows) != summary["episodes"]:
            raise ValueError(f"episode count mismatch at update {update}")
        if summary["invalid_count"] or summary["nonfinite_count"]:
            raise ValueError(f"invalid/nonfinite validation at update {update}")
        if (
            summary["task_config_sha256"] != manifest["task_config_sha256"]
            or binding["task_config_sha256"] != manifest["task_config_sha256"]
            or binding["config_sha256"] != manifest["config_sha256"]
        ):
            raise ValueError(f"validation provenance mismatch at update {update}")
        checkpoint = run / binding["checkpoint"]
        if _sha256_file(checkpoint) != binding["checkpoint_sha256"]:
            raise ValueError(f"checkpoint binding SHA256 mismatch at update {update}")
        success_by_layout = {
            str(row["layout_sha256"]): bool(row["success"]) for row in rows
        }
        if len(success_by_layout) != len(rows):
            raise ValueError(f"duplicate validation layout at update {update}")
        result[update] = {
            "update": update,
            "success_rate": float(summary["success_rate"]),
            "detect_all_rate": float(summary["detect_all_rate"]),
            "checkpoint": binding["checkpoint"],
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "success_by_layout": success_by_layout,
        }
    return result


def _paired_success(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    layouts = sorted(baseline["success_by_layout"])
    differences = np.asarray(
        [
            int(candidate["success_by_layout"][layout])
            - int(baseline["success_by_layout"][layout])
            for layout in layouts
        ],
        dtype=np.int8,
    )
    boot = differences[
        rng.integers(0, len(differences), size=(bootstrap_samples, len(differences)))
    ].mean(axis=1)
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return {
        "difference": float(differences.mean()),
        "candidate_only": int((differences == 1).sum()),
        "baseline_only": int((differences == -1).sum()),
        "ci95": [float(lower), float(upper)],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
