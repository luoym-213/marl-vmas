"""Finalize a completed Chapter 1 training run for frozen evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

from comm_spread.chapter1_config import PROJECT_ROOT, resolve_chapter1_config


SCHEMA_VERSION = 1
SELECTION_VERSION = "chapter1_checkpoint_finalizer_v1"
DECISION_NAME = "checkpoint_selection.json"
CONFIG_NAME = "derived_eval_final.yaml"


def select_chapter1_checkpoint(run_dir: str | Path) -> dict[str, Any]:
    """Validate a completed run and rank its non-baseline validation checkpoints."""
    run = Path(run_dir).resolve()
    manifest = _read_json(run / "run_manifest.json")
    resolved = _read_json(run / "resolved_config.json")
    fingerprints = resolved.get("fingerprints", {})
    training = resolved.get("resolved", {}).get("run", {}).get("training", {})
    target_update = int(training.get("updates", 0))
    if target_update <= 0:
        raise ValueError("resolved training target update must be positive")

    for key in ("task_config_sha256", "config_sha256"):
        if fingerprints.get(key) != manifest.get(key):
            raise ValueError(f"{key} mismatch between manifest and resolved config")
    _validate_complete_summary(run / "texts/run_summary.txt", target_update)
    _validate_low_checkpoint(manifest)

    rewards = _read_training_rewards(run / "scalars/train.csv", target_update)
    health = _read_terminal_health(run / "health_decisions.jsonl", target_update)
    candidates = _read_candidates(run, manifest, rewards)
    if not candidates:
        raise ValueError("no non-baseline validation checkpoints found")
    if target_update not in {item["update"] for item in candidates}:
        raise ValueError("target update has no bound validation checkpoint")
    selected = _select_best_candidate(candidates)
    health_status = _classify_health(health)
    low = manifest["low_level_checkpoint"]
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_version": SELECTION_VERSION,
        "status": "finalized",
        "health_status": health_status,
        "health_decision": health,
        "run_integrity_passed": True,
        "parameters": {
            "target_update": target_update,
            "ranking": [
                "success_rate",
                "detect_all_rate",
                "high_reward_mean",
                "earlier_update",
            ],
        },
        "inputs": {
            "run_dir": str(run),
            "task_config_sha256": manifest["task_config_sha256"],
            "config_sha256": manifest["config_sha256"],
            "low_level_checkpoint_sha256": low["sha256"],
            "base_eval_config": training.get(
                "base_eval_config", "configs/chapter1/eval_v1.yaml"
            ),
        },
        "selected_checkpoint": selected,
        "candidates": candidates,
    }


def write_finalizer_outputs(
    run_dir: str | Path,
    decision: dict[str, Any],
    *,
    seeds: Iterable[int] = (0, 1),
    episodes_per_seed: int = 500,
) -> tuple[Path, Path]:
    """Atomically write or idempotently reuse selection evidence and eval config."""
    run = Path(run_dir).resolve()
    seed_list = [int(seed) for seed in seeds]
    if not seed_list or len(seed_list) != len(set(seed_list)):
        raise ValueError("evaluation seeds must be non-empty and unique")
    if episodes_per_seed <= 0:
        raise ValueError("episodes_per_seed must be positive")
    decision_path = run / DECISION_NAME
    config_path = run / CONFIG_NAME
    if decision_path.exists() != config_path.exists():
        raise FileExistsError("partial finalizer outputs exist; refusing to overwrite")

    selected = decision["selected_checkpoint"]
    checkpoint = run / selected["checkpoint"]
    if _sha256_file(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("selected checkpoint SHA256 changed before finalization")
    source = Path(decision["inputs"]["base_eval_config"])
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    source = source.resolve()
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    task_source = (source.parent / document["task_ref"]).resolve()
    document["task_ref"] = os.path.relpath(task_source, config_path.parent)
    protocol = document["evaluation"]
    protocol["seeds"] = seed_list
    protocol["episodes_per_seed"] = episodes_per_seed
    protocol["total_episodes"] = episodes_per_seed * len(seed_list)
    output_dir = run / f"evaluation_final_{protocol['total_episodes']}"
    protocol["output_dir"] = _project_relative(output_dir)
    protocol["high_level_checkpoint"] = {
        "path": _project_relative(checkpoint),
        "sha256": selected["checkpoint_sha256"],
        "provenance": SELECTION_VERSION,
    }
    protocol["selection"] = {
        "version": SELECTION_VERSION,
        "selected_update": selected["update"],
        "decision_path": _project_relative(decision_path),
        "health_status": decision["health_status"],
    }

    decision_text = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    config_text = yaml.safe_dump(document, sort_keys=False)
    if decision_path.exists():
        if (
            decision_path.read_text(encoding="utf-8") != decision_text
            or config_path.read_text(encoding="utf-8") != config_text
        ):
            raise FileExistsError("finalizer outputs conflict with current selection")
        _verify_derived_config(config_path, decision)
        return decision_path, config_path

    decision_tmp = decision_path.with_suffix(decision_path.suffix + ".tmp")
    config_tmp = config_path.with_suffix(config_path.suffix + ".tmp")
    if decision_tmp.exists() or config_tmp.exists():
        raise FileExistsError("stale finalizer temporary output exists")
    try:
        decision_tmp.write_text(decision_text, encoding="utf-8")
        config_tmp.write_text(config_text, encoding="utf-8")
        _verify_derived_config(config_tmp, decision)
        os.replace(config_tmp, config_path)
        os.replace(decision_tmp, decision_path)
    finally:
        decision_tmp.unlink(missing_ok=True)
        config_tmp.unlink(missing_ok=True)
    return decision_path, config_path


def finalize_chapter1_run(
    run_dir: str | Path,
    *,
    seeds: Iterable[int] = (0, 1),
    episodes_per_seed: int = 500,
) -> tuple[dict[str, Any], Path, Path]:
    decision = select_chapter1_checkpoint(run_dir)
    decision_path, config_path = write_finalizer_outputs(
        run_dir,
        decision,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
    )
    return decision, decision_path, config_path


def _read_candidates(
    run: Path,
    manifest: dict[str, Any],
    rewards: dict[int, float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    for directory in sorted((run / "validation").glob("update_*")):
        summary = _read_json(directory / "summary.json")
        binding = _read_json(directory / "checkpoint_binding.json")
        update = int(binding["update"])
        if update == 0:
            continue
        if update in seen:
            raise ValueError(f"duplicate validation update {update}")
        seen.add(update)
        if int(summary.get("invalid_count", 0)) or int(summary.get("nonfinite_count", 0)):
            raise ValueError(f"invalid/nonfinite validation at update {update}")
        if (
            summary.get("task_config_sha256") != manifest["task_config_sha256"]
            or binding.get("task_config_sha256") != manifest["task_config_sha256"]
            or binding.get("config_sha256") != manifest["config_sha256"]
        ):
            raise ValueError(f"validation provenance mismatch at update {update}")
        if summary.get("low_level_checkpoint_sha256") != manifest["low_level_checkpoint"]["sha256"]:
            raise ValueError(f"low-level checkpoint mismatch at update {update}")
        checkpoint = run / binding["checkpoint"]
        if _sha256_file(checkpoint) != binding["checkpoint_sha256"]:
            raise ValueError(f"checkpoint binding SHA256 mismatch at update {update}")
        if update not in rewards:
            raise ValueError(f"missing training reward at validation update {update}")
        success_rate = float(summary["success_rate"])
        detect_all_rate = float(summary["detect_all_rate"])
        if not all(math.isfinite(value) for value in (success_rate, detect_all_rate)):
            raise ValueError(f"non-finite validation metric at update {update}")
        candidates.append({
            "update": update,
            "checkpoint": binding["checkpoint"],
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "success_rate": success_rate,
            "detect_all_rate": detect_all_rate,
            "high_reward_mean": rewards[update],
        })
    return candidates


def _read_training_rewards(path: Path, target_update: int) -> dict[int, float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rewards = {int(row["update"]): float(row["high_reward_mean"]) for row in rows}
    if (
        len(rows) != target_update
        or sorted(rewards) != list(range(1, target_update + 1))
        or not all(math.isfinite(value) for value in rewards.values())
    ):
        raise ValueError("training CSV must contain exactly updates 1..target")
    return rewards


def _select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("no checkpoint candidates")
    return max(
        candidates,
        key=lambda item: (
            item["success_rate"],
            item["detect_all_rate"],
            item["high_reward_mean"],
            -item["update"],
        ),
    )


def _classify_health(health: dict[str, Any]) -> str:
    status = health.get("status")
    if status == "passed_for_final_evaluation":
        return "passed"
    if status == "completed_but_gate_failed":
        return "gate_failed"
    raise ValueError("terminal health decision is not finalizable")


def _read_terminal_health(path: Path, target_update: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or int(rows[-1].get("update", -1)) != target_update:
        raise ValueError("terminal health decision is missing or not at target update")
    _classify_health(rows[-1])
    return rows[-1]


def _validate_complete_summary(path: Path, target_update: int) -> None:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            values.setdefault(key, value)
    if values.get("status") != "complete" or int(values.get("final_update", -1)) != target_update:
        raise ValueError("run did not complete its configured target update")


def _validate_low_checkpoint(manifest: dict[str, Any]) -> None:
    low = manifest.get("low_level_checkpoint", {})
    path = Path(low.get("path", ""))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if _sha256_file(path) != low.get("sha256"):
        raise ValueError("low-level checkpoint SHA256 mismatch")


def _verify_derived_config(path: Path, decision: dict[str, Any]) -> None:
    resolved = resolve_chapter1_config(path)
    if resolved.task_config_sha256 != decision["inputs"]["task_config_sha256"]:
        raise ValueError("derived eval task hash differs from training run")
    low_sha = resolved.run["evaluation"]["low_level_checkpoint"]["sha256"]
    if low_sha != decision["inputs"]["low_level_checkpoint_sha256"]:
        raise ValueError("derived eval low-level checkpoint differs from training run")


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError as error:
        raise ValueError(f"finalizer artifact is outside project root: {path}") from error


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
