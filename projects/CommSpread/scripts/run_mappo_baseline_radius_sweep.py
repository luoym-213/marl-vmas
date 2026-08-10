#!/usr/bin/env python3
"""Run the end-to-end MAPPO train/validation/evaluation radius sweep."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Callable
from urllib.parse import quote

import torch

from comm_spread.chapter1_config import PROJECT_ROOT, sha256_file
from comm_spread.chapter1_eval_metrics import (
    METRIC_SET,
    build_fixed_evaluation_metrics,
    write_fixed_evaluation_metrics,
)
from comm_spread.mappo_baseline import (
    load_mappo_baseline_config,
    validate_checkpoint_metadata,
)


SCHEMA_VERSION = 1
DEFAULT_BATCH_ID = "mappo-radius-sweep-seed0-v1"
DEFAULT_BARK_BASE_URL = "https://api.day.app/KTR8KhtV86QxMtX8Wvbu5V"
VALIDATION_SEED = 2
VALIDATION_EPISODES = 100
FINAL_EVAL_SEEDS = (0, 1)
EPISODES_PER_SEED = 500
VALIDATION_UPDATES = tuple(range(100, 201, 10))
EXPECTED_METRICS = {
    "success_rate",
    "task_completion_time",
    "mean_rescue_response_time",
    "search_efficiency",
    "last_target_detection_time",
}


@dataclass(frozen=True)
class RadiusSpec:
    code: str
    label: str
    value: float

    config_filename: str | None = None
    output_family: str = "mappo"

    @property
    def config(self) -> Path:
        filename = self.config_filename or (
            f"baseline_mappo_r{self.code}_seed0_v1.yaml"
        )
        return PROJECT_ROOT / "configs/chapter1" / filename

    @property
    def train_output_root(self) -> Path:
        return (
            PROJECT_ROOT
            / "outputs/chapter1_baselines"
            / self.output_family
            / f"r{self.code}/seed0"
        )


RADIUS_SPECS = (
    RadiusSpec("030", "0.3", 0.3),
    RadiusSpec("040", "0.4", 0.4),
    RadiusSpec("050", "0.5", 0.5),
)
R100_SPEC = RadiusSpec(
    "100",
    "1.0",
    1.0,
    config_filename="baseline_mappo_reward_v2_r100_seed0_v1.yaml",
    output_family="mappo_reward_v2",
)
RADIUS_REGISTRY = {
    spec.code: spec for spec in (*RADIUS_SPECS, R100_SPEC)
}
DENSE_V2_RADIUS_SPECS = tuple(
    RadiusSpec(
        code,
        label,
        value,
        config_filename=(
            f"baseline_mappo_reward_v2_r{code}_seed0_v1.yaml"
        ),
        output_family="mappo_reward_v2",
    )
    for code, label, value in (
        ("030", "0.3", 0.3),
        ("040", "0.4", 0.4),
        ("050", "0.5", 0.5),
        ("100", "1.0", 1.0),
    )
)
ASSIGNMENT_PRIOR_RADIUS_SPECS = tuple(
    RadiusSpec(
        code,
        label,
        value,
        config_filename=(
            f"baseline_mappo_assignment_prior_r{code}_seed0_v1.yaml"
        ),
        output_family="mappo_assignment_prior",
    )
    for code, label, value in (
        ("030", "0.3", 0.3),
        ("040", "0.4", 0.4),
        ("050", "0.5", 0.5),
    )
)
EXPERIMENT_SETS = {
    "current": RADIUS_REGISTRY,
    "dense-v2": {
        spec.code: spec for spec in DENSE_V2_RADIUS_SPECS
    },
    "assignment-prior": {
        spec.code: spec for spec in ASSIGNMENT_PRIOR_RADIUS_SPECS
    },
}


def resolve_radius_specs(
    value: str | None,
    experiment_set: str = "current",
) -> tuple[RadiusSpec, ...]:
    if experiment_set not in EXPERIMENT_SETS:
        raise ValueError(f"unknown experiment set: {experiment_set}")
    registry = EXPERIMENT_SETS[experiment_set]
    if value is None:
        if experiment_set == "current":
            return RADIUS_SPECS
        return tuple(registry.values())
    codes = [item.strip() for item in value.split(",") if item.strip()]
    if not codes:
        raise ValueError("--radii must select at least one radius code")
    if len(codes) != len(set(codes)):
        raise ValueError("--radii contains duplicate radius codes")
    unknown = [code for code in codes if code not in registry]
    if unknown:
        raise ValueError(f"unknown radius code(s): {', '.join(unknown)}")
    return tuple(registry[code] for code in codes)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/chapter1_baselines/mappo/radius_sweep"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--radii",
        default=None,
        help="comma-separated radius codes; defaults to 030,040,050",
    )
    parser.add_argument(
        "--experiment-set",
        choices=tuple(EXPERIMENT_SETS),
        default="current",
        help="configuration family; current preserves historical behavior",
    )
    parser.add_argument(
        "--bark-base-url",
        default=os.environ.get("BARK_BASE_URL", DEFAULT_BARK_BASE_URL),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_short_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def validate_batch_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError(
            "batch-id may contain only letters, digits, dot, underscore, "
            "and hyphen"
        )
    return value


def preflight_configs(
    specs: tuple[RadiusSpec, ...] = RADIUS_SPECS,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    normalized_tasks: list[dict[str, Any]] = []
    for spec in specs:
        config = load_mappo_baseline_config(spec.config)
        task = config["_task"]
        training = config["training"]
        evaluation = config["evaluation"]
        radius = float(task["perception"]["sensor_radius"])
        if radius != spec.value:
            raise ValueError(
                f"r{spec.code} sensor radius is {radius}, expected {spec.value}"
            )
        if int(training["seed"]) != 0:
            raise ValueError(f"r{spec.code} training seed must be 0")
        if int(training["updates"]) != 200:
            raise ValueError(f"r{spec.code} training updates must be 200")
        if int(training["checkpoint_interval"]) != 10:
            raise ValueError(
                f"r{spec.code} checkpoint interval must be 10"
            )
        if int(evaluation["episodes"]) != EPISODES_PER_SEED:
            raise ValueError(
                f"r{spec.code} evaluation episodes must be "
                f"{EPISODES_PER_SEED}"
            )
        if int(evaluation["batch_size"]) != EPISODES_PER_SEED:
            raise ValueError(
                f"r{spec.code} paired evaluation batch size mismatch"
            )

        normalized = deepcopy(task)
        normalized["perception"]["sensor_radius"] = "<radius>"
        normalized_tasks.append(normalized)
        result[spec.code] = {
            "radius": spec.value,
            "task_version": task["task_version"],
            "task_config_sha256": config["_task_sha256"],
            "baseline_config_sha256": config["_config_sha256"],
            "target_update": int(training["updates"]),
            "checkpoint_interval": int(training["checkpoint_interval"]),
            "horizon": int(task["scenario"]["horizon"]),
            "physics_dt": float(task["physics"]["dt"]),
        }
    if any(item != normalized_tasks[0] for item in normalized_tasks[1:]):
        raise ValueError(
            "MAPPO radius task configs differ outside sensor_radius"
        )
    return result


def stage_names() -> list[str]:
    return (
        ["train"]
        + [f"validate_{update:06d}" for update in VALIDATION_UPDATES]
        + ["select", "eval_seed_0", "eval_seed_1", "pool"]
    )


def initial_state(
    batch_id: str,
    config_info: dict[str, dict[str, Any]],
    specs: tuple[RadiusSpec, ...] = RADIUS_SPECS,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "batch_id": batch_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "git_head": git_short_head(),
        "overall_status": "pending",
        "overall_notified_status": None,
        "radii": {
            spec.code: {
                "radius": spec.value,
                "config": config_info[spec.code],
                "attempts": [],
                "current_run_id": None,
                "current_run_dir": None,
                "stages": {
                    name: {"status": "pending", "attempts": 0}
                    for name in stage_names()
                },
                "last_notified_outcome": None,
            }
            for spec in specs
        },
    }


def load_state(
    path: Path,
    batch_id: str,
    config_info: dict[str, dict[str, Any]],
    specs: tuple[RadiusSpec, ...] = RADIUS_SPECS,
) -> dict[str, Any]:
    if not path.exists():
        return initial_state(batch_id, config_info, specs)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported MAPPO radius-sweep state schema")
    if state.get("batch_id") != batch_id:
        raise ValueError("pipeline state batch-id mismatch")
    expected_stages = set(stage_names())
    for spec in specs:
        saved_radius = state["radii"][spec.code]
        if set(saved_radius["stages"]) != expected_stages:
            raise ValueError("pipeline state stage schema mismatch")
        saved = saved_radius["config"]
        current = config_info[spec.code]
        if (
            saved["task_config_sha256"] != current["task_config_sha256"]
            or saved["baseline_config_sha256"]
            != current["baseline_config_sha256"]
        ):
            raise ValueError(
                f"r{spec.code} config changed since batch creation"
            )
    return state


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_write_json(path, state)


def write_summary(path: Path, state: dict[str, Any]) -> None:
    summary = {
        "schema_version": state["schema_version"],
        "batch_id": state["batch_id"],
        "git_head": state["git_head"],
        "overall_status": state["overall_status"],
        "updated_at": state["updated_at"],
        "radii": {
            code: {
                "radius": item["radius"],
                "current_run_id": item["current_run_id"],
                "current_run_dir": item["current_run_dir"],
                "stages": {
                    name: stage["status"]
                    for name, stage in item["stages"].items()
                },
                "last_notified_outcome": item[
                    "last_notified_outcome"
                ],
            }
            for code, item in state["radii"].items()
        },
    }
    atomic_write_json(path, summary)


def pooled_policy_diagnostics(run_dir: Path) -> dict[str, Any]:
    documents = [
        json.loads(
            (
                final_eval_dir(run_dir, seed)
                / "policy_diagnostics.json"
            ).read_text(encoding="utf-8")
        )
        for seed in FINAL_EVAL_SEEDS
    ]
    action_counts = [
        sum(int(item["action_counts"][index]) for item in documents)
        for index in range(5)
    ]
    active_count = sum(action_counts)
    episodes = sum(int(item["episodes"]) for item in documents)

    def action_weighted(key: str) -> float:
        numerator = sum(
            float(item[key]) * sum(item["action_counts"])
            for item in documents
        )
        return numerator / max(active_count, 1)

    def episode_weighted(key: str) -> float:
        numerator = sum(
            float(item[key]) * int(item["episodes"])
            for item in documents
        )
        return numerator / max(episodes, 1)

    return {
        "episodes": episodes,
        "action_counts": action_counts,
        "action_fractions": [
            count / max(active_count, 1) for count in action_counts
        ],
        "noop_fraction": action_counts[0] / max(active_count, 1),
        "mean_policy_entropy": action_weighted("mean_policy_entropy"),
        "mean_top2_logit_margin": action_weighted(
            "mean_top2_logit_margin"
        ),
        "mean_detected_target_count": episode_weighted(
            "mean_detected_target_count"
        ),
        "mean_rescued_target_count": episode_weighted(
            "mean_rescued_target_count"
        ),
        "all_targets_detected_rate": episode_weighted(
            "all_targets_detected_rate"
        ),
        "all_targets_rescued_rate": episode_weighted(
            "all_targets_rescued_rate"
        ),
    }


def write_radius_comparison(
    path: Path,
    state: dict[str, Any],
    specs: tuple[RadiusSpec, ...],
    experiment_set: str,
) -> Path:
    radii = {}
    for spec in specs:
        radius_state = state["radii"][spec.code]
        run_dir = Path(radius_state["current_run_dir"])
        selection = json.loads(
            (run_dir / "checkpoint_selection.json").read_text(
                encoding="utf-8"
            )
        )
        pooled = json.loads(
            (
                run_dir
                / "evaluation_final_1000"
                / "evaluation_metrics_pooled.json"
            ).read_text(encoding="utf-8")
        )
        radii[spec.code] = {
            "radius": spec.value,
            "run_dir": str(run_dir),
            "task_config_sha256": radius_state["config"][
                "task_config_sha256"
            ],
            "baseline_config_sha256": radius_state["config"][
                "baseline_config_sha256"
            ],
            "selected_checkpoint": selection["selected"],
            "metrics": pooled["metrics"],
            "policy_diagnostics": pooled_policy_diagnostics(run_dir),
        }
    document = {
        "schema_version": 1,
        "batch_id": state["batch_id"],
        "git_head": state["git_head"],
        "experiment_set": experiment_set,
        "protocol": {
            "training_seed": 0,
            "validation_seed": VALIDATION_SEED,
            "validation_episodes": VALIDATION_EPISODES,
            "final_evaluation_seeds": list(FINAL_EVAL_SEEDS),
            "episodes_per_final_seed": EPISODES_PER_SEED,
            "test_seeds_used_for_selection": False,
        },
        "radii": radii,
    }
    atomic_write_json(path, document)
    return path


def stream_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {shlex.join(command)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] $ {shlex.join(command)}\n")
        log.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            message = f"failed to launch command: {error}\n"
            print(message, end="", flush=True)
            log.write(message)
            return 127
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


CommandRunner = Callable[[list[str], Path], int]
Validator = Callable[[], None]
Notifier = Callable[[str], bool]


def execute_stage(
    *,
    stage_name: str,
    command: list[str],
    log_path: Path,
    radius_state: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    runner: CommandRunner,
    validator: Validator,
) -> bool:
    record = radius_state["stages"][stage_name]
    record["status"] = "running"
    record["attempts"] = int(record.get("attempts", 0)) + 1
    record["started_at"] = utc_now()
    record["command"] = command
    record["log_path"] = str(log_path)
    save_state(state_path, state)

    return_code = runner(command, log_path)
    validation_error = None
    if return_code == 0:
        try:
            validator()
        except Exception as error:
            validation_error = str(error)
            return_code = 1
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"\nvalidation failed: {validation_error}\n"
                )
            print(
                f"validation failed: {validation_error}", flush=True
            )

    record["return_code"] = return_code
    record["finished_at"] = utc_now()
    record["status"] = "succeeded" if return_code == 0 else "failed"
    if validation_error is None:
        record.pop("validation_error", None)
    else:
        record["validation_error"] = validation_error
    save_state(state_path, state)
    return return_code == 0


def execute_internal_stage(
    *,
    stage_name: str,
    operation: Callable[[], None],
    radius_state: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
) -> bool:
    record = radius_state["stages"][stage_name]
    record["status"] = "running"
    record["attempts"] = int(record.get("attempts", 0)) + 1
    record["started_at"] = utc_now()
    save_state(state_path, state)
    try:
        operation()
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
        record["finished_at"] = utc_now()
        save_state(state_path, state)
        print(f"{stage_name} failed: {error}", flush=True)
        return False
    record["status"] = "succeeded"
    record.pop("error", None)
    record["finished_at"] = utc_now()
    save_state(state_path, state)
    return True


def mark_stale_success(
    radius_state: dict[str, Any],
    stage_name: str,
    validator: Validator,
) -> bool:
    record = radius_state["stages"][stage_name]
    if record["status"] != "succeeded":
        return False
    try:
        validator()
        return True
    except Exception as error:
        record["status"] = "failed"
        record["validation_error"] = f"saved success is stale: {error}"
        return False


def new_training_attempt(
    spec: RadiusSpec,
    radius_state: dict[str, Any],
    *,
    batch_id: str,
) -> tuple[str, Path]:
    attempt = len(radius_state["attempts"]) + 1
    run_id = (
        f"chapter1-mappo-r{spec.code}-seed0-{batch_id}-"
        f"attempt{attempt}-{utc_stamp()}-{git_short_head()}"
    )
    run_dir = spec.train_output_root / "runs" / run_id
    radius_state["attempts"].append(
        {
            "attempt": attempt,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "created_at": utc_now(),
        }
    )
    radius_state["current_run_id"] = run_id
    radius_state["current_run_dir"] = str(run_dir)
    for stage in radius_state["stages"].values():
        stage["status"] = "pending"
        stage.pop("validation_error", None)
        stage.pop("error", None)
    return run_id, run_dir


def checkpoint_path(run_dir: Path, update: int) -> Path:
    return run_dir / f"checkpoint_{update:06d}.pt"


def validation_dir(run_dir: Path, update: int) -> Path:
    return (
        run_dir
        / "validation"
        / f"seed_{VALIDATION_SEED}"
        / f"checkpoint_{update:06d}"
    )


def final_eval_dir(run_dir: Path, seed: int) -> Path:
    return run_dir / "evaluation_final_1000" / f"seed_{seed}"


def train_command(
    spec: RadiusSpec, run_dir: Path, device: str
) -> list[str]:
    return [
        sys.executable,
        "scripts/train_mappo_baseline_sar.py",
        "--config",
        str(spec.config.relative_to(PROJECT_ROOT)),
        "--device",
        device,
        "--output-dir",
        str(run_dir),
    ]


def evaluation_command(
    spec: RadiusSpec,
    *,
    checkpoint: Path,
    seed: int,
    episodes: int,
    output_dir: Path,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluate_mappo_baseline_sar.py",
        "--config",
        str(spec.config.relative_to(PROJECT_ROOT)),
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
        "--episodes",
        str(episodes),
        "--batch-size",
        str(episodes),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]


def command_plan(
    spec: RadiusSpec, *, run_dir: Path, device: str
) -> list[tuple[str, list[str]]]:
    commands = [("train", train_command(spec, run_dir, device))]
    for update in VALIDATION_UPDATES:
        commands.append(
            (
                f"validate_{update:06d}",
                evaluation_command(
                    spec,
                    checkpoint=checkpoint_path(run_dir, update),
                    seed=VALIDATION_SEED,
                    episodes=VALIDATION_EPISODES,
                    output_dir=validation_dir(run_dir, update),
                    device=device,
                ),
            )
        )
    selected = run_dir / "<selected-checkpoint-from-selection.json>"
    for seed in FINAL_EVAL_SEEDS:
        commands.append(
            (
                f"eval_seed_{seed}",
                evaluation_command(
                    spec,
                    checkpoint=selected,
                    seed=seed,
                    episodes=EPISODES_PER_SEED,
                    output_dir=final_eval_dir(run_dir, seed),
                    device=device,
                ),
            )
        )
    return commands


def require_complete_training(
    run_dir: Path,
    config: dict[str, Any],
    target_update: int,
) -> None:
    history = json.loads(
        (run_dir / "training_history.json").read_text(encoding="utf-8")
    )
    if not isinstance(history, list) or len(history) != target_update:
        raise ValueError("training history did not reach target update")
    if int(history[-1]["update"]) != target_update:
        raise ValueError("training final update mismatch")
    latest = torch.load(
        run_dir / "checkpoint_latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    validate_checkpoint_metadata(latest, config)
    if int(latest["update"]) != target_update:
        raise ValueError("latest checkpoint update mismatch")
    for update in VALIDATION_UPDATES:
        if not checkpoint_path(run_dir, update).is_file():
            raise ValueError(f"missing checkpoint at update {update}")


def require_evaluation_outputs(
    output_dir: Path,
    *,
    seed: int,
    episodes: int,
    horizon: int,
    expected_update: int,
) -> dict[str, Any]:
    metrics = json.loads(
        (output_dir / "evaluation_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    rows = json.loads(
        (output_dir / "evaluation_episodes.json").read_text(
            encoding="utf-8"
        )
    )
    if metrics.get("metric_set") != METRIC_SET:
        raise ValueError("unexpected fixed metric set")
    evaluation = metrics["evaluation"]
    if (
        int(evaluation["seed"]) != seed
        or int(evaluation["episodes"]) != episodes
        or int(evaluation["horizon"]) != horizon
    ):
        raise ValueError("evaluation metadata mismatch")
    if set(metrics["metrics"]) != EXPECTED_METRICS:
        raise ValueError("fixed metric keys mismatch")
    efficiency = metrics["metrics"]["search_efficiency"]
    if (
        len(efficiency["timesteps"]) != horizon + 1
        or len(efficiency["mean_curve"]) != horizon + 1
    ):
        raise ValueError("search efficiency curve length mismatch")
    if not isinstance(rows, list) or len(rows) != episodes:
        raise ValueError("evaluation episode row count mismatch")
    baseline = metrics.get("baseline", {})
    if int(baseline.get("checkpoint_update", -1)) != expected_update:
        raise ValueError("evaluation checkpoint update mismatch")
    return metrics


def nullable_min(value: Any) -> float:
    if value is None:
        return math.inf
    result = float(value)
    return result if math.isfinite(result) else math.inf


def validation_rank(metrics: dict[str, Any], update: int) -> tuple[float, ...]:
    values = metrics["metrics"]
    success = float(values["success_rate"]["mean"])
    completion = nullable_min(values["task_completion_time"]["mean"])
    response = nullable_min(
        values["mean_rescue_response_time"]["mean"]
    )
    return (-success, completion, response, -float(update))


def candidate_rank(candidate: dict[str, Any]) -> tuple[float, ...]:
    metrics = candidate["validation_metrics"]
    return (
        -float(metrics["success_rate"]),
        nullable_min(metrics["task_completion_time"]),
        nullable_min(metrics["mean_rescue_response_time"]),
        -float(candidate["update"]),
    )


def select_best_checkpoint(
    run_dir: Path,
    *,
    horizon: int,
    task_hash: str,
) -> dict[str, Any]:
    candidates = []
    for update in VALIDATION_UPDATES:
        output_dir = validation_dir(run_dir, update)
        metrics = require_evaluation_outputs(
            output_dir,
            seed=VALIDATION_SEED,
            episodes=VALIDATION_EPISODES,
            horizon=horizon,
            expected_update=update,
        )
        checkpoint = checkpoint_path(run_dir, update)
        values = metrics["metrics"]
        candidates.append(
            {
                "update": update,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "validation_metrics": {
                    "success_rate": values["success_rate"]["mean"],
                    "task_completion_time": values[
                        "task_completion_time"
                    ]["mean"],
                    "mean_rescue_response_time": values[
                        "mean_rescue_response_time"
                    ]["mean"],
                },
            }
        )
    selected = min(candidates, key=candidate_rank)
    decision = {
        "schema_version": 1,
        "selection_protocol": {
            "candidate_updates": list(VALIDATION_UPDATES),
            "validation_seed": VALIDATION_SEED,
            "episodes": VALIDATION_EPISODES,
            "ordering": [
                "success_rate_desc",
                "task_completion_time_asc_null_inf",
                "mean_rescue_response_time_asc_null_inf",
                "update_desc",
            ],
            "test_seeds_used": False,
        },
        "task_config_sha256": task_hash,
        "candidates": candidates,
        "selected": selected,
    }
    atomic_write_json(run_dir / "checkpoint_selection.json", decision)
    return decision


def require_selection(
    run_dir: Path,
    *,
    horizon: int,
    task_hash: str,
) -> dict[str, Any]:
    path = run_dir / "checkpoint_selection.json"
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision["task_config_sha256"] != task_hash:
        raise ValueError("selection task hash mismatch")
    if decision["selection_protocol"]["candidate_updates"] != list(
        VALIDATION_UPDATES
    ):
        raise ValueError("selection candidate protocol mismatch")
    selected = decision["selected"]
    update = int(selected["update"])
    if update not in VALIDATION_UPDATES:
        raise ValueError("selected update is outside candidate set")
    selected_path = Path(selected["checkpoint"])
    if not selected_path.is_file():
        raise ValueError("selected checkpoint is missing")
    if sha256_file(selected_path) != selected["checkpoint_sha256"]:
        raise ValueError("selected checkpoint hash mismatch")
    for candidate in decision["candidates"]:
        require_evaluation_outputs(
            validation_dir(run_dir, int(candidate["update"])),
            seed=VALIDATION_SEED,
            episodes=VALIDATION_EPISODES,
            horizon=horizon,
            expected_update=int(candidate["update"]),
        )
    expected = min(
        decision["candidates"],
        key=candidate_rank,
    )
    if int(expected["update"]) != update:
        raise ValueError("saved checkpoint selection is not optimal")
    return decision


def build_pooled_metrics(
    rows_by_seed: dict[int, list[dict[str, Any]]],
    *,
    horizon: int,
    physics_dt: float,
    checkpoint: Path,
    checkpoint_update: int,
    task_hash: str,
) -> dict[str, Any]:
    rows = [
        row
        for seed in FINAL_EVAL_SEEDS
        for row in rows_by_seed[seed]
    ]
    document = build_fixed_evaluation_metrics(
        rows,
        seed=-1,
        horizon=horizon,
        physics_dt=physics_dt,
    )
    document["evaluation"].pop("seed")
    document["evaluation"].update(
        {
            "seeds": list(FINAL_EVAL_SEEDS),
            "episodes_per_seed": EPISODES_PER_SEED,
            "aggregation": "pooled_episode_rows",
        }
    )
    document["baseline"] = {
        "name": "end_to_end_mappo",
        "checkpoint": str(checkpoint),
        "checkpoint_update": int(checkpoint_update),
        "task_config_sha256": task_hash,
        "hidden_target_truth": False,
        "hierarchy_used": False,
    }
    return document


def write_pooled_outputs(
    run_dir: Path,
    *,
    horizon: int,
    physics_dt: float,
    checkpoint: Path,
    checkpoint_update: int,
    task_hash: str,
) -> Path:
    rows_by_seed = {}
    for seed in FINAL_EVAL_SEEDS:
        rows = json.loads(
            (
                final_eval_dir(run_dir, seed)
                / "evaluation_episodes.json"
            ).read_text(encoding="utf-8")
        )
        if not isinstance(rows, list) or len(rows) != EPISODES_PER_SEED:
            raise ValueError(f"seed {seed} episode rows are incomplete")
        rows_by_seed[seed] = rows
    document = build_pooled_metrics(
        rows_by_seed,
        horizon=horizon,
        physics_dt=physics_dt,
        checkpoint=checkpoint,
        checkpoint_update=checkpoint_update,
        task_hash=task_hash,
    )
    destination = (
        run_dir
        / "evaluation_final_1000"
        / "evaluation_metrics_pooled.json"
    )
    write_fixed_evaluation_metrics(document, destination)
    return destination


def require_pooled_output(
    run_dir: Path,
    *,
    horizon: int,
    checkpoint_update: int,
    task_hash: str,
) -> None:
    path = (
        run_dir
        / "evaluation_final_1000"
        / "evaluation_metrics_pooled.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("metric_set") != METRIC_SET:
        raise ValueError("pooled metric set mismatch")
    evaluation = document["evaluation"]
    if (
        evaluation.get("seeds") != list(FINAL_EVAL_SEEDS)
        or int(evaluation["episodes"]) != 2 * EPISODES_PER_SEED
        or int(evaluation["episodes_per_seed"]) != EPISODES_PER_SEED
        or evaluation.get("aggregation") != "pooled_episode_rows"
    ):
        raise ValueError("pooled evaluation metadata mismatch")
    efficiency = document["metrics"]["search_efficiency"]
    if len(efficiency["mean_curve"]) != horizon + 1:
        raise ValueError("pooled search-efficiency curve mismatch")
    baseline = document["baseline"]
    if (
        int(baseline["checkpoint_update"]) != checkpoint_update
        or baseline["task_config_sha256"] != task_hash
    ):
        raise ValueError("pooled checkpoint metadata mismatch")


def send_bark_notification(message: str, base_url: str) -> bool:
    url = f"{base_url.rstrip('/')}/{quote(message, safe='')}"
    command = [
        "curl",
        "-fsS",
        "--retry",
        "3",
        "--retry-delay",
        "5",
        "--max-time",
        "30",
        url,
    ]
    print(f"$ {shlex.join(command)}", flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(
            f"warning: Bark notification failed for {message!r} "
            f"with exit code {result.returncode}",
            file=sys.stderr,
            flush=True,
        )
        return False
    return True


def run_dry_plan(
    config_info: dict[str, dict[str, Any]],
    device: str,
    specs: tuple[RadiusSpec, ...] = RADIUS_SPECS,
) -> int:
    print("Configuration preflight passed.")
    for spec in specs:
        run_dir = spec.train_output_root / "runs/<persisted-run-id>"
        print(
            f"\nr{spec.code}: radius={spec.value}, "
            f"task_sha={config_info[spec.code]['task_config_sha256']}"
        )
        for stage, command in command_plan(
            spec, run_dir=run_dir, device=device
        ):
            print(f"[{stage}] {shlex.join(command)}")
        print("[select] validation success -> T_all -> response -> update")
        print("[pool] seed 0/1 raw rows -> 1000-episode fixed metrics")
        print(f"[notify] {spec.label}success/fail")
    print("\n[notify overall] train_done")
    return 0


def run_pipeline(
    args: argparse.Namespace,
    *,
    runner: CommandRunner = stream_command,
    notifier: Notifier | None = None,
) -> int:
    batch_id = validate_batch_id(args.batch_id)
    specs = resolve_radius_specs(
        getattr(args, "radii", None),
        getattr(args, "experiment_set", "current"),
    )
    config_info = preflight_configs(specs)
    if args.dry_run:
        return run_dry_plan(config_info, args.device, specs)

    state_dir = args.state_root.resolve() / batch_id
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "pipeline.lock"
    state_path = state_dir / "pipeline_state.json"
    summary_path = state_dir / "pipeline_summary.json"
    notify = notifier or (
        lambda message: send_bark_notification(
            message, args.bark_base_url
        )
    )

    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"batch {batch_id!r} is already running"
            ) from error

        state = load_state(state_path, batch_id, config_info, specs)
        save_state(state_path, state)
        for spec in specs:
            info = config_info[spec.code]
            config = load_mappo_baseline_config(spec.config)
            radius_state = state["radii"][spec.code]
            run_dir_value = radius_state.get("current_run_dir")
            run_dir = Path(run_dir_value) if run_dir_value else None

            train_valid = False
            if run_dir is not None:
                train_valid = mark_stale_success(
                    radius_state,
                    "train",
                    lambda run_dir=run_dir, config=config, info=info: (
                        require_complete_training(
                            run_dir,
                            config,
                            int(info["target_update"]),
                        )
                    ),
                )
            if not train_valid:
                if run_dir is None or run_dir.exists():
                    _, run_dir = new_training_attempt(
                        spec, radius_state, batch_id=batch_id
                    )
                    save_state(state_path, state)
                train_valid = execute_stage(
                    stage_name="train",
                    command=train_command(spec, run_dir, args.device),
                    log_path=(
                        state_dir
                        / "logs"
                        / f"r{spec.code}"
                        / f"train-attempt{len(radius_state['attempts'])}.log"
                    ),
                    radius_state=radius_state,
                    state=state,
                    state_path=state_path,
                    runner=runner,
                    validator=(
                        lambda run_dir=run_dir, config=config, info=info: (
                            require_complete_training(
                                run_dir,
                                config,
                                int(info["target_update"]),
                            )
                        )
                    ),
                )

            validation_results: dict[int, bool] = {}
            if train_valid:
                for update in VALIDATION_UPDATES:
                    stage_name = f"validate_{update:06d}"
                    validator = (
                        lambda update=update, run_dir=run_dir, info=info: (
                            require_evaluation_outputs(
                                validation_dir(run_dir, update),
                                seed=VALIDATION_SEED,
                                episodes=VALIDATION_EPISODES,
                                horizon=int(info["horizon"]),
                                expected_update=update,
                            )
                        )
                    )
                    valid = mark_stale_success(
                        radius_state, stage_name, validator
                    )
                    if not valid:
                        valid = execute_stage(
                            stage_name=stage_name,
                            command=evaluation_command(
                                spec,
                                checkpoint=checkpoint_path(
                                    run_dir, update
                                ),
                                seed=VALIDATION_SEED,
                                episodes=VALIDATION_EPISODES,
                                output_dir=validation_dir(
                                    run_dir, update
                                ),
                                device=args.device,
                            ),
                            log_path=(
                                state_dir
                                / "logs"
                                / f"r{spec.code}"
                                / f"validate-{update:06d}.log"
                            ),
                            radius_state=radius_state,
                            state=state,
                            state_path=state_path,
                            runner=runner,
                            validator=validator,
                        )
                    validation_results[update] = valid
            else:
                validation_results = {
                    update: False for update in VALIDATION_UPDATES
                }

            selection_valid = False
            if train_valid and all(validation_results.values()):
                selection_validator = (
                    lambda run_dir=run_dir, info=info: require_selection(
                        run_dir,
                        horizon=int(info["horizon"]),
                        task_hash=str(info["task_config_sha256"]),
                    )
                )
                selection_valid = mark_stale_success(
                    radius_state, "select", selection_validator
                )
                if not selection_valid:
                    selection_valid = execute_internal_stage(
                        stage_name="select",
                        operation=(
                            lambda run_dir=run_dir, info=info: (
                                select_best_checkpoint(
                                    run_dir,
                                    horizon=int(info["horizon"]),
                                    task_hash=str(
                                        info["task_config_sha256"]
                                    ),
                                )
                            )
                        ),
                        radius_state=radius_state,
                        state=state,
                        state_path=state_path,
                    )
                    if selection_valid:
                        try:
                            selection_validator()
                        except Exception as error:
                            radius_state["stages"]["select"][
                                "status"
                            ] = "failed"
                            radius_state["stages"]["select"][
                                "validation_error"
                            ] = str(error)
                            selection_valid = False
                            save_state(state_path, state)

            eval_results: dict[int, bool] = {}
            selected_checkpoint = None
            selected_update = None
            if selection_valid:
                decision = require_selection(
                    run_dir,
                    horizon=int(info["horizon"]),
                    task_hash=str(info["task_config_sha256"]),
                )
                selected_checkpoint = Path(
                    decision["selected"]["checkpoint"]
                )
                selected_update = int(decision["selected"]["update"])
                for seed in FINAL_EVAL_SEEDS:
                    stage_name = f"eval_seed_{seed}"
                    validator = (
                        lambda seed=seed,
                        run_dir=run_dir,
                        info=info,
                        selected_update=selected_update: (
                            require_evaluation_outputs(
                                final_eval_dir(run_dir, seed),
                                seed=seed,
                                episodes=EPISODES_PER_SEED,
                                horizon=int(info["horizon"]),
                                expected_update=selected_update,
                            )
                        )
                    )
                    valid = mark_stale_success(
                        radius_state, stage_name, validator
                    )
                    if not valid:
                        valid = execute_stage(
                            stage_name=stage_name,
                            command=evaluation_command(
                                spec,
                                checkpoint=selected_checkpoint,
                                seed=seed,
                                episodes=EPISODES_PER_SEED,
                                output_dir=final_eval_dir(
                                    run_dir, seed
                                ),
                                device=args.device,
                            ),
                            log_path=(
                                state_dir
                                / "logs"
                                / f"r{spec.code}"
                                / f"eval-seed-{seed}.log"
                            ),
                            radius_state=radius_state,
                            state=state,
                            state_path=state_path,
                            runner=runner,
                            validator=validator,
                        )
                    eval_results[seed] = valid
            else:
                eval_results = {
                    seed: False for seed in FINAL_EVAL_SEEDS
                }

            pool_valid = False
            if (
                selected_checkpoint is not None
                and selected_update is not None
                and all(eval_results.values())
            ):
                pool_validator = (
                    lambda run_dir=run_dir,
                    info=info,
                    selected_update=selected_update: (
                        require_pooled_output(
                            run_dir,
                            horizon=int(info["horizon"]),
                            checkpoint_update=selected_update,
                            task_hash=str(info["task_config_sha256"]),
                        )
                    )
                )
                pool_valid = mark_stale_success(
                    radius_state, "pool", pool_validator
                )
                if not pool_valid:
                    pool_valid = execute_internal_stage(
                        stage_name="pool",
                        operation=(
                            lambda run_dir=run_dir,
                            info=info,
                            selected_checkpoint=selected_checkpoint,
                            selected_update=selected_update: (
                                write_pooled_outputs(
                                    run_dir,
                                    horizon=int(info["horizon"]),
                                    physics_dt=float(info["physics_dt"]),
                                    checkpoint=selected_checkpoint,
                                    checkpoint_update=selected_update,
                                    task_hash=str(
                                        info["task_config_sha256"]
                                    ),
                                )
                            )
                        ),
                        radius_state=radius_state,
                        state=state,
                        state_path=state_path,
                    )
                    if pool_valid:
                        try:
                            pool_validator()
                        except Exception as error:
                            radius_state["stages"]["pool"][
                                "status"
                            ] = "failed"
                            radius_state["stages"]["pool"][
                                "validation_error"
                            ] = str(error)
                            pool_valid = False
                            save_state(state_path, state)

            radius_success = (
                train_valid
                and all(validation_results.values())
                and selection_valid
                and all(eval_results.values())
                and pool_valid
            )
            outcome = "success" if radius_success else "fail"
            if radius_state.get("last_notified_outcome") != outcome:
                if notify(f"{spec.label}{outcome}"):
                    radius_state["last_notified_outcome"] = outcome
                    save_state(state_path, state)
            write_summary(summary_path, state)

        all_success = all(
            all(
                stage["status"] == "succeeded"
                for stage in item["stages"].values()
            )
            for item in state["radii"].values()
        )
        overall_status = "success" if all_success else "partial_failure"
        state["overall_status"] = overall_status
        if state.get("overall_notified_status") != overall_status:
            if notify("train_done"):
                state["overall_notified_status"] = overall_status
        save_state(state_path, state)
        write_summary(summary_path, state)
        comparison_path = None
        if all_success:
            comparison_path = write_radius_comparison(
                state_dir / "radius_comparison.json",
                state,
                specs,
                getattr(args, "experiment_set", "current"),
            )
        print(f"pipeline summary: {summary_path}", flush=True)
        if comparison_path is not None:
            print(f"radius comparison: {comparison_path}", flush=True)
        print(f"overall_status: {overall_status}", flush=True)
        return 0 if all_success else 1


def main() -> int:
    args = parse_args()
    try:
        return run_pipeline(args)
    except Exception as error:
        print(
            f"MAPPO radius sweep failed before completion: {error}",
            file=sys.stderr,
        )
        if not args.dry_run:
            send_bark_notification("train_done", args.bark_base_url)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
