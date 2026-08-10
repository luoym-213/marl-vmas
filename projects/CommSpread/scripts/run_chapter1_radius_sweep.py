#!/usr/bin/env python3
"""Run the Chapter 1 immediate-open sensor-radius training/evaluation sweep."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Callable
from urllib.parse import quote

from comm_spread.chapter1_config import (
    PROJECT_ROOT,
    resolve_chapter1_config,
    verify_checkpoint_sha256,
)


SCHEMA_VERSION = 1
DEFAULT_BATCH_ID = "immediate-open-radius-sweep-seed0-v1"
DEFAULT_BARK_BASE_URL = "https://api.day.app/KTR8KhtV86QxMtX8Wvbu5V"
FINAL_EVAL_SEEDS = (0, 1)
EPISODES_PER_SEED = 500
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

    @property
    def task_config(self) -> Path:
        return PROJECT_ROOT / "configs/chapter1" / (
            f"task_immediate_open_r{self.code}_v1.yaml"
        )

    @property
    def eval_config(self) -> Path:
        return PROJECT_ROOT / "configs/chapter1" / (
            f"eval_immediate_open_r{self.code}_v1.yaml"
        )

    @property
    def train_config(self) -> Path:
        return PROJECT_ROOT / "configs/chapter1" / (
            f"high_train_immediate_open_r{self.code}_seed0_v1.yaml"
        )

    @property
    def train_output_root(self) -> Path:
        return PROJECT_ROOT / "outputs/chapter1" / (
            f"high_train_immediate_open_r{self.code}_v1/seed_0"
        )


RADIUS_SPECS = (
    RadiusSpec("030", "0.3", 0.3),
    RadiusSpec("040", "0.4", 0.4),
    RadiusSpec("050", "0.5", 0.5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/chapter1/radius_sweep",
    )
    parser.add_argument("--device", default="cuda:0")
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
            "batch-id may contain only letters, digits, dot, underscore, and hyphen"
        )
    return value


def preflight_configs() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    normalized_tasks: list[dict[str, Any]] = []
    for spec in RADIUS_SPECS:
        task = resolve_chapter1_config(spec.task_config)
        evaluation = resolve_chapter1_config(spec.eval_config)
        training = resolve_chapter1_config(spec.train_config)
        if task.run_type != "task":
            raise ValueError(f"{spec.task_config} is not a task config")
        if evaluation.run_type != "eval":
            raise ValueError(f"{spec.eval_config} is not an eval config")
        if training.run_type != "high_train":
            raise ValueError(f"{spec.train_config} is not a high_train config")
        hashes = {
            task.task_config_sha256,
            evaluation.task_config_sha256,
            training.task_config_sha256,
        }
        if len(hashes) != 1:
            raise ValueError(f"r{spec.code} task hashes differ across role configs")
        radius = float(task.task["perception"]["sensor_radius"])
        if radius != spec.value:
            raise ValueError(
                f"r{spec.code} sensor radius is {radius}, expected {spec.value}"
            )
        train_protocol = training.run["training"]
        eval_protocol = evaluation.run["evaluation"]
        expected_eval = f"configs/chapter1/eval_immediate_open_r{spec.code}_v1.yaml"
        if train_protocol["base_eval_config"] != expected_eval:
            raise ValueError(f"r{spec.code} base_eval_config mismatch")
        expected_output = (
            f"outputs/chapter1/high_train_immediate_open_r{spec.code}_v1/seed_0"
        )
        if train_protocol["output_dir"] != expected_output:
            raise ValueError(f"r{spec.code} training output_dir mismatch")
        if train_protocol["seed"] != 0:
            raise ValueError(f"r{spec.code} training seed must be 0")
        if eval_protocol["seeds"] != [0, 1]:
            raise ValueError(f"r{spec.code} final eval seeds must be [0, 1]")
        if (
            eval_protocol["episodes_per_seed"] != EPISODES_PER_SEED
            or eval_protocol["total_episodes"] != 2 * EPISODES_PER_SEED
        ):
            raise ValueError(f"r{spec.code} eval episode protocol mismatch")
        low = train_protocol["low_level_checkpoint"]
        verify_checkpoint_sha256(PROJECT_ROOT / low["path"], low["sha256"])

        normalized = deepcopy(task.task)
        normalized["perception"]["sensor_radius"] = "<radius>"
        normalized_tasks.append(normalized)
        result[spec.code] = {
            "radius": spec.value,
            "task_config_sha256": task.task_config_sha256,
            "train_config_sha256": training.config_sha256,
            "target_update": int(train_protocol["updates"]),
            "horizon": int(task.task["scenario"]["horizon"]),
        }
    if any(item != normalized_tasks[0] for item in normalized_tasks[1:]):
        raise ValueError("radius task configs differ in fields other than sensor_radius")
    return result


def initial_state(batch_id: str, config_info: dict[str, dict[str, Any]]) -> dict[str, Any]:
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
                    "train": {"status": "pending", "attempts": 0},
                    "finalize": {"status": "pending", "attempts": 0},
                    "eval_seed_0": {"status": "pending", "attempts": 0},
                    "eval_seed_1": {"status": "pending", "attempts": 0},
                },
                "last_notified_outcome": None,
            }
            for spec in RADIUS_SPECS
        },
    }


def load_state(
    path: Path,
    batch_id: str,
    config_info: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not path.exists():
        return initial_state(batch_id, config_info)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported radius-sweep state schema")
    if state.get("batch_id") != batch_id:
        raise ValueError("pipeline state batch-id mismatch")
    for spec in RADIUS_SPECS:
        saved = state["radii"][spec.code]["config"]
        current = config_info[spec.code]
        if (
            saved["task_config_sha256"] != current["task_config_sha256"]
            or saved["train_config_sha256"] != current["train_config_sha256"]
        ):
            raise ValueError(
                f"r{spec.code} config changed since this batch was created"
            )
    return state


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
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
                    name: record["status"]
                    for name, record in item["stages"].items()
                },
                "last_notified_outcome": item["last_notified_outcome"],
            }
            for code, item in state["radii"].items()
        },
    }
    atomic_write_json(path, summary)


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
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\nvalidation failed: {validation_error}\n")
            print(f"validation failed: {validation_error}", flush=True)

    record["return_code"] = return_code
    record["finished_at"] = utc_now()
    record["status"] = "succeeded" if return_code == 0 else "failed"
    if validation_error is not None:
        record["validation_error"] = validation_error
    else:
        record.pop("validation_error", None)
    save_state(state_path, state)
    return return_code == 0


def require_complete_training(run_dir: Path, target_update: int) -> None:
    summary = (run_dir / "texts/run_summary.txt").read_text(encoding="utf-8")
    if "status: complete" not in summary:
        raise ValueError("training run is not complete")
    if f"final_update: {target_update}" not in summary:
        raise ValueError("training run did not reach configured target update")


def require_finalizer_outputs(run_dir: Path, task_hash: str) -> None:
    decision_path = run_dir / "checkpoint_selection.json"
    config_path = run_dir / "derived_eval_final.yaml"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision["inputs"]["task_config_sha256"] != task_hash:
        raise ValueError("finalizer task hash mismatch")
    resolved = resolve_chapter1_config(config_path)
    if resolved.task_config_sha256 != task_hash:
        raise ValueError("derived eval task hash mismatch")


def require_evaluation_outputs(
    eval_dir: Path,
    *,
    seed: int,
    horizon: int,
) -> None:
    metrics_path = eval_dir / "evaluation_metrics.json"
    failure_path = eval_dir / "failure_modes.json"
    document = json.loads(metrics_path.read_text(encoding="utf-8"))
    if document.get("metric_set") != "chapter1_sar_fixed_v1":
        raise ValueError("unexpected fixed metric set")
    evaluation = document["evaluation"]
    if (
        evaluation["seed"] != seed
        or evaluation["episodes"] != EPISODES_PER_SEED
        or evaluation["horizon"] != horizon
    ):
        raise ValueError("evaluation metadata mismatch")
    metrics = document["metrics"]
    if set(metrics) != EXPECTED_METRICS:
        raise ValueError("fixed metric keys mismatch")
    efficiency = metrics["search_efficiency"]
    if (
        len(efficiency["timesteps"]) != horizon + 1
        or len(efficiency["mean_curve"]) != horizon + 1
    ):
        raise ValueError("search efficiency curve length mismatch")
    json.loads(failure_path.read_text(encoding="utf-8"))


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
    attempt_number = len(radius_state["attempts"]) + 1
    run_id = (
        f"chapter1-immediate-open-r{spec.code}-seed0-"
        f"{batch_id}-attempt{attempt_number}-{utc_stamp()}-{git_short_head()}"
    )
    run_dir = spec.train_output_root / "runs" / run_id
    radius_state["attempts"].append(
        {
            "attempt": attempt_number,
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
    return run_id, run_dir


def command_plan(
    spec: RadiusSpec,
    *,
    run_id: str,
    run_dir: Path,
    device: str,
) -> list[tuple[str, list[str]]]:
    python = sys.executable
    commands: list[tuple[str, list[str]]] = [
        (
            "train",
            [
                python,
                "scripts/train_high_ppo_sar.py",
                "--chapter1-config",
                str(spec.train_config.relative_to(PROJECT_ROOT)),
                "--run-id",
                run_id,
            ],
        ),
        (
            "finalize",
            [
                python,
                "scripts/finalize_chapter1_run.py",
                "--run-dir",
                str(run_dir.relative_to(PROJECT_ROOT)),
                "--seeds",
                "0",
                "1",
                "--episodes-per-seed",
                str(EPISODES_PER_SEED),
            ],
        ),
    ]
    for seed in FINAL_EVAL_SEEDS:
        eval_dir = run_dir / "evaluation_final_1000" / f"seed_{seed}"
        commands.append(
            (
                f"eval_seed_{seed}",
                [
                    python,
                    "scripts/analyze_sar_failure_modes_mpe_physics.py",
                    "--chapter1-config",
                    str(run_dir / "derived_eval_final.yaml"),
                    "--policy",
                    "hgsar",
                    "--seed",
                    str(seed),
                    "--num-envs",
                    str(EPISODES_PER_SEED),
                    "--sensor-radius",
                    str(spec.value),
                    "--device",
                    device,
                    "--output",
                    str(eval_dir / "failure_modes.json"),
                    "--csv-output",
                    str(eval_dir / "failure_modes.csv"),
                    "--markdown",
                    str(eval_dir / "failure_modes.md"),
                ],
            )
        )
    return commands


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


def run_dry_plan(config_info: dict[str, dict[str, Any]], device: str) -> int:
    print("Configuration preflight passed.")
    for spec in RADIUS_SPECS:
        run_id = f"<persisted-run-id-r{spec.code}>"
        run_dir = spec.train_output_root / "runs" / run_id
        print(
            f"\nr{spec.code}: radius={spec.value}, "
            f"task_sha={config_info[spec.code]['task_config_sha256']}"
        )
        for stage, command in command_plan(
            spec, run_id=run_id, run_dir=run_dir, device=device
        ):
            print(f"[{stage}] {shlex.join(command)}")
        print(f"[notify success] {spec.label}success")
        print(f"[notify failure] {spec.label}fail")
    print("\n[notify overall] train_done")
    return 0


def run_pipeline(
    args: argparse.Namespace,
    *,
    runner: CommandRunner = stream_command,
    notifier: Notifier | None = None,
) -> int:
    batch_id = validate_batch_id(args.batch_id)
    config_info = preflight_configs()
    if args.dry_run:
        return run_dry_plan(config_info, args.device)

    state_dir = args.state_root.resolve() / batch_id
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "pipeline.lock"
    state_path = state_dir / "pipeline_state.json"
    summary_path = state_dir / "pipeline_summary.json"
    notify = notifier or (
        lambda message: send_bark_notification(message, args.bark_base_url)
    )

    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"batch {batch_id!r} is already running") from error

        state = load_state(state_path, batch_id, config_info)
        save_state(state_path, state)
        for spec in RADIUS_SPECS:
            radius_state = state["radii"][spec.code]
            target_update = config_info[spec.code]["target_update"]
            task_hash = config_info[spec.code]["task_config_sha256"]
            horizon = config_info[spec.code]["horizon"]

            run_dir_value = radius_state.get("current_run_dir")
            run_dir = Path(run_dir_value) if run_dir_value else None
            train_valid = False
            if run_dir is not None:
                train_valid = mark_stale_success(
                    radius_state,
                    "train",
                    lambda run_dir=run_dir: require_complete_training(
                        run_dir, target_update
                    ),
                )
            if not train_valid:
                if run_dir is None or run_dir.exists():
                    run_id, run_dir = new_training_attempt(
                        spec, radius_state, batch_id=batch_id
                    )
                    save_state(state_path, state)
                else:
                    run_id = str(radius_state["current_run_id"])
                train_command = command_plan(
                    spec, run_id=run_id, run_dir=run_dir, device=args.device
                )[0][1]
                train_log = (
                    state_dir
                    / "logs"
                    / f"r{spec.code}"
                    / f"train-attempt{len(radius_state['attempts'])}.log"
                )
                train_valid = execute_stage(
                    stage_name="train",
                    command=train_command,
                    log_path=train_log,
                    radius_state=radius_state,
                    state=state,
                    state_path=state_path,
                    runner=runner,
                    validator=lambda run_dir=run_dir: require_complete_training(
                        run_dir, target_update
                    ),
                )

            final_valid = False
            if train_valid:
                final_valid = mark_stale_success(
                    radius_state,
                    "finalize",
                    lambda run_dir=run_dir: require_finalizer_outputs(
                        run_dir, task_hash
                    ),
                )
                if not final_valid:
                    final_command = command_plan(
                        spec,
                        run_id=str(radius_state["current_run_id"]),
                        run_dir=run_dir,
                        device=args.device,
                    )[1][1]
                    final_valid = execute_stage(
                        stage_name="finalize",
                        command=final_command,
                        log_path=state_dir / "logs" / f"r{spec.code}" / "finalize.log",
                        radius_state=radius_state,
                        state=state,
                        state_path=state_path,
                        runner=runner,
                        validator=lambda run_dir=run_dir: require_finalizer_outputs(
                            run_dir, task_hash
                        ),
                    )

            eval_results: dict[int, bool] = {}
            if train_valid and final_valid:
                plans = dict(
                    command_plan(
                        spec,
                        run_id=str(radius_state["current_run_id"]),
                        run_dir=run_dir,
                        device=args.device,
                    )
                )
                for seed in FINAL_EVAL_SEEDS:
                    stage_name = f"eval_seed_{seed}"
                    eval_dir = run_dir / "evaluation_final_1000" / f"seed_{seed}"
                    validator = (
                        lambda eval_dir=eval_dir, seed=seed: require_evaluation_outputs(
                            eval_dir, seed=seed, horizon=horizon
                        )
                    )
                    valid = mark_stale_success(
                        radius_state, stage_name, validator
                    )
                    if not valid:
                        valid = execute_stage(
                            stage_name=stage_name,
                            command=plans[stage_name],
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
                eval_results = {seed: False for seed in FINAL_EVAL_SEEDS}

            radius_success = (
                train_valid
                and final_valid
                and all(eval_results.values())
            )
            outcome = "success" if radius_success else "fail"
            if radius_state.get("last_notified_outcome") != outcome:
                if notify(f"{spec.label}{outcome}"):
                    radius_state["last_notified_outcome"] = outcome
                    save_state(state_path, state)
            write_summary(summary_path, state)

        all_success = all(
            item["stages"]["train"]["status"] == "succeeded"
            and item["stages"]["finalize"]["status"] == "succeeded"
            and item["stages"]["eval_seed_0"]["status"] == "succeeded"
            and item["stages"]["eval_seed_1"]["status"] == "succeeded"
            for item in state["radii"].values()
        )
        overall_status = "success" if all_success else "partial_failure"
        state["overall_status"] = overall_status
        if state.get("overall_notified_status") != overall_status:
            if notify("train_done"):
                state["overall_notified_status"] = overall_status
        save_state(state_path, state)
        write_summary(summary_path, state)
        print(f"pipeline summary: {summary_path}", flush=True)
        print(f"overall_status: {overall_status}", flush=True)
        return 0 if all_success else 1


def main() -> int:
    args = parse_args()
    try:
        return run_pipeline(args)
    except Exception as error:
        print(f"radius sweep failed before completion: {error}", file=sys.stderr)
        if not args.dry_run:
            send_bark_notification("train_done", args.bark_base_url)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
