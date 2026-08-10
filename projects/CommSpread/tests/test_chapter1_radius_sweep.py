"""Regression tests for the Chapter 1 sensor-radius sweep."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from comm_spread.chapter1_config import PROJECT_ROOT, resolve_chapter1_config
from scripts import run_chapter1_radius_sweep as sweep


def test_radius_configs_change_only_sensor_radius():
    base = resolve_chapter1_config(
        PROJECT_ROOT / "configs/chapter1/task_immediate_open_v1.yaml"
    )
    information = sweep.preflight_configs()
    assert list(information) == ["030", "040", "050"]

    for spec in sweep.RADIUS_SPECS:
        task = resolve_chapter1_config(spec.task_config)
        evaluation = resolve_chapter1_config(spec.eval_config)
        training = resolve_chapter1_config(spec.train_config)
        expected = deepcopy(base.task)
        expected["perception"]["sensor_radius"] = spec.value
        assert task.task == expected
        assert {
            task.task_config_sha256,
            evaluation.task_config_sha256,
            training.task_config_sha256,
        } == {information[spec.code]["task_config_sha256"]}
        protocol = evaluation.run["evaluation"]
        assert protocol["seeds"] == [0, 1]
        assert protocol["episodes_per_seed"] == 500
        assert protocol["total_episodes"] == 1000
        assert (
            training.run["training"]["base_eval_config"]
            == f"configs/chapter1/eval_immediate_open_r{spec.code}_v1.yaml"
        )


def test_command_plan_is_train_finalize_then_two_evaluations():
    spec = sweep.RADIUS_SPECS[0]
    run_dir = spec.train_output_root / "runs/test-run"
    plans = sweep.command_plan(
        spec, run_id="test-run", run_dir=run_dir, device="cuda:0"
    )
    assert [name for name, _ in plans] == [
        "train",
        "finalize",
        "eval_seed_0",
        "eval_seed_1",
    ]
    commands = dict(plans)
    assert "--run-id" in commands["train"]
    assert commands["train"][-1] == "test-run"
    assert commands["finalize"][-5:] == [
        "--seeds",
        "0",
        "1",
        "--episodes-per-seed",
        "500",
    ]
    for seed in (0, 1):
        command = commands[f"eval_seed_{seed}"]
        assert command[command.index("--seed") + 1] == str(seed)
        assert command[command.index("--sensor-radius") + 1] == "0.3"
        assert command[command.index("--num-envs") + 1] == "500"
        assert command[command.index("--device") + 1] == "cuda:0"
        assert "--csv-output" in command
        assert "--markdown" in command


def test_execute_stage_persists_success_and_failure():
    config = {
        spec.code: {
            "radius": spec.value,
            "task_config_sha256": f"task-{spec.code}",
            "train_config_sha256": f"train-{spec.code}",
            "target_update": 200,
            "horizon": 100,
        }
        for spec in sweep.RADIUS_SPECS
    }
    state = sweep.initial_state("test-batch", config)
    radius_state = state["radii"]["030"]

    with TemporaryDirectory() as directory:
        root = Path(directory)
        state_path = root / "state.json"
        log_path = root / "stage.log"
        calls = []

        def success_runner(command, log):
            calls.append((command, log))
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("ok\n", encoding="utf-8")
            return 0

        assert sweep.execute_stage(
            stage_name="train",
            command=["fake", "train"],
            log_path=log_path,
            radius_state=radius_state,
            state=state,
            state_path=state_path,
            runner=success_runner,
            validator=lambda: None,
        )
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        assert persisted["radii"]["030"]["stages"]["train"]["status"] == "succeeded"
        assert calls == [(["fake", "train"], log_path)]

        def failed_runner(command, log):
            return 9

        assert not sweep.execute_stage(
            stage_name="finalize",
            command=["fake", "finalize"],
            log_path=root / "failed.log",
            radius_state=radius_state,
            state=state,
            state_path=state_path,
            runner=failed_runner,
            validator=lambda: (_ for _ in ()).throw(
                AssertionError("validator must not run")
            ),
        )
        assert radius_state["stages"]["finalize"]["status"] == "failed"
        assert radius_state["stages"]["finalize"]["return_code"] == 9


def test_new_attempt_preserves_history_and_resets_stages():
    config = {
        spec.code: {
            "radius": spec.value,
            "task_config_sha256": f"task-{spec.code}",
            "train_config_sha256": f"train-{spec.code}",
            "target_update": 200,
            "horizon": 100,
        }
        for spec in sweep.RADIUS_SPECS
    }
    state = sweep.initial_state("test-batch", config)
    radius_state = state["radii"]["030"]
    radius_state["stages"]["train"]["status"] = "failed"
    first_id, _ = sweep.new_training_attempt(
        sweep.RADIUS_SPECS[0], radius_state, batch_id="test-batch"
    )
    radius_state["stages"]["train"]["status"] = "failed"
    second_id, _ = sweep.new_training_attempt(
        sweep.RADIUS_SPECS[0], radius_state, batch_id="test-batch"
    )
    assert first_id != second_id
    assert len(radius_state["attempts"]) == 2
    assert radius_state["stages"]["train"]["status"] == "pending"


def test_evaluation_output_validation_checks_fixed_metrics():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        metrics = {
            "metric_set": "chapter1_sar_fixed_v1",
            "evaluation": {"seed": 0, "episodes": 500, "horizon": 2},
            "metrics": {
                name: {} for name in sweep.EXPECTED_METRICS
            },
        }
        metrics["metrics"]["search_efficiency"] = {
            "timesteps": [0, 1, 2],
            "mean_curve": [0.0, 0.2, 0.4],
        }
        (root / "evaluation_metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        (root / "failure_modes.json").write_text("{}", encoding="utf-8")
        sweep.require_evaluation_outputs(root, seed=0, horizon=2)
        metrics["metrics"]["search_efficiency"]["mean_curve"] = [0.0]
        (root / "evaluation_metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        try:
            sweep.require_evaluation_outputs(root, seed=0, horizon=2)
        except ValueError as error:
            assert "curve length" in str(error)
        else:
            raise AssertionError("expected invalid search curve to fail")


def test_dry_plan_and_bark_messages_are_stable():
    information = {
        spec.code: {"task_config_sha256": f"sha-{spec.code}"}
        for spec in sweep.RADIUS_SPECS
    }
    output = io.StringIO()
    with redirect_stdout(output):
        assert sweep.run_dry_plan(information, "cuda:0") == 0
    text = output.getvalue()
    assert text.index("r030:") < text.index("r040:") < text.index("r050:")
    for message in (
        "0.3success",
        "0.3fail",
        "0.4success",
        "0.4fail",
        "0.5success",
        "0.5fail",
        "train_done",
    ):
        assert message in text

    captured = []
    original = sweep.subprocess.run
    try:
        sweep.subprocess.run = lambda command, cwd: (
            captured.append(command) or SimpleNamespace(returncode=0)
        )
        assert sweep.send_bark_notification(
            "0.3success", "https://api.day.app/example"
        )
    finally:
        sweep.subprocess.run = original
    command = captured[0]
    assert command[0] == "curl"
    assert "--retry" in command
    assert command[-1] == "https://api.day.app/example/0.3success"


def _mock_config_information():
    return {
        spec.code: {
            "radius": spec.value,
            "task_config_sha256": f"task-{spec.code}",
            "train_config_sha256": f"train-{spec.code}",
            "target_update": 200,
            "horizon": 100,
        }
        for spec in sweep.RADIUS_SPECS
    }


def _run_with_mocked_artifact_checks(args, runner, notifier):
    originals = (
        sweep.preflight_configs,
        sweep.require_complete_training,
        sweep.require_finalizer_outputs,
        sweep.require_evaluation_outputs,
    )
    try:
        sweep.preflight_configs = _mock_config_information
        sweep.require_complete_training = lambda *args, **kwargs: None
        sweep.require_finalizer_outputs = lambda *args, **kwargs: None
        sweep.require_evaluation_outputs = lambda *args, **kwargs: None
        return sweep.run_pipeline(args, runner=runner, notifier=notifier)
    finally:
        (
            sweep.preflight_configs,
            sweep.require_complete_training,
            sweep.require_finalizer_outputs,
            sweep.require_evaluation_outputs,
        ) = originals


def test_successful_pipeline_rerun_skips_every_stage_and_notification():
    with TemporaryDirectory() as directory:
        args = SimpleNamespace(
            batch_id="mock-success",
            state_root=Path(directory),
            device="cuda:0",
            bark_base_url="https://api.day.app/example",
            dry_run=False,
        )
        calls = []
        notifications = []

        def runner(command, log_path):
            calls.append((command, log_path))
            return 0

        def notifier(message):
            notifications.append(message)
            return True

        assert _run_with_mocked_artifact_checks(args, runner, notifier) == 0
        assert len(calls) == 12
        assert notifications == [
            "0.3success",
            "0.4success",
            "0.5success",
            "train_done",
        ]

        assert _run_with_mocked_artifact_checks(args, runner, notifier) == 0
        assert len(calls) == 12
        assert len(notifications) == 4


def test_failed_middle_training_continues_to_next_radius():
    with TemporaryDirectory() as directory:
        args = SimpleNamespace(
            batch_id="mock-middle-failure",
            state_root=Path(directory),
            device="cuda:0",
            bark_base_url="https://api.day.app/example",
            dry_run=False,
        )
        calls = []
        notifications = []

        def runner(command, log_path):
            calls.append((command, log_path))
            joined = " ".join(command)
            if "train_high_ppo_sar.py" in joined and "r040" in joined:
                return 7
            return 0

        def notifier(message):
            notifications.append(message)
            return True

        assert _run_with_mocked_artifact_checks(args, runner, notifier) == 1
        assert len(calls) == 9
        assert notifications == [
            "0.3success",
            "0.4fail",
            "0.5success",
            "train_done",
        ]
        state = json.loads(
            (Path(directory) / "mock-middle-failure/pipeline_state.json").read_text()
        )
        assert state["radii"]["040"]["stages"]["train"]["status"] == "failed"
        assert all(
            stage["status"] == "succeeded"
            for stage in state["radii"]["050"]["stages"].values()
        )


if __name__ == "__main__":
    test_radius_configs_change_only_sensor_radius()
    test_command_plan_is_train_finalize_then_two_evaluations()
    test_execute_stage_persists_success_and_failure()
    test_new_attempt_preserves_history_and_resets_stages()
    test_evaluation_output_validation_checks_fixed_metrics()
    test_dry_plan_and_bark_messages_are_stable()
    test_successful_pipeline_rerun_skips_every_stage_and_notification()
    test_failed_middle_training_continues_to_next_radius()
    print("chapter1 radius sweep tests passed")
