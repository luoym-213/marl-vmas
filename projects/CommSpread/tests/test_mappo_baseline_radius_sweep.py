"""Tests for the end-to-end MAPPO sensor-radius sweep."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import torch

from comm_spread.mappo_baseline import (
    checkpoint_metadata,
    load_mappo_baseline_config,
)
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import run_mappo_baseline_radius_sweep as sweep


def metric_document(
    *,
    success: float,
    completion: float | None,
    response: float | None,
    update: int,
    seed: int = sweep.VALIDATION_SEED,
    episodes: int = sweep.VALIDATION_EPISODES,
    horizon: int = 2,
) -> dict:
    return {
        "schema_version": 1,
        "metric_set": sweep.METRIC_SET,
        "evaluation": {
            "seed": seed,
            "episodes": episodes,
            "horizon": horizon,
        },
        "metrics": {
            "success_rate": {
                "mean": success,
                "std": 0.0,
                "sample_count": episodes,
            },
            "task_completion_time": {
                "mean": completion,
                "std": 0.0 if completion is not None else None,
                "sample_count": episodes if completion is not None else 0,
            },
            "mean_rescue_response_time": {
                "mean": response,
                "std": 0.0 if response is not None else None,
                "sample_count": episodes if response is not None else 0,
            },
            "search_efficiency": {
                "timesteps": list(range(horizon + 1)),
                "mean_curve": [0.0] * (horizon + 1),
                "sample_count": episodes,
            },
            "last_target_detection_time": {
                "mean": 5.0,
                "std": 0.0,
                "sample_count": episodes,
            },
        },
        "baseline": {"checkpoint_update": update},
    }


def episode_row(
    *,
    success: int,
    horizon: int,
    entropy_end: float,
) -> dict:
    trace = [
        10.0 - (10.0 - entropy_end) * step / horizon
        for step in range(horizon + 1)
    ]
    return {
        "success": success,
        "success_step": 10 if success else -1,
        "rescue_response_time": 2.0 if success else 0.0,
        "last_target_detection_time": 5,
        "global_entropy_trace": trace,
    }


def write_evaluation(
    output_dir: Path,
    *,
    update: int,
    success: float,
    completion: float | None,
    response: float | None,
    seed: int = sweep.VALIDATION_SEED,
    episodes: int = sweep.VALIDATION_EPISODES,
    horizon: int = 2,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = metric_document(
        success=success,
        completion=completion,
        response=response,
        update=update,
        seed=seed,
        episodes=episodes,
        horizon=horizon,
    )
    rows = [
        episode_row(success=0, horizon=horizon, entropy_end=5.0)
        for _ in range(episodes)
    ]
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (output_dir / "evaluation_episodes.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    active_actions = episodes * horizon * 3
    diagnostics = {
        "episodes": episodes,
        "action_counts": [active_actions, 0, 0, 0, 0],
        "action_fractions": [1.0, 0.0, 0.0, 0.0, 0.0],
        "noop_fraction": 1.0,
        "mean_policy_entropy": 1.0,
        "mean_top2_logit_margin": 0.5,
        "mean_detected_target_count": 2.0,
        "mean_rescued_target_count": 1.5,
        "all_targets_detected_rate": 0.5,
        "all_targets_rescued_rate": success,
    }
    (output_dir / "policy_diagnostics.json").write_text(
        json.dumps(diagnostics), encoding="utf-8"
    )


def test_preflight_and_stage_schema():
    information = sweep.preflight_configs()
    assert list(information) == ["030", "040", "050"]
    assert sweep.VALIDATION_UPDATES == tuple(range(100, 201, 10))
    names = sweep.stage_names()
    assert names[0] == "train"
    assert names[1] == "validate_000100"
    assert names[11] == "validate_000200"
    assert names[-4:] == [
        "select",
        "eval_seed_0",
        "eval_seed_1",
        "pool",
    ]
    for spec in sweep.RADIUS_SPECS:
        assert information[spec.code]["radius"] == spec.value
        assert information[spec.code]["target_update"] == 200


def test_command_plan_has_validation_and_paired_final_evaluations():
    spec = sweep.RADIUS_SPECS[0]
    run_dir = Path("/tmp/mappo-radius-test")
    plan = dict(
        sweep.command_plan(spec, run_dir=run_dir, device="cuda:7")
    )
    assert len(plan) == 14
    assert "--output-dir" in plan["train"]
    for update in sweep.VALIDATION_UPDATES:
        command = plan[f"validate_{update:06d}"]
        assert command[command.index("--seed") + 1] == "2"
        assert command[command.index("--episodes") + 1] == "100"
        assert command[command.index("--batch-size") + 1] == "100"
        assert command[command.index("--device") + 1] == "cuda:7"
    for seed in sweep.FINAL_EVAL_SEEDS:
        command = plan[f"eval_seed_{seed}"]
        assert command[command.index("--seed") + 1] == str(seed)
        assert command[command.index("--episodes") + 1] == "500"
        assert command[command.index("--batch-size") + 1] == "500"


def test_validation_ranking_handles_nulls_and_deterministic_ties():
    null_metrics = metric_document(
        success=0.0,
        completion=None,
        response=None,
        update=100,
    )
    successful = metric_document(
        success=0.2,
        completion=50.0,
        response=8.0,
        update=110,
    )
    assert sweep.validation_rank(successful, 110) < sweep.validation_rank(
        null_metrics, 100
    )

    earlier = {
        "update": 150,
        "validation_metrics": {
            "success_rate": 0.8,
            "task_completion_time": 40.0,
            "mean_rescue_response_time": 6.0,
        },
    }
    later = {**earlier, "update": 170}
    assert sweep.candidate_rank(later) < sweep.candidate_rank(earlier)


def test_selection_uses_all_candidates_and_writes_standard_json():
    with TemporaryDirectory() as directory:
        run_dir = Path(directory)
        for update in sweep.VALIDATION_UPDATES:
            sweep.checkpoint_path(run_dir, update).write_bytes(
                f"checkpoint-{update}".encode()
            )
            success = 0.4
            completion = 60.0
            response = 9.0
            if update in (160, 170):
                success = 0.9
                completion = 40.0
                response = 6.0
            write_evaluation(
                sweep.validation_dir(run_dir, update),
                update=update,
                success=success,
                completion=completion,
                response=response,
            )

        decision = sweep.select_best_checkpoint(
            run_dir, horizon=2, task_hash="task-hash"
        )
        assert decision["selected"]["update"] == 170
        assert len(decision["candidates"]) == 11
        text = (run_dir / "checkpoint_selection.json").read_text(
            encoding="utf-8"
        )
        assert "Infinity" not in text
        saved = sweep.require_selection(
            run_dir, horizon=2, task_hash="task-hash"
        )
        assert saved["selected"]["update"] == 170


def test_pooled_metrics_recompute_from_1000_episode_rows():
    horizon = 2
    rows_by_seed = {
        0: [
            episode_row(success=1, horizon=horizon, entropy_end=5.0)
            for _ in range(500)
        ],
        1: [
            episode_row(success=0, horizon=horizon, entropy_end=0.0)
            for _ in range(500)
        ],
    }
    document = sweep.build_pooled_metrics(
        rows_by_seed,
        horizon=horizon,
        physics_dt=0.1,
        checkpoint=Path("/tmp/checkpoint.pt"),
        checkpoint_update=170,
        task_hash="task-hash",
    )
    assert document["evaluation"]["seeds"] == [0, 1]
    assert document["evaluation"]["episodes"] == 1000
    assert document["evaluation"]["episodes_per_seed"] == 500
    assert document["evaluation"]["aggregation"] == "pooled_episode_rows"
    assert document["metrics"]["success_rate"]["mean"] == 0.5
    assert (
        document["metrics"]["task_completion_time"]["sample_count"]
        == 500
    )
    curve = document["metrics"]["search_efficiency"]["mean_curve"]
    assert curve == [0.0, 0.375, 0.75]


def test_execute_stage_persists_failure_without_running_validator():
    information = {
        spec.code: {
            "radius": spec.value,
            "task_config_sha256": f"task-{spec.code}",
            "baseline_config_sha256": f"base-{spec.code}",
            "target_update": 200,
            "checkpoint_interval": 10,
            "horizon": 100,
            "physics_dt": 0.1,
        }
        for spec in sweep.RADIUS_SPECS
    }
    state = sweep.initial_state("test-batch", information)
    radius_state = state["radii"]["030"]
    validator_called = False

    with TemporaryDirectory() as directory:
        root = Path(directory)

        def runner(command, log_path):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("failed\n", encoding="utf-8")
            return 9

        def validator():
            nonlocal validator_called
            validator_called = True

        result = sweep.execute_stage(
            stage_name="train",
            command=["fake", "train"],
            log_path=root / "train.log",
            radius_state=radius_state,
            state=state,
            state_path=root / "state.json",
            runner=runner,
            validator=validator,
        )
        assert not result
        assert not validator_called
        assert radius_state["stages"]["train"]["status"] == "failed"
        assert radius_state["stages"]["train"]["return_code"] == 9


def test_state_rejects_changed_config_and_new_attempt_resets_stages():
    information = sweep.preflight_configs()
    with TemporaryDirectory() as directory:
        state_path = Path(directory) / "state.json"
        state = sweep.initial_state("batch", information)
        sweep.save_state(state_path, state)
        changed = json.loads(json.dumps(information))
        changed["030"]["baseline_config_sha256"] = "changed"
        try:
            sweep.load_state(state_path, "batch", changed)
        except ValueError as error:
            assert "config changed" in str(error)
        else:
            raise AssertionError("expected changed config rejection")

        radius_state = state["radii"]["030"]
        radius_state["stages"]["train"]["status"] = "failed"
        _, run_dir = sweep.new_training_attempt(
            sweep.RADIUS_SPECS[0],
            radius_state,
            batch_id="batch",
        )
        assert "attempt1" in run_dir.name
        assert all(
            stage["status"] == "pending"
            for stage in radius_state["stages"].values()
        )


def test_pipeline_continues_to_later_radii_after_training_failures():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        proxies = tuple(
            SimpleNamespace(
                code=spec.code,
                label=spec.label,
                value=spec.value,
                config=spec.config,
                train_output_root=root / f"runs-r{spec.code}",
            )
            for spec in sweep.RADIUS_SPECS
        )
        commands = []
        notifications = []

        def failed_runner(command, log_path):
            commands.append(command)
            return 7

        def notifier(message):
            notifications.append(message)
            return True

        args = SimpleNamespace(
            batch_id="failure-continuation",
            state_root=root / "state",
            device="cpu",
            bark_base_url="https://example.invalid",
            dry_run=False,
        )
        with mock.patch.object(sweep, "RADIUS_SPECS", proxies):
            return_code = sweep.run_pipeline(
                args, runner=failed_runner, notifier=notifier
            )

        assert return_code == 1
        assert len(commands) == 3
        assert all(
            command[1] == "scripts/train_mappo_baseline_sar.py"
            for command in commands
        )
        assert notifications == [
            "0.3fail",
            "0.4fail",
            "0.5fail",
            "train_done",
        ]
        summary = json.loads(
            (
                root
                / "state"
                / "failure-continuation"
                / "pipeline_summary.json"
            ).read_text(encoding="utf-8")
        )
        assert summary["overall_status"] == "partial_failure"


def test_mocked_success_pipeline_selects_and_pools_all_radii():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        proxies = tuple(
            SimpleNamespace(
                code=spec.code,
                label=spec.label,
                value=spec.value,
                config=spec.config,
                train_output_root=root / f"runs-r{spec.code}",
            )
            for spec in sweep.RADIUS_SPECS
        )
        commands = []
        notifications = []

        def argument(command, flag):
            return command[command.index(flag) + 1]

        def successful_runner(command, log_path):
            commands.append(command)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("ok\n", encoding="utf-8")
            if command[1] == "scripts/train_mappo_baseline_sar.py":
                output_dir = Path(argument(command, "--output-dir"))
                output_dir.mkdir(parents=True, exist_ok=True)
                config = load_mappo_baseline_config(
                    Path(argument(command, "--config"))
                )
                history = [
                    {"update": update} for update in range(1, 201)
                ]
                (output_dir / "training_history.json").write_text(
                    json.dumps(history), encoding="utf-8"
                )
                for update in sweep.VALIDATION_UPDATES:
                    torch.save(
                        {
                            "metadata": checkpoint_metadata(config),
                            "update": update,
                        },
                        sweep.checkpoint_path(output_dir, update),
                    )
                torch.save(
                    {
                        "metadata": checkpoint_metadata(config),
                        "update": 200,
                    },
                    output_dir / "checkpoint_latest.pt",
                )
                return 0

            output_dir = Path(argument(command, "--output-dir"))
            checkpoint = Path(argument(command, "--checkpoint"))
            update = int(checkpoint.stem.rsplit("_", 1)[-1])
            seed = int(argument(command, "--seed"))
            episodes = int(argument(command, "--episodes"))
            horizon = 100
            success = update / 1000.0 if seed == 2 else 0.5
            write_evaluation(
                output_dir,
                update=update,
                success=success,
                completion=50.0,
                response=8.0,
                seed=seed,
                episodes=episodes,
                horizon=horizon,
            )
            return 0

        def notifier(message):
            notifications.append(message)
            return True

        args = SimpleNamespace(
            batch_id="success",
            state_root=root / "state",
            device="cpu",
            bark_base_url="https://example.invalid",
            dry_run=False,
        )
        with mock.patch.object(sweep, "RADIUS_SPECS", proxies):
            return_code = sweep.run_pipeline(
                args, runner=successful_runner, notifier=notifier
            )

        assert return_code == 0
        assert len(commands) == 42
        assert notifications == [
            "0.3success",
            "0.4success",
            "0.5success",
            "train_done",
        ]
        for spec in proxies:
            run_root = spec.train_output_root / "runs"
            run_dir = next(run_root.iterdir())
            decision = json.loads(
                (run_dir / "checkpoint_selection.json").read_text(
                    encoding="utf-8"
                )
            )
            assert decision["selected"]["update"] == 200
            pooled = json.loads(
                (
                    run_dir
                    / "evaluation_final_1000"
                    / "evaluation_metrics_pooled.json"
                ).read_text(encoding="utf-8")
            )
            assert pooled["evaluation"]["episodes"] == 1000
            assert pooled["evaluation"]["seeds"] == [0, 1]
        comparison = json.loads(
            (
                root / "state" / "success" / "radius_comparison.json"
            ).read_text(encoding="utf-8")
        )
        assert list(comparison["radii"]) == ["030", "040", "050"]
        assert comparison["protocol"]["test_seeds_used_for_selection"] is False
        assert comparison["radii"]["030"]["metrics"]["success_rate"]["mean"] == 0.0




def test_r100_dense_reward_sweep_is_explicit_and_isolated():
    specs = sweep.resolve_radius_specs("100")
    assert specs == (sweep.R100_SPEC,)
    assert sweep.resolve_radius_specs(None) == sweep.RADIUS_SPECS
    assert specs[0].output_family == "mappo_reward_v2"
    information = sweep.preflight_configs(specs)
    assert list(information) == ["100"]
    plan = dict(
        sweep.command_plan(specs[0], run_dir=Path("/tmp/r100"), device="cpu")
    )
    assert "baseline_mappo_reward_v2_r100_seed0_v1.yaml" in " ".join(plan["train"])


def test_dense_v2_experiment_set_uses_controlled_radius_configs():
    specs = sweep.resolve_radius_specs("030,040,050", "dense-v2")
    assert [spec.code for spec in specs] == ["030", "040", "050"]
    assert all(spec.output_family == "mappo_reward_v2" for spec in specs)
    information = sweep.preflight_configs(specs)
    assert list(information) == ["030", "040", "050"]
    for spec in specs:
        config = load_mappo_baseline_config(spec.config)
        assert config["_reward"]["profile"] == "observable_dense_v2"
        assert config["_reward"]["detected_target_progress_scale"] == 5.0
        assert config["training"]["entropy_coefficient"] == 0.001
