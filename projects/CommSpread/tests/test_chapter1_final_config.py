"""Regression tests for the frozen Chapter 1 task configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from comm_spread.chapter1_config import (
    Chapter1ConfigError,
    FINAL_TASK_VERSION,
    IMMEDIATE_OPEN_TASK_VERSION,
    chapter1_sar_kwargs,
    config_sha256,
    resolve_chapter1_config,
    save_resolved_config,
    validate_runtime_task_values,
    validate_task,
    verify_checkpoint_sha256,
)
from comm_spread.env_factory import make_sar_env
from comm_spread.low_level_policy import BenchMARLLowLevelPolicy


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "chapter1"



def _expect_config_error(function, message: str) -> None:
    try:
        function()
    except Chapter1ConfigError as error:
        assert message in str(error)
    else:
        raise AssertionError(f"expected Chapter1ConfigError containing {message!r}")

def _resolve(name: str):
    return resolve_chapter1_config(CONFIG_DIR / name)


def test_final_task_parses_and_explicit_core_fields_exist() -> None:
    resolved = _resolve("task_final_v1.yaml")
    task = resolved.task
    assert task["task_version"] == FINAL_TASK_VERSION
    assert task["scenario"] == {
        "name": "comm_spread_sar",
        "implementation": "comm_spread.sar_scenario.SarScenario",
        "num_agents": 3,
        "num_targets": 3,
        "horizon": 100,
    }
    assert task["physics"]["profile"] == "mpe_strict"
    assert task["physics"]["action_space"]["continuous"] is False
    assert task["perception"]["sensor_radius"] == 0.6
    assert task["task_semantics"]["retire_on_rescue"] is True
    assert task["hierarchy"]["replanning"]["event_triggered"] is True
    assert task["hierarchy"]["finder_first"]["enabled"] is True


def test_low_high_eval_resolve_identical_task_core() -> None:
    resolved = [
        _resolve("low_train_v1.yaml"),
        _resolve("high_train_v1.yaml"),
        _resolve("eval_v1.yaml"),
    ]
    assert {item.task["task_version"] for item in resolved} == {FINAL_TASK_VERSION}
    assert len({item.task_config_sha256 for item in resolved}) == 1
    assert resolved[0].task == resolved[1].task == resolved[2].task
    assert chapter1_sar_kwargs(resolved[0].task, mode="high") == chapter1_sar_kwargs(
        resolved[2].task, mode="high"
    )


def test_immediate_open_task_and_role_configs_are_consistent() -> None:
    task = _resolve("task_immediate_open_v1.yaml")
    assert task.task["task_version"] == IMMEDIATE_OPEN_TASK_VERSION
    assert task.task["hierarchy"]["finder_first"]["cascade_mode"] == "immediate_open"

    roles = [
        _resolve("eval_immediate_open_v1.yaml"),
        *[
            _resolve(f"high_train_immediate_open_seed{seed}_v1.yaml")
            for seed in range(5)
        ],
    ]
    assert {item.task_config_sha256 for item in roles} == {
        task.task_config_sha256
    }
    assert [item.run["training"]["seed"] for item in roles[1:]] == list(range(5))
    assert all(
        item.run["training"]["base_eval_config"]
        == "configs/chapter1/eval_immediate_open_v1.yaml"
        for item in roles[1:]
    )

    original = _resolve("task_final_v1.yaml")
    assert task.task_config_sha256 != original.task_config_sha256
    changed = deepcopy(original.task)
    changed["task_version"] = IMMEDIATE_OPEN_TASK_VERSION
    changed["hierarchy"]["finder_first"]["cascade_mode"] = "immediate_open"
    assert task.task == changed


def test_paired_finder_only_seed_configs_preserve_frozen_task() -> None:
    original = _resolve("task_final_v1.yaml")
    roles = [
        _resolve(f"high_train_finder_only_seed{seed}_v1.yaml")
        for seed in range(1, 5)
    ]
    assert {item.task_config_sha256 for item in roles} == {
        original.task_config_sha256
    }
    assert [item.run["training"]["seed"] for item in roles] == [1, 2, 3, 4]


def test_task_version_rejects_mismatched_cascade_semantics() -> None:
    task = deepcopy(_resolve("task_final_v1.yaml").task)
    task["hierarchy"]["finder_first"]["cascade_mode"] = "immediate_open"
    _expect_config_error(lambda: validate_task(task), "requires hierarchy.finder_first")


def test_mpe_strict_rejects_continuous_actions() -> None:
    task = deepcopy(_resolve("task_final_v1.yaml").task)
    task["physics"]["action_space"]["continuous"] = True
    _expect_config_error(lambda: validate_task(task), "continuous_actions=False")


def test_final_task_rejects_legacy_profile() -> None:
    task = deepcopy(_resolve("task_final_v1.yaml").task)
    task["physics"]["profile"] = "legacy"
    _expect_config_error(lambda: validate_task(task), "forbids legacy")


def test_canonical_task_hash_is_repeatable_and_sensitive_to_core_change() -> None:
    first = _resolve("task_final_v1.yaml")
    second = _resolve("task_final_v1.yaml")
    assert first.task_config_sha256 == second.task_config_sha256
    changed = deepcopy(first.task)
    changed["perception"]["sensor_radius"] = 0.59
    assert config_sha256(changed) != first.task_config_sha256


def test_volatile_run_fields_do_not_change_task_or_config_hash() -> None:
    base = _resolve("eval_v1.yaml")
    moved = resolve_chapter1_config(
        CONFIG_DIR / "eval_v1.yaml",
        cli_overrides={
            "run.evaluation.output_dir": "/tmp/different-output",
            "run.timestamp": "2099-01-01T00:00:00",
        },
    )
    assert moved.task_config_sha256 == base.task_config_sha256
    assert moved.config_sha256 == base.config_sha256


def test_eval_rejects_core_cli_override_and_runtime_mismatch() -> None:
    _expect_config_error(
        lambda: resolve_chapter1_config(
            CONFIG_DIR / "eval_v1.yaml",
            cli_overrides={"task.perception.sensor_radius": 0.5},
        ),
        "cannot override frozen core",
    )
    resolved = _resolve("eval_v1.yaml")
    _expect_config_error(
        lambda: validate_runtime_task_values(resolved, {"sensor_radius": 0.5}),
        "frozen task mismatch",
    )


def test_resolved_config_records_all_layers() -> None:
    with TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        resolved = resolve_chapter1_config(
            CONFIG_DIR / "high_train_v1.yaml",
            cli_overrides={"run.training.output_dir": str(tmp_path)},
        )
        destination = save_resolved_config(resolved, tmp_path)
        text = destination.read_text(encoding="utf-8")
    assert '"yaml"' in text
    assert '"program_defaults"' in text
    assert '"cli_overrides"' in text
    assert '"resolved"' in text
    assert '"task_config_sha256"' in text


def test_resolved_task_constructs_matching_runtime_environment() -> None:
    resolved = _resolve("task_final_v1.yaml")
    env = make_sar_env(
        num_envs=1,
        device="cpu",
        seed=42,
        **chapter1_sar_kwargs(resolved.task, mode="high"),
    )
    scenario = env.scenario
    assert scenario.physics_profile == "mpe_strict"
    assert env.continuous_actions is False
    assert scenario.n_agents == 3
    assert scenario.n_targets == 3
    assert scenario.max_steps == 100
    assert scenario.world._substeps == 1
    assert scenario.world._dt == 0.1
    assert scenario.world._drag == 0.25
    assert scenario.sensor_radius == 0.6
    assert scenario.rrt_top_k == 5
    assert scenario.rrt_max_iter == 40
    assert scenario.finder_cascade_mode == "finder_only"


def test_low_checkpoint_profile_and_sha_guards_fail_closed() -> None:
    policy = BenchMARLLowLevelPolicy.__new__(BenchMARLLowLevelPolicy)
    policy.task_variant = "sar_low_mpe_physics"
    policy.physics_profile = "mpe_strict"
    wrong_env = SimpleNamespace(
        scenario=SimpleNamespace(physics_profile="legacy")
    )
    try:
        policy(wrong_env)
    except ValueError as error:
        assert "expects physics_profile='mpe_strict'" in str(error)
    else:
        raise AssertionError("low-level profile guard accepted legacy runtime")

    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        checkpoint.write_bytes(b"not a real checkpoint")
        _expect_config_error(
            lambda: verify_checkpoint_sha256(checkpoint, "0" * 64),
            "checkpoint SHA256 mismatch",
        )




if __name__ == "__main__":
    test_final_task_parses_and_explicit_core_fields_exist()
    test_low_high_eval_resolve_identical_task_core()
    test_mpe_strict_rejects_continuous_actions()
    test_final_task_rejects_legacy_profile()
    test_canonical_task_hash_is_repeatable_and_sensitive_to_core_change()
    test_volatile_run_fields_do_not_change_task_or_config_hash()
    test_eval_rejects_core_cli_override_and_runtime_mismatch()
    test_resolved_config_records_all_layers()
    test_resolved_task_constructs_matching_runtime_environment()
    test_low_checkpoint_profile_and_sha_guards_fail_closed()
    print("chapter1 final config tests passed")
