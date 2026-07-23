"""Unit tests for the standalone Chapter 1 validation artifact contract."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from comm_spread.chapter1_config import resolve_chapter1_config
from comm_spread.chapter1_validation import (
    run_chapter1_validation,
    validation_summary,
    write_validation_outputs,
)


def sample_rows():
    return [
        {
            "env_id": 0,
            "success": True,
            "success_step": 80,
            "all_detected_step": 40,
            "failure_reason": "success",
            "invalid_assignment_count": 0,
            "nonfinite_state_count": 0,
        },
        {
            "env_id": 1,
            "success": False,
            "success_step": -1,
            "all_detected_step": 60,
            "failure_reason": "detected_unvisited_remaining",
            "invalid_assignment_count": 2,
            "nonfinite_state_count": 1,
        },
        {
            "env_id": 2,
            "success": False,
            "success_step": -1,
            "all_detected_step": -1,
            "failure_reason": "undiscovered_remaining",
            "invalid_assignment_count": 0,
            "nonfinite_state_count": 0,
        },
    ]


def test_validation_summary_contract() -> None:
    summary = validation_summary(sample_rows())
    assert summary["episodes"] == 3
    assert summary["success_count"] == 1
    assert summary["success_rate"] == 1 / 3
    assert summary["detect_all_count"] == 2
    assert summary["detect_all_rate"] == 2 / 3
    assert summary["timeout_count"] == 2
    assert summary["success_step"] == {"count": 1, "mean": 80.0, "min": 80.0, "max": 80.0}
    assert summary["all_detected_step"]["mean"] == 50.0
    assert summary["invalid_count"] == 2
    assert summary["nonfinite_count"] == 1
    assert summary["failure_classification"] == {
        "detected_unvisited_remaining": 1,
        "success": 1,
        "undiscovered_remaining": 1,
    }


def test_outputs_are_deterministic_and_non_overwriting() -> None:
    rows = sample_rows()
    summary = validation_summary(rows)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        first = root / "first"
        second = root / "second"
        first_paths = write_validation_outputs(first, rows, summary)
        second_paths = write_validation_outputs(second, rows, summary)
        assert first_paths[0].read_bytes() == second_paths[0].read_bytes()
        assert first_paths[1].read_bytes() == second_paths[1].read_bytes()
        assert len(first_paths[0].read_text(encoding="utf-8").splitlines()) == 3
        assert json.loads(first_paths[1].read_text(encoding="utf-8"))["episodes"] == 3
        try:
            write_validation_outputs(first, rows, summary)
        except FileExistsError:
            pass
        else:
            raise AssertionError("validation output directory was overwritten")


def test_callable_policy_validation_is_deterministic() -> None:
    from scripts import analyze_sar_failure_modes as analysis

    class Policy:
        deterministic = False

    policy = Policy()
    original_run = analysis.run

    def fake_run(args, *, high_level_policy=None):
        assert high_level_policy is policy
        assert policy.deterministic is True
        assert args.seed == 10000
        assert args.num_envs == 3
        return sample_rows(), {}

    analysis.run = fake_run
    try:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            low = root / "low.pt"
            low.write_bytes(b"frozen-low")
            sha = hashlib.sha256(low.read_bytes()).hexdigest()
            config = Path(__file__).resolve().parents[1] / "configs/chapter1/high_train_v1.yaml"
            resolved = resolve_chapter1_config(config)
            first_rows, first_summary = run_chapter1_validation(
                resolved=resolved, low_level_checkpoint=low,
                low_level_sha256=sha, output_dir=root / "first",
                high_level_policy=policy, episodes=3, device="cpu",
            )
            second_rows, second_summary = run_chapter1_validation(
                resolved=resolved, low_level_checkpoint=low,
                low_level_sha256=sha, output_dir=root / "second",
                high_level_policy=policy, episodes=3, device="cpu",
            )
            assert first_rows == second_rows
            assert first_summary == second_summary
            assert policy.deterministic is False
    finally:
        analysis.run = original_run


if __name__ == "__main__":
    test_validation_summary_contract()
    test_outputs_are_deterministic_and_non_overwriting()
    test_callable_policy_validation_is_deterministic()
    print("chapter1 validation tests passed")
