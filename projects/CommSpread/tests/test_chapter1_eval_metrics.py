"""Tests for the fixed Chapter 1 SAR evaluation metrics."""

from __future__ import annotations

import json
import math
from tempfile import TemporaryDirectory
from pathlib import Path

from comm_spread.chapter1_eval_metrics import (
    build_fixed_evaluation_metrics,
    format_fixed_evaluation_metrics,
    sample_statistics,
    write_fixed_evaluation_metrics,
)


def row(success, completion, response, detection, entropy):
    return {
        "success": success,
        "success_step": completion,
        "rescue_response_time": response,
        "last_target_detection_time": detection,
        "global_entropy_trace": entropy,
    }


def test_fixed_metrics_use_declared_populations_and_sample_std():
    rows = [
        row(1, 10, 4, 6, [10, 8, 5]),
        row(1, 14, 6, 8, [20, 10, 0]),
        row(0, -1, 99, 9, [5, 5, 4]),
    ]
    document = build_fixed_evaluation_metrics(
        rows, seed=7, horizon=2, physics_dt=0.1
    )
    metrics = document["metrics"]

    assert math.isclose(metrics["success_rate"]["mean"], 2 / 3)
    assert math.isclose(metrics["success_rate"]["std"], math.sqrt(1 / 3))
    assert metrics["task_completion_time"] == {
        "symbol": "T_all",
        "unit": "environment_step",
        "conditioning": "successful_episodes",
        "mean": 12.0,
        "std": math.sqrt(8.0),
        "sample_count": 2,
    }
    response = metrics["mean_rescue_response_time"]
    assert response["mean"] == 5.0
    assert math.isclose(response["std"], math.sqrt(2.0))
    assert response["sample_count"] == 2
    detection = metrics["last_target_detection_time"]
    assert math.isclose(detection["mean"], 23 / 3)
    assert detection["sample_count"] == 3

    efficiency = metrics["search_efficiency"]
    assert efficiency["timesteps"] == [0, 1, 2]
    expected = [0.0, (0.2 + 0.5 + 0.0) / 3, (0.5 + 1.0 + 0.2) / 3]
    assert all(
        math.isclose(actual, target)
        for actual, target in zip(efficiency["mean_curve"], expected)
    )


def test_conditioned_metrics_emit_null_for_insufficient_samples():
    document = build_fixed_evaluation_metrics(
        [row(0, -1, 0, -1, [4, 3])],
        seed=0,
        horizon=1,
        physics_dt=None,
    )
    metrics = document["metrics"]
    assert metrics["task_completion_time"]["mean"] is None
    assert metrics["task_completion_time"]["std"] is None
    assert metrics["mean_rescue_response_time"]["sample_count"] == 0
    assert metrics["last_target_detection_time"]["sample_count"] == 0
    assert sample_statistics([3]) == {
        "mean": 3.0,
        "std": None,
        "sample_count": 1,
    }


def test_entropy_trace_validation_fails_closed():
    for entropy, message in (
        ([1], "horizon + 1"),
        ([0, 0], "must be positive"),
        ([1, float("nan")], "finite"),
    ):
        try:
            build_fixed_evaluation_metrics(
                [row(1, 1, 1, 1, entropy)],
                seed=0,
                horizon=1,
                physics_dt=0.1,
            )
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("expected invalid entropy trace to fail")


def test_json_output_and_terminal_format_are_stable():
    document = build_fixed_evaluation_metrics(
        [row(1, 1, 1, 1, [2, 1])],
        seed=3,
        horizon=1,
        physics_dt=0.1,
    )
    with TemporaryDirectory() as directory:
        destination = Path(directory) / "evaluation_metrics.json"
        assert write_fixed_evaluation_metrics(document, destination) == destination
        assert json.loads(destination.read_text(encoding="utf-8")) == document

    rendered = format_fixed_evaluation_metrics(document)
    for token in (
        "Fixed evaluation metrics:",
        "success_rate",
        "T_all",
        "mean_delta_t_res",
        "eta_search mean_curve",
        "T_det_last",
    ):
        assert token in rendered


if __name__ == "__main__":
    test_fixed_metrics_use_declared_populations_and_sample_std()
    test_conditioned_metrics_emit_null_for_insufficient_samples()
    test_entropy_trace_validation_fails_closed()
    test_json_output_and_terminal_format_are_stable()
    print("chapter1 eval metrics tests passed")
