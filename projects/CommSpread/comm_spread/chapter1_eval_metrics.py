"""Fixed paper-facing metrics for Chapter 1 SAR evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


METRIC_SCHEMA_VERSION = 1
METRIC_SET = "chapter1_sar_fixed_v1"


def _finite_values(values: Iterable[Any]) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("fixed evaluation metrics require finite samples")
    return result


def sample_statistics(values: Iterable[Any]) -> dict[str, float | int | None]:
    samples = _finite_values(values)
    count = len(samples)
    if count == 0:
        return {"mean": None, "std": None, "sample_count": 0}
    mean = sum(samples) / count
    std = None
    if count > 1:
        variance = sum((value - mean) ** 2 for value in samples) / (count - 1)
        std = math.sqrt(variance)
    return {"mean": mean, "std": std, "sample_count": count}


def build_fixed_evaluation_metrics(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    horizon: int,
    physics_dt: float | None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("fixed evaluation metrics require at least one episode")
    if horizon <= 0:
        raise ValueError("evaluation horizon must be positive")
    expected_trace_length = horizon + 1

    success_indicators = [int(bool(row["success"])) for row in rows]
    successful = [row for row in rows if int(bool(row["success"]))]
    detected_all = [
        row for row in rows if int(row["last_target_detection_time"]) >= 0
    ]
    traces: list[list[float]] = []
    for row in rows:
        trace = _finite_values(row["global_entropy_trace"])
        if len(trace) != expected_trace_length:
            raise ValueError(
                "global entropy trace length must equal evaluation horizon + 1"
            )
        if trace[0] <= 0:
            raise ValueError("initial global entropy must be positive")
        traces.append(trace)

    mean_curve = []
    for step in range(expected_trace_length):
        efficiencies = [
            (trace[0] - trace[step]) / trace[0]
            for trace in traces
        ]
        mean_curve.append(sum(efficiencies) / len(efficiencies))

    return {
        "schema_version": METRIC_SCHEMA_VERSION,
        "metric_set": METRIC_SET,
        "evaluation": {
            "seed": int(seed),
            "episodes": len(rows),
            "horizon": int(horizon),
            "time_unit": "environment_step",
            "physics_dt": (
                float(physics_dt) if physics_dt is not None else None
            ),
            "standard_deviation": "sample_ddof_1",
        },
        "metrics": {
            "success_rate": {
                "symbol": "success_rate",
                "conditioning": "all_episodes",
                **sample_statistics(success_indicators),
            },
            "task_completion_time": {
                "symbol": "T_all",
                "unit": "environment_step",
                "conditioning": "successful_episodes",
                **sample_statistics(
                    row["success_step"] for row in successful
                ),
            },
            "mean_rescue_response_time": {
                "symbol": "mean_delta_t_res",
                "unit": "environment_step",
                "conditioning": "successful_episodes",
                "episode_value": (
                    "mean_over_targets(first_rescue_step-first_detection_step)"
                ),
                **sample_statistics(
                    row["rescue_response_time"] for row in successful
                ),
            },
            "search_efficiency": {
                "symbol": "eta_search",
                "conditioning": "all_episodes",
                "formula": "(H_0-H_t)/H_0",
                "uncertainty": "team_global_belief_entropy",
                "sample_count": len(rows),
                "timesteps": list(range(expected_trace_length)),
                "mean_curve": mean_curve,
            },
            "last_target_detection_time": {
                "symbol": "T_det_last",
                "unit": "environment_step",
                "conditioning": "episodes_detecting_all_targets",
                **sample_statistics(
                    row["last_target_detection_time"] for row in detected_all
                ),
            },
        },
    }


def write_fixed_evaluation_metrics(
    document: dict[str, Any],
    destination: str | Path,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def format_fixed_evaluation_metrics(document: dict[str, Any]) -> str:
    metrics = document["metrics"]
    lines = ["Fixed evaluation metrics:"]

    def scalar(label: str, key: str) -> None:
        item = metrics[key]
        mean = "null" if item["mean"] is None else f"{item['mean']:.6g}"
        std = "null" if item["std"] is None else f"{item['std']:.6g}"
        lines.append(
            f"  {label}: mean={mean}, std={std}, n={item['sample_count']}"
        )

    scalar("success_rate", "success_rate")
    scalar("T_all [step]", "task_completion_time")
    scalar("mean_delta_t_res [step]", "mean_rescue_response_time")
    efficiency = metrics["search_efficiency"]
    lines.append(
        "  eta_search mean_curve: "
        + json.dumps(efficiency["mean_curve"], separators=(",", ":"))
    )
    scalar("T_det_last [step]", "last_target_detection_time")
    return "\n".join(lines)
