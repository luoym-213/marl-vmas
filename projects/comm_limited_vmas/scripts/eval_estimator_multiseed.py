"""Run multi-seed evals for the hidden-goal estimator baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval.py"

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_TASK = "comm_hidden_goal_navigation"
DEFAULT_ALGORITHM = "mappo"
DEFAULT_LIMITED_COMM = "radius_delay_dropout_mean8_dropout03"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "eval"
    / "baseline_radius_delay_dropout_mean8_dropout03_spawn0.9_multiseed"
)

METRIC_KEYS = [
    "success_rate",
    "timeout_rate",
    "episode_len_mean",
    "success_episode_len_mean",
    "episode_return_mean",
    "team_return_mean",
    "final_team_distance_mean",
    "mean_team_distance",
    "mean_leader_distance",
    "mean_follow_error",
    "mean_aoi",
    "mean_comm_mask",
]

EVAL_REGISTRY = {
    "comm_full": {
        "comm": "comm_full",
        "estimator": "stale",
        "checkpoint": (
            "outputs/comm_hidden_goal_navigation_mappo_comm_full_stale_seed0/"
            "spawn0.9/checkpoints/checkpoint_20040000.pt"
        ),
    },
    "stale": {
        "comm": DEFAULT_LIMITED_COMM,
        "estimator": "stale",
        "checkpoint": (
            "outputs/comm_hidden_goal_navigation_mappo_radius_delay_dropout_mean8_dropout03_"
            "stale_seed0/spawn0.9/checkpoints/checkpoint_20040000.pt"
        ),
    },
    "kinematic": {
        "comm": DEFAULT_LIMITED_COMM,
        "estimator": "kinematic",
        "checkpoint": (
            "outputs/comm_hidden_goal_navigation_mappo_radius_delay_dropout_mean8_dropout03_"
            "kinematic_seed0/spawn0.9/checkpoints/checkpoint_20040000.pt"
        ),
    },
    "aoi_residual": {
        "comm": DEFAULT_LIMITED_COMM,
        "estimator": "aoi_residual",
        "checkpoint": (
            "outputs/comm_hidden_goal_navigation_mappo_radius_delay_dropout_mean8_dropout03_"
            "aoi_residual_seed0/spawn0.9/checkpoints/checkpoint_20040000.pt"
        ),
    },
    "gnn_residual": {
        "comm": DEFAULT_LIMITED_COMM,
        "estimator": "gnn_residual",
        "checkpoint": (
            "outputs/comm_hidden_goal_navigation_mappo_radius_delay_dropout_mean8_dropout03_"
            "gnn_residual_seed0/spawn0.9/checkpoints/checkpoint_20040000.pt"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate estimator baselines over multiple eval seeds."
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--algorithm", default=DEFAULT_ALGORITHM)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(EVAL_REGISTRY),
        default=None,
        help="Subset of estimator groups to evaluate.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--render-gif", action="store_true")
    parser.add_argument("--render-episodes", type=int, default=3)
    parser.add_argument("--render-fps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args.output_root)
    groups = args.only or list(EVAL_REGISTRY)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for group in groups:
        spec = EVAL_REGISTRY[group]
        for seed in args.seeds:
            command, eval_root, summary_path = build_eval_command(
                args=args,
                group=group,
                spec=spec,
                seed=seed,
                output_root=output_root,
            )
            if args.dry_run:
                print(" ".join(command))
                continue

            if args.skip_existing and summary_path.exists():
                print(f"[skip] {group} seed={seed}: {summary_path}")
            else:
                print(f"[run] {group} seed={seed}")
                try:
                    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                except subprocess.CalledProcessError as err:
                    failure = {
                        "group": group,
                        "seed": seed,
                        "returncode": err.returncode,
                        "command": command,
                    }
                    failures.append(failure)
                    if not args.continue_on_error:
                        write_failure_report(output_root, failures)
                        raise
                    print(f"[failed] {group} seed={seed} returncode={err.returncode}")
                    continue

            if summary_path.exists():
                results.append(load_result_row(group, seed, eval_root, summary_path))
            else:
                failures.append(
                    {
                        "group": group,
                        "seed": seed,
                        "returncode": None,
                        "command": command,
                        "missing_summary": str(summary_path),
                    }
                )

    if args.dry_run:
        expected = len(groups) * len(args.seeds)
        print(f"Dry run commands: {expected}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    write_per_seed_csv(output_root / "per_seed_summary.csv", results)
    aggregate = aggregate_results(results)
    write_aggregate_csv(output_root / "aggregate_summary.csv", aggregate)
    write_json(
        output_root / "aggregate_summary.json",
        {
            "metadata": {
                "task": args.task,
                "algorithm": args.algorithm,
                "episodes": args.episodes,
                "seeds": args.seeds,
                "groups": groups,
                "device": args.device,
            },
            "per_seed": results,
            "aggregate": aggregate,
            "failures": failures,
        },
    )
    if failures:
        write_failure_report(output_root, failures)
    print(f"Wrote: {output_root / 'per_seed_summary.csv'}")
    print(f"Wrote: {output_root / 'aggregate_summary.csv'}")
    print(f"Wrote: {output_root / 'aggregate_summary.json'}")


def build_eval_command(
    args: argparse.Namespace,
    group: str,
    spec: dict[str, str],
    seed: int,
    output_root: Path,
) -> tuple[list[str], Path, Path]:
    eval_root = output_root / group
    checkpoint = resolve_project_path(Path(spec["checkpoint"]))
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--task",
        args.task,
        "--checkpoint",
        str(checkpoint),
        "--algorithm",
        args.algorithm,
        "--comm",
        spec["comm"],
        "--estimator",
        spec["estimator"],
        "--episodes",
        str(args.episodes),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--output-dir",
        str(eval_root),
    ]
    if args.render_gif:
        command.extend(["--render-gif", "--render-episodes", str(args.render_episodes)])
        if args.render_fps is not None:
            command.extend(["--render-fps", str(args.render_fps)])

    summary_path = expected_summary_path(
        eval_root=eval_root,
        task=args.task,
        algorithm=args.algorithm,
        comm=spec["comm"],
        checkpoint=checkpoint,
        seed=seed,
    )
    return command, eval_root, summary_path


def expected_summary_path(
    eval_root: Path,
    task: str,
    algorithm: str,
    comm: str,
    checkpoint: Path,
    seed: int,
) -> Path:
    return (
        eval_root
        / f"{task}_{algorithm}_{comm}_{checkpoint.stem}_seed{seed}"
        / "eval_summary.json"
    )


def load_result_row(
    group: str,
    seed: int,
    eval_root: Path,
    summary_path: Path,
) -> dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    metadata = result["metadata"]
    metrics = result["metrics"]
    row: dict[str, Any] = {
        "group": group,
        "seed": seed,
        "eval_dir": str(summary_path.parent),
        "task": metadata.get("task"),
        "algorithm": metadata.get("algorithm"),
        "comm": metadata.get("comm"),
        "estimator": metadata.get("estimator"),
        "checkpoint": metadata.get("checkpoint"),
        "episodes": metadata.get("episodes"),
    }
    for key, value in metrics.items():
        row[key] = value
    return row


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["group"]), []).append(row)

    aggregate = []
    for group, group_rows in grouped.items():
        out: dict[str, Any] = {"group": group, "n_seeds": len(group_rows)}
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in group_rows if is_number(row.get(key))]
            if not values:
                continue
            mean_value = sum(values) / len(values)
            std_value = sample_std(values)
            ci95 = 1.96 * std_value / math.sqrt(len(values)) if values else float("nan")
            out[f"{key}_mean"] = mean_value
            out[f"{key}_std"] = std_value
            out[f"{key}_min"] = min(values)
            out[f"{key}_max"] = max(values)
            out[f"{key}_ci95"] = ci95
        aggregate.append(out)
    return aggregate


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def write_per_seed_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        write_csv(path, [])
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "group",
        "seed",
        "episodes",
        "comm",
        "estimator",
        *METRIC_KEYS,
        "eval_dir",
        "checkpoint",
    ]
    ordered = [key for key in preferred if key in fieldnames]
    ordered.extend(key for key in fieldnames if key not in ordered)
    write_csv(path, rows, ordered)


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        write_csv(path, [])
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["group", "n_seeds"]
    for metric in METRIC_KEYS:
        preferred.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_ci95",
                f"{metric}_min",
                f"{metric}_max",
            ]
        )
    ordered = [key for key in preferred if key in fieldnames]
    ordered.extend(key for key in fieldnames if key not in ordered)
    write_csv(path, rows, ordered)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_failure_report(output_root: Path, failures: list[dict[str, Any]]) -> None:
    write_json(output_root / "failures.json", {"failures": failures})


def resolve_project_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return (PROJECT_ROOT / expanded).resolve()


def resolve_output_root(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return (PROJECT_ROOT / expanded).resolve()


if __name__ == "__main__":
    main()
