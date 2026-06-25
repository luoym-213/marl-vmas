"""Summarize BenchMARL CSV scalars for comm-limited VMAS runs."""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path
from typing import Any


METRIC_FILES = {
    "eval_reward_mean": "eval_reward_episode_reward_mean.csv",
    "eval_agents_reward_mean": "eval_agents_reward_episode_reward_mean.csv",
    "collection_reward_mean": "collection_reward_episode_reward_mean.csv",
    "success_rate": "collection_agents_info_all_landmarks_covered.csv",
    "mean_aoi": "collection_agents_info_mean_aoi.csv",
    "mean_comm_mask": "collection_agents_info_mean_comm_mask.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize comm-limited VMAS BenchMARL run outputs."
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path(__file__).resolve().parents[1] / "outputs"],
        help="Output roots or run folders to scan.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write the summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    warnings = []

    for root in args.roots:
        for run_dir in find_run_dirs(root):
            row, row_warnings = summarize_run(run_dir)
            rows.append(row)
            warnings.extend(row_warnings)

    rows.sort(key=lambda row: str(row["run_dir"]))
    write_rows(rows, args.output_csv)

    for warning in warnings:
        print(f"[Warning] {warning}")


def find_run_dirs(root: Path) -> list[Path]:
    if (root / "scalars").is_dir():
        return [root]
    return sorted({path.parent.parent for path in root.glob("**/scalars/*.csv")})


def summarize_run(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    warnings = []
    hparams = read_hparams(run_dir)
    task_config = hparams.get("task_config", {})

    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "comm": task_config.get("comm", {}).get("mode", infer_comm_from_path(run_dir)),
        "estimator": task_config.get("estimator", {}).get("name", ""),
        "seed": hparams.get("seed", ""),
    }

    scalars_dir = run_dir / "scalars"
    for metric_name, file_name in METRIC_FILES.items():
        metric_path = scalars_dir / file_name
        if not metric_path.exists():
            row[metric_name] = ""
            warnings.append(f"Missing {metric_path}")
            continue
        row[metric_name] = read_last_scalar(metric_path)

    return row, warnings


def read_hparams(run_dir: Path) -> dict[str, Any]:
    hparams_path = run_dir / "texts" / "hparams0.txt"
    if not hparams_path.exists():
        return {}

    data: dict[str, Any] = {}
    for line in hparams_path.read_text(encoding="utf-8").splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        try:
            data[key] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            data[key] = value
    return data


def read_last_scalar(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return ""

    row = rows[-1]
    for key in ("value", "Value", "y"):
        if key in row:
            return row[key]

    numeric_keys = [key for key in row if key.lower() not in {"step", "wall_time"}]
    return row[numeric_keys[-1]] if numeric_keys else ""


def infer_comm_from_path(run_dir: Path) -> str:
    parts = set(run_dir.parts)
    for comm in ("comm_full", "delay", "radius_delay_dropout"):
        if comm in parts or any(comm in part for part in parts):
            return comm
    return ""


def write_rows(rows: list[dict[str, Any]], output_csv: Path | None) -> None:
    fieldnames = [
        "run_dir",
        "comm",
        "estimator",
        "seed",
        *METRIC_FILES.keys(),
    ]

    if output_csv is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary: {output_csv}")

if __name__ == "__main__":
    main()
