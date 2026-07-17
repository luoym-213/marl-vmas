"""Compare two SAR diagnostic JSON files on exactly paired layouts."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--additional-pair",
        nargs=2,
        action="append",
        type=Path,
        default=[],
        metavar=("BASELINE", "CANDIDATE"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = round(probability * (len(ordered) - 1))
    return ordered[index]


def main() -> None:
    args = parse_args()
    rows: list[dict[str, int | str]] = []
    differences: list[int] = []
    layout_sets: list[str] = []
    pairs = [(args.baseline, args.candidate), *args.additional_pair]
    for pair_set, (baseline_path, candidate_path) in enumerate(pairs):
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        baseline_rows = {int(row["env_id"]): row for row in baseline["episodes"]}
        candidate_rows = {int(row["env_id"]): row for row in candidate["episodes"]}
        if baseline_rows.keys() != candidate_rows.keys():
            raise ValueError("paired files have different environment ids")
        layout_sets.append(str(baseline["summary"]["layout_sha256"]))
        for env_id in sorted(baseline_rows):
            base = baseline_rows[env_id]
            cand = candidate_rows[env_id]
            if base["layout_sha256"] != cand["layout_sha256"]:
                raise ValueError(f"layout mismatch for env {env_id}")
            base_success = int(base["success"])
            candidate_success = int(cand["success"])
            differences.append(candidate_success - base_success)
            if base_success and candidate_success:
                transition = "both_success"
            elif base_success:
                transition = "baseline_only"
            elif candidate_success:
                transition = "candidate_only"
            else:
                transition = "both_fail"
            rows.append(
                {
                    "pair_set": pair_set,
                    "env_id": env_id,
                    "layout_sha256": str(base["layout_sha256"]),
                    "baseline_success": base_success,
                    "candidate_success": candidate_success,
                    "transition": transition,
                }
            )

    rng = random.Random(args.seed)
    n = len(differences)
    bootstrap = [
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(args.bootstrap_samples)
    ]
    counts = {
        name: sum(row["transition"] == name for row in rows)
        for name in ("candidate_only", "baseline_only", "both_success", "both_fail")
    }
    baseline_successes = sum(int(row["baseline_success"]) for row in rows)
    candidate_successes = sum(int(row["candidate_success"]) for row in rows)
    summary = {
        "episodes": n,
        "layout_sha256": layout_sets,
        "baseline_successes": baseline_successes,
        "baseline_success_rate": baseline_successes / n,
        "candidate_successes": candidate_successes,
        "candidate_success_rate": candidate_successes / n,
        "paired_success_difference": (candidate_successes - baseline_successes) / n,
        "paired_difference_ci95": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
        **counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "episodes": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.csv_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Paired SAR comparison", ""]
    lines.extend(f"- `{key}`: {value}" for key, value in summary.items())
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
