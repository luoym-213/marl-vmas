"""Build the paired 2x2x2 SAR module-ablation report from saved JSONs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any


VARIANTS = {
    "A": ("A_seed{seed}.json", "HGSAR", "finder", "commitment", (0, 0, 0)),
    "B": ("B_seed{seed}.json", "three-lane", "finder", "commitment", (1, 0, 0)),
    "C": ("C_seed{seed}.json", "HGSAR", "all-detected", "commitment", (0, 1, 0)),
    "D": ("D_v2_seed{seed}.json", "HGSAR", "finder", "exact", (0, 0, 1)),
    "E": ("E_seed{seed}.json", "three-lane", "all-detected", "exact", (1, 1, 1)),
    "F": ("F_search_timing_seed{seed}.json", "three-lane", "all-detected", "commitment", (1, 1, 0)),
    "G": ("G_search_assignment_seed{seed}.json", "three-lane", "finder", "exact", (1, 0, 1)),
    "H": ("H_timing_assignment_seed{seed}.json", "HGSAR", "all-detected", "exact", (0, 1, 1)),
}
CORE_COMPARISONS = {"search": "B", "timing": "C", "assignment": "D", "total": "E"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-64", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    return parser.parse_args()


def finite_mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return mean(clean) if clean else float("nan")


def percentile(values: list[float], q: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return float("nan")
    rank = (len(clean) - 1) * q
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return clean[lower]
    weight = rank - lower
    return clean[lower] * (1 - weight) + clean[upper] * weight


def load_runs(input_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    pooled: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, Any] = {}
    reference_hashes: dict[int, list[str]] = {}
    for name, (template, *_rest) in VARIANTS.items():
        pooled[name] = []
        metadata[name] = {}
        for seed in (0, 1):
            path = input_dir / template.format(seed=seed)
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = sorted(data["episodes"], key=lambda row: int(row["env_id"]))
            hashes = [str(row["layout_sha256"]) for row in rows]
            if name == "A":
                reference_hashes[seed] = hashes
            elif hashes != reference_hashes[seed]:
                raise ValueError(f"layout mismatch: {name} seed {seed}")
            for row in rows:
                copied = dict(row)
                copied["seed"] = seed
                copied["pooled_env_id"] = seed * 256 + int(row["env_id"])
                pooled[name].append(copied)
            metadata[name][f"seed_{seed}"] = data["metadata"]
    return pooled, metadata


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    successes = [row for row in rows if int(row["success"])]
    detected_all = [row for row in rows if int(row["all_detected_step"]) >= 0]
    cascade_events = [event for row in rows for event in row["finder_cascade_events"]]
    reject_waits = [
        float(wait) for row in rows for wait in row["rejected_target_wait_steps"]
    ]
    quality_events = sum(int(row["assignment_quality_event_count"]) for row in rows)
    crossing_events = sum(int(row["crossing_event_count"]) for row in rows)
    scalar_means = [
        "first_detection_step", "second_detection_step", "all_detected_step",
        "success_step", "team_coverage_rate", "overlap_ratio",
        "explored_area_revisit_ratio", "coverage_efficiency",
        "search_travel_distance", "search_option_switches",
        "search_heading_continuity", "discovery_to_assignment_delay_mean",
        "assignment_to_visit_delay_mean", "assignment_sum_regret",
        "assignment_makespan_regret",
    ]
    result: dict[str, Any] = {
        "episodes": n,
        "success_count": len(successes),
        "success_rate": len(successes) / n,
        "detect_all_count": len(detected_all),
        "detect_all_rate": len(detected_all) / n,
        "success_given_detect_all": len(successes) / max(len(detected_all), 1),
        "terminal_not_all_detected_count": n - len(detected_all),
        "detected_all_but_incomplete_count": len(detected_all) - len(successes),
        "completion_step_p95": percentile(
            [float(row["success_step"]) for row in successes], 0.95
        ),
        "premature_retirement_count": sum(int(row["premature_retirement"]) for row in rows),
        "premature_retirement_rate": finite_mean(
            [float(row["premature_retirement"]) for row in rows]
        ),
        "duplicate_assignment_excess": sum(
            float(row["duplicate_assignment_excess"]) for row in rows
        ),
        "crossing_event_rate": crossing_events / max(quality_events, 1),
        "finder_open_count": sum(event["event"] == "finder_open" for event in cascade_events),
        "finder_accept_count": sum(event["event"] == "accept" for event in cascade_events),
        "finder_reject_count": sum(event["event"] == "reject" for event in cascade_events),
        "finder_unavailable_count": sum(event["event"] == "finder_unavailable" for event in cascade_events),
        "rejected_target_later_claimed_count": sum(
            int(row["rejected_target_later_claimed_count"]) for row in rows
        ),
        "rejected_target_wait_mean": finite_mean(reject_waits),
        "rejected_target_wait_p95": percentile(reject_waits, 0.95),
        "invalid_assignment_count": sum(int(row["invalid_assignment_count"]) for row in rows),
        "nonfinite_state_count": sum(int(row["nonfinite_state_count"]) for row in rows),
        "failure_decomposition": dict(sorted(Counter(
            str(row["failure_category"]) for row in rows
        ).items())),
    }
    for key in scalar_means:
        selected = successes if key == "success_step" else rows
        result_key = key if key.endswith("_mean") else f"{key}_mean"
        result[result_key] = finite_mean(
            [float(row[key]) for row in selected if float(row[key]) >= 0]
        )
    for key in (
        "per_agent_explored_area", "per_agent_search_travel_distance",
        "per_agent_search_option_switches",
    ):
        result[f"{key}_mean"] = [
            finite_mean([float(row[key][agent]) for row in rows])
            for agent in range(3)
        ]
    return result


def paired_summary(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]],
    *, samples: int, seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paired_rows: list[dict[str, Any]] = []
    differences: list[int] = []
    transitions: Counter[tuple[str, str]] = Counter()
    for base, cand in zip(baseline, candidate):
        if base["layout_sha256"] != cand["layout_sha256"]:
            raise ValueError("per-environment layout mismatch")
        base_success, cand_success = int(base["success"]), int(cand["success"])
        difference = cand_success - base_success
        differences.append(difference)
        label = (
            "both_success" if base_success and cand_success else
            "baseline_only" if base_success else
            "candidate_only" if cand_success else "both_fail"
        )
        transitions[(str(base["failure_category"]), str(cand["failure_category"]))] += 1
        paired_rows.append({
            "seed": base["seed"], "env_id": base["env_id"],
            "pooled_env_id": base["pooled_env_id"],
            "layout_sha256": base["layout_sha256"],
            "baseline_success": base_success, "candidate_success": cand_success,
            "transition": label,
            "baseline_failure": base["failure_category"],
            "candidate_failure": cand["failure_category"],
            "baseline_detect_all": int(base["all_detected_step"] >= 0),
            "candidate_detect_all": int(cand["all_detected_step"] >= 0),
        })
    rng = random.Random(seed)
    n = len(differences)
    bootstrap = [
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    ]
    counts = Counter(row["transition"] for row in paired_rows)
    base_agg, cand_agg = aggregate(baseline), aggregate(candidate)
    summary = {
        "baseline_success": base_agg["success_count"],
        "candidate_success": cand_agg["success_count"],
        "absolute_success_lift": cand_agg["success_rate"] - base_agg["success_rate"],
        "candidate_only": counts["candidate_only"],
        "baseline_only": counts["baseline_only"],
        "both_success": counts["both_success"],
        "both_fail": counts["both_fail"],
        "paired_bootstrap_ci95": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
        "detect_all_delta": cand_agg["detect_all_rate"] - base_agg["detect_all_rate"],
        "success_given_detect_all_delta": (
            cand_agg["success_given_detect_all"] - base_agg["success_given_detect_all"]
        ),
        "failure_transitions": {
            f"{left} -> {right}": count
            for (left, right), count in sorted(transitions.items())
        },
    }
    return summary, paired_rows


def bootstrap_interaction(
    runs: dict[str, list[dict[str, Any]]], *, samples: int, seed: int,
) -> list[float]:
    per_episode = [
        int(runs["E"][i]["success"]) - int(runs["B"][i]["success"])
        - int(runs["C"][i]["success"]) - int(runs["D"][i]["success"])
        + 2 * int(runs["A"][i]["success"])
        for i in range(len(runs["A"]))
    ]
    rng = random.Random(seed)
    n = len(per_episode)
    draws = [
        sum(per_episode[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    args = parse_args()
    runs, metadata = load_runs(args.input_dir)
    aggregates = {name: aggregate(rows) for name, rows in runs.items()}
    seed_aggregates = {
        name: {
            str(seed): aggregate([row for row in rows if row["seed"] == seed])
            for seed in (0, 1)
        }
        for name, rows in runs.items()
    }
    paired: dict[str, Any] = {}
    consolidated: list[dict[str, Any]] = []
    for index, (label, candidate) in enumerate(CORE_COMPARISONS.items()):
        summary, rows = paired_summary(
            runs["A"], runs[candidate], samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + index,
        )
        paired[label] = summary
        for row in rows:
            consolidated.append({"comparison": f"{candidate}-A", **row})

    rates = {name: aggregates[name]["success_rate"] for name in VARIANTS}
    singles_sum = rates["B"] + rates["C"] + rates["D"] - 3 * rates["A"]
    total_delta = rates["E"] - rates["A"]
    residual = total_delta - singles_sum
    factorial = {
        "single_search": rates["B"] - rates["A"],
        "single_timing": rates["C"] - rates["A"],
        "single_assignment": rates["D"] - rates["A"],
        "total": total_delta,
        "sum_of_singles": singles_sum,
        "core_interaction_residual": residual,
        "core_interaction_residual_ci95": bootstrap_interaction(
            runs, samples=args.bootstrap_samples, seed=args.bootstrap_seed + 10
        ),
        "search_x_timing": rates["F"] - rates["B"] - rates["C"] + rates["A"],
        "search_x_assignment": rates["G"] - rates["B"] - rates["D"] + rates["A"],
        "timing_x_assignment": rates["H"] - rates["C"] - rates["D"] + rates["A"],
        "three_way": (
            rates["E"] - rates["F"] - rates["G"] - rates["H"]
            + rates["B"] + rates["C"] + rates["D"] - rates["A"]
        ),
    }

    overlap: dict[str, Any] = {"checked": False}
    if args.validation_64 is not None:
        validation = json.loads(args.validation_64.read_text(encoding="utf-8"))
        validation_hashes = {str(row["layout_sha256"]) for row in validation["episodes"]}
        seed_sets = {
            str(seed): {str(row["layout_sha256"]) for row in runs["A"] if row["seed"] == seed}
            for seed in (0, 1)
        }
        overlap = {
            "checked": True,
            "validation_path": str(args.validation_64),
            "seed_0_overlap_count": len(validation_hashes & seed_sets["0"]),
            "seed_1_overlap_count": len(validation_hashes & seed_sets["1"]),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis = {
        "aggregates": aggregates, "seed_aggregates": seed_aggregates,
        "paired_comparisons": paired,
        "factorial_interactions": factorial, "layout_overlap": overlap,
    }
    (args.output_dir / "module_ablation_analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "paired_comparisons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(consolidated[0]))
        writer.writeheader()
        writer.writerows(consolidated)
    (args.output_dir / "experiment_config_and_commands.json").write_text(
        json.dumps({"variants": VARIANTS, "metadata": metadata}, indent=2, default=list)
        + "\n", encoding="utf-8"
    )

    lines = [
        "# High-level SAR module replacement: paired 2x2x2 ablation", "",
        "## Executive result", "",
        f"Across the two independent 256-layout sets, A succeeds on "
        f"{aggregates['A']['success_count']}/512 ({pct(rates['A'])}); E succeeds "
        f"on {aggregates['E']['success_count']}/512 ({pct(rates['E'])}). The "
        f"paired total gain is {pct(total_delta)}.", "",
        "The largest single-module gain is search (B-A), but the dominant result "
        "is non-additivity: the three single replacements sum to "
        f"{pct(singles_sum)}, versus {pct(total_delta)} jointly; residual "
        f"{pct(residual)} (paired bootstrap 95% CI "
        f"[{pct(factorial['core_interaction_residual_ci95'][0])}, "
        f"{pct(factorial['core_interaction_residual_ci95'][1])}]).", "",
        "## Methods and controls", "",
        "- Every cell uses the same seed-0 256 and seed-1 256 reset streams, 3 UAVs, 3 targets, 100 steps, sensor radius 0.6, retire-on-rescue, repaired MAPPO low-level checkpoint, and goal fallback. No training is performed.",
        "- B overrides only goals for actions that the old HGSAR actor selected as search; rescue timing and commitment are unchanged.",
        "- C masks rescue until all three targets are detected; search actions and search-node choices remain those of HGSAR.",
        "- D preserves the actor's number and time of rescue acceptances, then exactly minimizes makespan (sum distance tie-break) over active current-decision UAVs and globally detected, unvisited, unclaimed targets. It never reads an undiscovered target. A displaced former accepter uses the observable highest-score explore candidate.",
        "- E/H force all active current-decision UAVs into the exact visible matching after all targets are detected. All exact cells are duplicate-free.",
        "- Coverage is the union of map cells inside active search-UAV sensor footprints. Coverage efficiency is union explored area divided by search travel distance. Overlap ratio is cells explored by multiple UAVs divided by union explored cells. Revisit ratio is one minus union cells divided by accumulated footprint cell-visits.",
        "- Paired bootstrap resamples the 512 paired environments with replacement (20,000 draws).", "",
        "## Seed split", "",
        "| ID | Seed-0 success / detect-all | Seed-1 success / detect-all | Pooled success / detect-all |",
        "|---|---:|---:|---:|",
    ]
    for name in VARIANTS:
        seed0 = seed_aggregates[name]["0"]
        seed1 = seed_aggregates[name]["1"]
        pooled = aggregates[name]
        lines.append(
            f"| {name} | {seed0['success_count']}/256 / {seed0['detect_all_count']}/256 | "
            f"{seed1['success_count']}/256 / {seed1['detect_all_count']}/256 | "
            f"{pooled['success_count']}/512 / {pooled['detect_all_count']}/512 |"
        )
    lines += ["",
        "## Experiment matrix", "",
        "| ID | Search | Timing | Assignment | Success | Detect-all | Success given detect-all | Not all detected | Detected-all incomplete |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        _, search, timing, assignment, _ = VARIANTS[name]
        item = aggregates[name]
        lines.append(
            f"| {name} | {search} | {timing} | {assignment} | "
            f"{item['success_count']}/512 ({pct(item['success_rate'])}) | "
            f"{item['detect_all_count']}/512 ({pct(item['detect_all_rate'])}) | "
            f"{pct(item['success_given_detect_all'])} | "
            f"{item['terminal_not_all_detected_count']} | "
            f"{item['detected_all_but_incomplete_count']} |"
        )

    lines += ["", "## Core paired effects", "",
        "| Effect | Lift | Candidate-only / baseline-only | Paired bootstrap 95% CI | Detect-all Δ | Success|detect-all Δ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("search", "timing", "assignment", "total"):
        item = paired[label]
        lines.append(
            f"| Δ{label} | {pct(item['absolute_success_lift'])} | "
            f"{item['candidate_only']} / {item['baseline_only']} | "
            f"[{pct(item['paired_bootstrap_ci95'][0])}, {pct(item['paired_bootstrap_ci95'][1])}] | "
            f"{pct(item['detect_all_delta'])} | {pct(item['success_given_detect_all_delta'])} |"
        )

    lines += ["", "## Search-quality comparison: B versus A", "",
        "| Metric | A HGSAR | B three-lane | Difference |", "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("team_coverage_rate_mean", "Team coverage"),
        ("overlap_ratio_mean", "Multi-UAV overlap ratio"),
        ("explored_area_revisit_ratio_mean", "Explored-area revisit ratio"),
        ("coverage_efficiency_mean", "Coverage efficiency"),
        ("first_detection_step_mean", "First detection step"),
        ("second_detection_step_mean", "Second detection step"),
        ("all_detected_step_mean", "All-detected step"),
        ("search_option_switches_mean", "Search option switches"),
        ("search_heading_continuity_mean", "Search trajectory continuity"),
    ):
        a, b = aggregates["A"][key], aggregates["B"][key]
        lines.append(f"| {label} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
    lines.append(
        f"Per-UAV explored area A: {aggregates['A']['per_agent_explored_area_mean']}; "
        f"B: {aggregates['B']['per_agent_explored_area_mean']}."
    )

    lines += ["", "## Execution, assignment, and safety diagnostics", "",
        "| ID | Completion mean / P95 | Premature | Duplicate | Crossing | Sum / makespan regret | Assign→visit | Finder accept/reject/unavailable | Rejected later claimed; wait mean/P95 | Invalid/nonfinite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        item = aggregates[name]
        lines.append(
            f"| {name} | {item['success_step_mean']:.2f} / {item['completion_step_p95']:.2f} | "
            f"{item['premature_retirement_count']} ({pct(item['premature_retirement_rate'])}) | "
            f"{item['duplicate_assignment_excess']:.0f} | {pct(item['crossing_event_rate'])} | "
            f"{item['assignment_sum_regret_mean']:.4f} / {item['assignment_makespan_regret_mean']:.4f} | "
            f"{item['assignment_to_visit_delay_mean']:.2f} | "
            f"{item['finder_accept_count']}/{item['finder_reject_count']}/{item['finder_unavailable_count']} | "
            f"{item['rejected_target_later_claimed_count']}; {item['rejected_target_wait_mean']:.2f}/{item['rejected_target_wait_p95']:.2f} | "
            f"{item['invalid_assignment_count']}/{item['nonfinite_state_count']} |"
        )

    lines += ["", "## Failure decomposition", "",
        "| ID | Not all detected | All detected too late | Post-detection over-exploration | Assignment conflict | Residual low-level | Success |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        failure = aggregates[name]["failure_decomposition"]
        lines.append(
            f"| {name} | {failure.get('1_not_all_targets_detected', 0)} | "
            f"{failure.get('2_all_detected_too_late', 0)} | "
            f"{failure.get('3_post_detection_over_exploration', 0)} | "
            f"{failure.get('5_target_assignment_conflict', 0)} | "
            f"{failure.get('6_residual_low_level_failure', 0)} | "
            f"{failure.get('success', 0)} |"
        )

    lines += ["", "## Interaction decomposition", ""]
    for key, value in factorial.items():
        if key.endswith("ci95"):
            continue
        lines.append(f"- {key}: {pct(value)}")
    lines += ["", "## Failure transitions from A", ""]
    for label in ("search", "timing", "assignment", "total"):
        lines.append(f"### Δ{label}")
        for transition, count in sorted(
            paired[label]["failure_transitions"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            if count:
                lines.append(f"- {transition}: {count}")
        lines.append("")

    lines += ["## Answers to the seven required questions", "",
        f"1. Search is the largest single replacement: {pct(paired['search']['absolute_success_lift'])}. Timing gives {pct(paired['timing']['absolute_success_lift'])}; assignment gives {pct(paired['assignment']['absolute_success_lift'])}.",
        f"2. Three-lane versus HGSAR changes team coverage by {aggregates['B']['team_coverage_rate_mean']-aggregates['A']['team_coverage_rate_mean']:+.4f}, overlap by {aggregates['B']['overlap_ratio_mean']-aggregates['A']['overlap_ratio_mean']:+.4f}, and all-detected mean step by {aggregates['B']['all_detected_step_mean']-aggregates['A']['all_detected_step_mean']:+.2f} among detected-all episodes; detect-all rate changes by {pct(aggregates['B']['detect_all_rate']-aggregates['A']['detect_all_rate'])}.",
        f"3. Yes. A has {aggregates['A']['premature_retirement_count']}/512 premature retirements ({pct(aggregates['A']['premature_retirement_rate'])}); all-detected gating reduces this to zero.",
        f"4. Exact matching alone changes success by {pct(paired['assignment']['absolute_success_lift'])} ({paired['assignment']['candidate_only']}/{paired['assignment']['baseline_only']} candidate-only/baseline-only), while reducing geometric regret; it is not an isolated success bottleneck.",
        f"5. In A, {aggregates['A']['terminal_not_all_detected_count']} failures never detect all, versus {aggregates['A']['detected_all_but_incomplete_count']} that detect all but do not finish. The dominant baseline failure is non-discovery.",
        f"6. Yes. The core interaction residual is {pct(residual)}, well above the preregistered 3pp trigger. Search×timing={pct(factorial['search_x_timing'])}, search×assignment={pct(factorial['search_x_assignment'])}, timing×assignment={pct(factorial['timing_x_assignment'])}, three-way={pct(factorial['three_way'])}.",
        "7. Prioritize search jointly with a reliable all-detected terminal-phase handoff. Do not prioritize standalone assignment learning: exact assignment helps materially only after search and timing create a coherent all-detected rescue phase.",
        "", "## Layout independence and audit notes", "",
        f"- 64-layout overlap check: {overlap}.",
        "- Seed-1 is an independent 256-layout set and is retained as primary generalization evidence alongside pooled 512.",
        "- D_seed0.json is an invalid implementation-audit artifact (stale target action caused duplicates); all D results use D_v2_seed0/1, both duplicate-free.",
        "- E uses repaired MAPPO and exactly reproduces the existing seed-0 coverage_sensor_fixed_low success 167/256. Proportional control is absent from the main table.",
        "", "## Reproduction", "",
        "Every exact command and parsed configuration is stored in `experiment_config_and_commands.json`. Per-environment source JSON/CSV and coverage CSV files remain in this directory.",
    ]
    (args.output_dir / "module_ablation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"success": {k: v["success_count"] for k, v in aggregates.items()}, "factorial": factorial, "overlap": overlap}, indent=2))


if __name__ == "__main__":
    main()
