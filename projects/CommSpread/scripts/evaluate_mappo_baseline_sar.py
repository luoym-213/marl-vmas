#!/usr/bin/env python
"""Evaluate an end-to-end MAPPO SAR checkpoint with the fixed five metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from comm_spread.chapter1_eval_metrics import (
    build_fixed_evaluation_metrics,
    format_fixed_evaluation_metrics,
    write_fixed_evaluation_metrics,
)
from comm_spread.mappo_baseline import (
    load_mappo_baseline_config,
    make_mappo_baseline_env,
    model_from_config,
    scenario_snapshot,
    validate_checkpoint_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def global_entropy(scenario: Any) -> torch.Tensor:
    maps = scenario._compute_entropy(scenario.belief_maps)
    return maps.min(dim=1).values.sum(dim=(-1, -2))


def evaluate_batch(
    config: dict[str, Any],
    model: torch.nn.Module,
    *,
    num_envs: int,
    device: str,
    seed: int,
) -> list[dict[str, Any]]:
    task = config["_task"]
    horizon = int(task["scenario"]["horizon"])
    env = make_mappo_baseline_env(
        config, num_envs=num_envs, device=device, seed=seed
    )
    env.reset()
    scenario = env.scenario
    n_targets = int(task["scenario"]["num_targets"])

    detection_first = torch.full(
        (num_envs, n_targets), -1, dtype=torch.long, device=device
    )
    visit_first = torch.full_like(detection_first, -1)
    initially_detected = scenario.target_detected.any(dim=1)
    detection_first = torch.where(
        initially_detected, torch.zeros_like(detection_first), detection_first
    )
    entropy_trace = torch.zeros(
        num_envs, horizon + 1, dtype=torch.float32, device=device
    )
    entropy_trace[:, 0] = global_entropy(scenario)
    success_step = torch.full(
        (num_envs,), -1, dtype=torch.long, device=device
    )
    action_counts = torch.zeros(num_envs, 5, dtype=torch.long, device=device)
    policy_entropy_sum = torch.zeros(num_envs, device=device)
    policy_margin_sum = torch.zeros(num_envs, device=device)
    active_action_count = torch.zeros(num_envs, device=device)

    for step in range(horizon):
        pre_detected = scenario.target_detected.any(dim=1).clone()
        pre_visited = scenario.target_visited.clone()
        snapshot = scenario_snapshot(scenario)
        with torch.no_grad():
            actions, _, _ = model.act(snapshot, deterministic=True)
            logits = model.actor_logits(snapshot)
            distribution = torch.distributions.Categorical(logits=logits)
            active = snapshot["active"]
            action_counts.scatter_add_(1, actions, active.long())
            policy_entropy_sum += (distribution.entropy() * active).sum(dim=1)
            top_two = logits.topk(k=2, dim=-1).values
            policy_margin_sum += (
                (top_two[..., 0] - top_two[..., 1]) * active
            ).sum(dim=1)
            active_action_count += active.sum(dim=1)
        env.step([actions[:, index] for index in range(scenario.n_agents)])

        detected = scenario.target_detected.any(dim=1)
        visited = scenario.target_visited
        step_value = torch.full_like(detection_first, step + 1)
        detection_first = torch.where(
            detected & ~pre_detected & (detection_first < 0),
            step_value,
            detection_first,
        )
        visit_first = torch.where(
            visited & ~pre_visited & (visit_first < 0),
            step_value,
            visit_first,
        )
        newly_successful = visited.all(dim=-1) & (success_step < 0)
        success_step = torch.where(
            newly_successful,
            torch.full_like(success_step, step + 1),
            success_step,
        )
        entropy_trace[:, step + 1] = global_entropy(scenario)

    rows: list[dict[str, Any]] = []
    success = scenario.target_visited.all(dim=-1)
    for env_index in range(num_envs):
        detected_all = bool((detection_first[env_index] >= 0).all())
        is_success = bool(success[env_index])
        if is_success:
            response = (
                visit_first[env_index] - detection_first[env_index]
            ).float().mean().item()
        else:
            response = 0.0
        rows.append(
            {
                "episode_seed": seed,
                "success": int(is_success),
                "success_step": int(success_step[env_index].item()),
                "rescue_response_time": float(response),
                "last_target_detection_time": (
                    int(detection_first[env_index].max().item())
                    if detected_all
                    else -1
                ),
                "target_detection_steps": detection_first[
                    env_index
                ].cpu().tolist(),
                "target_rescue_steps": visit_first[env_index].cpu().tolist(),
                "global_entropy_trace": entropy_trace[
                    env_index
                ].cpu().tolist(),
                "policy_action_counts": action_counts[env_index].cpu().tolist(),
                "policy_active_action_count": int(
                    active_action_count[env_index].item()
                ),
                "policy_entropy_sum": float(policy_entropy_sum[env_index].item()),
                "policy_top2_logit_margin_sum": float(
                    policy_margin_sum[env_index].item()
                ),
                "detected_target_count": int((detection_first[env_index] >= 0).sum()),
                "rescued_target_count": int((visit_first[env_index] >= 0).sum()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_mappo_baseline_config(args.config)
    evaluation = config["evaluation"]
    device = args.device or evaluation["device"]
    episodes = args.episodes or int(evaluation["episodes"])
    batch_size = args.batch_size or int(evaluation["batch_size"])
    if batch_size != episodes:
        raise ValueError(
            "paired Chapter 1 evaluation requires batch_size == episodes; "
            "splitting a seed changes the initial-layout stream"
        )
    seed = args.seed if args.seed is not None else int(evaluation["seed"])
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(evaluation["output_dir"]).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    validate_checkpoint_metadata(checkpoint, config)
    model = model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows: list[dict[str, Any]] = []
    processed = 0
    while processed < episodes:
        current = min(batch_size, episodes - processed)
        rows.extend(
            evaluate_batch(
                config,
                model,
                num_envs=current,
                device=device,
                seed=seed + processed,
            )
        )
        processed += current
    action_counts = [
        sum(int(row["policy_action_counts"][index]) for row in rows)
        for index in range(5)
    ]
    active_count = sum(int(row["policy_active_action_count"]) for row in rows)
    entropy_sum = sum(float(row["policy_entropy_sum"]) for row in rows)
    margin_sum = sum(float(row["policy_top2_logit_margin_sum"]) for row in rows)
    diagnostics = {
        "episodes": episodes,
        "action_counts": action_counts,
        "action_fractions": [
            count / max(active_count, 1) for count in action_counts
        ],
        "noop_fraction": action_counts[0] / max(active_count, 1),
        "mean_policy_entropy": entropy_sum / max(active_count, 1),
        "mean_top2_logit_margin": margin_sum / max(active_count, 1),
        "mean_detected_target_count": sum(row["detected_target_count"] for row in rows) / episodes,
        "mean_rescued_target_count": sum(row["rescued_target_count"] for row in rows) / episodes,
        "all_targets_detected_rate": sum(row["detected_target_count"] == 3 for row in rows) / episodes,
        "all_targets_rescued_rate": sum(row["rescued_target_count"] == 3 for row in rows) / episodes,
    }

    horizon = int(config["_task"]["scenario"]["horizon"])
    physics_dt = float(config["_task"]["physics"]["dt"])
    metrics = build_fixed_evaluation_metrics(
        rows, seed=seed, horizon=horizon, physics_dt=physics_dt
    )
    metrics["baseline"] = {
        "name": "end_to_end_mappo",
        "checkpoint": str(checkpoint_path),
        "checkpoint_update": int(checkpoint["update"]),
        "task_config_sha256": config["_task_sha256"],
        "hidden_target_truth": False,
        "hierarchy_used": False,
        "reward_profile": config["_reward"]["profile"],
    }
    metrics_path = write_fixed_evaluation_metrics(
        metrics, output_dir / "evaluation_metrics.json"
    )
    (output_dir / "evaluation_episodes.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostics_path = output_dir / "policy_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(format_fixed_evaluation_metrics(metrics), flush=True)
    print(f"metrics={metrics_path}", flush=True)
    print(f"policy_diagnostics={diagnostics_path}", flush=True)


if __name__ == "__main__":
    main()
