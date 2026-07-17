"""Train SAR high-level policy with standalone PyTorch PPO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import torch
import torch.nn.functional as F

from comm_spread.async_smdp import AsyncSMDPCollector, HighLevelTransition
from comm_spread.env_factory import make_sar_env
from comm_spread.high_level_policy import HGSARActorCriticPolicy
from comm_spread.low_level_policy import BenchMARLLowLevelPolicy

try:
    torch.backends.nnpack.enabled = False
except AttributeError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/sar_low_full/low_safe_0.06/checkpoints/checkpoint_50040000.pt"
)

CSV_FIELDS = [
    "update",
    "transitions",
    "high_reward_mean",
    "high_reward_sum",
    "duration_mean",
    "detected_targets_mean",
    "visited_targets_mean",
    "retired_agents_mean",
    "success_rate",
    "entropy_reduction_mean",
    "actor_explore_actions",
    "actor_target_actions",
    "actor_invalid_actions",
    "actor_nonfinite_logits",
    "actor_valid_target_actions",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "intent_loss",
    "intent_top1_accuracy",
    "intent_none_accuracy",
    "intent_rescue_accuracy",
    "intent_simultaneous_accuracy",
    "intent_frequency_baseline_accuracy",
    "coordination_pressure_mean",
    "coordination_pressure_max",
    "coordination_suppressed_target_ratio",
    "coordination_logit_reduction_mean",
    "coordination_target_top1_changes",
    "coordination_gate_near_zero_ratio",
    "coordination_gate_saturated_ratio",
    "coordination_utility_abs_max",
    "rescue_probability_base_mean",
    "rescue_probability_final_mean",
    "intention_predicted_none_ratio",
    "intention_teacher_none_ratio",
    "intention_teacher_samples",
    "intention_rescue_samples",
    "intention_simultaneous_rescue_samples",
    "intention_rollout_kl_loss",
    "high_probability_target_conflicts",
    "mutual_yield_events",
    "phase_rescue_probability_mean",
    "phase_rescue_choice_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-level-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--enable-low-level-goal-fallback", action="store_true")
    parser.add_argument("--low-level-goal-near-threshold", type=float, default=0.25)
    parser.add_argument("--low-level-fallback-gain", type=float, default=2.0)
    parser.add_argument("--low-level-fallback-stagnation-steps", type=int, default=5)
    parser.add_argument("--low-level-fallback-progress-epsilon", type=float, default=0.005)
    parser.add_argument(
        "--low-level-controller",
        choices=("checkpoint", "proportional"),
        default="checkpoint",
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--low-level-steps-per-batch", type=int, default=400)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-folder", type=Path, default=Path("outputs/sar_high_ppo"))
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--sensor-radius", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--retire-on-rescue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-features", action="store_true")
    parser.add_argument("--target-assignment-features", action="store_true")
    parser.add_argument("--staged-rescue", action="store_true")
    parser.add_argument("--rescue-detected-threshold", type=int, default=2)
    parser.add_argument("--rescue-entropy-threshold", type=float, default=None)
    parser.add_argument("--dynamic-rescue-release", action="store_true")
    parser.add_argument("--dynamic-release-min-searchers", type=int, default=2)
    parser.add_argument("--dynamic-release-entropy-ratio-threshold", type=float, default=0.58)
    parser.add_argument("--dynamic-release-entropy-rate-threshold", type=float, default=0.0015)
    parser.add_argument("--dynamic-release-min-stagnation-step", type=int, default=25)
    parser.add_argument("--dynamic-release-search-steps-per-target", type=float, default=22.0)
    parser.add_argument("--dynamic-release-speed-per-step", type=float, default=0.035)
    parser.add_argument("--dynamic-release-time-margin", type=float, default=8.0)
    parser.add_argument("--dynamic-release-max-new-agents-per-event", type=int, default=1)
    parser.add_argument("--dynamic-rescue-only-after-all-detected", action="store_true")
    parser.add_argument("--redecide-on-detection-change", action="store_true")
    parser.add_argument("--redecide-on-assignment-change", action="store_true")
    parser.add_argument("--enable-finder-first-cascade", action="store_true")
    parser.add_argument(
        "--finder-cascade-mode",
        choices=("finder_only", "immediate", "one_event"),
        default="immediate",
    )
    parser.add_argument("--early-rescue-penalty", type=float, default=0.0)
    parser.add_argument("--early-rescue-detected-threshold", type=int, default=3)
    parser.add_argument("--search-capacity-discovery-bonus", type=float, default=0.0)
    parser.add_argument("--discovery-active-agents-threshold", type=int, default=2)
    parser.add_argument("--all-targets-detected-bonus", type=float, default=0.0)
    parser.add_argument(
        "--all-targets-detected-bonus-requires-no-rescue",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rescue-phase-explore-penalty", type=float, default=0.0)
    parser.add_argument("--rescue-phase-detected-threshold", type=int, default=2)
    parser.add_argument("--rescue-phase-min-search-agents", type=int, default=1)
    parser.add_argument("--unique-rescue-assignment-bonus", type=float, default=0.0)
    parser.add_argument("--duplicate-rescue-assignment-penalty", type=float, default=0.0)
    parser.add_argument("--detected-unassigned-target-penalty", type=float, default=0.0)
    parser.add_argument("--coordinated-target-selection", action="store_true")
    parser.add_argument("--enable-commitment-aware-actor", action="store_true")
    parser.add_argument("--enable-phase-policy", action="store_true")
    parser.add_argument("--phase-initial-rescue-logit", type=float, default=-1.5)
    parser.add_argument("--rescue-distance-logit-scale", type=float, default=0.0)
    parser.add_argument(
        "--enable-teammate-intention-coordination",
        action="store_true",
    )
    parser.add_argument("--intent-loss-coef", type=float, default=0.1)
    parser.add_argument("--coordination-beta", type=float, default=0.25)
    parser.add_argument("--coordination-margin", type=float, default=0.1)
    parser.add_argument("--coordination-temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.low_level_controller == "proportional" and args.enable_low_level_goal_fallback:
        raise ValueError(
            "low-level goal fallback applies only to the checkpoint controller"
        )
    if args.dynamic_rescue_release and args.staged_rescue:
        raise ValueError("dynamic rescue release cannot be combined with staged rescue")
    if args.dynamic_rescue_release and args.coordinated_target_selection:
        raise ValueError(
            "dynamic rescue release cannot be combined with sequential hard masking"
        )
    if args.enable_finder_first_cascade and not args.dynamic_rescue_release:
        raise ValueError("finder-first cascade requires --dynamic-rescue-release")
    if args.enable_phase_policy and not args.progress_features:
        raise ValueError("phase policy requires --progress-features")
    workspace_root = Path(__file__).resolve().parents[3]
    try:
        args.git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=workspace_root, text=True
        ).strip()
        git_diff = subprocess.check_output(
            ["git", "diff", "--binary"], cwd=workspace_root
        )
        args.git_diff_sha256 = hashlib.sha256(git_diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        args.git_head = "unknown"
        args.git_diff_sha256 = "unknown"
    args.command = [sys.executable, *sys.argv]
    learned_coordination_modes = int(args.enable_commitment_aware_actor) + int(
        args.enable_teammate_intention_coordination
    )
    if learned_coordination_modes > 1:
        raise ValueError(
            "commitment-aware actor and teammate intention coordination "
            "cannot be enabled together"
        )
    if args.coordinated_target_selection and learned_coordination_modes:
        raise ValueError(
            "sequential hard mask cannot be combined with a learned "
            "coordination experiment"
        )
    torch.manual_seed(args.seed)
    args.save_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.save_folder / "checkpoints"
    scalars_dir = args.save_folder / "scalars"
    texts_dir = args.save_folder / "texts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scalars_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)
    config = serialize_config(args)
    (texts_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    csv_path = scalars_dir / "train.csv"
    run_summary_path = texts_dir / "run_summary.txt"
    write_run_summary(
        run_summary_path,
        args=args,
        status="running",
        final_update=0,
        last_metrics={},
        latest_checkpoint=checkpoint_dir / "latest.pt",
    )

    env = make_sar_env(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        mode="high",
        emit_info=False,
        enable_high_level_state=True,
        enable_rrt_candidates=True,
        auto_resample_goals=False,
        sensor_radius=args.sensor_radius,
        max_steps=args.max_steps,
        retire_on_rescue=args.retire_on_rescue,
        high_level_progress_features=args.progress_features,
        target_assignment_features=args.target_assignment_features,
        staged_rescue=args.staged_rescue,
        rescue_detected_threshold=args.rescue_detected_threshold,
        rescue_entropy_threshold=args.rescue_entropy_threshold,
        dynamic_rescue_release=args.dynamic_rescue_release,
        dynamic_release_min_searchers=args.dynamic_release_min_searchers,
        dynamic_release_entropy_ratio_threshold=(
            args.dynamic_release_entropy_ratio_threshold
        ),
        dynamic_release_entropy_rate_threshold=(
            args.dynamic_release_entropy_rate_threshold
        ),
        dynamic_release_min_stagnation_step=(
            args.dynamic_release_min_stagnation_step
        ),
        dynamic_release_search_steps_per_target=(
            args.dynamic_release_search_steps_per_target
        ),
        dynamic_release_speed_per_step=args.dynamic_release_speed_per_step,
        dynamic_release_time_margin=args.dynamic_release_time_margin,
        dynamic_release_max_new_agents_per_event=(
            args.dynamic_release_max_new_agents_per_event
        ),
        dynamic_rescue_only_after_all_detected=(
            args.dynamic_rescue_only_after_all_detected
        ),
        redecide_on_detection_change=args.redecide_on_detection_change,
        redecide_on_assignment_change=args.redecide_on_assignment_change,
        enable_finder_first_cascade=args.enable_finder_first_cascade,
        finder_cascade_mode=args.finder_cascade_mode,
        early_rescue_penalty=args.early_rescue_penalty,
        early_rescue_detected_threshold=args.early_rescue_detected_threshold,
        search_capacity_discovery_bonus=args.search_capacity_discovery_bonus,
        discovery_active_agents_threshold=args.discovery_active_agents_threshold,
        all_targets_detected_bonus=args.all_targets_detected_bonus,
        all_targets_detected_bonus_requires_no_rescue=args.all_targets_detected_bonus_requires_no_rescue,
        rescue_phase_explore_penalty=args.rescue_phase_explore_penalty,
        rescue_phase_detected_threshold=args.rescue_phase_detected_threshold,
        rescue_phase_min_search_agents=args.rescue_phase_min_search_agents,
        unique_rescue_assignment_bonus=args.unique_rescue_assignment_bonus,
        duplicate_rescue_assignment_penalty=args.duplicate_rescue_assignment_penalty,
        detected_unassigned_target_penalty=args.detected_unassigned_target_penalty,
    )
    scenario = env.scenario
    low_policy = (
        BenchMARLLowLevelPolicy(
            args.low_level_checkpoint,
            device=args.device,
            seed=args.seed,
            deterministic=True,
            max_steps=scenario.max_steps,
            enable_goal_reaching_fallback=args.enable_low_level_goal_fallback,
            goal_near_threshold=args.low_level_goal_near_threshold,
            fallback_proportional_gain=args.low_level_fallback_gain,
            fallback_stagnation_steps=args.low_level_fallback_stagnation_steps,
            fallback_progress_epsilon=args.low_level_fallback_progress_epsilon,
        )
        if args.low_level_controller == "checkpoint"
        else None
    )
    low_level_source = (
        low_policy.source if low_policy is not None else "proportional_gain:2.0"
    )
    high_policy = HGSARActorCriticPolicy(
        n_agents=scenario.n_agents,
        rrt_top_k=scenario.rrt_top_k,
        device=args.device,
        deterministic=False,
        ego_features=9 if args.progress_features else 5,
        target_features=7 if args.target_assignment_features else 4,
        enable_commitment_aware_actor=args.enable_commitment_aware_actor,
        enable_phase_policy=args.enable_phase_policy,
        phase_initial_rescue_logit=args.phase_initial_rescue_logit,
        rescue_distance_logit_scale=args.rescue_distance_logit_scale,
        enable_teammate_intention_coordination=(
            args.enable_teammate_intention_coordination
        ),
        coordination_beta=args.coordination_beta,
        coordination_margin=args.coordination_margin,
        coordination_temperature=args.coordination_temperature,
    )
    optimizer = torch.optim.Adam(high_policy.parameters(), lr=args.lr)
    start_update = 0
    if args.resume_checkpoint is not None:
        start_update = load_checkpoint(args.resume_checkpoint, high_policy, optimizer, args.device)

    collector = AsyncSMDPCollector(
        env,
        high_level_policy=high_policy,
        low_level_policy=low_policy,
        coordinated_target_selection=args.coordinated_target_selection,
    )

    try:
        for update in range(start_update + 1, args.updates + 1):
            high_policy.reset_stats()
            collector.reset()
            if update == start_update + 1:
                initial_agents = torch.stack(
                    [agent.state.pos for agent in scenario.world.agents], dim=1
                ).detach().cpu().contiguous()
                initial_targets = torch.stack(
                    [target.state.pos for target in scenario.targets], dim=1
                ).detach().cpu().contiguous()
                initial_layout = torch.cat([initial_agents, initial_targets], dim=1)
                args.initial_layout_sha256 = hashlib.sha256(
                    initial_layout.numpy().tobytes()
                ).hexdigest()
                args.initial_layout_checksum = float(initial_layout.sum())
                (texts_dir / "config.json").write_text(
                    json.dumps(serialize_config(args), indent=2),
                    encoding="utf-8",
                )
            initial_entropy = entropy_total(scenario).detach().clone()
            transitions = collector.rollout(args.low_level_steps_per_batch)
            final_entropy = entropy_total(scenario).detach().clone()
            if not transitions:
                print(f"[update {update}] transitions=0, skipping")
                continue

            batch = transitions_to_batch(transitions, device=args.device)
            add_gae(batch, gamma=args.gamma, gae_lambda=args.gae_lambda)
            update_metrics = ppo_update(
                policy=high_policy,
                optimizer=optimizer,
                batch=batch,
                ppo_epochs=args.ppo_epochs,
                minibatch_size=args.minibatch_size,
                clip_eps=args.clip_eps,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                max_grad_norm=args.max_grad_norm,
                intent_loss_coef=args.intent_loss_coef,
            )
            rollout_metrics = summarize_rollout(
                transitions=transitions,
                scenario=scenario,
                initial_entropy=initial_entropy,
                final_entropy=final_entropy,
                low_level_source=low_level_source,
                actor_metrics=high_policy.metrics(),
            )
            print_update(update, rollout_metrics | update_metrics)
            all_metrics = {"update": update} | rollout_metrics | update_metrics
            append_train_csv(csv_path, all_metrics)

            if update % args.save_interval == 0 or update == args.updates:
                save_checkpoint(checkpoint_dir / f"checkpoint_{update}.pt", high_policy, optimizer, update, args)
            save_checkpoint(checkpoint_dir / "latest.pt", high_policy, optimizer, update, args)
            write_run_summary(
                run_summary_path,
                args=args,
                status="running" if update < args.updates else "complete",
                final_update=update,
                last_metrics=all_metrics,
                latest_checkpoint=checkpoint_dir / "latest.pt",
            )
    finally:
        if low_policy is not None:
            low_policy.close()


def transitions_to_batch(
    transitions: list[HighLevelTransition],
    *,
    device: str,
) -> dict[str, torch.Tensor]:
    def stack(name: str) -> torch.Tensor:
        return torch.stack([getattr(transition, name) for transition in transitions]).to(device)

    return {
        "ego_node": stack("ego_node").float(),
        "teammate_nodes": stack("teammate_nodes").float(),
        "teammate_context_nodes": stack("teammate_context_nodes").float(),
        "teammate_mask": stack("teammate_mask").float(),
        "explore_nodes": stack("explore_nodes").float(),
        "target_nodes": stack("target_nodes").float(),
        "target_mask": stack("target_mask").bool(),
        "intent_target_mask": stack("intent_target_mask").bool(),
        "map_channels": stack("map_channels").float(),
        "global_map_channels": stack("global_map_channels").float(),
        "global_agent_nodes": stack("global_agent_nodes").float(),
        "explore_edges": stack("explore_edges").float(),
        "target_edges": stack("target_edges").float(),
        "action_mask": stack("action_mask").bool(),
        "coord_agent_nodes": stack("coord_agent_nodes").float(),
        "agent_active_mask": stack("agent_active_mask").bool(),
        "agent_decision_mask": stack("agent_decision_mask").bool(),
        "agent_commitment_target_id": stack(
            "agent_commitment_target_id"
        ).long(),
        "target_ids": stack("target_ids").long(),
        "intent_teacher": stack("intent_teacher").float(),
        "intent_teacher_mask": stack("intent_teacher_mask").bool(),
        "intent_teacher_simultaneous_mask": stack(
            "intent_teacher_simultaneous_mask"
        ).bool(),
        "action": stack("action").long().view(-1),
        "old_log_prob": stack("log_prob").float().view(-1, 1),
        "old_value": stack("value").float().view(-1, 1),
        "reward": stack("reward").float().view(-1, 1),
        "done": stack("done").float().view(-1, 1),
        "duration": torch.tensor(
            [transition.duration for transition in transitions],
            dtype=torch.float32,
            device=device,
        ).view(-1, 1),
        "env_id": torch.tensor(
            [transition.env_id for transition in transitions],
            dtype=torch.long,
            device=device,
        ),
        "agent_id": torch.tensor(
            [transition.agent_id for transition in transitions],
            dtype=torch.long,
            device=device,
        ),
        "decision_start_t": torch.tensor(
            [transition.decision_start_t for transition in transitions],
            dtype=torch.long,
            device=device,
        ),
    }


def add_gae(batch: dict[str, torch.Tensor], *, gamma: float, gae_lambda: float) -> None:
    n = batch["reward"].shape[0]
    advantages = torch.zeros(n, 1, device=batch["reward"].device)
    returns = torch.zeros_like(advantages)
    env_ids = batch["env_id"]
    agent_ids = batch["agent_id"]
    starts = batch["decision_start_t"]

    for env_id in env_ids.unique():
        for agent_id in agent_ids[env_ids == env_id].unique():
            mask = (env_ids == env_id) & (agent_ids == agent_id)
            indices = torch.nonzero(mask, as_tuple=False).view(-1)
            if indices.numel() == 0:
                continue
            indices = indices[starts[indices].argsort()]
            next_value = torch.zeros(1, 1, device=batch["reward"].device)
            next_advantage = torch.zeros(1, 1, device=batch["reward"].device)
            for idx in reversed(indices.tolist()):
                done_mask = 1.0 - batch["done"][idx : idx + 1]
                discount = gamma ** batch["duration"][idx : idx + 1]
                delta = (
                    batch["reward"][idx : idx + 1]
                    + discount * next_value * done_mask
                    - batch["old_value"][idx : idx + 1]
                )
                advantage = delta + discount * gae_lambda * next_advantage * done_mask
                advantages[idx : idx + 1] = advantage
                returns[idx : idx + 1] = advantage + batch["old_value"][idx : idx + 1]
                next_value = batch["old_value"][idx : idx + 1]
                next_advantage = advantage

    batch["advantage"] = advantages
    batch["return"] = returns


def ppo_update(
    *,
    policy: HGSARActorCriticPolicy,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    ppo_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    intent_loss_coef: float,
) -> dict[str, float]:
    policy.train()
    n = batch["action"].shape[0]
    advantages = batch["advantage"]
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
    metrics: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
        "intent_loss": [],
        "intent_top1_accuracy": [],
        "intent_none_accuracy": [],
        "intent_rescue_accuracy": [],
        "intent_simultaneous_accuracy": [],
        "intent_frequency_baseline_accuracy": [],
    }
    for _ in range(ppo_epochs):
        permutation = torch.randperm(n, device=batch["action"].device)
        for start in range(0, n, minibatch_size):
            mb_indices = permutation[start : start + minibatch_size]
            minibatch = index_batch(batch, mb_indices)
            mb_advantages = advantages[mb_indices]
            eval_out = policy.evaluate_actions(minibatch)
            ratio = torch.exp(eval_out["log_prob"] - minibatch["old_log_prob"])
            unclipped = ratio * mb_advantages
            clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(eval_out["value"], minibatch["return"])
            entropy = eval_out["entropy"].mean()
            intent_loss = eval_out["intent_loss"]
            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
                + intent_loss_coef * intent_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = eval_out["log_prob"] - minibatch["old_log_prob"]
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()
            metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
            metrics["value_loss"].append(float(value_loss.detach().cpu()))
            metrics["entropy"].append(float(entropy.detach().cpu()))
            metrics["approx_kl"].append(float(approx_kl.detach().cpu()))
            metrics["clip_fraction"].append(float(clip_fraction.detach().cpu()))
            for key in [
                "intent_loss",
                "intent_top1_accuracy",
                "intent_none_accuracy",
                "intent_rescue_accuracy",
                "intent_simultaneous_accuracy",
                "intent_frequency_baseline_accuracy",
            ]:
                metrics[key].append(float(eval_out[key].detach().cpu()))

    return {key: float(torch.tensor(values).mean()) for key, values in metrics.items()}


def index_batch(batch: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    indexed = {}
    batch_size = batch["action"].shape[0]
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.shape[:1] == (batch_size,):
            indexed[key] = value[indices]
        else:
            indexed[key] = value
    return indexed


def summarize_rollout(
    *,
    transitions: list[HighLevelTransition],
    scenario,
    initial_entropy: torch.Tensor,
    final_entropy: torch.Tensor,
    low_level_source: str,
    actor_metrics: dict[str, int],
) -> dict[str, float | int | str]:
    durations = torch.tensor([transition.duration for transition in transitions], dtype=torch.float32)
    rewards = torch.stack([transition.reward.float() for transition in transitions])
    detected = scenario.target_detected.any(dim=1).float().sum(dim=-1)
    visited = scenario.target_visited.float().sum(dim=-1)
    retired = (~scenario.active_agents).float().sum(dim=-1)
    entropy_reduction = initial_entropy - final_entropy
    metrics: dict[str, float | int | str] = {
        "low_level_action_source": low_level_source,
        "transitions": len(transitions),
        "duration_mean": float(durations.mean()),
        "high_reward_sum": float(rewards.sum()),
        "high_reward_mean": float(rewards.mean()),
        "detected_targets_mean": float(detected.mean().cpu()),
        "visited_targets_mean": float(visited.mean().cpu()),
        "retired_agents_mean": float(retired.mean().cpu()),
        "success_rate": float(scenario.success.float().mean().cpu()),
        "entropy_reduction_mean": float(entropy_reduction.mean().cpu()),
    }
    metrics.update(actor_metrics)
    return metrics


def print_update(update: int, metrics: dict[str, float | int | str]) -> None:
    print(
        "\n".join(
            [
                f"[high_ppo] update {update:04d}",
                (
                    "  rollout | "
                    f"transitions {fmt_int(metrics, 'transitions')}  "
                    f"duration {fmt_float(metrics, 'duration_mean')}  "
                    f"reward {fmt_float(metrics, 'high_reward_mean')} "
                    f"(sum {fmt_float(metrics, 'high_reward_sum')})  "
                    f"success {fmt_float(metrics, 'success_rate')}"
                ),
                (
                    "  targets | "
                    f"detected {fmt_float(metrics, 'detected_targets_mean')}  "
                    f"visited {fmt_float(metrics, 'visited_targets_mean')}  "
                    f"retired {fmt_float(metrics, 'retired_agents_mean')}  "
                    f"entropy_gain {fmt_float(metrics, 'entropy_reduction_mean')}"
                ),
                (
                    "  actor   | "
                    f"explore {fmt_int(metrics, 'actor_explore_actions')}  "
                    f"target {fmt_int(metrics, 'actor_target_actions')} "
                    f"(valid {fmt_int(metrics, 'actor_valid_target_actions')})  "
                    f"invalid {fmt_int(metrics, 'actor_invalid_actions')}  "
                    f"nonfinite {fmt_int(metrics, 'actor_nonfinite_logits')}"
                ),
                (
                    "  ppo     | "
                    f"policy {fmt_float(metrics, 'policy_loss')}  "
                    f"value {fmt_float(metrics, 'value_loss')}  "
                    f"entropy {fmt_float(metrics, 'entropy')}  "
                    f"kl {fmt_float(metrics, 'approx_kl')}  "
                    f"clip {fmt_float(metrics, 'clip_fraction')}"
                ),
                (
                    "  intent  | "
                    f"loss {fmt_float(metrics, 'intent_loss')}  "
                    f"top1 {fmt_float(metrics, 'intent_top1_accuracy')}  "
                    f"none {fmt_float(metrics, 'intent_none_accuracy')}  "
                    f"rescue {fmt_float(metrics, 'intent_rescue_accuracy')}  "
                    f"sim {fmt_float(metrics, 'intent_simultaneous_accuracy')}"
                    f"  freq {fmt_float(metrics, 'intent_frequency_baseline_accuracy')}"
                ),
                (
                    "  coord   | "
                    f"pressure {fmt_float(metrics, 'coordination_pressure_mean')} "
                    f"(max {fmt_float(metrics, 'coordination_pressure_max')})  "
                    f"reduction {fmt_float(metrics, 'coordination_logit_reduction_mean')}  "
                    f"suppressed {fmt_float(metrics, 'coordination_suppressed_target_ratio')}  "
                    f"mutual_yield {fmt_int(metrics, 'mutual_yield_events')}"
                ),
            ]
        ),
        flush=True,
    )


def fmt_float(metrics: dict[str, float | int | str], key: str) -> str:
    value = metrics.get(key, 0.0)
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return f"{float(value):.4f}"
    return str(value)


def fmt_int(metrics: dict[str, float | int | str], key: str) -> str:
    value = metrics.get(key, 0)
    if isinstance(value, float):
        return str(int(round(value)))
    if isinstance(value, int):
        return str(value)
    return str(value)


def entropy_total(scenario) -> torch.Tensor:
    return scenario._compute_entropy(scenario.belief_maps).sum(dim=(-1, -2))


def save_checkpoint(
    path: Path,
    policy: HGSARActorCriticPolicy,
    optimizer: torch.optim.Optimizer,
    update: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update": update,
            "config": serialize_config(args),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    policy: HGSARActorCriticPolicy,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> int:
    checkpoint: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("update", 0))


def serialize_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    return config


def append_train_csv(path: Path, metrics: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: metrics.get(field, "") for field in CSV_FIELDS})


def write_run_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    status: str,
    final_update: int,
    last_metrics: dict[str, float | int | str],
    latest_checkpoint: Path,
) -> None:
    lines = [
        "SAR High-Level PPO Run Summary",
        f"status: {status}",
        f"updated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"command: {' '.join(sys.argv)}",
        f"low_level_checkpoint: {args.low_level_checkpoint}",
        f"device: {args.device}",
        f"num_envs: {args.num_envs}",
        f"low_level_steps_per_batch: {args.low_level_steps_per_batch}",
        f"updates: {args.updates}",
        f"ppo_epochs: {args.ppo_epochs}",
        f"minibatch_size: {args.minibatch_size}",
        f"gamma: {args.gamma}",
        f"gae_lambda: {args.gae_lambda}",
        f"clip_eps: {args.clip_eps}",
        f"entropy_coef: {args.entropy_coef}",
        f"value_coef: {args.value_coef}",
        f"lr: {args.lr}",
        f"max_grad_norm: {args.max_grad_norm}",
        f"coordinated_target_selection: {args.coordinated_target_selection}",
        f"enable_commitment_aware_actor: {args.enable_commitment_aware_actor}",
        f"enable_phase_policy: {args.enable_phase_policy}",
        f"phase_initial_rescue_logit: {args.phase_initial_rescue_logit}",
        f"rescue_distance_logit_scale: {args.rescue_distance_logit_scale}",
        f"redecide_on_detection_change: {args.redecide_on_detection_change}",
        f"redecide_on_assignment_change: {args.redecide_on_assignment_change}",
        f"enable_finder_first_cascade: {args.enable_finder_first_cascade}",
        f"finder_cascade_mode: {args.finder_cascade_mode}",
        f"dynamic_release_max_new_agents_per_event: {args.dynamic_release_max_new_agents_per_event}",
        f"enable_teammate_intention_coordination: {args.enable_teammate_intention_coordination}",
        f"intent_loss_coef: {args.intent_loss_coef}",
        f"coordination_beta: {args.coordination_beta}",
        f"coordination_margin: {args.coordination_margin}",
        f"coordination_temperature: {args.coordination_temperature}",
        f"final_update: {final_update}",
        f"latest_checkpoint: {latest_checkpoint}",
        "",
        "last_metrics:",
    ]
    if last_metrics:
        for key in CSV_FIELDS:
            if key in last_metrics:
                lines.append(f"  {key}: {last_metrics[key]}")
    else:
        lines.append("  none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
