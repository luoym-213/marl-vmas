#!/usr/bin/env python
"""Train the end-to-end MAPPO baseline on the frozen Chapter 1 SAR task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor

from comm_spread.mappo_baseline import (
    checkpoint_metadata,
    compute_gae,
    load_mappo_baseline_config,
    make_mappo_baseline_env,
    model_from_config,
    scenario_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    return parser.parse_args()


def detach_snapshot(snapshot: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().clone() for key, value in snapshot.items()}


def stack_snapshots(items: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def flatten_time_batch(snapshot: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        key: value.flatten(0, 1)
        for key, value in snapshot.items()
    }


def select_snapshot(snapshot: dict[str, Tensor], index: Tensor) -> dict[str, Tensor]:
    return {key: value[index] for key, value in snapshot.items()}


def main() -> None:
    args = parse_args()
    config = load_mappo_baseline_config(args.config)
    training = config["training"]
    seed = int(training["seed"])
    device = args.device or training["device"]
    num_envs = args.num_envs or int(training["num_envs"])
    rollout_steps = args.rollout_steps or int(training["rollout_steps"])
    updates = args.updates or int(training["updates"])
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path(training["output_dir"]).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_mappo_baseline_env(
        config, num_envs=num_envs, device=device, seed=seed
    )
    env.reset()
    scenario = env.scenario
    model = model_from_config(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        eps=1e-5,
    )

    gamma = float(training["gamma"])
    gae_lambda = float(training["gae_lambda"])
    clip = float(training["clip_epsilon"])
    value_coef = float(training["value_coefficient"])
    entropy_coef = float(training["entropy_coefficient"])
    max_grad_norm = float(training["max_grad_norm"])
    ppo_epochs = int(training["ppo_epochs"])
    minibatch_size = int(training["minibatch_size"])
    history: list[dict[str, Any]] = []

    for update in range(1, updates + 1):
        snapshot_buffer: list[dict[str, Tensor]] = []
        action_buffer: list[Tensor] = []
        log_prob_buffer: list[Tensor] = []
        value_buffer: list[Tensor] = []
        reward_buffer: list[Tensor] = []
        done_buffer: list[Tensor] = []
        component_buffer = {
            key: [] for key in scenario.direct_reward_components
        }
        action_counts = torch.zeros(5, dtype=torch.long, device=device)
        rollout_entropy_sum = torch.zeros((), device=device)
        rollout_margin_sum = torch.zeros((), device=device)
        rollout_active_count = torch.zeros((), device=device)

        for _ in range(rollout_steps):
            snapshot = scenario_snapshot(scenario)
            with torch.no_grad():
                actions, log_prob, value = model.act(
                    snapshot, deterministic=False
                )
                logits = model.actor_logits(snapshot)
                distribution = torch.distributions.Categorical(logits=logits)
                active = snapshot["active"]
                action_counts += torch.bincount(
                    actions[active], minlength=action_counts.numel()
                )
                rollout_entropy_sum += distribution.entropy()[active].sum()
                top_two = logits.topk(k=2, dim=-1).values
                rollout_margin_sum += (top_two[..., 0] - top_two[..., 1])[active].sum()
                rollout_active_count += active.sum()
            _, rewards, dones, _ = env.step(
                [actions[:, index] for index in range(scenario.n_agents)]
            )
            team_reward = torch.stack(rewards, dim=1).sum(dim=1)
            for key, component in scenario.direct_reward_components.items():
                component_buffer[key].append(
                    component.sum(dim=1).detach().clone()
                )

            snapshot_buffer.append(detach_snapshot(snapshot))
            action_buffer.append(actions.detach())
            log_prob_buffer.append(log_prob.detach())
            value_buffer.append(value.detach())
            reward_buffer.append(team_reward.detach())
            done_buffer.append(dones.bool().detach())

            if bool(dones.all()):
                env.reset()

        with torch.no_grad():
            if bool(done_buffer[-1].all()):
                bootstrap = torch.zeros(num_envs, device=device)
            else:
                bootstrap = model.value(scenario_snapshot(scenario))

        snapshots = flatten_time_batch(stack_snapshots(snapshot_buffer))
        actions = torch.stack(action_buffer).flatten(0, 1)
        old_log_prob = torch.stack(log_prob_buffer).flatten(0, 1)
        values = torch.stack(value_buffer)
        rewards = torch.stack(reward_buffer)
        dones = torch.stack(done_buffer)
        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            bootstrap,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        advantages = advantages.flatten()
        returns = returns.flatten()
        advantages = (
            advantages - advantages.mean()
        ) / advantages.std(unbiased=False).clamp_min(1e-8)

        transition_count = advantages.shape[0]
        actor_loss_value = 0.0
        critic_loss_value = 0.0
        entropy_value = 0.0
        optimization_steps = 0
        for _ in range(ppo_epochs):
            permutation = torch.randperm(transition_count, device=device)
            for start in range(0, transition_count, minibatch_size):
                index = permutation[start : start + minibatch_size]
                batch = select_snapshot(snapshots, index)
                logits = model.actor_logits(batch)
                distribution = torch.distributions.Categorical(logits=logits)
                new_log_prob = distribution.log_prob(actions[index])
                entropy = distribution.entropy()
                ratio = (new_log_prob - old_log_prob[index]).exp()
                advantage = advantages[index].unsqueeze(-1)
                unclipped = ratio * advantage
                clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * advantage
                active = batch["active"].float()
                actor_loss = -(
                    torch.minimum(unclipped, clipped) * active
                ).sum() / active.sum().clamp_min(1.0)
                entropy_mean = (
                    entropy * active
                ).sum() / active.sum().clamp_min(1.0)
                critic_loss = 0.5 * (
                    model.value(batch) - returns[index]
                ).square().mean()
                loss = (
                    actor_loss
                    + value_coef * critic_loss
                    - entropy_coef * entropy_mean
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                optimizer.step()

                actor_loss_value += float(actor_loss.detach())
                critic_loss_value += float(critic_loss.detach())
                entropy_value += float(entropy_mean.detach())
                optimization_steps += 1
        action_total = action_counts.sum().clamp_min(1)
        active_total = rollout_active_count.clamp_min(1)
        reward_component_means = {
            key: float(torch.stack(values).mean().cpu())
            for key, values in component_buffer.items()
        }
        rollout_policy = {
            "action_counts": action_counts.cpu().tolist(),
            "action_fractions": (
                action_counts.float() / action_total
            ).cpu().tolist(),
            "mean_entropy": float((rollout_entropy_sum / active_total).cpu()),
            "mean_top2_logit_margin": float((rollout_margin_sum / active_total).cpu()),
        }

        record = {
            "update": update,
            "environment_steps": update * rollout_steps * num_envs,
            "mean_team_reward": float(rewards.mean().cpu()),
            "actor_loss": actor_loss_value / optimization_steps,
            "critic_loss": critic_loss_value / optimization_steps,
            "entropy": entropy_value / optimization_steps,
            "reward_components": reward_component_means,
            "rollout_policy": rollout_policy,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "environment_steps": record["environment_steps"],
            "metadata": checkpoint_metadata(config),
            "model_config": config["baseline"]["model"],
        }
        torch.save(checkpoint, output_dir / "checkpoint_latest.pt")
        save_interval = int(training["checkpoint_interval"])
        if update % save_interval == 0 or update == updates:
            torch.save(
                checkpoint,
                output_dir / f"checkpoint_{update:06d}.pt",
            )

    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "resolved_baseline_config.json").write_text(
        json.dumps(
            {
                "source": config["_source"],
                "task_path": config["_task_path"],
                "task_config_sha256": config["_task_sha256"],
                "baseline_config_sha256": config["_config_sha256"],
                "baseline": config["baseline"],
                "training": training,
                "reward": config["_reward"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"checkpoint={output_dir / 'checkpoint_latest.pt'}", flush=True)


if __name__ == "__main__":
    main()
