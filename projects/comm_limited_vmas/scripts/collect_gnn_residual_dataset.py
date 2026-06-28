"""Collect receiver-centered graph samples for GNN residual estimators."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

try:
    from torchrl.envs.libs.vmas import VmasEnv
except ImportError:
    from torchrl.envs import VmasEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

CONFIG_ROOT = PROJECT_ROOT / "configs"

from comm_limited_vmas.scenarios.comm_hidden_goal_navigation import Scenario
from comm_limited_vmas.scripts.collect_estimator_dataset import (
    _num_envs,
    clamp_with_norm,
    current_agent_states,
    random_action,
    scripted_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect graph-level supervised samples for GNN residual estimators."
    )
    parser.add_argument("--task", default="comm_hidden_goal_navigation")
    parser.add_argument("--comm", default="radius_delay_dropout_mean8_dropout03")
    parser.add_argument("--estimator-config", default="gnn_residual")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--policy",
        choices=["scripted", "random", "checkpoint"],
        default="scripted",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--policy-estimator", default="stale")
    parser.add_argument("--algorithm", default="mappo")
    parser.add_argument("--gain", type=float, default=2.0)
    parser.add_argument("--action-noise", type=float, default=0.15)
    parser.add_argument(
        "--include-zero-aoi",
        action="store_true",
        help="Keep graphs where every teammate message is fresh.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task != "comm_hidden_goal_navigation":
        raise ValueError("Only comm_hidden_goal_navigation is supported")

    dataset = collect_dataset(args)
    output_path = args.output or default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)

    summary = dataset_summary(dataset)
    print(f"Wrote GNN residual dataset: {output_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


def collect_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.policy == "checkpoint":
        return collect_with_checkpoint_policy(args)
    return collect_with_manual_policy(args)


def collect_with_manual_policy(args: argparse.Namespace) -> dict[str, Any]:
    task_config = load_yaml("task", args.task)
    comm_config = load_yaml("comm", args.comm)
    estimator_config = load_yaml("estimator", args.estimator_config)
    max_steps = int(args.max_steps or task_config["max_steps"])
    num_envs = int(args.num_envs or args.episodes)

    env_kwargs = dict(task_config)
    env_kwargs.pop("scenario", None)
    continuous_actions = bool(env_kwargs.pop("continuous_actions", True))
    env = VmasEnv(
        scenario=Scenario(
            comm_config=comm_config,
            estimator_config={"name": "stale"},
        ),
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        categorical_actions=not continuous_actions,
        clamp_actions=True,
        seed=args.seed,
        device=args.device,
        **env_kwargs,
    )

    try:
        td = env.reset()
        collector = GraphSampleCollector(env._env, include_zero_aoi=args.include_zero_aoi)
        done = torch.zeros(num_envs, dtype=torch.bool, device=td.device)
        collector.collect(step=0, valid_env_mask=~done)

        generator = torch.Generator(device=args.device).manual_seed(args.seed + 17)
        for step in range(1, max_steps + 1):
            if args.policy == "random":
                action = random_action(td[("agents", "observation")], generator)
            else:
                action = scripted_action(td[("agents", "observation")], args.gain)
                if args.action_noise > 0:
                    noise = torch.randn(
                        action.shape,
                        generator=generator,
                        dtype=action.dtype,
                        device=action.device,
                    )
                    action = clamp_with_norm(action + args.action_noise * noise, 1.0)

            td.set(("agents", "action"), action)
            td = env.step(td)["next"]
            done = done | td["done"].squeeze(-1).bool()
            collector.collect(step=step, valid_env_mask=~done)
            if done.all():
                break

        return build_dataset_dict(args, task_config, comm_config, estimator_config, collector)
    finally:
        env.close()


def collect_with_checkpoint_policy(args: argparse.Namespace) -> dict[str, Any]:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required when --policy checkpoint")

    from comm_limited_vmas.scripts.eval import (
        build_experiment,
        load_checkpoint_weights,
    )

    task_config = load_yaml("task", args.task)
    comm_config = load_yaml("comm", args.comm)
    estimator_config = load_yaml("estimator", args.estimator_config)
    max_steps = int(args.max_steps or task_config["max_steps"])
    experiment = build_experiment(
        task_name=args.task,
        algorithm_name=args.algorithm,
        comm_name=args.comm,
        estimator_name=args.policy_estimator,
        episodes=int(args.num_envs or args.episodes),
        device=args.device,
        seed=args.seed,
    )
    checkpoint = args.checkpoint.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()

    try:
        load_checkpoint_weights(experiment, checkpoint, args.device)
        env = experiment.test_env
        td = env.reset()
        done = torch.zeros(_num_envs(env), dtype=torch.bool, device=td.device)
        collector = GraphSampleCollector(env._env, include_zero_aoi=args.include_zero_aoi)
        collector.collect(step=0, valid_env_mask=~done)

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for step in range(1, max_steps + 1):
                td = experiment.policy(td)
                step_td = env.step(td)
                td = step_mdp(step_td)
                done = done | td["done"].squeeze(-1).bool()
                collector.collect(step=step, valid_env_mask=~done)
                if done.all():
                    break

        return build_dataset_dict(args, task_config, comm_config, estimator_config, collector)
    finally:
        experiment.close()


class GraphSampleCollector:
    def __init__(self, native_env: Any, include_zero_aoi: bool):
        self.native_env = native_env
        self.include_zero_aoi = bool(include_zero_aoi)
        self.chunks: dict[str, list[Tensor]] = {
            "cached_states": [],
            "true_states": [],
            "aoi": [],
            "comm_mask": [],
            "receiver": [],
            "env_index": [],
            "episode_step": [],
        }

    def collect(self, step: int, valid_env_mask: Tensor) -> None:
        scenario = self.native_env.scenario
        comm_manager = scenario._comm_manager
        true_states = current_agent_states(self.native_env.world)
        _, n_agents, _ = true_states.shape
        valid_env_mask = valid_env_mask.to(device=true_states.device)
        peer_template = ~torch.eye(n_agents, dtype=torch.bool, device=true_states.device)

        for receiver in range(n_agents):
            cached = comm_manager.get_receiver_states(receiver).clone()
            aoi = comm_manager.get_aoi(receiver).clone()
            comm_mask = comm_manager.get_current_comm_mask(receiver).clone()
            cached[:, receiver] = true_states[:, receiver]
            aoi[:, receiver] = 0
            comm_mask[:, receiver] = 1.0

            sample_mask = valid_env_mask
            if not self.include_zero_aoi:
                peer_mask = peer_template[receiver].unsqueeze(0)
                sample_mask = sample_mask & ((aoi > 0) & peer_mask).any(dim=-1)
            if not sample_mask.any():
                continue

            env_indices = sample_mask.nonzero(as_tuple=False).squeeze(-1)
            n_samples = env_indices.numel()
            self.chunks["cached_states"].append(cached[env_indices])
            self.chunks["true_states"].append(true_states[env_indices])
            self.chunks["aoi"].append(aoi[env_indices])
            self.chunks["comm_mask"].append(comm_mask[env_indices])
            self.chunks["receiver"].append(
                torch.full((n_samples,), receiver, dtype=torch.long, device=true_states.device)
            )
            self.chunks["env_index"].append(env_indices.long())
            self.chunks["episode_step"].append(
                torch.full((n_samples,), step, dtype=torch.long, device=true_states.device)
            )

    def tensors(self) -> dict[str, Tensor]:
        tensors = {}
        for key, chunks in self.chunks.items():
            if not chunks:
                raise RuntimeError(f"No samples collected for key={key}")
            tensors[key] = torch.cat(chunks, dim=0).detach().cpu()
        return tensors


def build_dataset_dict(
    args: argparse.Namespace,
    task_config: dict[str, Any],
    comm_config: dict[str, Any],
    estimator_config: dict[str, Any],
    collector: GraphSampleCollector,
) -> dict[str, Any]:
    return {
        "metadata": {
            "task": args.task,
            "comm": args.comm,
            "estimator_config": args.estimator_config,
            "policy": args.policy,
            "episodes": args.episodes,
            "max_steps": int(args.max_steps or task_config["max_steps"]),
            "seed": args.seed,
            "include_zero_aoi": args.include_zero_aoi,
        },
        "task_config": task_config,
        "comm_config": comm_config,
        "estimator_config_values": estimator_config,
        "tensors": collector.tensors(),
    }


def dataset_summary(dataset: dict[str, Any]) -> dict[str, Any]:
    tensors = dataset["tensors"]
    aoi = tensors["aoi"].float()
    peer_mask = ~torch.nn.functional.one_hot(
        tensors["receiver"].long(),
        num_classes=tensors["aoi"].shape[-1],
    ).bool()
    peer_aoi = aoi[peer_mask]
    stale_mse = ((tensors["cached_states"] - tensors["true_states"]) ** 2)[
        peer_mask
    ].mean()
    return {
        "samples": int(tensors["receiver"].numel()),
        "n_agents": int(tensors["aoi"].shape[-1]),
        "mean_peer_aoi": float(peer_aoi.mean().item()),
        "max_peer_aoi": float(peer_aoi.max().item()),
        "nonzero_peer_aoi_fraction": float((peer_aoi > 0).float().mean().item()),
        "mean_peer_comm_mask": float(tensors["comm_mask"][peer_mask].float().mean().item()),
        "peer_stale_mse": float(stale_mse.item()),
    }


def default_output_path(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%y_%m_%d-%H_%M_%S")
    name = f"{args.task}_{args.comm}_{args.policy}_gnn_seed{args.seed}_{stamp}.pt"
    return PROJECT_ROOT / "outputs" / "estimator_datasets" / name


def load_yaml(group: str, name: str) -> dict[str, Any]:
    path = CONFIG_ROOT / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


if __name__ == "__main__":
    main()
