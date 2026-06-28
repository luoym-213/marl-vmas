"""Collect supervised samples for communication state estimators."""

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

from comm_limited_vmas.estimators.kinematic import KinematicEstimator
from comm_limited_vmas.scenarios.comm_hidden_goal_navigation import Scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect supervised estimator samples from VMAS rollouts."
    )
    parser.add_argument("--task", default="comm_hidden_goal_navigation")
    parser.add_argument("--comm", default="radius_delay_dropout_mean8_dropout03")
    parser.add_argument(
        "--estimator-config",
        default="aoi_residual",
        help="Estimator config whose kinematic parameters should be used.",
    )
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
    parser.add_argument("--include-zero-aoi", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task != "comm_hidden_goal_navigation":
        raise ValueError("Only comm_hidden_goal_navigation is supported for scripted collection")

    dataset = collect_dataset(args)
    output_path = args.output or default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)

    summary = dataset_summary(dataset)
    print(f"Wrote estimator dataset: {output_path}")
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
        native_env = env._env
        collector = SampleCollector(native_env, estimator_config, args.include_zero_aoi)
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
        native_env = env._env
        td = env.reset()
        done = torch.zeros(_num_envs(env), dtype=torch.bool, device=td.device)
        collector = SampleCollector(native_env, estimator_config, args.include_zero_aoi)
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


class SampleCollector:
    def __init__(
        self,
        native_env: Any,
        estimator_config: dict[str, Any],
        include_zero_aoi: bool,
    ):
        self.native_env = native_env
        self.include_zero_aoi = include_zero_aoi
        self.kinematic = KinematicEstimator(
            state_dim=4,
            device=native_env.world.device,
            dt=float(native_env.world.dt),
            sigma_p=float(estimator_config.get("sigma_p", 0.01)),
            sigma_v=float(estimator_config.get("sigma_v", 0.01)),
            process_noise_q=float(estimator_config.get("process_noise_q", 1.0)),
            max_extrapolation_steps=estimator_config.get("max_extrapolation_steps", 15),
        )
        self.chunks: dict[str, list[Tensor]] = {
            "cached_state": [],
            "kinematic_state": [],
            "true_state": [],
            "aoi": [],
            "comm_mask": [],
            "delta_t": [],
            "receiver": [],
            "sender": [],
            "env_index": [],
            "episode_step": [],
        }

    def collect(self, step: int, valid_env_mask: Tensor) -> None:
        scenario = self.native_env.scenario
        comm_manager = scenario._comm_manager
        true_states = current_agent_states(self.native_env.world)
        num_envs, n_agents, _ = true_states.shape
        valid_env_mask = valid_env_mask.to(device=true_states.device)

        for receiver in range(n_agents):
            cached = comm_manager.get_receiver_states(receiver)
            aoi = comm_manager.get_aoi(receiver)
            comm_mask = comm_manager.get_current_comm_mask(receiver)
            kin = self.kinematic.estimate_state(cached, aoi)
            delta_t = self.kinematic._delta_t(aoi, clamp=True)

            for sender in range(n_agents):
                if sender == receiver:
                    continue
                sample_mask = valid_env_mask
                if not self.include_zero_aoi:
                    sample_mask = sample_mask & (aoi[:, sender] > 0)
                if not sample_mask.any():
                    continue

                env_indices = sample_mask.nonzero(as_tuple=False).squeeze(-1)
                n_samples = env_indices.numel()
                self.chunks["cached_state"].append(cached[env_indices, sender])
                self.chunks["kinematic_state"].append(kin[env_indices, sender])
                self.chunks["true_state"].append(true_states[env_indices, sender])
                self.chunks["aoi"].append(aoi[env_indices, sender])
                self.chunks["comm_mask"].append(comm_mask[env_indices, sender])
                self.chunks["delta_t"].append(delta_t[env_indices, sender])
                self.chunks["receiver"].append(
                    torch.full((n_samples,), receiver, dtype=torch.long, device=true_states.device)
                )
                self.chunks["sender"].append(
                    torch.full((n_samples,), sender, dtype=torch.long, device=true_states.device)
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
    collector: SampleCollector,
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


def scripted_action(observation: Tensor, gain: float) -> Tensor:
    action = torch.zeros(
        observation.shape[0],
        observation.shape[1],
        2,
        dtype=observation.dtype,
        device=observation.device,
    )
    action[:, 0] = observation[:, 0, 4:6]
    action[:, 1] = observation[:, 1, 9:11]
    action[:, 2] = observation[:, 2, 9:11]
    return clamp_with_norm(action * gain, 1.0)


def random_action(observation: Tensor, generator: torch.Generator) -> Tensor:
    action = torch.rand(
        observation.shape[0],
        observation.shape[1],
        2,
        generator=generator,
        dtype=observation.dtype,
        device=observation.device,
    )
    return action * 2.0 - 1.0


def clamp_with_norm(value: Tensor, max_norm: float) -> Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-6)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return value * scale


def current_agent_states(world: Any) -> Tensor:
    return torch.cat(
        [
            torch.stack([agent.state.pos for agent in world.agents], dim=1),
            torch.stack([agent.state.vel for agent in world.agents], dim=1),
        ],
        dim=-1,
    )


def dataset_summary(dataset: dict[str, Any]) -> dict[str, Any]:
    tensors = dataset["tensors"]
    aoi = tensors["aoi"].float()
    stale_mse = ((tensors["cached_state"] - tensors["true_state"]) ** 2).mean().item()
    kin_mse = ((tensors["kinematic_state"] - tensors["true_state"]) ** 2).mean().item()
    return {
        "samples": int(aoi.numel()),
        "mean_aoi": float(aoi.mean().item()),
        "max_aoi": float(aoi.max().item()),
        "nonzero_aoi_fraction": float((aoi > 0).float().mean().item()),
        "mean_comm_mask": float(tensors["comm_mask"].float().mean().item()),
        "stale_mse": stale_mse,
        "kinematic_mse": kin_mse,
    }


def default_output_path(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%y_%m_%d-%H_%M_%S")
    name = f"{args.task}_{args.comm}_{args.policy}_seed{args.seed}_{stamp}.pt"
    return PROJECT_ROOT / "outputs" / "estimator_datasets" / name


def load_yaml(group: str, name: str) -> dict[str, Any]:
    path = CONFIG_ROOT / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _num_envs(env: Any) -> int:
    if len(env.batch_size) > 0:
        return int(env.batch_size[0])
    native_env = getattr(env, "_env", None)
    if native_env is not None and hasattr(native_env, "num_envs"):
        return int(native_env.num_envs)
    return 1


if __name__ == "__main__":
    main()
