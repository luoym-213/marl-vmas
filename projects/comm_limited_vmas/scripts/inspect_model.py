"""Inspect BenchMARL actor and critic modules without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

CONFIG_ROOT = PROJECT_ROOT / "configs"

from comm_limited_vmas.benchmarl_task import build_comm_vmas_task


ALGORITHM_REGISTRY = {
    "mappo": MappoConfig,
    "ippo": IppoConfig,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect current BenchMARL actor/critic architecture."
    )
    parser.add_argument("--task", default="comm_hidden_goal_navigation")
    parser.add_argument(
        "--algorithm",
        default="mappo",
        choices=sorted(ALGORITHM_REGISTRY),
    )
    parser.add_argument("--comm", default="comm_full")
    parser.add_argument("--estimator", default="stale")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a JSON architecture report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_model(args)
    print_report(report)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"\nWrote JSON report to: {args.json_output}")


def inspect_model(args: argparse.Namespace) -> dict[str, Any]:
    algorithm_config = build_algorithm_config(args.algorithm)
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    task = build_comm_vmas_task(
        task_name=args.task,
        task_config=load_yaml("task", args.task),
        comm_config=load_yaml("comm", args.comm),
        estimator_config=load_yaml("estimator", args.estimator),
    )

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = args.device
    experiment_config.train_device = args.device
    experiment_config.buffer_device = args.device
    experiment_config.evaluation_episodes = 2
    experiment_config.render = False
    experiment_config.loggers = []
    experiment_config.create_json = False
    experiment_config.checkpoint_interval = 0
    experiment_config.checkpoint_at_end = False
    experiment_config.restore_file = None
    experiment_config.save_folder = None

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )

    try:
        policy_parameters = parameter_shapes(experiment.policy)
        loss_reports = {}
        for group, loss in experiment.losses.items():
            loss_reports[group] = {
                "repr": repr(loss),
                "parameters": parameter_shapes(loss),
            }

        actor_shapes = select_parameter_shapes(
            loss_reports,
            prefix="actor_network_params",
        )
        critic_shapes = select_parameter_shapes(
            loss_reports,
            prefix="critic_network_params",
        )

        return {
            "task": args.task,
            "algorithm": args.algorithm,
            "comm": args.comm,
            "estimator": args.estimator,
            "group_map": experiment.group_map,
            "model_config": model_config_report(model_config),
            "critic_model_config": model_config_report(critic_model_config),
            "algorithm_config": config_object_report(algorithm_config),
            "policy": {
                "type": type(experiment.policy).__name__,
                "repr": repr(experiment.policy),
                "parameters": policy_parameters,
            },
            "losses": loss_reports,
            "derived_architecture": {
                "actor": derive_mlp_architecture(actor_shapes),
                "critic": derive_mlp_architecture(critic_shapes),
            },
        }
    finally:
        experiment.close()


def print_report(report: dict[str, Any]) -> None:
    print("Task:", report["task"])
    print("Algorithm:", report["algorithm"])
    print("Comm:", report["comm"])
    print("Group map:", report["group_map"])
    print("\nModel config:")
    print(json.dumps(report["model_config"], indent=2, sort_keys=True))
    print("\nAlgorithm config:")
    print(json.dumps(report["algorithm_config"], indent=2, sort_keys=True))

    actor = report["derived_architecture"]["actor"]
    critic = report["derived_architecture"]["critic"]
    print("\nDerived actor architecture:")
    print(json.dumps(actor, indent=2, sort_keys=True))
    print("\nDerived critic architecture:")
    print(json.dumps(critic, indent=2, sort_keys=True))

    print("\nPolicy module:")
    print(report["policy"]["repr"])
    print("\nPolicy parameter shapes:")
    for name, shape in report["policy"]["parameters"].items():
        print(f"  {name}: {shape}")

    print("\nLoss parameter shapes:")
    for group, loss_report in report["losses"].items():
        print(f"  [{group}]")
        for name, shape in loss_report["parameters"].items():
            print(f"    {name}: {shape}")


def derive_mlp_architecture(parameters: dict[str, list[int]]) -> dict[str, Any]:
    linear_weights = [
        (name, shape)
        for name, shape in parameters.items()
        if name.endswith(".weight") and len(shape) == 2
    ]
    linear_weights.sort(key=lambda item: item[0])
    layers = [
        {
            "name": name,
            "in_features": shape[1],
            "out_features": shape[0],
        }
        for name, shape in linear_weights
    ]
    return {
        "linear_layers": layers,
        "num_linear_layers": len(layers),
        "input_dim": layers[0]["in_features"] if layers else None,
        "output_dim": layers[-1]["out_features"] if layers else None,
        "hidden_dims": [layer["out_features"] for layer in layers[:-1]],
        "activation": "Tanh",
        "normalization": None,
    }


def select_parameter_shapes(
    loss_reports: dict[str, dict[str, Any]],
    prefix: str,
) -> dict[str, list[int]]:
    selected: dict[str, list[int]] = {}
    for loss_report in loss_reports.values():
        for name, shape in loss_report["parameters"].items():
            if name.startswith(prefix):
                selected[name] = shape
    return selected


def parameter_shapes(module: Any) -> dict[str, list[int]]:
    return {name: list(param.shape) for name, param in module.named_parameters()}


def model_config_report(config: MlpConfig) -> dict[str, Any]:
    return {
        "class": type(config).__name__,
        "num_cells": list(config.num_cells),
        "layer_class": class_name(config.layer_class),
        "activation_class": class_name(config.activation_class),
        "activation_kwargs": config.activation_kwargs,
        "norm_class": class_name(config.norm_class),
        "norm_kwargs": config.norm_kwargs,
    }


def config_object_report(config: Any) -> dict[str, Any]:
    report = {}
    for key, value in config.__dict__.items():
        if key.startswith("_"):
            continue
        report[key] = to_jsonable(value)
    return report


def class_name(value: Any) -> str | None:
    if value is None:
        return None
    return f"{value.__module__}.{value.__name__}"


def build_algorithm_config(name: str) -> Any:
    algorithm_cls = ALGORITHM_REGISTRY[name]
    algorithm_config = algorithm_cls.get_from_yaml()
    apply_overrides(algorithm_config, load_yaml("algorithm", name))
    return algorithm_config


def apply_overrides(config_obj: Any, overrides: dict[str, Any]) -> Any:
    for key, value in overrides.items():
        if hasattr(config_obj, key):
            setattr(config_obj, key, value)
        else:
            print(
                f"[Warning] Skip unknown config field: {key} "
                f"for {type(config_obj).__name__}"
            )
    return config_obj


def load_yaml(group: str, name: str) -> dict[str, Any]:
    path = CONFIG_ROOT / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return class_name(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "__class__") and value.__class__.__module__.startswith("torch"):
        return str(value)
    return value


if __name__ == "__main__":
    main()
