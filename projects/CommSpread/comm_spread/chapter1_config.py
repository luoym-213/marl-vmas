"""Versioned resolver and fingerprinting for the frozen Chapter 1 SAR task."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "chapter1"
FINAL_TASK_VERSION = "chapter1_sar_final_v1"
IMMEDIATE_OPEN_TASK_VERSION = "chapter1_sar_immediate_open_v1"
SUPPORTED_TASK_VERSIONS = frozenset({
    FINAL_TASK_VERSION,
    IMMEDIATE_OPEN_TASK_VERSION,
})
TASK_VERSION_CASCADE_MODES = {
    FINAL_TASK_VERSION: "finder_only",
    IMMEDIATE_OPEN_TASK_VERSION: "immediate_open",
}

PROGRAM_DEFAULTS: dict[str, dict[str, Any]] = {
    "low_train": {"restore_file": None, "restore_map_location": None},
    "high_train": {"resume_checkpoint": None, "intent_loss_coefficient": 0.1},
    "eval": {"write_resolved_config": True},
}

VOLATILE_HASH_KEYS = {
    "command",
    "created_at",
    "output_dir",
    "save_folder",
    "timestamp",
    "updated_at",
}

REQUIRED_TASK_PATHS = (
    "task_version",
    "scenario.name",
    "scenario.num_agents",
    "scenario.num_targets",
    "scenario.horizon",
    "world.semidim_x",
    "world.semidim_y",
    "physics.profile",
    "physics.dt",
    "physics.substeps",
    "physics.drag",
    "physics.action_space.continuous",
    "physics.action_space.cardinality",
    "physics.agent_radius",
    "physics.target_radius",
    "initial_distribution.agent_agent_min_distance",
    "initial_distribution.target_target_min_distance",
    "initial_distribution.agent_target_min_distance",
    "perception.sensor_radius",
    "perception.team_shared_belief",
    "perception.belief.detection_threshold",
    "perception.detection.persistent_until_rescue",
    "task_semantics.rescue_distance_threshold",
    "task_semantics.success",
    "task_semantics.early_done_on_success",
    "task_semantics.retire_on_rescue",
    "hierarchy.rrt.top_k",
    "hierarchy.rrt.max_iterations",
    "hierarchy.replanning.event_triggered",
    "hierarchy.finder_first.enabled",
    "hierarchy.finder_first.cascade_mode",
    "low_level_interface.variant",
    "low_level_interface.observation.width",
    "low_level_interface.goal.threshold",
)


class Chapter1ConfigError(ValueError):
    """Raised when a frozen task invariant is violated."""


@dataclass(frozen=True)
class ResolvedChapter1Config:
    source: Path
    run_type: str
    document: dict[str, Any]

    @property
    def task(self) -> dict[str, Any]:
        return self.document["resolved"]["task"]

    @property
    def run(self) -> dict[str, Any]:
        return self.document["resolved"]["run"]

    @property
    def task_config_sha256(self) -> str:
        return self.document["fingerprints"]["task_config_sha256"]

    @property
    def config_sha256(self) -> str:
        return self.document["fingerprints"]["config_sha256"]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise Chapter1ConfigError(f"config must be a mapping: {path}")
    return value


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _set_dotted(config: dict[str, Any], path: str, value: Any) -> None:
    cursor = config
    parts = path.split(".")
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise Chapter1ConfigError(f"cannot override non-mapping path: {path}")
        cursor = child
    cursor[parts[-1]] = value


def _get_dotted(config: Mapping[str, Any], path: str) -> Any:
    cursor: Any = config
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise Chapter1ConfigError(f"missing required task field: {path}")
        cursor = cursor[part]
    return cursor


def _canonical_value(value: Any, *, exclude_volatile: bool) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child, exclude_volatile=exclude_volatile)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if not (exclude_volatile and str(key) in VOLATILE_HASH_KEYS)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(child, exclude_volatile=exclude_volatile) for child in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Chapter1ConfigError("non-finite floats are not canonicalizable")
        # A tagged decimal string makes the representation independent of JSON
        # encoder whitespace and preserves the distinction from YAML strings.
        return {"__float__": format(value, ".17g")}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise Chapter1ConfigError(f"unsupported canonical value: {type(value)!r}")


def canonical_bytes(value: Any, *, exclude_volatile: bool = False) -> bytes:
    normalized = _canonical_value(value, exclude_volatile=exclude_volatile)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def config_sha256(value: Any, *, exclude_volatile: bool = False) -> str:
    return hashlib.sha256(
        canonical_bytes(value, exclude_volatile=exclude_volatile)
    ).hexdigest()


def validate_task(task: Mapping[str, Any]) -> None:
    for path in REQUIRED_TASK_PATHS:
        _get_dotted(task, path)
    task_version = task["task_version"]
    if task_version not in SUPPORTED_TASK_VERSIONS:
        raise Chapter1ConfigError(
            f"unsupported Chapter 1 task_version={task_version!r}; "
            f"expected one of {sorted(SUPPORTED_TASK_VERSIONS)!r}"
        )
    cascade_mode = _get_dotted(task, "hierarchy.finder_first.cascade_mode")
    expected_cascade_mode = TASK_VERSION_CASCADE_MODES[task_version]
    if cascade_mode != expected_cascade_mode:
        raise Chapter1ConfigError(
            f"task_version={task_version!r} requires "
            f"hierarchy.finder_first.cascade_mode={expected_cascade_mode!r}"
        )
    profile = _get_dotted(task, "physics.profile")
    continuous = _get_dotted(task, "physics.action_space.continuous")
    if profile == "mpe_strict" and continuous:
        raise Chapter1ConfigError("mpe_strict requires continuous_actions=False")
    if profile != "mpe_strict":
        raise Chapter1ConfigError("Chapter 1 final task forbids legacy physics")
    if _get_dotted(task, "hierarchy.execution") != "asynchronous_event_triggered_smdp":
        raise Chapter1ConfigError("Chapter 1 final task requires AsyncSMDP execution")


def validate_high_training(run: Mapping[str, Any], task: Mapping[str, Any]) -> None:
    """Fail closed on the frozen Chapter 1 high-level training semantics."""

    training = run.get("training")
    if not isinstance(training, Mapping):
        raise Chapter1ConfigError("high_train config requires training mapping")
    high = training.get("high_level")
    if not isinstance(high, Mapping):
        raise Chapter1ConfigError("high_train config requires training.high_level")
    required = {
        "execution_mode": "event_triggered_async",
        "return_mode": "strict_smdp",
        "gae_duration_mode": "environment_time",
        "training_success_terminal": True,
        "environment_continue_after_success": True,
        "exclude_post_success_steps_from_training": True,
        "time_limit_bootstrap": True,
        "recurrent_state_mode": "feedforward_empty",
    }
    for key, expected in required.items():
        if high.get(key) != expected:
            raise Chapter1ConfigError(
                f"Chapter 1 high_train requires high_level.{key}={expected!r}"
            )
    if training.get("low_level_steps_per_batch", 0) < task["scenario"]["horizon"]:
        raise Chapter1ConfigError(
            "strict SMDP collection cannot discard an unfinished episode at a batch boundary"
        )


def resolve_chapter1_config(
    path: str | Path,
    *,
    cli_overrides: Mapping[str, Any] | None = None,
) -> ResolvedChapter1Config:
    source = Path(path).resolve()
    role_yaml = _load_yaml(source)
    run_type = str(role_yaml.get("run_type", "task"))
    if run_type == "task":
        task_yaml = role_yaml
        run_yaml: dict[str, Any] = {}
        task_source = source
    else:
        task_ref = role_yaml.get("task_ref")
        if not isinstance(task_ref, str):
            raise Chapter1ConfigError(f"missing task_ref in {source}")
        task_source = (source.parent / task_ref).resolve()
        task_yaml = _load_yaml(task_source)
        if role_yaml.get("task_version") != task_yaml.get("task_version"):
            raise Chapter1ConfigError("role config and referenced task_version differ")
        run_yaml = {
            key: deepcopy(value)
            for key, value in role_yaml.items()
            if key not in {"schema_version", "run_type", "task_ref", "task_version"}
        }

    validate_task(task_yaml)
    defaults = deepcopy(PROGRAM_DEFAULTS.get(run_type, {}))
    resolved_run = _merge(defaults, run_yaml)
    overrides = dict(cli_overrides or {})
    for dotted_path, value in overrides.items():
        if dotted_path.startswith("task."):
            if run_type == "eval":
                raise Chapter1ConfigError(
                    f"evaluation cannot override frozen core field: {dotted_path}"
                )
            _set_dotted(task_yaml, dotted_path.removeprefix("task."), value)
        elif dotted_path.startswith("run."):
            _set_dotted(resolved_run, dotted_path.removeprefix("run."), value)
        else:
            raise Chapter1ConfigError(
                f"override must start with task. or run.: {dotted_path}"
            )
    validate_task(task_yaml)
    if run_type == "high_train":
        validate_high_training(resolved_run, task_yaml)
    resolved = {"task": task_yaml, "run": resolved_run}
    task_hash = config_sha256(task_yaml)
    full_hash = config_sha256(resolved, exclude_volatile=True)
    document = {
        "schema_version": 1,
        "sources": {
            "run_config": source.as_posix(),
            "task_config": task_source.as_posix(),
        },
        "yaml": {"task": task_yaml, "run": run_yaml},
        "program_defaults": defaults,
        "cli_overrides": overrides,
        "resolved": resolved,
        "fingerprints": {
            "canonical_float_format": ".17g tagged decimal",
            "excluded_volatile_keys": sorted(VOLATILE_HASH_KEYS),
            "task_config_sha256": task_hash,
            "config_sha256": full_hash,
        },
    }
    return ResolvedChapter1Config(source, run_type, document)


def save_resolved_config(
    resolved: ResolvedChapter1Config,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir) / "resolved_config.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(resolved.document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def fingerprint_lines(resolved: ResolvedChapter1Config) -> list[str]:
    task = resolved.task
    return [
        f"task_version={task['task_version']}",
        f"physics_profile={_get_dotted(task, 'physics.profile')}",
        f"num_agents={_get_dotted(task, 'scenario.num_agents')}",
        f"num_targets={_get_dotted(task, 'scenario.num_targets')}",
        f"horizon={_get_dotted(task, 'scenario.horizon')}",
        f"sensor_radius={_get_dotted(task, 'perception.sensor_radius')}",
        f"retire_on_rescue={_get_dotted(task, 'task_semantics.retire_on_rescue')}",
        f"event_triggered_replanning={_get_dotted(task, 'hierarchy.replanning.event_triggered')}",
        f"finder_first={_get_dotted(task, 'hierarchy.finder_first.enabled')}",
        f"config_sha256={resolved.config_sha256}",
        f"task_config_sha256={resolved.task_config_sha256}",
    ]


def chapter1_sar_kwargs(task: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    """Translate the frozen task schema to :func:`make_sar_env` kwargs."""

    validate_task(task)
    perception = task["perception"]
    belief = perception["belief"]
    physics = task["physics"]
    collision = physics["collision"]
    hierarchy = task["hierarchy"]
    release = hierarchy["rescue_release"]
    rewards = task["rewards"]
    semantics = task["task_semantics"]
    return {
        "mode": mode,
        "emit_info": mode != "low",
        "enable_high_level_state": mode != "low",
        "n_agents": task["scenario"]["num_agents"],
        "n_targets": task["scenario"]["num_targets"],
        "max_steps": task["scenario"]["horizon"],
        "physics_profile": physics["profile"],
        "continuous_actions": physics["action_space"]["continuous"],
        "mpe_action_force_scale": physics["action_space"]["force_sensitivity"],
        "world_substeps": physics["substeps"],
        "world_collision_force": collision["collision_force"],
        "world_contact_margin": collision["contact_margin"],
        "world_dt": physics["dt"],
        "world_drag": physics["drag"],
        "world_linear_friction": physics["linear_friction"],
        "world_angular_friction": physics["angular_friction"],
        "world_hard_bounds": task["world"]["boundary"]["hard_clamp"],
        "world_spawning_x": task["world"]["semidim_x"],
        "world_spawning_y": task["world"]["semidim_y"],
        "belief_world_size": task["world"]["belief_world_size"],
        "belief_cell_size": belief["cell_size"],
        "sensor_radius": perception["sensor_radius"],
        "sensor_fidelity": belief["sensor_fidelity"],
        "initial_belief": belief["initial_probability"],
        "belief_detection_threshold": belief["detection_threshold"],
        "belief_include_inactive_agents": perception["belief_include_inactive_agents"],
        "agent_radius": physics["agent_radius"],
        "target_radius": physics["target_radius"],
        "goal_radius": semantics["rescue_distance_threshold"],
        "goal_reward": rewards["low_level_goal"],
        "rescue_reward": rewards["rescue_base_by_order"],
        "discovery_reward": rewards["discovery"],
        "distance_reward_scale": rewards["distance_progress_scale"],
        "time_penalty": rewards["time_penalty"],
        "collision_penalty": collision["reward_coefficient"],
        "collision_safe_distance": collision["reward_safe_distance"],
        "max_collision_penalty": collision["reward_floor"],
        "boundary_penalty": task["world"]["boundary"]["penalty"],
        "retire_on_rescue": semantics["retire_on_rescue"],
        "done_when_all_targets_visited": semantics["early_done_on_success"],
        "auto_resample_goals": mode == "low",
        "enable_rrt_candidates": mode != "low",
        "rrt_top_k": hierarchy["rrt"]["top_k"],
        "rrt_max_iter": hierarchy["rrt"]["max_iterations"],
        "rrt_seed": hierarchy["rrt"]["seed"],
        "dynamic_rescue_release": release["dynamic_staggered"],
        "dynamic_release_min_searchers": release["min_searchers"],
        "dynamic_release_entropy_ratio_threshold": release["entropy_ratio_threshold"],
        "dynamic_release_entropy_rate_threshold": release["entropy_rate_threshold"],
        "dynamic_release_min_stagnation_step": release["min_stagnation_step"],
        "dynamic_release_search_steps_per_target": release["search_steps_per_target"],
        "dynamic_release_speed_per_step": release["assumed_speed_per_step"],
        "dynamic_release_time_margin": release["time_margin"],
        "dynamic_release_max_new_agents_per_event": release["max_new_agents_per_event"],
        "dynamic_rescue_only_after_all_detected": release["only_after_all_targets_detected"],
        "redecide_on_detection_change": hierarchy["replanning"]["on_detection_change"],
        "redecide_on_assignment_change": hierarchy["replanning"]["on_assignment_change"],
        "enable_finder_first_cascade": hierarchy["finder_first"]["enabled"],
        "finder_cascade_mode": hierarchy["finder_first"]["cascade_mode"],
    }


def validate_runtime_task_values(
    resolved: ResolvedChapter1Config,
    runtime_values: Mapping[str, Any],
) -> None:
    """Reject legacy CLI task values that differ from the frozen eval task."""

    expected = {
        "physics_profile": _get_dotted(resolved.task, "physics.profile"),
        "continuous_actions": _get_dotted(
            resolved.task, "physics.action_space.continuous"
        ),
        "num_agents": _get_dotted(resolved.task, "scenario.num_agents"),
        "num_targets": _get_dotted(resolved.task, "scenario.num_targets"),
        "max_steps": _get_dotted(resolved.task, "scenario.horizon"),
        "sensor_radius": _get_dotted(resolved.task, "perception.sensor_radius"),
        "retire_on_rescue": _get_dotted(
            resolved.task, "task_semantics.retire_on_rescue"
        ),
    }
    for key, actual in runtime_values.items():
        if key in expected and actual is not None and actual != expected[key]:
            raise Chapter1ConfigError(
                f"frozen task mismatch for {key}: expected {expected[key]!r}, got {actual!r}"
            )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_sha256(path: str | Path, expected: str) -> Path:
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise Chapter1ConfigError(f"checkpoint does not exist: {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise Chapter1ConfigError(
            f"checkpoint SHA256 mismatch for {checkpoint}: expected {expected}, got {actual}"
        )
    return checkpoint


def checkpoint_metadata(
    resolved: ResolvedChapter1Config,
    checkpoint: str | Path,
) -> dict[str, Any]:
    interface = resolved.task["low_level_interface"]
    action = interface["action"]
    path = Path(checkpoint).resolve()
    return {
        "task_version": resolved.task["task_version"],
        "task_config_sha256": resolved.task_config_sha256,
        "physics_profile": resolved.task["physics"]["profile"],
        "low_level_variant": interface["variant"],
        "observation_width": interface["observation"]["width"],
        "action_space_kind": "categorical" if not action["continuous"] else "continuous",
        "action_cardinality": action["categorical_cardinality"],
        "goal_threshold": interface["goal"]["threshold"],
        "checkpoint": path.as_posix(),
        "checkpoint_sha256": sha256_file(path),
    }


def write_checkpoint_metadata_sidecars(
    resolved: ResolvedChapter1Config,
    output_dir: str | Path,
) -> list[Path]:
    """Write immutable metadata next to completed low-level checkpoints."""

    written: list[Path] = []
    for checkpoint in sorted(Path(output_dir).rglob("checkpoint_*.pt")):
        sidecar = checkpoint.with_suffix(checkpoint.suffix + ".metadata.json")
        if sidecar.exists():
            continue
        sidecar.write_text(
            json.dumps(
                checkpoint_metadata(resolved, checkpoint),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(sidecar)
    return written
