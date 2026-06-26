"""BenchMARL task adapter for communication-limited VMAS scenarios."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

try:
    from torchrl.envs.libs.vmas import VmasEnv
except ImportError:
    from torchrl.envs import VmasEnv

from torchrl.data import Composite
from torchrl.envs import EnvBase

from benchmarl.environments.common import Task
from benchmarl.utils import DEVICE_TYPING

from comm_limited_vmas.scenarios.comm_discovery import (
    Scenario as CommDiscoveryScenario,
)
from comm_limited_vmas.scenarios.comm_dispersion import (
    Scenario as CommDispersionScenario,
)
from comm_limited_vmas.scenarios.comm_flocking import (
    Scenario as CommFlockingScenario,
)
from comm_limited_vmas.scenarios.comm_navigation import (
    Scenario as CommNavigationScenario,
)
from comm_limited_vmas.scenarios.comm_spread import Scenario as CommSpreadScenario


class CommLimitedVmasTask(Task):
    """Enum wrapper so BenchMARL can consume communication-limited VMAS tasks."""

    # 注意：如果以后有多个任务，不要都写成 None。
    # Python Enum 中多个相同值会变成 alias。
    COMM_NAVIGATION = "comm_navigation"
    COMM_SPREAD = "comm_spread"
    COMM_DISCOVERY = "comm_discovery"
    COMM_DISPERSION = "comm_dispersion"
    COMM_FLOCKING = "comm_flocking"

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)

        # 从总 config 中拆出通信配置和 estimator 配置
        comm_config = config.pop("comm", {})
        estimator_config = config.pop("estimator", {})
        config.pop("scenario", None)
        config.pop("continuous_actions", None)

        if self.name == "COMM_NAVIGATION":
            scenario_cls = CommNavigationScenario
        elif self.name == "COMM_SPREAD":
            scenario_cls = CommSpreadScenario
        elif self.name == "COMM_DISCOVERY":
            scenario_cls = CommDiscoveryScenario
        elif self.name == "COMM_DISPERSION":
            scenario_cls = CommDispersionScenario
        elif self.name == "COMM_FLOCKING":
            scenario_cls = CommFlockingScenario
        else:
            raise ValueError(f"Unknown task: {self.name}")

        def make_env() -> EnvBase:
            return VmasEnv(
                scenario=scenario_cls(
                    comm_config=copy.deepcopy(comm_config),
                    estimator_config=copy.deepcopy(estimator_config),
                ),
                num_envs=num_envs,
                continuous_actions=continuous_actions,
                seed=seed,
                device=device,
                categorical_actions=not continuous_actions,
                clamp_actions=True,
                **config,
            )

        return make_env

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return env.group_map
        return {"agents": [agent.name for agent in env.agents]}

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = _clone_unbatched_spec(env, "full_observation_spec")
        for group in self.group_map(env):
            if "info" in observation_spec[group]:
                del observation_spec[(group, "info")]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        info_spec = _clone_unbatched_spec(env, "full_observation_spec")
        for group in self.group_map(env):
            if "observation" in info_spec[group]:
                del info_spec[(group, "observation")]

        for group in self.group_map(env):
            if "info" in info_spec[group]:
                return info_spec

        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return _clone_unbatched_spec(env, "full_action_spec")

    @staticmethod
    def env_name() -> str:
        return "comm_limited_vmas"


def _clone_unbatched_spec(env: EnvBase, spec_name: str) -> Composite:
    unbatched_name = f"{spec_name}_unbatched"
    if hasattr(env, unbatched_name):
        spec = getattr(env, unbatched_name)
    else:
        spec = getattr(env, spec_name)
    return spec.clone()


def build_comm_vmas_task(
    task_name: str,
    task_config: dict[str, Any],
    comm_config: dict[str, Any],
    estimator_config: dict[str, Any],
) -> CommLimitedVmasTask:
    """Build a BenchMARL-compatible communication-limited VMAS task."""

    task_map = {
        "comm_navigation": CommLimitedVmasTask.COMM_NAVIGATION,
        "comm_spread": CommLimitedVmasTask.COMM_SPREAD,
        "comm_discovery": CommLimitedVmasTask.COMM_DISCOVERY,
        "comm_dispersion": CommLimitedVmasTask.COMM_DISPERSION,
        "comm_flocking": CommLimitedVmasTask.COMM_FLOCKING,
    }

    if task_name not in task_map:
        raise ValueError(
            f"Unknown task_name={task_name}. "
            f"Available tasks: {list(task_map.keys())}"
        )

    merged_config = copy.deepcopy(task_config)
    merged_config["comm"] = copy.deepcopy(comm_config)
    merged_config["estimator"] = copy.deepcopy(estimator_config)

    task = task_map[task_name]

    # The project already loaded its task yaml in scripts/train.py.
    # Do not call BenchMARL get_from_yaml(), which looks under benchmarl/conf/task.
    task.config = None
    return task.update_config(merged_config)
