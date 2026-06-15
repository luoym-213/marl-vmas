"""BenchMARL task adapter for the CommSpread VMAS scenario."""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional

try:
    from torchrl.envs.libs.vmas import VmasEnv
except ImportError:
    from torchrl.envs import VmasEnv

from torchrl.data import Composite
from torchrl.envs import EnvBase

from benchmarl.environments.common import Task
from benchmarl.utils import DEVICE_TYPING

from comm_spread.scenario import CommSpreadScenario


class CommSpreadTask(Task):
    """Enum wrapper so BenchMARL can consume CommSpread."""

    SPREAD = None

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)

        return lambda: VmasEnv(
            scenario=CommSpreadScenario(),
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            categorical_actions=True,
            clamp_actions=True,
            **config,
        )

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
        return "comm_spread"


def _clone_unbatched_spec(env: EnvBase, spec_name: str) -> Composite:
    unbatched_name = f"{spec_name}_unbatched"
    if hasattr(env, unbatched_name):
        spec = getattr(env, unbatched_name)
    else:
        spec = getattr(env, spec_name)
    return spec.clone()
