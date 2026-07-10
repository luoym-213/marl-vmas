"""BenchMARL task adapter for the CommSpread VMAS scenario."""

from __future__ import annotations

import copy

import torch
from typing import Callable, Dict, List, Optional

try:
    from torchrl.envs.libs.vmas import VmasEnv
except ImportError:
    from torchrl.envs import VmasEnv

from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase, Transform, TransformedEnv

from benchmarl.environments.common import Task
from benchmarl.utils import DEVICE_TYPING

from comm_spread.sar_scenario import SarScenario
from comm_spread.scenario import CommSpreadScenario


class CommSpreadTask(Task):
    """Enum wrapper so BenchMARL can consume CommSpread."""

    SPREAD = None
    SAR_LOW = None
    SAR_HIGH_FIXED = None

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        scenario_type = config.pop("scenario_type", "spread")
        env_continuous_actions = config.pop("continuous_actions", continuous_actions)
        reward_return_norm = config.pop("reward_return_norm", False)
        reward_return_clip = config.pop("reward_return_clip", 10.0)
        reward_return_gamma = config.pop("reward_return_gamma", 0.99)

        def _make_env() -> EnvBase:
            env = VmasEnv(
                scenario=SarScenario()
                if scenario_type in {"sar_low", "sar_high_fixed"}
                else CommSpreadScenario(),
                num_envs=num_envs,
                continuous_actions=env_continuous_actions,
                seed=seed,
                device=device,
                categorical_actions=True,
                clamp_actions=True,
                **config,
            )
            transforms = []
            if config.get("emit_global_state", False):
                transforms.append(GlobalStateFromAgentInfo(group="agents"))
            if reward_return_norm:
                transforms.append(RewardReturnNormalize(
                    group="agents",
                    gamma=reward_return_gamma,
                    clip=reward_return_clip,
                ))
            for transform in transforms:
                env = TransformedEnv(env, transform)
            return env

        return _make_env

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
        if self.config.get("emit_global_state", False):
            n_agents = self.config["n_agents"]
            n_landmarks = self.config["n_landmarks"]
            state_dim = 2 * n_agents + 2 * n_agents + 2 * n_landmarks
            if self.config.get("global_state_include_pairwise", True):
                state_dim += n_agents * n_landmarks
            device = getattr(env, "device", None)
            return Composite(
                {"global_state": Unbounded(shape=(state_dim,), device=device)},
                device=device,
            )
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
        if self.config.get("scenario_type") == "sar_low":
            return None
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

class GlobalStateFromAgentInfo(Transform):
    """Expose per-agent scenario global_state info as a root-level critic state."""

    def __init__(self, group: str):
        self.group = group
        super().__init__(
            in_keys=[(group, "info", "global_state")],
            out_keys=["global_state"],
        )

    def _apply_transform(self, global_state):
        return global_state[..., 0, :].clone()

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        observation_spec = observation_spec.clone()
        info_spec = observation_spec[(self.group, "info", "global_state")]
        observation_spec.set(
            "global_state",
            Unbounded(
                shape=(*observation_spec.shape, *info_spec.shape[-1:]),
                device=info_spec.device,
            ),
        )
        return observation_spec

class RewardReturnNormalize(Transform):
    """Normalize rewards by running discounted-return RMS like old VecNormalize."""

    def __init__(self, group: str, gamma: float = 0.99, clip: float = 10.0, eps: float = 1e-8):
        self.group = group
        self.gamma = gamma
        self.clip = clip
        self.eps = eps
        self.ret = None
        self.mean = None
        self.var = None
        self.count = 1e-4
        super().__init__(in_keys=[], out_keys=[])

    def _ensure_state(self, reward):
        if self.ret is None or self.ret.shape != reward.shape or self.ret.device != reward.device:
            self.ret = torch.zeros_like(reward)
            self.mean = torch.zeros((), device=reward.device, dtype=reward.dtype)
            self.var = torch.ones((), device=reward.device, dtype=reward.dtype)
            self.count = 1e-4

    def _update_rms(self, values):
        flat = values.reshape(-1)
        batch_count = flat.numel()
        if batch_count == 0:
            return
        batch_mean = flat.mean()
        batch_var = flat.var(unbiased=False)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def _step(self, tensordict, next_tensordict):
        reward_key = (self.group, "reward")
        reward = next_tensordict.get(reward_key, default=None)
        if reward is None:
            return next_tensordict
        self._ensure_state(reward)
        self.ret = self.ret * self.gamma + reward
        self._update_rms(self.ret.detach())
        normalized = (reward / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)
        next_tensordict.set(reward_key, normalized)
        done = next_tensordict.get("done", default=None)
        if done is not None:
            done = done.to(dtype=torch.bool)
            while done.ndim < self.ret.ndim:
                done = done.unsqueeze(-1)
            self.ret = torch.where(done, torch.zeros_like(self.ret), self.ret)
        return next_tensordict

