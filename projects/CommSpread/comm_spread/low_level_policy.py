"""Adapters for using a trained BenchMARL sar_low policy in raw VMAS rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.experiment import Experiment

from comm_spread.benchmarl_task import CommSpreadTask
from comm_spread.training_config import (
    TASK_VARIANTS,
    build_experiment_config,
    build_mappo_config,
    build_mlp_configs,
)


class BenchMARLLowLevelPolicy:
    """Load a trained sar_low MAPPO checkpoint and produce raw VMAS actions."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        seed: int = 0,
        deterministic: bool = True,
        max_steps: int | None = None,
        enable_goal_reaching_fallback: bool = False,
        goal_near_threshold: float = 0.25,
        fallback_proportional_gain: float = 2.0,
        fallback_stagnation_steps: int = 5,
        fallback_progress_epsilon: float = 0.005,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.seed = seed
        self.deterministic = deterministic
        self.enable_goal_reaching_fallback = enable_goal_reaching_fallback
        self.goal_near_threshold = goal_near_threshold
        self.fallback_proportional_gain = fallback_proportional_gain
        self.fallback_stagnation_steps = fallback_stagnation_steps
        self.fallback_progress_epsilon = fallback_progress_epsilon
        self._fallback_last_goal: Tensor | None = None
        self._fallback_best_distance: Tensor | None = None
        self._fallback_stagnant_steps: Tensor | None = None
        self._fallback_latched: Tensor | None = None
        self.fallback_action_count = 0
        self.total_action_count = 0
        self.experiment = self._make_experiment(max_steps=max_steps)

    @property
    def source(self) -> str:
        return f"checkpoint:{self.checkpoint}"

    def close(self) -> None:
        self.experiment.close()

    @torch.no_grad()
    def __call__(self, env) -> list[Tensor]:
        scenario = env.scenario
        obs = torch.stack(
            [scenario.observation(agent)["obs"] for agent in env.agents],
            dim=1,
        )
        td = TensorDict(
            {
                "agents": TensorDict(
                    {"observation": TensorDict({"obs": obs}, batch_size=obs.shape[:2])},
                    batch_size=obs.shape[:2],
                ),
                "done": torch.zeros(
                    scenario.world.batch_dim,
                    1,
                    dtype=torch.bool,
                    device=scenario.world.device,
                ),
                "terminated": torch.zeros(
                    scenario.world.batch_dim,
                    1,
                    dtype=torch.bool,
                    device=scenario.world.device,
                ),
            },
            batch_size=[scenario.world.batch_dim],
            device=scenario.world.device,
        )
        exploration = (
            ExplorationType.DETERMINISTIC
            if self.deterministic
            else ExplorationType.RANDOM
        )
        with set_exploration_type(exploration):
            td = self.experiment.policy(td)
        actions = td.get(("agents", "action"))
        actions = actions * scenario.active_agents.unsqueeze(-1).float()
        if self.enable_goal_reaching_fallback:
            actions = self._apply_goal_reaching_fallback(scenario, actions)
        return [actions[:, agent_index] for agent_index in range(scenario.n_agents)]

    def _apply_goal_reaching_fallback(self, scenario: Any, actions: Tensor) -> Tensor:
        """Use a latched proportional fallback without changing the trained policy.

        The fallback reads only the same fresh goal, position, velocity-independent
        navigation state available at execution. It resets whenever a high-level
        goal changes and is disabled by default for exact baseline parity.
        """

        positions = torch.stack([agent.state.pos for agent in scenario.world.agents], dim=1)
        goals = scenario.assigned_goals
        distances = torch.linalg.vector_norm(goals - positions, dim=-1)
        reset_all = bool(torch.all(scenario.world_steps == 0))
        if self._fallback_last_goal is None or reset_all:
            self._fallback_last_goal = goals.clone()
            self._fallback_best_distance = distances.clone()
            self._fallback_stagnant_steps = torch.zeros_like(distances, dtype=torch.long)
            self._fallback_latched = torch.zeros_like(distances, dtype=torch.bool)

        assert self._fallback_best_distance is not None
        assert self._fallback_stagnant_steps is not None
        assert self._fallback_latched is not None
        goal_changed = torch.linalg.vector_norm(
            goals - self._fallback_last_goal, dim=-1
        ) > 1e-4
        improved = distances <= (
            self._fallback_best_distance - self.fallback_progress_epsilon
        )
        reset_progress = goal_changed | improved
        self._fallback_stagnant_steps = torch.where(
            reset_progress,
            torch.zeros_like(self._fallback_stagnant_steps),
            self._fallback_stagnant_steps + 1,
        )
        self._fallback_best_distance = torch.where(
            goal_changed,
            distances,
            torch.minimum(self._fallback_best_distance, distances),
        )
        self._fallback_latched = torch.where(
            goal_changed,
            torch.zeros_like(self._fallback_latched),
            self._fallback_latched,
        )
        trigger = (distances <= self.goal_near_threshold) | (
            self._fallback_stagnant_steps >= self.fallback_stagnation_steps
        )
        self._fallback_latched |= trigger & scenario.active_agents
        fallback_actions = torch.clamp(
            (goals - positions) * self.fallback_proportional_gain,
            -1.0,
            1.0,
        )
        fallback_mask = self._fallback_latched & scenario.active_agents
        self.fallback_action_count += int(fallback_mask.sum().cpu())
        self.total_action_count += int(scenario.active_agents.sum().cpu())
        self._fallback_last_goal = goals.clone()
        return torch.where(fallback_mask.unsqueeze(-1), fallback_actions, actions)

    @property
    def fallback_action_fraction(self) -> float:
        return self.fallback_action_count / max(self.total_action_count, 1)

    def _make_experiment(self, *, max_steps: int | None) -> Experiment:
        task_config: dict[str, Any] = dict(TASK_VARIANTS["sar_low"])
        if max_steps is not None:
            task_config["max_steps"] = max_steps
        model_config, critic_model_config = build_mlp_configs()
        task = CommSpreadTask.SAR_LOW.update_config(task_config)
        Path("outputs/hierarchical_sar_debug/low_level_policy").mkdir(
            parents=True,
            exist_ok=True,
        )
        config = build_experiment_config(
            train_device=self.device,
            sampling_device=self.device,
            quick=True,
            save_folder="outputs/hierarchical_sar_debug/low_level_policy",
            restore_file=str(self.checkpoint),
            restore_map_location=self.device,
            max_n_frames=1_000,
        )
        config.collect_with_grad = True
        config.evaluation = False
        config.render = False
        config.loggers = []
        config.create_json = False

        return Experiment(
            task=task,
            algorithm_config=build_mappo_config(),
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=self.seed,
            config=config,
        )
