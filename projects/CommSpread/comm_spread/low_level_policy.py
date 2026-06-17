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
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.seed = seed
        self.deterministic = deterministic
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
        return [actions[:, agent_index] for agent_index in range(scenario.n_agents)]

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
