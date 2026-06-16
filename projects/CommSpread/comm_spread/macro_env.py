"""Fixed-interval high-level wrapper for SAR macro actions."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


LowLevelPolicy = Callable[[dict[str, Tensor]], list[Tensor]]


class SarFixedIntervalMacroEnv:
    """Wrap a VMAS SAR env with discrete high-level node actions.

    This wrapper is intentionally lightweight. It provides the fixed-interval
    high-level semantics needed to train/debug HGSAR policies before replacing
    BenchMARL's default collector with a custom macro-action collector.
    """

    def __init__(
        self,
        env,
        *,
        high_level_interval: int = 5,
        low_level_policy: LowLevelPolicy | None = None,
        proportional_gain: float = 2.0,
    ) -> None:
        self.env = env
        self.high_level_interval = high_level_interval
        self.low_level_policy = low_level_policy
        self.proportional_gain = proportional_gain

    @property
    def scenario(self):
        return self.env.scenario

    def reset(self):
        return self.env.reset()

    def high_level_observation(self) -> dict[str, Tensor]:
        scenario = self.scenario
        agent_pos = torch.stack([agent.state.pos for agent in self.env.agents], dim=1)
        agent_vel = torch.stack([agent.state.vel for agent in self.env.agents], dim=1)
        battery = 1.0 - (
            scenario.world_steps.float() / max(float(scenario.max_steps), 1.0)
        )
        battery = battery.view(-1, 1, 1).expand(-1, scenario.n_agents, 1)
        ego_nodes = torch.cat([agent_pos, agent_vel, battery], dim=-1)

        dist_to_goal = torch.linalg.vector_norm(
            scenario.assigned_goals - agent_pos,
            dim=-1,
            keepdim=True,
        )
        teammate_nodes = torch.cat([agent_pos, agent_vel, dist_to_goal], dim=-1)

        target_pos = scenario.detected_targets[:, None, :, 0:2] - agent_pos[:, :, None, :]
        target_meta = scenario.detected_targets[:, None, :, 2:4].expand(
            -1,
            scenario.n_agents,
            -1,
            -1,
        )
        target_nodes = torch.cat([target_pos, target_meta], dim=-1)
        target_mask = (
            scenario.target_detected[:, None, :]
            & ~scenario.target_visited[:, None, :]
        ).expand(-1, scenario.n_agents, -1)

        return {
            "ego_nodes": ego_nodes,
            "teammate_nodes": teammate_nodes,
            "explore_nodes": scenario.explore_candidates,
            "target_nodes": target_nodes,
            "target_mask": target_mask,
            "maps": torch.stack(
                [
                    scenario.entropy_map,
                    scenario.agent_heatmap,
                    scenario.landmark_heatmap,
                ],
                dim=1,
            ),
            "active_mask": scenario.active_agents,
        }

    def step(self, high_level_actions: Tensor):
        """Apply high-level node indices and execute low-level actions.

        Args:
            high_level_actions: Long tensor [batch, agents]. Indices smaller
                than ``rrt_top_k`` select exploration candidates. Larger indices
                select detected targets by ``index - rrt_top_k``.
        """

        goals, tasks = self._actions_to_goals(high_level_actions)
        self.scenario.set_high_level_assignments(goals, tasks, self.scenario.active_agents)

        total_rewards = None
        last_obs = last_dones = last_info = None
        for _ in range(self.high_level_interval):
            actions = self._low_level_actions()
            last_obs, rewards, last_dones, last_info = self.env.step(actions)
            stacked_rewards = torch.stack(rewards, dim=1)
            total_rewards = stacked_rewards if total_rewards is None else total_rewards + stacked_rewards
            if bool(last_dones.all()):
                break
        return last_obs, total_rewards, last_dones, last_info

    def _actions_to_goals(self, high_level_actions: Tensor) -> tuple[Tensor, Tensor]:
        scenario = self.scenario
        actions = high_level_actions.to(scenario.assigned_goals.device).long()
        actions = actions.clamp(min=0, max=scenario.rrt_top_k + scenario.n_targets - 1)
        batch_dim = actions.shape[0]
        agent_ids = torch.arange(scenario.n_agents, device=actions.device).view(1, -1)
        batch_ids = torch.arange(batch_dim, device=actions.device).view(-1, 1)

        explore_index = actions.clamp(max=scenario.rrt_top_k - 1)
        rel_goals = scenario.explore_candidates[
            batch_ids,
            agent_ids,
            explore_index,
            0:2,
        ]
        agent_pos = torch.stack([agent.state.pos for agent in self.env.agents], dim=1)
        explore_goals = agent_pos + rel_goals

        target_index = (actions - scenario.rrt_top_k).clamp(min=0, max=scenario.n_targets - 1)
        target_pos = torch.stack([target.state.pos for target in scenario.targets], dim=1)
        target_goals = target_pos[batch_ids, target_index]

        is_target = actions >= scenario.rrt_top_k
        goals = torch.where(is_target.unsqueeze(-1), target_goals, explore_goals)
        tasks = is_target.float()
        return goals, tasks

    def _low_level_actions(self) -> list[Tensor]:
        if self.low_level_policy is not None:
            return self.low_level_policy(self._low_level_policy_input())

        scenario = self.scenario
        actions = []
        for agent_index, agent in enumerate(self.env.agents):
            delta = scenario.assigned_goals[:, agent_index] - agent.state.pos
            action = torch.clamp(delta * self.proportional_gain, -1.0, 1.0)
            action = action * scenario.active_agents[:, agent_index].unsqueeze(-1)
            actions.append(action)
        return actions

    def _low_level_policy_input(self) -> dict[str, Tensor]:
        scenario = self.scenario
        return {
            "assigned_goals": scenario.assigned_goals,
            "assigned_tasks": scenario.assigned_tasks,
            "active_agents": scenario.active_agents,
            "observations": [scenario.observation(agent) for agent in self.env.agents],
        }
