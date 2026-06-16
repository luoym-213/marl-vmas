"""Asynchronous high-level SMDP rollout utilities for SAR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


@dataclass
class HighLevelTransition:
    env_id: int
    agent_id: int
    decision_start_t: int
    decision_end_t: int
    action: Tensor
    log_prob: Tensor
    value: Tensor
    reward: Tensor
    done: Tensor
    duration: int
    ego_node: Tensor
    teammate_nodes: Tensor
    explore_nodes: Tensor
    target_nodes: Tensor
    target_mask: Tensor
    map_channels: Tensor


class HighLevelPolicy(Protocol):
    def __call__(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Return action, log_prob, value for each requested decision."""


class FirstExploreNodePolicy:
    """Deterministic placeholder policy for collector validation."""

    def __call__(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        n = observation["ego_node"].shape[0]
        device = observation["ego_node"].device
        return (
            torch.zeros(n, dtype=torch.long, device=device),
            torch.zeros(n, 1, device=device),
            torch.zeros(n, 1, device=device),
        )


class AsyncSMDPCollector:
    """Collect per-agent high-level transitions at asynchronous decision times."""

    def __init__(
        self,
        env,
        *,
        high_level_policy: HighLevelPolicy | None = None,
        proportional_gain: float = 2.0,
    ) -> None:
        self.env = env
        self.policy = high_level_policy or FirstExploreNodePolicy()
        self.proportional_gain = proportional_gain
        self.transitions: list[HighLevelTransition] = []
        self._pending: dict[tuple[int, int], dict[str, Tensor | int]] = {}
        self._time = torch.zeros(env.scenario.world.batch_dim, dtype=torch.long, device=env.scenario.world.device)
        self._return_accumulator = torch.zeros_like(env.scenario.high_rewards)

    @property
    def scenario(self):
        return self.env.scenario

    def reset(self):
        self.transitions.clear()
        self._pending.clear()
        self.env.reset()
        self._time.zero_()
        self._return_accumulator.zero_()
        self._decide(self.scenario.active_agents)

    def rollout(self, max_low_level_steps: int) -> list[HighLevelTransition]:
        if not self._pending:
            self.reset()

        for _ in range(max_low_level_steps):
            actions = self._low_level_actions()
            _, _, dones, _ = self.env.step(actions)
            self._return_accumulator += self.scenario.high_rewards
            self._time += 1

            decision_mask = (
                self.scenario.goal_done
                | ~self.scenario.active_agents
                | dones.unsqueeze(-1)
            )
            decision_mask &= torch.as_tensor(
                [
                    [((env_id, agent_id) in self._pending) for agent_id in range(self.scenario.n_agents)]
                    for env_id in range(self.scenario.world.batch_dim)
                ],
                dtype=torch.bool,
                device=self.scenario.world.device,
            )
            if decision_mask.any():
                self._finalize(decision_mask, dones)
                self._decide(decision_mask & self.scenario.active_agents & ~dones.unsqueeze(-1))
        return self.transitions

    def _decide(self, decision_mask: Tensor) -> None:
        if not decision_mask.any():
            return
        obs = self._high_level_observation(decision_mask)
        actions, log_probs, values = self.policy(obs)
        goals, tasks = self._actions_to_goals(actions, obs)

        env_ids = obs["env_id"].long()
        agent_ids = obs["agent_id"].long()
        full_goals = self.scenario.assigned_goals.clone()
        full_tasks = self.scenario.assigned_tasks.squeeze(-1).clone()
        full_goals[env_ids, agent_ids] = goals
        full_tasks[env_ids, agent_ids] = tasks.float()
        self.scenario.set_high_level_assignments(full_goals, full_tasks, decision_mask)

        for i in range(len(env_ids)):
            key = (int(env_ids[i]), int(agent_ids[i]))
            self._pending[key] = {
                "start_t": int(self._time[env_ids[i]]),
                "action": actions[i].detach().clone(),
                "log_prob": log_probs[i].detach().clone(),
                "value": values[i].detach().clone(),
                "ego_node": obs["ego_node"][i].detach().clone(),
                "teammate_nodes": obs["teammate_nodes"][i].detach().clone(),
                "explore_nodes": obs["explore_nodes"][i].detach().clone(),
                "target_nodes": obs["target_nodes"][i].detach().clone(),
                "target_mask": obs["target_mask"][i].detach().clone(),
                "map_channels": obs["map_channels"][i].detach().clone(),
            }
            self._return_accumulator[env_ids[i], agent_ids[i]] = 0

    def _finalize(self, decision_mask: Tensor, dones: Tensor) -> None:
        env_ids, agent_ids = torch.nonzero(decision_mask, as_tuple=True)
        for env_id_t, agent_id_t in zip(env_ids, agent_ids):
            env_id = int(env_id_t)
            agent_id = int(agent_id_t)
            key = (env_id, agent_id)
            pending = self._pending.pop(key, None)
            if pending is None:
                continue
            end_t = int(self._time[env_id_t])
            start_t = int(pending["start_t"])
            self.transitions.append(
                HighLevelTransition(
                    env_id=env_id,
                    agent_id=agent_id,
                    decision_start_t=start_t,
                    decision_end_t=end_t,
                    action=pending["action"],
                    log_prob=pending["log_prob"],
                    value=pending["value"],
                    reward=self._return_accumulator[env_id_t, agent_id_t].detach().clone(),
                    done=dones[env_id_t].detach().clone(),
                    duration=max(end_t - start_t, 1),
                    ego_node=pending["ego_node"],
                    teammate_nodes=pending["teammate_nodes"],
                    explore_nodes=pending["explore_nodes"],
                    target_nodes=pending["target_nodes"],
                    target_mask=pending["target_mask"],
                    map_channels=pending["map_channels"],
                )
            )
            self._return_accumulator[env_id_t, agent_id_t] = 0

    def _high_level_observation(self, decision_mask: Tensor) -> dict[str, Tensor]:
        scenario = self.scenario
        env_ids, agent_ids = torch.nonzero(decision_mask, as_tuple=True)
        agent_pos = torch.stack([agent.state.pos for agent in self.env.agents], dim=1)
        agent_vel = torch.stack([agent.state.vel for agent in self.env.agents], dim=1)
        battery = 1.0 - scenario.world_steps.float() / max(float(scenario.max_steps), 1.0)
        ego_nodes_all = torch.cat([agent_pos, agent_vel, battery.view(-1, 1, 1).expand(-1, scenario.n_agents, 1)], dim=-1)
        dist_to_goal = torch.linalg.vector_norm(scenario.assigned_goals - agent_pos, dim=-1, keepdim=True)
        teammate_nodes_all = torch.cat([agent_pos, agent_vel, dist_to_goal], dim=-1)

        entropy_maps = scenario._compute_entropy(scenario.belief_maps)
        target_heatmaps = scenario._target_heatmaps()
        map_channels_all = torch.stack([entropy_maps, scenario.belief_maps, target_heatmaps], dim=2)
        target_pos = scenario.detected_targets[:, :, :, 0:2] - agent_pos[:, :, None, :]
        target_nodes_all = torch.cat([target_pos, scenario.detected_targets[:, :, :, 2:4]], dim=-1)
        target_mask_all = scenario.target_detected & ~scenario.target_visited[:, None, :]

        return {
            "env_id": env_ids,
            "agent_id": agent_ids,
            "ego_node": ego_nodes_all[env_ids, agent_ids],
            "teammate_nodes": teammate_nodes_all[env_ids],
            "explore_nodes": scenario.explore_candidates[env_ids, agent_ids],
            "target_nodes": target_nodes_all[env_ids, agent_ids],
            "target_mask": target_mask_all[env_ids, agent_ids],
            "map_channels": map_channels_all[env_ids, agent_ids],
        }

    def _actions_to_goals(self, actions: Tensor, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        scenario = self.scenario
        actions = actions.long().clamp(min=0, max=scenario.rrt_top_k + scenario.n_targets - 1)
        explore_index = actions.clamp(max=scenario.rrt_top_k - 1)
        rel_goals = obs["explore_nodes"][torch.arange(len(actions), device=actions.device), explore_index, 0:2]
        agent_pos = torch.stack([agent.state.pos for agent in self.env.agents], dim=1)
        base_pos = agent_pos[obs["env_id"].long(), obs["agent_id"].long()]
        explore_goals = base_pos + rel_goals

        target_index = (actions - scenario.rrt_top_k).clamp(min=0, max=scenario.n_targets - 1)
        target_pos = torch.stack([target.state.pos for target in scenario.targets], dim=1)
        target_goals = target_pos[obs["env_id"].long(), target_index]
        is_target = actions >= scenario.rrt_top_k
        return torch.where(is_target.unsqueeze(-1), target_goals, explore_goals), is_target

    def _low_level_actions(self) -> list[Tensor]:
        scenario = self.scenario
        actions = []
        for agent_index, agent in enumerate(self.env.agents):
            delta = scenario.assigned_goals[:, agent_index] - agent.state.pos
            action = torch.clamp(delta * self.proportional_gain, -1.0, 1.0)
            action = action * scenario.active_agents[:, agent_index].unsqueeze(-1)
            actions.append(action)
        return actions
