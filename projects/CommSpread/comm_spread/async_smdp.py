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
    teammate_mask: Tensor
    explore_nodes: Tensor
    target_nodes: Tensor
    target_mask: Tensor
    map_channels: Tensor
    global_map_channels: Tensor
    global_agent_nodes: Tensor
    explore_edges: Tensor
    target_edges: Tensor
    action_mask: Tensor


class HighLevelPolicy(Protocol):
    def __call__(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Return action, log_prob, value for each requested decision."""


class LowLevelPolicy(Protocol):
    source: str

    def __call__(self, env) -> list[Tensor]:
        """Return one raw VMAS action tensor per agent."""


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


class GreedyExplorePolicy:
    """Select the exploration node with the highest utility feature."""

    def __call__(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        utilities = observation["explore_nodes"][..., 2]
        actions = utilities.argmax(dim=-1)
        n = actions.shape[0]
        device = actions.device
        return (
            actions.long(),
            torch.zeros(n, 1, device=device),
            torch.zeros(n, 1, device=device),
        )


class TargetFirstPolicy:
    """Prioritize detected targets; otherwise select the highest-utility explore node."""

    def __call__(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        target_mask = observation["target_mask"].bool()
        has_target = target_mask.any(dim=-1)
        first_target = target_mask.float().argmax(dim=-1)
        greedy_explore = observation["explore_nodes"][..., 2].argmax(dim=-1)
        rrt_top_k = observation["explore_nodes"].shape[1]
        actions = torch.where(has_target, rrt_top_k + first_target, greedy_explore)
        n = actions.shape[0]
        device = actions.device
        return (
            actions.long(),
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
        low_level_policy: LowLevelPolicy | None = None,
        proportional_gain: float = 2.0,
    ) -> None:
        self.env = env
        self.policy = high_level_policy or FirstExploreNodePolicy()
        self.low_level_policy = low_level_policy
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
            dones = self.step()
            if bool(dones.all()):
                break
        return self.transitions

    def step(self) -> Tensor:
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
        return dones

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
        if hasattr(self.scenario, "mark_recent_high_level_decisions"):
            candidate_world = self.scenario.recent_rrt_candidate_world.clone()
            agent_pos = torch.stack([agent.state.pos for agent in self.env.agents], dim=1)
            base_pos = agent_pos[env_ids, agent_ids].unsqueeze(1)
            candidate_world[env_ids, agent_ids] = base_pos + obs["explore_nodes"][..., 0:2]
            self.scenario.mark_recent_high_level_decisions(
                decision_mask,
                candidate_world,
            )

        for i in range(len(env_ids)):
            key = (int(env_ids[i]), int(agent_ids[i]))
            self._pending[key] = {
                "start_t": int(self._time[env_ids[i]]),
                "action": actions[i].detach().clone(),
                "log_prob": log_probs[i].detach().clone(),
                "value": values[i].detach().clone(),
                "ego_node": obs["ego_node"][i].detach().clone(),
                "teammate_nodes": obs["teammate_nodes"][i].detach().clone(),
                "teammate_mask": obs["teammate_mask"][i].detach().clone(),
                "explore_nodes": obs["explore_nodes"][i].detach().clone(),
                "target_nodes": obs["target_nodes"][i].detach().clone(),
                "target_mask": obs["target_mask"][i].detach().clone(),
                "map_channels": obs["map_channels"][i].detach().clone(),
                "global_map_channels": obs["global_map_channels"][i].detach().clone(),
                "global_agent_nodes": obs["global_agent_nodes"][i].detach().clone(),
                "explore_edges": obs["explore_edges"][i].detach().clone(),
                "target_edges": obs["target_edges"][i].detach().clone(),
                "action_mask": obs["action_mask"][i].detach().clone(),
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
                    teammate_mask=pending["teammate_mask"],
                    explore_nodes=pending["explore_nodes"],
                    target_nodes=pending["target_nodes"],
                    target_mask=pending["target_mask"],
                    map_channels=pending["map_channels"],
                    global_map_channels=pending["global_map_channels"],
                    global_agent_nodes=pending["global_agent_nodes"],
                    explore_edges=pending["explore_edges"],
                    target_edges=pending["target_edges"],
                    action_mask=pending["action_mask"],
                )
            )
            self._return_accumulator[env_id_t, agent_id_t] = 0

    def _high_level_observation(self, decision_mask: Tensor) -> dict[str, Tensor]:
        scenario = self.scenario
        env_ids, agent_ids = torch.nonzero(decision_mask, as_tuple=True)
        agent_pos = torch.stack([agent.state.pos for agent in self.env.agents], dim=1)
        agent_vel = torch.stack([agent.state.vel for agent in self.env.agents], dim=1)
        battery = 1.0 - scenario.world_steps.float() / max(float(scenario.max_steps), 1.0)
        ego_parts = [
            agent_pos,
            agent_vel,
            battery.view(-1, 1, 1).expand(-1, scenario.n_agents, 1),
        ]
        if getattr(scenario, "high_level_progress_features", False):
            detected_count = scenario.target_detected.any(dim=1).float().sum(dim=-1)
            visited_count = scenario.target_visited.float().sum(dim=-1)
            active_count = scenario.active_agents.float().sum(dim=-1)
            entropy_remaining = scenario._compute_entropy(scenario.belief_maps).mean(dim=(-1, -2, -3))
            progress = torch.stack(
                [
                    detected_count / max(float(scenario.n_targets), 1.0),
                    visited_count / max(float(scenario.n_targets), 1.0),
                    active_count / max(float(scenario.n_agents), 1.0),
                    entropy_remaining,
                ],
                dim=-1,
            )
            ego_parts.append(progress.view(-1, 1, 4).expand(-1, scenario.n_agents, 4))
        ego_nodes_all = torch.cat(ego_parts, dim=-1)
        dist_to_goal = torch.linalg.vector_norm(scenario.assigned_goals - agent_pos, dim=-1, keepdim=True)
        teammate_nodes_all = torch.cat([agent_pos, agent_vel, dist_to_goal], dim=-1)
        teammate_mask_all = scenario.active_agents.unsqueeze(-1).float()
        self_mask = torch.eye(
            scenario.n_agents,
            dtype=torch.bool,
            device=scenario.world.device,
        ).view(1, scenario.n_agents, scenario.n_agents, 1)
        teammate_mask_all = teammate_mask_all.unsqueeze(1).expand(
            -1,
            scenario.n_agents,
            -1,
            -1,
        ).clone()
        teammate_mask_all = teammate_mask_all.masked_fill(self_mask, 0.0)

        entropy_maps = scenario._compute_entropy(scenario.belief_maps)
        target_heatmaps = scenario._target_heatmaps()
        map_channels_all = torch.stack([entropy_maps, scenario.belief_maps, target_heatmaps], dim=2)
        target_pos = scenario.detected_targets[:, :, :, 0:2] - agent_pos[:, :, None, :]
        target_parts = [target_pos, scenario.detected_targets[:, :, :, 2:4]]
        target_claimed = scenario._target_claimed().bool()
        if getattr(scenario, "target_assignment_features", False):
            target_abs_pos = torch.stack([target.state.pos for target in scenario.targets], dim=1)
            collect_task = scenario.assigned_tasks[..., 0] > 0.5
            assigned_dists = torch.cdist(scenario.assigned_goals, target_abs_pos)
            assigned_min_dist, assigned_index = assigned_dists.min(dim=-1)
            assigned_valid = collect_task & (assigned_min_dist <= scenario.goal_radius) & scenario.active_agents
            assigned_one_hot = torch.nn.functional.one_hot(
                assigned_index.clamp_min(0),
                num_classes=scenario.n_targets,
            ).bool()
            assigned_one_hot = assigned_one_hot & assigned_valid.unsqueeze(-1)
            claim_count = assigned_one_hot.float().sum(dim=1) / max(float(scenario.n_agents), 1.0)
            active_target_dists = torch.cdist(agent_pos, target_abs_pos).masked_fill(
                ~scenario.active_agents.unsqueeze(-1),
                torch.inf,
            )
            nearest_active_dist = active_target_dists.min(dim=1).values
            nearest_active_dist = torch.where(
                torch.isfinite(nearest_active_dist),
                nearest_active_dist / max(float(scenario.world_size), 1e-6),
                torch.ones_like(nearest_active_dist),
            )
            detected_age = torch.where(
                scenario.target_detected_step >= 0,
                (scenario.world_steps.view(-1, 1) - scenario.target_detected_step).float()
                / max(float(scenario.max_steps), 1.0),
                torch.zeros_like(nearest_active_dist),
            )
            assignment_features = torch.stack(
                [claim_count, nearest_active_dist, detected_age],
                dim=-1,
            )
            target_parts.append(
                assignment_features.unsqueeze(1).expand(-1, scenario.n_agents, -1, -1)
            )
        target_nodes_all = torch.cat(target_parts, dim=-1)
        target_mask_all = (
            scenario.target_detected
            & ~scenario.target_visited[:, None, :]
            & ~target_claimed
        )
        explore_edges_all = _relative_edges(scenario.explore_candidates[..., 0:2])
        target_edges_all = _relative_edges(target_pos)
        explore_mask_all = torch.ones(
            *scenario.explore_candidates.shape[:3],
            dtype=torch.bool,
            device=scenario.world.device,
        )
        if getattr(scenario, "staged_rescue", False):
            detected_count = scenario.target_detected.any(dim=1).float().sum(dim=-1)
            allow_rescue = detected_count >= float(getattr(scenario, "rescue_detected_threshold", scenario.n_targets))
            entropy_threshold = getattr(scenario, "rescue_entropy_threshold", None)
            if entropy_threshold is not None:
                entropy_remaining = scenario._compute_entropy(scenario.belief_maps).mean(dim=(-1, -2, -3))
                allow_rescue = allow_rescue | (entropy_remaining <= float(entropy_threshold))
            allow_rescue = allow_rescue.view(-1, 1, 1)
            target_mask_all = target_mask_all & allow_rescue
        action_mask_all = torch.cat([explore_mask_all, target_mask_all], dim=-1)

        return {
            "env_id": env_ids,
            "agent_id": agent_ids,
            "ego_node": ego_nodes_all[env_ids, agent_ids],
            "teammate_nodes": teammate_nodes_all[env_ids],
            "teammate_mask": teammate_mask_all[env_ids, agent_ids],
            "explore_nodes": scenario.explore_candidates[env_ids, agent_ids],
            "target_nodes": target_nodes_all[env_ids, agent_ids],
            "target_mask": target_mask_all[env_ids, agent_ids],
            "map_channels": map_channels_all[env_ids, agent_ids],
            "global_map_channels": map_channels_all[env_ids],
            "global_agent_nodes": ego_nodes_all[env_ids],
            "explore_edges": explore_edges_all[env_ids, agent_ids],
            "target_edges": target_edges_all[env_ids, agent_ids],
            "action_mask": action_mask_all[env_ids, agent_ids],
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
        if self.low_level_policy is not None:
            return self.low_level_policy(self.env)

        scenario = self.scenario
        actions = []
        for agent_index, agent in enumerate(self.env.agents):
            delta = scenario.assigned_goals[:, agent_index] - agent.state.pos
            action = torch.clamp(delta * self.proportional_gain, -1.0, 1.0)
            action = action * scenario.active_agents[:, agent_index].unsqueeze(-1)
            actions.append(action)
        return actions


def _relative_edges(relative_xy: Tensor) -> Tensor:
    distance = torch.linalg.vector_norm(relative_xy, dim=-1, keepdim=True)
    safe_distance = distance.clamp_min(1e-6)
    cos_theta = relative_xy[..., 0:1] / safe_distance
    sin_theta = relative_xy[..., 1:2] / safe_distance
    return torch.cat([distance, cos_theta, sin_theta], dim=-1)
