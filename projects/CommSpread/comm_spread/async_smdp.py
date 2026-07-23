"""Asynchronous high-level SMDP rollout utilities for SAR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from comm_spread.smdp_returns import LEGACY_EVENT_STEP, RETURN_MODES, STRICT_SMDP


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
    terminal: Tensor
    task_success_terminal: Tensor
    environment_done: Tensor
    time_limit_truncated: Tensor
    train_active_mask: Tensor
    agent_terminal: Tensor
    next_value: Tensor
    duration: int
    ego_node: Tensor
    teammate_nodes: Tensor
    teammate_context_nodes: Tensor
    teammate_mask: Tensor
    explore_nodes: Tensor
    target_nodes: Tensor
    target_mask: Tensor
    intent_target_mask: Tensor
    map_channels: Tensor
    global_map_channels: Tensor
    global_agent_nodes: Tensor
    explore_edges: Tensor
    target_edges: Tensor
    action_mask: Tensor
    coord_agent_nodes: Tensor
    agent_active_mask: Tensor
    agent_decision_mask: Tensor
    agent_commitment_target_id: Tensor
    target_ids: Tensor
    intent_teacher: Tensor
    intent_teacher_mask: Tensor
    intent_teacher_simultaneous_mask: Tensor
    next_ego_node: Tensor
    next_global_map_channels: Tensor
    next_global_agent_nodes: Tensor
    recurrent_state: Tensor
    next_recurrent_state: Tensor


class HighLevelPolicy(Protocol):
    def __call__(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Return action, log_prob, value for each requested decision."""

    def value(self, observation: dict[str, Tensor]) -> Tensor:
        """Return critic values without sampling or updating actor state."""


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


    def value(self, observation: dict[str, Tensor]) -> Tensor:
        return torch.zeros(
            observation["ego_node"].shape[0], 1,
            device=observation["ego_node"].device,
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


    def value(self, observation: dict[str, Tensor]) -> Tensor:
        return torch.zeros(
            observation["ego_node"].shape[0], 1,
            device=observation["ego_node"].device,
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


    def value(self, observation: dict[str, Tensor]) -> Tensor:
        return torch.zeros(
            observation["ego_node"].shape[0], 1,
            device=observation["ego_node"].device,
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
        coordinated_target_selection: bool = False,
        gamma: float = 0.99,
        return_mode: str = LEGACY_EVENT_STEP,
        training_success_terminal: bool = True,
        environment_continue_after_success: bool = True,
        exclude_post_success_steps_from_training: bool = True,
    ) -> None:
        if return_mode not in RETURN_MODES:
            raise ValueError(f"unknown high-level return mode: {return_mode!r}")
        if not training_success_terminal:
            raise ValueError("task success must be a high-level training terminal")
        if not environment_continue_after_success:
            raise ValueError("Chapter 1 requires environment continuation after success")
        if not exclude_post_success_steps_from_training:
            raise ValueError("post-success environment steps must be excluded from training")
        self.env = env
        self.policy = high_level_policy or FirstExploreNodePolicy()
        self.low_level_policy = low_level_policy
        self.proportional_gain = proportional_gain
        self.coordinated_target_selection = coordinated_target_selection
        self.gamma = float(gamma)
        self.return_mode = return_mode
        self.transitions: list[HighLevelTransition] = []
        self._pending: dict[tuple[int, int], dict[str, Tensor | int]] = {}
        self._time = torch.zeros(env.scenario.world.batch_dim, dtype=torch.long, device=env.scenario.world.device)
        self._return_accumulator = torch.zeros_like(env.scenario.high_rewards)
        self._train_active_mask = torch.ones(
            env.scenario.world.batch_dim, dtype=torch.bool,
            device=env.scenario.world.device,
        )
        self._dynamic_initial_entropy = torch.full_like(self._time, torch.nan, dtype=torch.float)
        self._dynamic_previous_entropy = torch.full_like(
            self._dynamic_initial_entropy, torch.nan
        )
        self._dynamic_entropy_rate = torch.zeros_like(self._dynamic_initial_entropy)
        self._previous_global_detected = torch.zeros(
            env.scenario.world.batch_dim,
            env.scenario.n_targets,
            dtype=torch.bool,
            device=env.scenario.world.device,
        )
        self._previous_rescue_commitment = torch.zeros(
            env.scenario.world.batch_dim,
            env.scenario.n_agents,
            dtype=torch.bool,
            device=env.scenario.world.device,
        )
        self._cascade_active = torch.zeros(
            env.scenario.world.batch_dim,
            env.scenario.n_targets,
            dtype=torch.bool,
            device=env.scenario.world.device,
        )
        self._cascade_stage = torch.zeros_like(self._cascade_active, dtype=torch.long)
        self._cascade_wait = torch.zeros_like(self._cascade_stage)
        self._cascade_finders = torch.zeros(
            env.scenario.world.batch_dim,
            env.scenario.n_targets,
            env.scenario.n_agents,
            dtype=torch.bool,
            device=env.scenario.world.device,
        )
        self._cascade_force_mask = torch.zeros_like(env.scenario.active_agents)
        self.cascade_events: list[list[dict[str, object]]] = [
            [] for _ in range(env.scenario.world.batch_dim)
        ]

    @property
    def scenario(self):
        return self.env.scenario

    def reset(self):
        self.transitions.clear()
        self._pending.clear()
        self.env.reset()
        self._time.zero_()
        self._return_accumulator.zero_()
        self._train_active_mask.fill_(True)
        entropy = self.scenario._compute_entropy(self.scenario.belief_maps).mean(
            dim=(-1, -2, -3)
        )
        self._dynamic_initial_entropy.copy_(entropy)
        self._dynamic_previous_entropy.copy_(entropy)
        self._dynamic_entropy_rate.zero_()
        self._previous_global_detected.copy_(
            self.scenario.target_detected.any(dim=1)
        )
        self._decide(self.scenario.active_agents)
        self._previous_rescue_commitment.copy_(
            self.scenario.assigned_tasks[..., 0] > 0.5
        )
        self._cascade_active.zero_()
        self._cascade_stage.zero_()
        self._cascade_wait.zero_()
        self._cascade_finders.zero_()
        self._cascade_force_mask.zero_()
        self.cascade_events = [
            [] for _ in range(self.scenario.world.batch_dim)
        ]

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
        pending_mask = self._pending_mask()
        active_pending = pending_mask & self._train_active_mask.unsqueeze(-1)
        if self.return_mode == STRICT_SMDP:
            elapsed = torch.zeros_like(self._return_accumulator)
            for (env_id, agent_id), pending in self._pending.items():
                elapsed[env_id, agent_id] = self._time[env_id] - int(pending["start_t"])
            reward_weight = self.gamma ** elapsed
        else:
            reward_weight = torch.ones_like(self._return_accumulator)
        self._return_accumulator += (
            self.scenario.high_rewards * reward_weight * active_pending.float()
        )
        self._time += 1
        if getattr(self.scenario, "dynamic_rescue_release", False):
            entropy = self.scenario._compute_entropy(self.scenario.belief_maps).mean(
                dim=(-1, -2, -3)
            )
            instantaneous_rate = (self._dynamic_previous_entropy - entropy).clamp_min(0)
            self._dynamic_entropy_rate.mul_(0.8).add_(instantaneous_rate * 0.2)
            self._dynamic_previous_entropy.copy_(entropy)

        task_success = self.scenario.success.bool()
        task_success_terminal = self._train_active_mask & task_success
        environment_done = dones.bool()
        time_limit_truncated = environment_done & ~task_success
        decision_mask = self._train_active_mask.unsqueeze(-1) & (
            self.scenario.goal_done
            | ~self.scenario.active_agents
            | environment_done.unsqueeze(-1)
        )
        current_global_detected = self.scenario.target_detected.any(dim=1)
        current_rescue_commitment = self.scenario.assigned_tasks[..., 0] > 0.5
        cascade_decision_mask = torch.zeros_like(self.scenario.active_agents)
        if getattr(self.scenario, "enable_finder_first_cascade", False):
            cascade_decision_mask |= self._advance_finder_cascades(
                current_global_detected, current_rescue_commitment,
            )
            cascade_decision_mask |= self._register_new_finders(current_rescue_commitment)
        if getattr(self.scenario, "redecide_on_detection_change", False):
            if getattr(self.scenario, "enable_finder_first_cascade", False):
                decision_mask |= cascade_decision_mask
            else:
                detection_changed = (
                    current_global_detected != self._previous_global_detected
                ).any(dim=-1)
                decision_mask |= (
                    detection_changed.unsqueeze(-1)
                    & self.scenario.active_agents
                    & ~current_rescue_commitment
                    & self._train_active_mask.unsqueeze(-1)
                )
        self._previous_global_detected.copy_(current_global_detected)
        if getattr(self.scenario, "redecide_on_assignment_change", False):
            changed_agents = current_rescue_commitment != self._previous_rescue_commitment
            decision_mask |= (
                changed_agents
                & self.scenario.active_agents
                & ~current_rescue_commitment
                & self._train_active_mask.unsqueeze(-1)
            )
        self._previous_rescue_commitment.copy_(current_rescue_commitment)
        decision_mask |= task_success_terminal.unsqueeze(-1) & pending_mask
        decision_mask &= pending_mask
        if decision_mask.any():
            self._finalize(
                decision_mask,
                task_success_terminal=task_success_terminal,
                environment_done=environment_done,
                time_limit_truncated=time_limit_truncated,
            )
        self._train_active_mask &= ~task_success_terminal
        redecision_mask = (
            decision_mask
            & self.scenario.active_agents
            & ~environment_done.unsqueeze(-1)
            & self._train_active_mask.unsqueeze(-1)
        )
        if redecision_mask.any():
            self._decide(redecision_mask)
        return dones

    @property
    def train_active_mask(self) -> Tensor:
        return self._train_active_mask.clone()

    def _pending_mask(self) -> Tensor:
        return torch.as_tensor(
            [[(env_id, agent_id) in self._pending
              for agent_id in range(self.scenario.n_agents)]
             for env_id in range(self.scenario.world.batch_dim)],
            dtype=torch.bool, device=self.scenario.world.device,
        )

    def _current_commitment_target_ids(self) -> Tensor:
        scenario = self.scenario
        target_positions = scenario.detected_target_positions
        distances = torch.cdist(scenario.assigned_goals, target_positions)
        minimum, target_ids = distances.min(dim=-1)
        valid = (
            (scenario.assigned_tasks[..., 0] > 0.5)
            & scenario.active_agents
            & (minimum <= scenario.goal_radius)
        )
        return torch.where(valid, target_ids, torch.full_like(target_ids, -1))

    def _cascade_uncommitted_mask(self) -> Tensor:
        return (
            self.scenario.active_agents
            & ~(self.scenario.assigned_tasks[..., 0] > 0.5)
        )

    def _schedule_cascade_diffusion(
        self,
        env_id: int,
        target_id: int,
        *,
        defer_decision: bool = True,
    ) -> Tensor:
        self._cascade_stage[env_id, target_id] = 2
        eligible = self._cascade_uncommitted_mask()[env_id]
        if defer_decision:
            # Rejection is observed after the current policy call, so its
            # diffusion must be scheduled for the next low-level event. Calls
            # made while constructing the current event return the mask
            # directly and must not leave a duplicate decision behind.
            self._cascade_force_mask[env_id] |= eligible
        self.cascade_events[env_id].append(
            {
                "step": int(self._time[env_id]),
                "target_id": target_id,
                "event": "diffuse",
                "eligible_agent_ids": torch.nonzero(
                    eligible, as_tuple=False
                ).flatten().cpu().tolist(),
            }
        )
        return eligible

    def _advance_finder_cascades(
        self,
        globally_detected: Tensor,
        current_commitment: Tensor,
    ) -> Tensor:
        scheduled = self._cascade_force_mask.clone()
        self._cascade_force_mask.zero_()
        waiting = self._cascade_active & (self._cascade_stage == 1) & (self._cascade_wait > 0)
        self._cascade_wait = torch.where(
            waiting,
            self._cascade_wait - 1,
            self._cascade_wait,
        )
        ready = waiting & (self._cascade_wait == 0)
        for env_id, target_id in zip(*torch.nonzero(ready, as_tuple=True)):
            scheduled[env_id] |= self._schedule_cascade_diffusion(
                int(env_id), int(target_id), defer_decision=False
            )
        all_detected = globally_detected.all(dim=-1)
        force_terminal = self._cascade_active & all_detected.unsqueeze(-1)
        for env_id, target_id in zip(*torch.nonzero(force_terminal, as_tuple=True)):
            if self._cascade_stage[env_id, target_id] != 2:
                scheduled[env_id] |= self._schedule_cascade_diffusion(
                    int(env_id), int(target_id), defer_decision=False
                )
        return scheduled & self.scenario.active_agents & ~current_commitment

    def _register_new_finders(self, current_commitment: Tensor) -> Tensor:
        finder_edges = self.scenario.last_new_target_finders
        new_targets = finder_edges.any(dim=1)
        if not bool(new_targets.any()):
            return torch.zeros_like(self.scenario.active_agents)
        agent_positions = torch.stack(
            [agent.state.pos for agent in self.scenario.world.agents], dim=1
        )
        target_positions = torch.stack(
            [target.state.pos for target in self.scenario.targets], dim=1
        )
        agent_target_distance = torch.cdist(agent_positions, target_positions)
        decision = torch.zeros_like(self.scenario.active_agents)
        for env_id, target_id in zip(*torch.nonzero(new_targets, as_tuple=True)):
            env_index = int(env_id)
            target_index = int(target_id)
            finders = finder_edges[env_id, :, target_id]
            self._cascade_active[env_id, target_id] = True
            self._cascade_stage[env_id, target_id] = 0
            self._cascade_wait[env_id, target_id] = 0
            self._cascade_finders[env_id, target_id] = finders
            eligible_finders = (
                finders
                & self.scenario.active_agents[env_id]
                & ~current_commitment[env_id]
            )
            active_distances = agent_target_distance[
                env_id, :, target_id
            ].masked_fill(~self.scenario.active_agents[env_id], torch.inf)
            finder_distance = agent_target_distance[
                env_id, finders, target_id
            ].min()
            self.cascade_events[env_index].append(
                {
                    "step": int(self._time[env_id]),
                    "target_id": target_index,
                    "event": "finder_open",
                    "finder_agent_ids": torch.nonzero(
                        finders, as_tuple=False
                    ).flatten().cpu().tolist(),
                    "finder_distance": float(finder_distance),
                    "nearest_active_distance": float(active_distances.min()),
                    "finder_is_nearest": bool(
                        finder_distance <= active_distances.min() + 1e-6
                    ),
                }
            )
            if bool(eligible_finders.any()):
                decision[env_id] |= eligible_finders
            else:
                self.cascade_events[env_index].append(
                    {
                        "step": int(self._time[env_id]),
                        "target_id": target_index,
                        "event": "finder_unavailable",
                    }
                )
                decision[env_id] |= self._schedule_cascade_diffusion(
                    env_index, target_index, defer_decision=False
                )
            if bool(
                self.scenario.target_detected[env_id].any(dim=0).all()
            ):
                # Terminal rescue phase takes precedence over finder waiting:
                # the staggered gate still exposes at most one new commitment.
                decision[env_id] |= self._schedule_cascade_diffusion(
                    env_index, target_index, defer_decision=False
                )
        return decision

    def _update_finder_cascades_after_decision(
        self,
        decision_mask: Tensor,
    ) -> None:
        if not getattr(self.scenario, "enable_finder_first_cascade", False):
            return
        commitment_target = self._current_commitment_target_ids()
        for env_id, target_id in zip(*torch.nonzero(self._cascade_active, as_tuple=True)):
            env_index = int(env_id)
            target_index = int(target_id)
            committed = torch.nonzero(
                commitment_target[env_id] == target_id, as_tuple=False
            ).flatten()
            if committed.numel():
                committer = int(committed[0])
                self.cascade_events[env_index].append(
                    {
                        "step": int(self._time[env_id]),
                        "target_id": target_index,
                        "event": "accept",
                        "committer_agent_id": committer,
                        "committer_is_finder": bool(
                            self._cascade_finders[env_id, target_id, committer]
                        ),
                    }
                )
                self._cascade_active[env_id, target_id] = False
                continue
            if self._cascade_stage[env_id, target_id] != 0:
                continue
            finder_decided = bool(
                (
                    decision_mask[env_id]
                    & self._cascade_finders[env_id, target_id]
                ).any()
            )
            if not finder_decided:
                continue
            self.cascade_events[env_index].append(
                {
                    "step": int(self._time[env_id]),
                    "target_id": target_index,
                    "event": "reject",
                }
            )
            mode = self.scenario.finder_cascade_mode
            if mode == "immediate":
                self._schedule_cascade_diffusion(env_index, target_index)
            elif mode == "one_event":
                self._cascade_stage[env_id, target_id] = 1
                self._cascade_wait[env_id, target_id] = 2
            else:
                self._cascade_stage[env_id, target_id] = 1
                self._cascade_wait[env_id, target_id] = -1

    def _decide(self, decision_mask: Tensor) -> None:
        if not decision_mask.any():
            return
        obs = self._high_level_observation(decision_mask)
        if self.coordinated_target_selection:
            actions, log_probs, values = self._coordinated_policy_actions(obs)
        else:
            actions, log_probs, values = self.policy(obs)
        teacher_labels = None
        build_teacher = getattr(self.policy, "build_intent_teacher", None)
        if build_teacher is not None:
            teacher_labels = build_teacher(obs)
        if teacher_labels is None:
            n_decisions = obs["ego_node"].shape[0]
            n_agents = self.scenario.n_agents
            n_targets = self.scenario.n_targets
            teacher_labels = {
                "intent_teacher": torch.zeros(
                    n_decisions,
                    n_agents,
                    n_targets + 1,
                    device=obs["ego_node"].device,
                ),
                "intent_teacher_mask": torch.zeros(
                    n_decisions,
                    n_agents,
                    dtype=torch.bool,
                    device=obs["ego_node"].device,
                ),
                "intent_teacher_simultaneous_mask": torch.zeros(
                    n_decisions,
                    n_agents,
                    dtype=torch.bool,
                    device=obs["ego_node"].device,
                ),
            }
        goals, tasks = self._actions_to_goals(actions, obs)

        env_ids = obs["env_id"].long()
        agent_ids = obs["agent_id"].long()
        full_goals = self.scenario.assigned_goals.clone()
        full_tasks = self.scenario.assigned_tasks.squeeze(-1).clone()
        full_goals[env_ids, agent_ids] = goals
        full_tasks[env_ids, agent_ids] = tasks.float()
        self.scenario.set_high_level_assignments(full_goals, full_tasks, decision_mask)
        self._update_finder_cascades_after_decision(decision_mask)
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
                "teammate_context_nodes": obs["teammate_context_nodes"][i].detach().clone(),
                "teammate_mask": obs["teammate_mask"][i].detach().clone(),
                "explore_nodes": obs["explore_nodes"][i].detach().clone(),
                "target_nodes": obs["target_nodes"][i].detach().clone(),
                "target_mask": obs["target_mask"][i].detach().clone(),
                "intent_target_mask": obs["intent_target_mask"][i].detach().clone(),
                "map_channels": obs["map_channels"][i].detach().clone(),
                "global_map_channels": obs["global_map_channels"][i].detach().clone(),
                "global_agent_nodes": obs["global_agent_nodes"][i].detach().clone(),
                "explore_edges": obs["explore_edges"][i].detach().clone(),
                "target_edges": obs["target_edges"][i].detach().clone(),
                "action_mask": obs["action_mask"][i].detach().clone(),
                "coord_agent_nodes": obs["coord_agent_nodes"][i].detach().clone(),
                "agent_active_mask": obs["agent_active_mask"][i].detach().clone(),
                "agent_decision_mask": obs["agent_decision_mask"][i].detach().clone(),
                "agent_commitment_target_id": obs[
                    "agent_commitment_target_id"
                ][i].detach().clone(),
                "target_ids": obs["target_ids"][i].detach().clone(),
                "intent_teacher": teacher_labels["intent_teacher"][i].detach().clone(),
                "intent_teacher_mask": teacher_labels[
                    "intent_teacher_mask"
                ][i].detach().clone(),
                "intent_teacher_simultaneous_mask": teacher_labels[
                    "intent_teacher_simultaneous_mask"
                ][i].detach().clone(),
            }
            self._return_accumulator[env_ids[i], agent_ids[i]] = 0

    def _coordinated_policy_actions(
        self,
        obs: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample simultaneous decisions in a random per-env agent order.

        Targets selected by an earlier agent in the randomized order are removed
        from both masks seen by later agents. The modified masks remain in
        ``obs`` and are therefore the exact masks stored in the PPO rollout.
        """
        n_decisions = obs["env_id"].shape[0]
        obs["target_mask"] = obs["target_mask"].clone()
        obs["action_mask"] = obs["action_mask"].clone()
        actions_by_row: list[Tensor | None] = [None] * n_decisions
        log_probs_by_row: list[Tensor | None] = [None] * n_decisions
        values_by_row: list[Tensor | None] = [None] * n_decisions
        env_ids = obs["env_id"].long()
        target_action_start = self.scenario.rrt_top_k

        for env_id in env_ids.unique(sorted=True):
            row_ids = torch.nonzero(env_ids == env_id, as_tuple=False).flatten()
            random_order = row_ids[
                torch.randperm(row_ids.numel(), device=row_ids.device)
            ]
            selected_targets = torch.zeros(
                self.scenario.n_targets,
                dtype=torch.bool,
                device=row_ids.device,
            )
            for row_tensor in random_order:
                row = int(row_tensor.item())
                available_targets = ~selected_targets
                obs["target_mask"][row] &= available_targets
                obs["action_mask"][row, target_action_start:] &= available_targets
                row_obs = self._observation_row(obs, row, n_decisions)
                row_actions, row_log_probs, row_values = self.policy(row_obs)
                actions_by_row[row] = row_actions
                log_probs_by_row[row] = row_log_probs
                values_by_row[row] = row_values

                action = int(row_actions.item())
                target_index = action - target_action_start
                if 0 <= target_index < self.scenario.n_targets:
                    selected_targets[target_index] = True

        if any(value is None for value in actions_by_row + log_probs_by_row + values_by_row):
            raise RuntimeError("coordinated target selection did not produce every decision")
        return (
            torch.cat(actions_by_row),  # type: ignore[arg-type]
            torch.cat(log_probs_by_row),  # type: ignore[arg-type]
            torch.cat(values_by_row),  # type: ignore[arg-type]
        )

    @staticmethod
    def _observation_row(
        obs: dict[str, Tensor],
        row: int,
        n_decisions: int,
    ) -> dict[str, Tensor]:
        return {
            key: value[row : row + 1]
            if value.ndim > 0 and value.shape[0] == n_decisions
            else value
            for key, value in obs.items()
        }

    def _finalize(
        self,
        decision_mask: Tensor,
        dones: Tensor | None = None,
        *,
        task_success_terminal: Tensor | None = None,
        environment_done: Tensor | None = None,
        time_limit_truncated: Tensor | None = None,
    ) -> None:
        # ``dones`` is retained only for diagnostics that historically called
        # this private method.  The training path always passes all three
        # explicit boundary tensors and never infers semantics from ``done``.
        if environment_done is None:
            environment_done = dones if dones is not None else torch.zeros_like(self._time, dtype=torch.bool)
        if task_success_terminal is None:
            task_success_terminal = torch.zeros_like(environment_done, dtype=torch.bool)
        if time_limit_truncated is None:
            time_limit_truncated = environment_done & ~task_success_terminal
        next_obs = self._high_level_observation(decision_mask)
        value_function = getattr(self.policy, "value", None)
        if value_function is None:
            next_values = torch.zeros(
                next_obs["ego_node"].shape[0], 1, device=self.scenario.world.device
            )
        else:
            next_values = value_function(next_obs).detach()
        next_rows = {
            (int(env_id), int(agent_id)): row
            for row, (env_id, agent_id) in enumerate(
                zip(next_obs["env_id"], next_obs["agent_id"])
            )
        }
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
            row = next_rows[key]
            success_terminal = task_success_terminal[env_id_t]
            agent_terminal = ~self.scenario.active_agents[env_id_t, agent_id_t] & ~success_terminal
            terminal = success_terminal | agent_terminal
            next_value = torch.where(terminal, torch.zeros_like(next_values[row]), next_values[row])
            empty_recurrent = torch.empty(0, dtype=torch.float32, device=self.scenario.world.device)
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
                    done=(terminal | environment_done[env_id_t]).detach().clone(),
                    terminal=terminal.detach().clone(),
                    task_success_terminal=success_terminal.detach().clone(),
                    environment_done=environment_done[env_id_t].detach().clone(),
                    time_limit_truncated=time_limit_truncated[env_id_t].detach().clone(),
                    train_active_mask=torch.ones((), dtype=torch.bool, device=self.scenario.world.device),
                    agent_terminal=agent_terminal.detach().clone(),
                    next_value=next_value.detach().clone(),
                    duration=max(end_t - start_t, 1),
                    ego_node=pending["ego_node"],
                    teammate_nodes=pending["teammate_nodes"],
                    teammate_context_nodes=pending["teammate_context_nodes"],
                    teammate_mask=pending["teammate_mask"],
                    explore_nodes=pending["explore_nodes"],
                    target_nodes=pending["target_nodes"],
                    target_mask=pending["target_mask"],
                    intent_target_mask=pending["intent_target_mask"],
                    map_channels=pending["map_channels"],
                    global_map_channels=pending["global_map_channels"],
                    global_agent_nodes=pending["global_agent_nodes"],
                    explore_edges=pending["explore_edges"],
                    target_edges=pending["target_edges"],
                    action_mask=pending["action_mask"],
                    coord_agent_nodes=pending["coord_agent_nodes"],
                    agent_active_mask=pending["agent_active_mask"],
                    agent_decision_mask=pending["agent_decision_mask"],
                    agent_commitment_target_id=pending[
                        "agent_commitment_target_id"
                    ],
                    target_ids=pending["target_ids"],
                    intent_teacher=pending["intent_teacher"],
                    intent_teacher_mask=pending["intent_teacher_mask"],
                    intent_teacher_simultaneous_mask=pending[
                        "intent_teacher_simultaneous_mask"
                    ],
                    next_ego_node=next_obs["ego_node"][row].detach().clone(),
                    next_global_map_channels=next_obs["global_map_channels"][row].detach().clone(),
                    next_global_agent_nodes=next_obs["global_agent_nodes"][row].detach().clone(),
                    recurrent_state=empty_recurrent.clone(),
                    next_recurrent_state=empty_recurrent.clone(),
                )
            )
            self._return_accumulator[env_id_t, agent_id_t] = 0

    def _dynamic_release_agent_mask(self, decision_mask: Tensor) -> Tensor:
        """Select which agents may choose rescue targets from observable state.

        All inputs are current task state: time, belief entropy, active/committed
        teammates, and positions of targets already detected by at least one UAV.
        Undiscovered target positions are never indexed. The same deterministic
        computation can therefore be executed independently by every agent when
        communication is fresh and unrestricted.
        """
        scenario = self.scenario
        batch = scenario.world.batch_dim
        allowed = torch.zeros_like(scenario.active_agents)
        reason_code = torch.zeros(batch, dtype=torch.long, device=scenario.world.device)
        globally_detected = scenario.target_detected.any(dim=1)
        known_unvisited = globally_detected & ~scenario.target_visited
        target_claimed = scenario._target_claimed().bool()
        observable_detected = scenario.target_detected
        if getattr(scenario, "enable_finder_first_cascade", False):
            observable_detected = globally_detected.unsqueeze(1).expand_as(
                scenario.target_detected
            )
        local_available = (
            observable_detected
            & ~scenario.target_visited.unsqueeze(1)
            & ~target_claimed
        )
        agent_positions = torch.stack(
            [agent.state.pos for agent in scenario.world.agents], dim=1
        )
        target_positions = torch.stack(
            [target.state.pos for target in scenario.targets], dim=1
        )
        collect_task = scenario.assigned_tasks[..., 0].bool()
        assigned_dists = torch.cdist(scenario.assigned_goals, target_positions)
        commitment_valid = (
            collect_task
            & scenario.active_agents
            & (assigned_dists.min(dim=-1).values <= scenario.goal_radius)
        )
        allowed |= commitment_valid

        entropy_ratio = self._dynamic_previous_entropy / self._dynamic_initial_entropy.clamp_min(1e-8)
        for env_id in range(batch):
            active_count = int(scenario.active_agents[env_id].sum())
            committed_count = int(commitment_valid[env_id].sum())
            detected_count = int(known_unvisited[env_id].sum())
            undiscovered = int((~globally_detected[env_id]).sum())
            if detected_count == 0:
                reason_code[env_id] = 0
                continue

            known_positions = target_positions[env_id, known_unvisited[env_id]]
            if undiscovered == 0:
                max_total_rescue = min(detected_count, active_count)
                reason_code[env_id] = 5
            else:
                active_positions = agent_positions[
                    env_id, scenario.active_agents[env_id]
                ]
                nearest_distance = torch.cdist(
                    active_positions.unsqueeze(0), known_positions.unsqueeze(0)
                ).min()
                rescue_eta = nearest_distance / max(
                    float(scenario.dynamic_release_speed_per_step), 1e-6
                )
                search_eta = (
                    float(scenario.dynamic_release_search_steps_per_target)
                    * undiscovered
                    * entropy_ratio[env_id].clamp_min(0.1)
                )
                remaining = float(
                    scenario.max_steps - int(scenario.world_steps[env_id])
                )
                time_pressure = remaining <= float(
                    rescue_eta + search_eta + scenario.dynamic_release_time_margin
                )
                low_unknown = bool(
                    entropy_ratio[env_id]
                    <= float(scenario.dynamic_release_entropy_ratio_threshold)
                )
                stagnated = (
                    int(scenario.world_steps[env_id])
                    >= int(scenario.dynamic_release_min_stagnation_step)
                    and bool(
                        self._dynamic_entropy_rate[env_id]
                        <= float(scenario.dynamic_release_entropy_rate_threshold)
                    )
                )
                normal_capacity = max(
                    0, active_count - int(scenario.dynamic_release_min_searchers)
                )
                release_extra = time_pressure or low_unknown or stagnated
                max_total_rescue = (
                    min(detected_count, active_count)
                    if release_extra
                    else min(detected_count, normal_capacity)
                )
            budget = max(0, max_total_rescue - committed_count)
            if undiscovered > 0:
                if time_pressure:
                    reason_code[env_id] = 2
                elif low_unknown:
                    reason_code[env_id] = 3
                elif stagnated:
                    reason_code[env_id] = 4
                else:
                    reason_code[env_id] = 1
            if budget == 0:
                continue
            budget = min(
                budget,
                max(
                    int(
                        getattr(
                            scenario,
                            "dynamic_release_max_new_agents_per_event",
                            1,
                        )
                    ),
                    1,
                ),
            )

            candidates = torch.nonzero(
                decision_mask[env_id]
                & scenario.active_agents[env_id]
                & ~commitment_valid[env_id]
                & local_available[env_id].any(dim=-1),
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                continue
            candidate_distances = torch.cdist(
                agent_positions[env_id, candidates].unsqueeze(0),
                known_positions.unsqueeze(0),
            ).squeeze(0).min(dim=-1).values
            candidate_positions = agent_positions[env_id, candidates]
            permutation_invariant_tie_break = (
                candidate_positions[:, 0] * 1e-7
                + candidate_positions[:, 1] * 1e-9
            )
            selected = candidates[
                (candidate_distances + permutation_invariant_tie_break)
                .argsort(stable=True)[:budget]
            ]
            allowed[env_id, selected] = True

        scenario.dynamic_release_agent_allowed = allowed
        scenario.dynamic_release_reason_code = reason_code
        scenario.dynamic_release_entropy_ratio = entropy_ratio
        scenario.dynamic_release_entropy_rate = self._dynamic_entropy_rate.clone()
        return allowed

    def _high_level_observation(self, decision_mask: Tensor) -> dict[str, Tensor]:
        scenario = self.scenario
        env_ids, agent_ids = torch.nonzero(decision_mask, as_tuple=True)
        agent_pos = torch.stack([agent.state.pos for agent in self.env.agents], dim=1)
        agent_vel = torch.stack([agent.state.vel for agent in self.env.agents], dim=1)
        target_abs_pos = torch.stack(
            [target.state.pos for target in scenario.targets], dim=1
        )
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
        commitment_rel = scenario.assigned_goals - agent_pos
        collect_task = scenario.assigned_tasks[..., 0] > 0.5
        assigned_dists = torch.cdist(scenario.assigned_goals, target_abs_pos)
        assigned_min_dist, assigned_index = assigned_dists.min(dim=-1)
        commitment_valid = (
            collect_task
            & scenario.active_agents
            & (assigned_min_dist <= scenario.goal_radius)
        )
        commitment_target_id_all = torch.where(
            commitment_valid,
            assigned_index,
            torch.full_like(assigned_index, -1),
        )
        teammate_context_nodes_all = torch.cat(
            [
                commitment_rel,
                collect_task.unsqueeze(-1).float(),
                scenario.active_agents.unsqueeze(-1).float(),
                decision_mask.unsqueeze(-1).float(),
            ],
            dim=-1,
        )
        coord_agent_nodes_all = torch.cat(
            [teammate_nodes_all, teammate_context_nodes_all],
            dim=-1,
        )
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
        if getattr(scenario, "enable_finder_first_cascade", False):
            globally_detected = scenario.target_detected.any(dim=1)
            intent_target_mask_all = (
                globally_detected[:, None, :].expand_as(scenario.target_detected)
                & ~scenario.target_visited[:, None, :]
            )
        else:
            intent_target_mask_all = (
                scenario.target_detected
                & ~scenario.target_visited[:, None, :]
            )
        target_mask_all = (
            intent_target_mask_all
            & ~target_claimed
        )
        explore_edges_all = _relative_edges(scenario.explore_candidates[..., 0:2])
        target_edges_all = _relative_edges(target_pos)
        explore_mask_all = torch.ones(
            *scenario.explore_candidates.shape[:3],
            dtype=torch.bool,
            device=scenario.world.device,
        )
        if getattr(scenario, "dynamic_rescue_release", False):
            released_agents = self._dynamic_release_agent_mask(decision_mask)
            target_mask_all = target_mask_all & released_agents.unsqueeze(-1)
        if getattr(scenario, "enable_finder_first_cascade", False):
            permission = torch.ones_like(target_mask_all)
            for env_id, target_id in zip(
                *torch.nonzero(self._cascade_active, as_tuple=True)
            ):
                if self._cascade_stage[env_id, target_id] < 2:
                    permission[env_id, :, target_id] = self._cascade_finders[
                        env_id, target_id
                    ]
                else:
                    permission[env_id, :, target_id] = (
                        scenario.active_agents[env_id]
                        & ~(scenario.assigned_tasks[env_id, :, 0] > 0.5)
                    )
            target_mask_all &= permission
            scenario.finder_cascade_active = self._cascade_active.clone()
            scenario.finder_cascade_stage = self._cascade_stage.clone()
        if getattr(scenario, "staged_rescue", False):
            detected_count = scenario.target_detected.any(dim=1).float().sum(dim=-1)
            allow_rescue = detected_count >= float(getattr(scenario, "rescue_detected_threshold", scenario.n_targets))
            entropy_threshold = getattr(scenario, "rescue_entropy_threshold", None)
            if entropy_threshold is not None:
                entropy_remaining = scenario._compute_entropy(scenario.belief_maps).mean(dim=(-1, -2, -3))
                allow_rescue = allow_rescue | (entropy_remaining <= float(entropy_threshold))
            allow_rescue = allow_rescue.view(-1, 1, 1)
            target_mask_all = target_mask_all & allow_rescue
        if getattr(scenario, "dynamic_rescue_only_after_all_detected", False):
            # This observable terminal-phase rule is intentionally independent
            # of how rescue was released before all targets became known.
            all_detected = scenario.target_detected.any(dim=1).all(dim=-1)
            rescue_available = target_mask_all.any(dim=-1)
            rescue_only = all_detected.unsqueeze(-1) & rescue_available
            explore_mask_all = explore_mask_all & ~rescue_only.unsqueeze(-1)
        action_mask_all = torch.cat([explore_mask_all, target_mask_all], dim=-1)
        target_ids_all = torch.arange(
            scenario.n_targets,
            device=scenario.world.device,
        ).view(1, 1, -1).expand(
            scenario.world.batch_dim,
            scenario.n_agents,
            -1,
        )

        return {
            "env_id": env_ids,
            "agent_id": agent_ids,
            "ego_node": ego_nodes_all[env_ids, agent_ids],
            "teammate_nodes": teammate_nodes_all[env_ids],
            "teammate_context_nodes": teammate_context_nodes_all[env_ids],
            "teammate_mask": teammate_mask_all[env_ids, agent_ids],
            "explore_nodes": scenario.explore_candidates[env_ids, agent_ids],
            "target_nodes": target_nodes_all[env_ids, agent_ids],
            "target_mask": target_mask_all[env_ids, agent_ids],
            "intent_target_mask": intent_target_mask_all[env_ids, agent_ids],
            "map_channels": map_channels_all[env_ids, agent_ids],
            "global_map_channels": map_channels_all[env_ids],
            "global_agent_nodes": ego_nodes_all[env_ids],
            "explore_edges": explore_edges_all[env_ids, agent_ids],
            "target_edges": target_edges_all[env_ids, agent_ids],
            "action_mask": action_mask_all[env_ids, agent_ids],
            "coord_agent_nodes": coord_agent_nodes_all[env_ids],
            "agent_active_mask": scenario.active_agents[env_ids],
            "agent_decision_mask": decision_mask[env_ids],
            "agent_commitment_target_id": commitment_target_id_all[env_ids],
            "target_ids": target_ids_all[env_ids, agent_ids],
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
        global_target_id = obs["target_ids"].gather(
            1, target_index.view(-1, 1)
        ).squeeze(1)
        target_pos = torch.stack([target.state.pos for target in scenario.targets], dim=1)
        target_goals = target_pos[obs["env_id"].long(), global_target_id]
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
