from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from comm_spread.low_level_policy import BenchMARLLowLevelPolicy


def _policy(*, stagnation_steps: int = 2) -> BenchMARLLowLevelPolicy:
    policy = BenchMARLLowLevelPolicy.__new__(BenchMARLLowLevelPolicy)
    policy.goal_near_threshold = 0.25
    policy.fallback_proportional_gain = 2.0
    policy.fallback_stagnation_steps = stagnation_steps
    policy.fallback_progress_epsilon = 0.005
    policy._fallback_last_goal = None
    policy._fallback_best_distance = None
    policy._fallback_stagnant_steps = None
    policy._fallback_latched = None
    policy.fallback_action_count = 0
    policy.total_action_count = 0
    return policy


def _scenario(position: float, goal: float) -> SimpleNamespace:
    agent = SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[position, 0.0]])))
    return SimpleNamespace(
        world=SimpleNamespace(agents=[agent]),
        assigned_goals=torch.tensor([[[goal, 0.0]]]),
        active_agents=torch.ones(1, 1, dtype=torch.bool),
        world_steps=torch.ones(1, dtype=torch.long),
    )


def test_goal_reaching_fallback_is_disabled_by_default() -> None:
    parameter = inspect.signature(BenchMARLLowLevelPolicy).parameters[
        "enable_goal_reaching_fallback"
    ]
    assert parameter.default is False


def test_near_goal_fallback_is_proportional_and_latched() -> None:
    policy = _policy()
    scenario = _scenario(0.0, 0.2)
    base = torch.tensor([[[-0.7, 0.4]]])
    result = policy._apply_goal_reaching_fallback(scenario, base)
    torch.testing.assert_close(result, torch.tensor([[[0.4, 0.0]]]))

    scenario.world.agents[0].state.pos = torch.tensor([[0.05, 0.0]])
    result = policy._apply_goal_reaching_fallback(scenario, base)
    torch.testing.assert_close(result, torch.tensor([[[0.3, 0.0]]]))


def test_goal_change_resets_latched_fallback() -> None:
    policy = _policy()
    scenario = _scenario(0.0, 0.2)
    base = torch.tensor([[[-0.7, 0.4]]])
    policy._apply_goal_reaching_fallback(scenario, base)

    scenario.assigned_goals = torch.tensor([[[0.8, 0.0]]])
    result = policy._apply_goal_reaching_fallback(scenario, base)
    torch.testing.assert_close(result, base)


def test_stagnation_triggers_fallback() -> None:
    policy = _policy(stagnation_steps=2)
    scenario = _scenario(0.0, 0.8)
    base = torch.zeros(1, 1, 2)
    torch.testing.assert_close(policy._apply_goal_reaching_fallback(scenario, base), base)
    result = policy._apply_goal_reaching_fallback(scenario, base)
    torch.testing.assert_close(result, torch.tensor([[[1.0, 0.0]]]))
