"""Regression tests for the strict MPE-compatible SAR physics profile."""

from __future__ import annotations

import math

import torch

from comm_spread.env_factory import make_sar_env
from comm_spread.mpe_physics import MPEParityWorld


def _strict_env(*, max_steps: int = 20):
    return make_sar_env(
        num_envs=1,
        device="cpu",
        seed=7,
        max_steps=max_steps,
        physics_profile="mpe_strict",
        enable_rrt_candidates=False,
    )


def _separate_entities(env) -> None:
    scenario = env.scenario
    agent_positions = [(0.0, 0.0), (-8.0, 8.0), (8.0, 8.0)]
    target_positions = [(-8.0, -8.0), (8.0, -8.0), (0.0, 8.0)]
    for agent, position in zip(scenario.world.agents, agent_positions):
        agent.set_pos(torch.tensor(position), batch_index=0)
        agent.set_vel(torch.zeros(2), batch_index=0)
    for target, position in zip(scenario.targets, target_positions):
        target.set_pos(torch.tensor(position), batch_index=0)


def _actions(action_id: int) -> list[torch.Tensor]:
    return [torch.tensor([action_id]), torch.tensor([0]), torch.tensor([0])]


def test_strict_profile_configuration_and_action_guard() -> None:
    env = _strict_env()
    scenario = env.scenario
    assert isinstance(scenario.world, MPEParityWorld)
    assert env.continuous_actions is False
    assert scenario.world._substeps == 1
    assert scenario.world._dt == 0.1
    assert scenario.world._drag == 0.25
    assert scenario.world._x_semidim is None
    assert scenario.world._y_semidim is None
    assert scenario.world.agents[0].action_size == 1
    assert scenario.world.agents[0].discrete_action_nvec == [5]

    try:
        make_sar_env(
            num_envs=1,
            physics_profile="mpe_strict",
            continuous_actions=True,
        )
    except ValueError as error:
        assert "requires continuous_actions=False" in str(error)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("strict profile accepted a continuous action space")


def test_all_five_actions_match_mpe_first_step() -> None:
    expected = {
        0: (0.0, 0.0),
        1: (-0.05, 0.0),
        2: (0.05, 0.0),
        3: (0.0, -0.05),
        4: (0.0, 0.05),
    }
    for action_id, expected_delta in expected.items():
        env = _strict_env(max_steps=2)
        env.reset()
        _separate_entities(env)
        env.step(_actions(action_id))
        observed = env.scenario.world.agents[0].state.pos[0]
        assert torch.allclose(
            observed,
            torch.tensor(expected_delta),
            atol=1e-7,
            rtol=0.0,
        )


def test_repeated_right_matches_mpe_recurrence() -> None:
    env = _strict_env(max_steps=12)
    env.reset()
    _separate_entities(env)
    agent = env.scenario.world.agents[0]
    velocity = 0.0
    position = 0.0
    for _ in range(10):
        env.step(_actions(2))
        velocity = velocity * 0.75 + 5.0 * 0.1
        position += velocity * 0.1
        assert math.isclose(float(agent.state.vel[0, 0]), velocity, abs_tol=1e-6)
        assert math.isclose(float(agent.state.pos[0, 0]), position, abs_tol=1e-6)


def test_soft_collision_matches_mpe_equation() -> None:
    env = _strict_env(max_steps=2)
    env.reset()
    scenario = env.scenario
    left, right, distant = scenario.world.agents
    left.set_pos(torch.tensor([-0.04, 0.0]), batch_index=0)
    right.set_pos(torch.tensor([0.04, 0.0]), batch_index=0)
    distant.set_pos(torch.tensor([8.0, 8.0]), batch_index=0)
    for agent in scenario.world.agents:
        agent.set_vel(torch.zeros(2), batch_index=0)
    for target in scenario.targets:
        target.set_pos(torch.tensor([8.0, -8.0]), batch_index=0)

    env.step([torch.tensor([0]), torch.tensor([0]), torch.tensor([0])])
    penetration = math.log1p(math.exp((0.1 - 0.08) / 0.001)) * 0.001
    expected_velocity = -100.0 * penetration * 0.1
    expected_position = -0.04 + expected_velocity * 0.1
    assert math.isclose(float(left.state.vel[0, 0]), expected_velocity, abs_tol=1e-6)
    assert math.isclose(float(left.state.pos[0, 0]), expected_position, abs_tol=1e-6)
    assert math.isclose(float(right.state.vel[0, 0]), -expected_velocity, abs_tol=1e-6)


def test_boundary_is_unclamped_and_uses_mpe_center_penalty() -> None:
    env = _strict_env(max_steps=2)
    env.reset()
    _separate_entities(env)
    scenario = env.scenario
    agent = scenario.world.agents[0]
    agent.set_pos(torch.tensor([0.99, 0.0]), batch_index=0)
    agent.set_vel(torch.tensor([0.5, 0.0]), batch_index=0)
    env.step(_actions(2))
    assert float(agent.state.pos[0, 0]) > 1.0
    penalties = scenario._boundary_penalties()
    assert float(penalties[0, 0]) == -2.0


def test_legacy_first_step_is_unchanged() -> None:
    env = make_sar_env(
        num_envs=1,
        device="cpu",
        seed=7,
        max_steps=2,
        physics_profile="legacy",
        enable_rrt_candidates=False,
    )
    env.reset()
    _separate_entities(env)
    env.step([torch.tensor([[1.0, 0.0]]), torch.zeros(1, 2), torch.zeros(1, 2)])
    agent = env.scenario.world.agents[0]
    assert math.isclose(float(agent.state.pos[0, 0]), 0.006, abs_tol=1e-7)
    assert math.isclose(float(agent.state.vel[0, 0]), 0.1, abs_tol=1e-7)
