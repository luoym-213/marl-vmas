"""Tests for the end-to-end MAPPO Chapter 1 baseline."""

from __future__ import annotations

from pathlib import Path

import torch

from comm_spread.mappo_baseline import (
    compute_gae,
    load_mappo_baseline_config,
    make_mappo_baseline_env,
    model_from_config,
    scenario_snapshot,
    checkpoint_metadata,
    validate_checkpoint_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PROJECT_ROOT
    / "configs/chapter1/baseline_mappo_r030_seed0_v1.yaml"
)

DENSE_CONFIG = (
    PROJECT_ROOT
    / "configs/chapter1/baseline_mappo_reward_v2_r100_seed0_v1.yaml"
)

def make_env():
    config = load_mappo_baseline_config(CONFIG)
    env = make_mappo_baseline_env(
        config, num_envs=1, device="cpu", seed=7
    )
    env.reset()
    return config, env


def test_snapshot_does_not_expose_undetected_target_truth():
    _, env = make_env()
    scenario = env.scenario
    before = scenario_snapshot(scenario)
    for target in scenario.targets:
        target.set_pos(
            torch.tensor([[0.91, -0.87]], dtype=torch.float32),
            batch_index=None,
        )
    after = scenario_snapshot(scenario)

    assert torch.equal(before["target_mask"], after["target_mask"])
    assert torch.equal(before["target_features"], after["target_features"])
    assert not before["target_mask"].any()


def test_detected_target_is_visible_and_directly_rescuable():
    _, env = make_env()
    scenario = env.scenario
    position = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    scenario.world.agents[0].set_pos(position, batch_index=None)
    scenario.world.agents[0].set_vel(torch.zeros_like(position), batch_index=None)
    scenario.targets[0].set_pos(position, batch_index=None)
    scenario.detected_target_positions[:, 0] = position
    scenario.target_detected[:, :, 0] = True
    scenario.target_visited.zero_()

    snapshot = scenario_snapshot(scenario)
    assert snapshot["target_mask"][0, 0, 0]
    env.step([torch.zeros(1, dtype=torch.long) for _ in range(3)])

    assert scenario.target_visited[0, 0]
    assert not scenario.active_agents[0, 0]


def test_model_shapes_and_inactive_action_mask():
    config, env = make_env()
    scenario = env.scenario
    scenario.active_agents[:, 2] = False
    snapshot = scenario_snapshot(scenario)
    model = model_from_config(config)
    actions, log_prob, value = model.act(snapshot, deterministic=False)

    assert actions.shape == (1, 3)
    assert log_prob.shape == (1, 3)
    assert value.shape == (1,)
    assert actions[0, 2].item() == 0
    assert model.actor_logits(snapshot).shape == (1, 3, 5)


def test_gae_respects_terminal_boundary():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.25]])
    dones = torch.tensor([[False], [True]])
    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        torch.tensor([99.0]),
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert torch.allclose(advantages, torch.tensor([[2.5], [1.75]]))
    assert torch.allclose(returns, torch.tensor([[3.0], [2.0]]))


def test_radius_configs_resolve_to_matching_frozen_tasks():
    for tag, radius in (("030", 0.3), ("040", 0.4), ("050", 0.5)):
        config = load_mappo_baseline_config(
            PROJECT_ROOT
            / f"configs/chapter1/baseline_mappo_r{tag}_seed0_v1.yaml"
        )
        assert config["_task"]["perception"]["sensor_radius"] == radius
        assert config["baseline"]["model"]["communication_radius"] == 2.0
        assert config["evaluation"]["batch_size"] == config["evaluation"]["episodes"]


def make_dense_env():
    config = load_mappo_baseline_config(DENSE_CONFIG)
    env = make_mappo_baseline_env(
        config, num_envs=1, device="cpu", seed=11
    )
    env.reset()
    return config, env


def test_dense_reward_config_and_environment_overrides():
    config, env = make_dense_env()
    scenario = env.scenario
    assert config["_task"]["perception"]["sensor_radius"] == 1.0
    assert config["_reward"]["profile"] == "observable_dense_v2"
    assert scenario.direct_target_progress_scale == 5.0
    assert scenario.collision_penalty == -0.2
    assert scenario.max_collision_penalty == -0.2
    assert scenario.boundary_penalty == -0.05
    assert scenario.time_penalty == 0.01


def test_dense_progress_uses_only_persistent_detected_centroids():
    _, env = make_dense_env()
    scenario = env.scenario
    scenario.target_detected.zero_()
    scenario.target_visited.zero_()
    scenario.previous_direct_agent_positions.fill_(0.75)
    before = scenario._direct_target_progress_rewards().clone()
    scenario.targets[0].set_pos(
        torch.tensor([[-0.91, 0.88]]), batch_index=None
    )
    after_hidden_truth_change = scenario._direct_target_progress_rewards()
    assert torch.equal(before, torch.zeros_like(before))
    assert torch.equal(before, after_hidden_truth_change)

    scenario.target_detected[:, :, 0] = True
    scenario.detected_target_positions[:, 0] = torch.tensor([[0.0, 0.0]])
    scenario.previous_direct_agent_positions[:] = torch.tensor([[[1.0, 0.0]]]).expand(-1, 3, -1)
    for agent in scenario.world.agents:
        agent.set_pos(torch.tensor([[0.8, 0.0]]), batch_index=None)
    toward = scenario._direct_target_progress_rewards()
    assert torch.allclose(toward, torch.full_like(toward, 1.0))

    scenario.previous_direct_agent_positions[:] = torch.tensor([[[0.8, 0.0]]]).expand(-1, 3, -1)
    for agent in scenario.world.agents:
        agent.set_pos(torch.tensor([[1.0, 0.0]]), batch_index=None)
    away = scenario._direct_target_progress_rewards()
    assert torch.allclose(away, torch.full_like(away, -1.0))


def test_dense_reward_components_sum_to_direct_agent_reward():
    _, env = make_dense_env()
    scenario = env.scenario
    actions = [torch.zeros(1, dtype=torch.long) for _ in range(3)]
    env.step(actions)
    component_sum = sum(scenario.direct_reward_components.values())
    assert torch.allclose(component_sum, scenario.agent_rewards)


def test_direct_progress_reset_has_no_cross_episode_jump():
    _, env = make_dense_env()
    scenario = env.scenario
    current = torch.stack(
        [agent.state.pos for agent in scenario.world.agents], dim=1
    )
    assert torch.allclose(scenario.previous_direct_agent_positions, current)


def test_sparse_profile_remains_the_default():
    config, env = make_env()
    assert config["_reward"]["profile"] == "sparse_v1"
    assert config["_reward"]["detected_target_progress_scale"] == 0.0
    assert env.scenario.direct_target_progress_scale == 0.0


def test_dense_checkpoint_requires_reward_profile_provenance():
    config = load_mappo_baseline_config(DENSE_CONFIG)
    checkpoint = {"metadata": checkpoint_metadata(config)}
    del checkpoint["metadata"]["reward_profile"]
    try:
        validate_checkpoint_metadata(checkpoint, config)
    except ValueError as error:
        assert "reward_profile" in str(error)
    else:
        raise AssertionError(
            "dense checkpoint without reward profile was accepted"
        )
