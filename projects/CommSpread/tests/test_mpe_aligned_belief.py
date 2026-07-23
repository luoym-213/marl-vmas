from __future__ import annotations

import torch

from comm_spread.env_factory import make_sar_env
from comm_spread.team_belief import label_8_connected


def _set_layout(scenario) -> None:
    agent_positions = [(-0.75, -0.75), (0.75, -0.75), (0.0, 0.75)]
    target_positions = [(-0.70, -0.75), (0.75, 0.75), (-0.75, 0.75)]
    for entity, position in zip(scenario.world.agents, agent_positions):
        entity.set_pos(torch.tensor(position), batch_index=0)
        entity.set_vel(torch.zeros(2), batch_index=0)
    for entity, position in zip(scenario.targets, target_positions):
        entity.set_pos(torch.tensor(position), batch_index=0)


def _clear_belief_state(scenario) -> None:
    scenario.belief_maps.fill_(scenario.initial_belief)
    scenario.target_detected.zero_()
    scenario.target_visited.zero_()
    scenario.target_detected_step.fill_(-1)
    scenario.detected_target_positions.zero_()
    scenario.last_new_target_finders.zero_()
    scenario.target_first_finder_mask.zero_()
    scenario.world_steps.zero_()


def test_eight_connected_labeling_merges_diagonal_cells() -> None:
    binary = torch.zeros(1, 5, 5, dtype=torch.bool)
    binary[0, 1, 1] = True
    binary[0, 2, 2] = True
    binary[0, 4, 4] = True
    labels = label_8_connected(binary)
    assert labels[0, 1, 1] == labels[0, 2, 2]
    assert labels[0, 4, 4] != labels[0, 2, 2]


def test_shared_bayesian_delay_and_persistent_target_nodes() -> None:
    env = make_sar_env(
        num_envs=1,
        device="cpu",
        seed=42,
        max_steps=10,
        sensor_radius=0.3,
        enable_rrt_candidates=False,
    )
    env.reset()
    scenario = env.scenario
    _set_layout(scenario)
    _clear_belief_state(scenario)

    scenario._update_beliefs_and_discoveries()
    assert not scenario.target_detected.any()
    assert torch.allclose(
        scenario.belief_maps[:, 0], scenario.belief_maps[:, 1]
    )
    positive_cell = (
        (scenario.targets[0].state.pos[0] + scenario.world_size / 2.0)
        / scenario.cell_size
    ).long().clamp(0, scenario.map_dim - 1)
    assert torch.isclose(
        scenario.belief_maps[0, 0, positive_cell[0], positive_cell[1]],
        torch.tensor(0.8),
        atol=1e-5,
    )

    scenario._update_beliefs_and_discoveries()
    assert not scenario.target_detected.any()
    assert torch.isclose(
        scenario.belief_maps[0, 0, positive_cell[0], positive_cell[1]],
        torch.tensor(0.94117647),
        atol=1e-5,
    )

    scenario._update_beliefs_and_discoveries()
    assert scenario.target_detected[0, :, 0].all()
    assert not scenario.target_detected[0, :, 1:].any()
    assert scenario.last_joint_fov[0].sum() > scenario.last_individual_fov[0].sum(dim=(-1, -2)).max()
    detected_position = scenario.detected_target_positions[0, 0].clone()
    true_position = scenario.targets[0].state.pos[0]
    assert torch.linalg.vector_norm(detected_position - true_position) <= 0.04

    scenario._refresh_high_level_state()
    assert torch.allclose(
        scenario.detected_targets[0, :, 0, :2],
        detected_position.expand(scenario.n_agents, -1),
    )

    for agent in scenario.world.agents:
        agent.set_pos(torch.tensor([0.75, 0.0]), batch_index=0)
    scenario._update_beliefs_and_discoveries()
    scenario._refresh_high_level_state()
    assert scenario.target_detected[0, :, 0].all()
    assert torch.allclose(
        scenario.detected_targets[0, :, 0, :2],
        detected_position.expand(scenario.n_agents, -1),
        atol=scenario.cell_size,
    )

    scenario.target_visited[0, 0] = True
    scenario._refresh_high_level_state()
    assert torch.count_nonzero(scenario.detected_targets[0, :, 0]) == 0


def test_inactive_agents_continue_updating_shared_belief_by_default() -> None:
    env = make_sar_env(
        num_envs=1,
        device="cpu",
        seed=7,
        max_steps=10,
        sensor_radius=0.3,
        enable_rrt_candidates=False,
        belief_include_inactive_agents=True,
    )
    env.reset()
    scenario = env.scenario
    _set_layout(scenario)
    _clear_belief_state(scenario)
    scenario.active_agents[0, 0] = False

    for _ in range(3):
        scenario._update_beliefs_and_discoveries()

    assert scenario.target_detected[0, :, 0].all()
    assert not scenario.last_new_target_finders[0, :, 0].any()
