from __future__ import annotations

from types import SimpleNamespace

import torch

from comm_spread.async_smdp import AsyncSMDPCollector
from comm_spread.env_factory import DEFAULT_SAR_CONFIG
from scripts.analyze_sar_failure_modes import assignment_quality_snapshot


def _collector(
    agent_x: tuple[float, float, float],
    hidden_target_x: tuple[float, float] = (10.0, -10.0),
) -> AsyncSMDPCollector:
    agents = [
        SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[x, 0.0]])))
        for x in agent_x
    ]
    target_x = (0.5, *hidden_target_x)
    targets = [
        SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[x, 0.0]])))
        for x in target_x
    ]
    scenario = SimpleNamespace(
        world=SimpleNamespace(batch_dim=1, device=torch.device("cpu"), agents=agents),
        targets=targets,
        active_agents=torch.ones(1, 3, dtype=torch.bool),
        target_detected=torch.tensor(
            [[[True, False, False], [True, False, False], [True, False, False]]]
        ),
        target_visited=torch.zeros(1, 3, dtype=torch.bool),
        assigned_tasks=torch.zeros(1, 3, 1),
        assigned_goals=torch.zeros(1, 3, 2),
        goal_radius=0.05,
        max_steps=100,
        world_steps=torch.tensor([10]),
        dynamic_release_speed_per_step=0.035,
        dynamic_release_search_steps_per_target=22.0,
        dynamic_release_time_margin=8.0,
        dynamic_release_entropy_ratio_threshold=0.58,
        dynamic_release_min_stagnation_step=25,
        dynamic_release_entropy_rate_threshold=0.0015,
        dynamic_release_min_searchers=2,
        _target_claimed=lambda: torch.zeros(1, 3, 3, dtype=torch.bool),
    )
    collector = AsyncSMDPCollector.__new__(AsyncSMDPCollector)
    collector.env = SimpleNamespace(scenario=scenario)
    collector._dynamic_initial_entropy = torch.ones(1)
    collector._dynamic_previous_entropy = torch.ones(1)
    collector._dynamic_entropy_rate = torch.ones(1)
    return collector


def test_dynamic_rescue_release_is_disabled_by_default() -> None:
    assert DEFAULT_SAR_CONFIG["dynamic_rescue_release"] is False
    assert DEFAULT_SAR_CONFIG["dynamic_rescue_only_after_all_detected"] is False
    assert DEFAULT_SAR_CONFIG["redecide_on_detection_change"] is False
    assert DEFAULT_SAR_CONFIG["redecide_on_assignment_change"] is False
    assert DEFAULT_SAR_CONFIG["dynamic_release_max_new_agents_per_event"] == 1
    assert DEFAULT_SAR_CONFIG["enable_finder_first_cascade"] is False
    assert DEFAULT_SAR_CONFIG["finder_cascade_mode"] == "immediate"


def test_dynamic_release_does_not_read_undiscovered_target_positions() -> None:
    decision_mask = torch.ones(1, 3, dtype=torch.bool)
    near_hidden = _collector((-0.5, 0.0, 0.4), (0.45, 0.55))
    far_hidden = _collector((-0.5, 0.0, 0.4), (100.0, -100.0))
    torch.testing.assert_close(
        near_hidden._dynamic_release_agent_mask(decision_mask),
        far_hidden._dynamic_release_agent_mask(decision_mask),
    )


def test_dynamic_release_is_agent_permutation_consistent() -> None:
    decision_mask = torch.ones(1, 3, dtype=torch.bool)
    original = _collector((-0.5, 0.0, 0.4))._dynamic_release_agent_mask(
        decision_mask
    )
    swapped = _collector((0.4, 0.0, -0.5))._dynamic_release_agent_mask(
        decision_mask
    )
    torch.testing.assert_close(original[:, [2, 1, 0]], swapped)


def test_all_detected_release_is_staggered() -> None:
    collector = _collector((-0.5, 0.0, 0.4))
    collector.scenario.target_detected[:] = True
    collector.scenario.dynamic_release_max_new_agents_per_event = 1
    allowed = collector._dynamic_release_agent_mask(
        torch.ones(1, 3, dtype=torch.bool)
    )
    assert int(allowed.sum()) == 1


def _cascade_collector(mode: str = "immediate") -> AsyncSMDPCollector:
    collector = _collector((-0.5, 0.0, 0.4))
    scenario = collector.scenario
    scenario.n_agents = 3
    scenario.n_targets = 3
    scenario.enable_finder_first_cascade = True
    scenario.finder_cascade_mode = mode
    scenario.last_new_target_finders = torch.zeros(1, 3, 3, dtype=torch.bool)
    scenario.high_rewards = torch.zeros(1, 3)
    collector._time = torch.tensor([10])
    collector._cascade_active = torch.zeros(1, 3, dtype=torch.bool)
    collector._cascade_stage = torch.zeros(1, 3, dtype=torch.long)
    collector._cascade_wait = torch.zeros(1, 3, dtype=torch.long)
    collector._cascade_finders = torch.zeros(1, 3, 3, dtype=torch.bool)
    collector._cascade_force_mask = torch.zeros(1, 3, dtype=torch.bool)
    collector.cascade_events = [[]]
    return collector


def test_finder_first_only_redecides_the_finder() -> None:
    collector = _cascade_collector()
    collector.scenario.last_new_target_finders[0, 2, 0] = True
    decision = collector._register_new_finders(
        torch.zeros(1, 3, dtype=torch.bool)
    )
    assert decision.tolist() == [[False, False, True]]
    assert collector._cascade_stage[0, 0] == 0


def test_immediate_rejection_diffuses_on_next_event_once() -> None:
    collector = _cascade_collector("immediate")
    collector.scenario.last_new_target_finders[0, 0, 0] = True
    collector._register_new_finders(torch.zeros(1, 3, dtype=torch.bool))
    collector._update_finder_cascades_after_decision(
        torch.tensor([[True, False, False]])
    )
    first = collector._advance_finder_cascades(
        torch.tensor([[True, False, False]]),
        torch.zeros(1, 3, dtype=torch.bool),
    )
    second = collector._advance_finder_cascades(
        torch.tensor([[True, False, False]]),
        torch.zeros(1, 3, dtype=torch.bool),
    )
    assert first.tolist() == [[True, True, True]]
    assert not bool(second.any())


def test_one_event_rejection_waits_one_intervening_event() -> None:
    collector = _cascade_collector("one_event")
    collector.scenario.last_new_target_finders[0, 1, 0] = True
    collector._register_new_finders(torch.zeros(1, 3, dtype=torch.bool))
    collector._update_finder_cascades_after_decision(
        torch.tensor([[False, True, False]])
    )
    commitment = torch.zeros(1, 3, dtype=torch.bool)
    detected = torch.tensor([[True, False, False]])
    assert not bool(collector._advance_finder_cascades(detected, commitment).any())
    assert bool(collector._advance_finder_cascades(detected, commitment).all())
    assert not bool(collector._cascade_force_mask.any())


def test_assignment_quality_detects_crossing_and_regret() -> None:
    event = assignment_quality_snapshot(
        env_id=0,
        step=10,
        agent_positions=torch.tensor([[0.0, 0.0], [10.0, 0.0]]),
        target_positions=torch.tensor([[1.0, 0.0], [9.0, 0.0]]),
        pre_active=torch.tensor([True, True]),
        pre_assigned_target=torch.tensor([-1, -1]),
        post_assigned_target=torch.tensor([1, 0]),
        detected_unvisited=torch.tensor([True, True]),
    )
    assert event is not None
    assert event["crossing_pair_count"] == 1
    assert event["sum_regret"] == 16.0
    assert event["makespan_regret"] == 8.0
    assert event["comparative_advantage_violation_count"] == 2
