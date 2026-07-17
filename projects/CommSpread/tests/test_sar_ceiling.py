from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.evaluate_sar_ceiling import (
    CoverageSensorOracle,
    DEFAULT_LOW_CHECKPOINT,
    DynamicCoverageSensorOracle,
    minimax_full_matching,
    optimal_partial_matching,
)


def test_exact_minimax_matching_recovers_zero_cost_permutation() -> None:
    agents = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]])
    targets = torch.tensor([[[4.0, 0.0], [0.0, 0.0], [2.0, 0.0]]])
    assignment = minimax_full_matching(agents, targets)
    assert assignment.tolist() == [[1, 2, 0]]


def test_partial_matching_selects_best_agents_and_unique_targets() -> None:
    distances = torch.tensor(
        [
            [9.0, 1.0],
            [1.0, 9.0],
            [5.0, 5.0],
        ]
    )
    matches = optimal_partial_matching(
        row_ids=[5, 6, 7],
        agent_ids=[0, 1, 2],
        target_ids=[0, 1],
        distances=distances,
    )
    assert matches == {5: 1, 6: 0}
    assert len(set(matches.values())) == len(matches)


def test_minimax_matching_rejects_non_bijective_team_sizes() -> None:
    try:
        minimax_full_matching(torch.zeros(1, 2, 2), torch.zeros(1, 3, 2))
    except ValueError as error:
        assert "n_agents == n_targets" in str(error)
    else:
        raise AssertionError("non-bijective full matching must be rejected")


def test_partial_matching_respects_rescue_budget() -> None:
    distances = torch.tensor([[1., 9., 9.], [9., 1., 9.], [9., 9., 1.]])
    matches = optimal_partial_matching([0, 1, 2], [0, 1, 2], [0, 1, 2], distances, max_matches=1)
    assert len(matches) == 1


class _BudgetScenario:
    def __init__(self) -> None:
        self.n_agents = 3
        self.world = type("World", (), {"batch_dim": 1, "device": torch.device("cpu")})()
        self.active_agents = torch.ones(1, 3, dtype=torch.bool)


def test_coverage_rescue_timing_budgets_do_not_use_target_positions() -> None:
    scenario = _BudgetScenario(); detected = torch.tensor([[True, False, False]])
    kwargs = dict(env_id=0, deciding_agent_ids=[0, 1, 2], committed_rescue_count=0, globally_detected=detected)
    assert CoverageSensorOracle(scenario, "immediate").new_rescue_budget(**kwargs) is None
    assert CoverageSensorOracle(scenario, "threshold2").new_rescue_budget(**kwargs) == 0
    assert CoverageSensorOracle(scenario, "all_detected").new_rescue_budget(**kwargs) == 0
    assert CoverageSensorOracle(scenario, "reserve_one").new_rescue_budget(**kwargs) == 2
    assert CoverageSensorOracle(scenario, "reserve_two").new_rescue_budget(**kwargs) == 1
    assert CoverageSensorOracle(scenario, "reserve_undiscovered").new_rescue_budget(**kwargs) == 1

    two_detected = torch.tensor([[True, True, False]])
    kwargs["globally_detected"] = two_detected
    assert CoverageSensorOracle(scenario, "threshold2").new_rescue_budget(**kwargs) is None


def test_default_low_checkpoint_is_project_root_relative() -> None:
    assert DEFAULT_LOW_CHECKPOINT.is_absolute()
    assert DEFAULT_LOW_CHECKPOINT.name == "checkpoint_50040000.pt"


class _DynamicScenario(_BudgetScenario):
    def __init__(self, hidden_target_x: float) -> None:
        super().__init__()
        self.max_steps = 100
        self.world_steps = torch.tensor([40])
        self.target_visited = torch.zeros(1, 3, dtype=torch.bool)
        self.belief_maps = torch.ones(1, 3, 2, 2)
        self._compute_entropy = lambda maps: maps
        positions = [0.2, hidden_target_x, -hidden_target_x]
        self.targets = [
            SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[x, 0.0]])))
            for x in positions
        ]
        self.world.agents = [
            SimpleNamespace(state=SimpleNamespace(pos=torch.tensor([[x, 0.0]])))
            for x in (-0.5, 0.0, 0.5)
        ]


def _dynamic_args() -> SimpleNamespace:
    return SimpleNamespace(
        dynamic_min_searchers=2,
        dynamic_entropy_ratio_threshold=0.58,
        dynamic_entropy_rate_threshold=0.0015,
        dynamic_min_stagnation_step=25,
        dynamic_search_steps_per_target=22.0,
        dynamic_speed_per_step=0.035,
        dynamic_time_margin=8.0,
    )


def test_dynamic_budget_does_not_read_undiscovered_target_positions() -> None:
    detected = torch.tensor([[True, False, False]])
    kwargs = dict(
        env_id=0,
        deciding_agent_ids=[0, 1, 2],
        committed_rescue_count=0,
        globally_detected=detected,
    )
    near_hidden = DynamicCoverageSensorOracle(_DynamicScenario(0.1), _dynamic_args())
    far_hidden = DynamicCoverageSensorOracle(_DynamicScenario(100.0), _dynamic_args())
    assert near_hidden.new_rescue_budget(**kwargs) == far_hidden.new_rescue_budget(**kwargs)
