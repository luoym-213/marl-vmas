from __future__ import annotations

import torch

from comm_spread.env_factory import DEFAULT_SAR_CONFIG
from scripts.analyze_sar_failure_modes import estimated_parallel_rescue_steps


def test_default_sar_horizon_is_100_when_cli_max_steps_is_none() -> None:
    assert DEFAULT_SAR_CONFIG["max_steps"] == 100


def test_parallel_rescue_workload_uses_exact_small_team_matching() -> None:
    agents = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    targets = torch.tensor([[2.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    steps = estimated_parallel_rescue_steps(
        agents,
        torch.tensor([True, True, True]),
        targets,
        torch.tensor([True, True, True]),
        speed_per_step=0.1,
    )
    assert steps == 0


def test_parallel_rescue_workload_reports_insufficient_active_agents() -> None:
    agents = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    targets = torch.tensor([[0.5, 0.0], [1.5, 0.0], [2.0, 0.0]])
    steps = estimated_parallel_rescue_steps(
        agents,
        torch.tensor([True, False, False]),
        targets,
        torch.tensor([True, True, False]),
        speed_per_step=0.1,
    )
    assert steps == float("inf")
