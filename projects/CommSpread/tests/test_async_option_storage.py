"""Independent agent-boundary tests for asynchronous option storage."""

from __future__ import annotations

import torch

from comm_spread.async_smdp import AsyncSMDPCollector
from comm_spread.env_factory import make_sar_env
from comm_spread.smdp_returns import STRICT_SMDP


def _collector() -> AsyncSMDPCollector:
    env = make_sar_env(num_envs=1, device="cpu", seed=7, mode="high", max_steps=100)
    collector = AsyncSMDPCollector(env, return_mode=STRICT_SMDP, gamma=0.9)
    collector.reset()
    return collector


def _finalize(collector: AsyncSMDPCollector, mask: torch.Tensor) -> None:
    collector._finalize(
        mask,
        task_success_terminal=torch.tensor([False]),
        environment_done=torch.tensor([False]),
        time_limit_truncated=torch.tensor([False]),
    )


def test_three_uavs_have_independent_durations_and_rewards() -> None:
    collector = _collector()
    collector._time[:] = 7
    for agent_id, start in enumerate((4, 0, 2)):
        collector._pending[(0, agent_id)]["start_t"] = start
    collector._return_accumulator[0] = torch.tensor([1.0, 2.0, 3.0])
    _finalize(collector, torch.ones(1, 3, dtype=torch.bool))
    assert [t.duration for t in collector.transitions] == [3, 7, 5]
    assert [float(t.reward) for t in collector.transitions] == [1.0, 2.0, 3.0]
    assert len({(t.env_id, t.agent_id) for t in collector.transitions}) == 3


def test_finder_interrupts_only_its_own_option() -> None:
    collector = _collector()
    collector._time[:] = 3
    collector._return_accumulator[0] = torch.tensor([1.0, 2.0, 3.0])
    _finalize(collector, torch.tensor([[False, False, True]]))
    assert [(t.env_id, t.agent_id) for t in collector.transitions] == [(0, 2)]
    assert set(collector._pending) == {(0, 0), (0, 1)}
    assert collector._return_accumulator[0, :2].tolist() == [1.0, 2.0]


def test_retirement_closes_only_retired_agent() -> None:
    collector = _collector()
    collector.scenario.active_agents[0, 1] = False
    collector._time[:] = 2
    _finalize(collector, torch.tensor([[False, True, False]]))
    transition = collector.transitions[0]
    assert transition.agent_id == 1
    assert bool(transition.agent_terminal)
    assert bool(transition.terminal)
    assert float(transition.next_value) == 0.0
    assert set(collector._pending) == {(0, 0), (0, 2)}


if __name__ == "__main__":
    test_three_uavs_have_independent_durations_and_rewards()
    test_finder_interrupts_only_its_own_option()
    test_retirement_closes_only_retired_agent()
    print("async option storage tests passed")
