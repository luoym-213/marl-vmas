"""Finder and assignment event-boundary regressions."""

from __future__ import annotations

import torch

from comm_spread.async_smdp import AsyncSMDPCollector
from comm_spread.env_factory import make_sar_env
from comm_spread.smdp_returns import STRICT_SMDP


def _collector() -> AsyncSMDPCollector:
    env = make_sar_env(
        num_envs=1, device="cpu", seed=11, mode="high", max_steps=100,
        redecide_on_assignment_change=True,
        enable_finder_first_cascade=True,
        dynamic_rescue_release=True,
        finder_cascade_mode="finder_only",
    )
    collector = AsyncSMDPCollector(env, return_mode=STRICT_SMDP)
    collector.reset()
    return collector


def test_finder_first_event_selects_only_finder() -> None:
    collector = _collector()
    scenario = collector.scenario
    scenario.last_new_target_finders.zero_()
    scenario.last_new_target_finders[0, 2, 0] = True
    decision = collector._register_new_finders(
        torch.zeros(1, 3, dtype=torch.bool)
    )
    assert decision.tolist() == [[False, False, True]]


def test_assignment_release_interrupts_only_affected_uav_once() -> None:
    collector = _collector()
    scenario = collector.scenario
    scenario.enable_finder_first_cascade = False
    scenario.dynamic_rescue_release = False
    scenario.redecide_on_detection_change = False
    scenario.redecide_on_assignment_change = True
    scenario.goal_done.zero_()
    scenario.success.zero_()
    scenario.high_rewards.zero_()
    scenario.assigned_tasks.zero_()
    collector._previous_rescue_commitment[:] = torch.tensor([[True, False, False]])
    captured: list[torch.Tensor] = []

    def fake_step(_actions):
        return None, None, torch.tensor([False]), None

    def capture(mask, **_flags):
        captured.append(mask.clone())

    collector.env.step = fake_step
    collector._finalize = capture
    collector._decide = lambda _mask: None
    collector.step()
    collector.step()
    assert len(captured) == 1
    assert captured[0].tolist() == [[True, False, False]]


if __name__ == "__main__":
    test_finder_first_event_selects_only_finder()
    test_assignment_release_interrupts_only_affected_uav_once()
    print("event trigger boundary tests passed")
