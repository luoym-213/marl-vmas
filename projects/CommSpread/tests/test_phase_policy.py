from __future__ import annotations

import torch

from comm_spread.high_level_policy import HGSARActorCriticPolicy
from comm_spread.models import PhaseConditionedActionHead


def test_phase_policy_is_absent_by_default() -> None:
    policy = HGSARActorCriticPolicy(n_agents=3, rrt_top_k=2)
    assert policy.enable_phase_policy is False
    assert policy.phase_head is None
    assert policy.rescue_distance_logit_scale == 0.0


def test_phase_head_preserves_conditional_node_ranking() -> None:
    head = PhaseConditionedActionHead(
        ego_features=3,
        initial_rescue_logit=0.0,
    )
    base_logits = torch.tensor([[1.0, 0.0, 2.0, -1.0]])
    action_mask = torch.ones_like(base_logits, dtype=torch.bool)
    final_logits, phase_probs = head(
        base_logits,
        torch.zeros(1, 3),
        action_mask,
        rrt_top_k=2,
    )
    final_probs = torch.softmax(final_logits, dim=-1)

    torch.testing.assert_close(phase_probs, torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(final_probs[:, :2].sum(-1), phase_probs[:, 0])
    torch.testing.assert_close(final_probs[:, 2:].sum(-1), phase_probs[:, 1])
    torch.testing.assert_close(
        final_probs[:, 0] / final_probs[:, 1],
        torch.exp(base_logits[:, 0] - base_logits[:, 1]),
    )
    torch.testing.assert_close(
        final_probs[:, 2] / final_probs[:, 3],
        torch.exp(base_logits[:, 2] - base_logits[:, 3]),
    )


def test_phase_head_removes_invalid_phase_without_nan() -> None:
    head = PhaseConditionedActionHead(ego_features=3)
    base_logits = torch.tensor([[0.2, -0.3, 0.7, 0.1]])
    action_mask = torch.tensor([[True, True, False, False]])
    final_logits, phase_probs = head(
        base_logits,
        torch.zeros(1, 3),
        action_mask,
        rrt_top_k=2,
    )
    final_probs = torch.softmax(final_logits, dim=-1)

    assert torch.isfinite(final_probs).all()
    torch.testing.assert_close(phase_probs, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(final_probs[:, 2:], torch.zeros(1, 2))
