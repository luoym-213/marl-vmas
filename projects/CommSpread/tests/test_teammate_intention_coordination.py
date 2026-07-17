from __future__ import annotations

import copy

import torch

from comm_spread.high_level_policy import (
    HGSARActorCriticPolicy,
    masked_categorical,
)
from comm_spread.models import TeammateIntentionCoordinator


def make_observation(batch_size: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(4)
    n_agents, n_explore, n_targets = 3, 2, 3
    agent_ids = torch.arange(batch_size) % n_agents
    teammate_mask = torch.ones(batch_size, n_agents, 1)
    teammate_mask[
        torch.arange(batch_size), agent_ids
    ] = 0.0
    target_mask = torch.ones(batch_size, n_targets, dtype=torch.bool)
    action_mask = torch.ones(
        batch_size, n_explore + n_targets, dtype=torch.bool
    )
    coord_agent_nodes = torch.randn(batch_size, n_agents, 10)
    coord_agent_nodes[..., 8] = 1.0
    coord_agent_nodes[..., 9] = 0.0
    coord_agent_nodes[
        torch.arange(batch_size), agent_ids, 9
    ] = 1.0
    return {
        "env_id": torch.zeros(batch_size, dtype=torch.long),
        "agent_id": agent_ids,
        "ego_node": torch.randn(batch_size, 5),
        "teammate_nodes": torch.randn(batch_size, n_agents, 5),
        "teammate_context_nodes": coord_agent_nodes[..., 5:].clone(),
        "teammate_mask": teammate_mask,
        "explore_nodes": torch.randn(batch_size, n_explore, 4),
        "target_nodes": torch.randn(batch_size, n_targets, 4),
        "target_mask": target_mask,
        "intent_target_mask": target_mask.clone(),
        "explore_edges": torch.randn(batch_size, n_explore, 3),
        "target_edges": torch.randn(batch_size, n_targets, 3),
        "action_mask": action_mask,
        "coord_agent_nodes": coord_agent_nodes,
        "agent_active_mask": torch.ones(
            batch_size, n_agents, dtype=torch.bool
        ),
        "agent_decision_mask": torch.ones(
            batch_size, n_agents, dtype=torch.bool
        ),
        "agent_commitment_target_id": torch.full(
            (batch_size, n_agents), -1, dtype=torch.long
        ),
        "target_ids": torch.arange(n_targets).expand(batch_size, -1).clone(),
        "map_channels": torch.randn(batch_size, 3, 16, 16),
        "global_map_channels": torch.randn(
            batch_size, n_agents, 3, 16, 16
        ),
        "global_agent_nodes": torch.randn(batch_size, n_agents, 5),
    }


def test_baseline_logits_are_exact_actor_logits_when_disabled() -> None:
    obs = make_observation()
    policy = HGSARActorCriticPolicy(
        n_agents=3,
        rrt_top_k=2,
        enable_teammate_intention_coordination=False,
    )
    expected = policy.actor(
        obs["ego_node"],
        obs["teammate_nodes"],
        obs["explore_nodes"],
        obs["target_nodes"],
        teammate_mask=obs["teammate_mask"],
        target_mask=obs["target_mask"],
        explore_edges=obs["explore_edges"],
        target_edges=obs["target_edges"],
        action_mask=obs["action_mask"],
    )
    actual = policy._logits(obs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_teacher_tensors_cannot_change_execution_logits() -> None:
    obs = make_observation()
    policy = HGSARActorCriticPolicy(
        n_agents=3,
        rrt_top_k=2,
        enable_teammate_intention_coordination=True,
    )
    expected = policy._logits(obs).detach().clone()
    changed = copy.deepcopy(obs)
    changed["intent_teacher"] = torch.rand(2, 3, 4)
    changed["intent_teacher_mask"] = torch.ones(2, 3, dtype=torch.bool)
    changed["intent_teacher_simultaneous_mask"] = torch.ones(
        2, 3, dtype=torch.bool
    )
    actual = policy._logits(changed).detach()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_soft_gate_suppresses_better_teammate_without_hard_mask() -> None:
    base_logits = torch.zeros(1, 5)
    intent_probs = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.95, 0.0, 0.05]]]
    )
    target_mask = torch.tensor([[True, True]])
    utilities = torch.tensor([[[0.0, 0.0], [3.0, -3.0]]])
    ego_utilities = torch.tensor([[0.0, 0.0]])
    final, pressure, _, _ = TeammateIntentionCoordinator.apply_soft_gate(
        base_logits,
        intent_probs=intent_probs,
        utilities=utilities,
        ego_utilities=ego_utilities,
        ego_agent_ids=torch.tensor([0]),
        agent_active_mask=torch.tensor([[True, True]]),
        target_mask=target_mask,
        rrt_top_k=3,
        beta=0.5,
        margin=0.1,
        temperature=1.0,
    )
    assert final[0, 3] < base_logits[0, 3]
    assert pressure[0, 0] > pressure[0, 1]
    assert final[0, 4] > final[0, 3]
    probs = masked_categorical(
        final, torch.ones_like(final, dtype=torch.bool)
    ).probs
    assert 0.0 < probs[0, 3] < 1.0


def test_async_teacher_and_global_target_identity_mapping() -> None:
    obs = make_observation()
    obs["target_ids"][0] = torch.tensor([0, 1, 2])
    obs["target_ids"][1] = torch.tensor([2, 0, 1])
    obs["agent_active_mask"][:, 2] = False
    policy = HGSARActorCriticPolicy(
        n_agents=3,
        rrt_top_k=2,
        enable_teammate_intention_coordination=True,
    )
    policy._logits(obs)
    base = torch.full((2, 5), -10.0)
    base[:, :2] = 0.0
    base[1, 2:] = torch.tensor([-10.0, 10.0, -10.0])
    policy._coord_output["base_logits"] = base
    labels = policy.build_intent_teacher(obs)
    assert labels is not None
    # Row 1 local slot 1 is global target 0, which maps to row 0 slot 0.
    assert labels["intent_teacher"][0, 1, 0] > 0.99
    assert labels["intent_teacher_simultaneous_mask"][0, 1]
    assert not labels["intent_teacher_mask"][0, 2]

    obs["agent_decision_mask"][:, 1] = False
    obs["agent_decision_mask"][:, 2] = False
    obs["agent_active_mask"][:, 2] = True
    obs["agent_commitment_target_id"][:, 1] = 2
    obs["agent_commitment_target_id"][:, 2] = -1
    policy._logits(obs)
    labels = policy.build_intent_teacher(obs)
    assert labels is not None
    assert labels["intent_teacher"][0, 1, 2] == 1.0
    assert labels["intent_teacher"][0, 2, -1] == 1.0


def test_ppo_log_prob_recompute_uses_same_final_distribution() -> None:
    torch.manual_seed(8)
    obs = make_observation()
    policy = HGSARActorCriticPolicy(
        n_agents=3,
        rrt_top_k=2,
        enable_teammate_intention_coordination=True,
    )
    actions, old_log_prob, _ = policy(obs)
    labels = policy.build_intent_teacher(obs)
    assert labels is not None
    batch = dict(obs)
    batch.update(labels)
    batch["action"] = actions
    evaluated = policy.evaluate_actions(batch)
    torch.testing.assert_close(
        evaluated["log_prob"], old_log_prob, rtol=1e-5, atol=1e-6
    )
    assert torch.isfinite(evaluated["log_prob"]).all()
    probabilities = masked_categorical(
        evaluated["logits"], obs["action_mask"]
    ).probs
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(2), rtol=1e-6, atol=1e-6
    )


def test_agent_permutation_consistency() -> None:
    obs = make_observation(batch_size=1)
    coordinator = TeammateIntentionCoordinator()
    base = torch.randn(1, 5)
    output = coordinator(
        base,
        coord_agent_nodes=obs["coord_agent_nodes"],
        ego_agent_ids=torch.tensor([0]),
        target_nodes=obs["target_nodes"],
        target_mask=obs["target_mask"],
        intent_target_mask=obs["intent_target_mask"],
        agent_active_mask=obs["agent_active_mask"],
        rrt_top_k=2,
    )
    permutation = torch.tensor([0, 2, 1])
    permuted = coordinator(
        base,
        coord_agent_nodes=obs["coord_agent_nodes"][:, permutation],
        ego_agent_ids=torch.tensor([0]),
        target_nodes=obs["target_nodes"],
        target_mask=obs["target_mask"],
        intent_target_mask=obs["intent_target_mask"],
        agent_active_mask=obs["agent_active_mask"][:, permutation],
        rrt_top_k=2,
    )
    torch.testing.assert_close(
        output["final_logits"], permuted["final_logits"], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        output["competition_pressure"],
        permuted["competition_pressure"],
        rtol=1e-5,
        atol=1e-6,
    )


def test_commitment_aware_actor_is_ppo_only_and_uses_fresh_context() -> None:
    obs = make_observation()
    policy = HGSARActorCriticPolicy(
        n_agents=3,
        rrt_top_k=2,
        enable_commitment_aware_actor=True,
    )
    original = policy._logits(obs).detach().clone()

    changed_teacher = copy.deepcopy(obs)
    changed_teacher["intent_teacher"] = torch.rand(2, 3, 4)
    changed_teacher["intent_teacher_mask"] = torch.ones(2, 3, dtype=torch.bool)
    torch.testing.assert_close(
        policy._logits(changed_teacher), original, rtol=0, atol=0
    )

    changed_context = copy.deepcopy(obs)
    changed_context["teammate_context_nodes"][:, 1, 0] += 5.0
    updated = policy._logits(changed_context).detach()
    assert not torch.allclose(updated, original)


def test_commitment_aware_actor_recomputes_log_prob_and_is_permutation_consistent() -> None:
    obs = make_observation(batch_size=1)
    policy = HGSARActorCriticPolicy(
        n_agents=3,
        rrt_top_k=2,
        enable_commitment_aware_actor=True,
    )
    actions, old_log_prob, _ = policy(obs)
    batch = dict(obs)
    batch["action"] = actions
    evaluated = policy.evaluate_actions(batch)
    torch.testing.assert_close(
        evaluated["log_prob"], old_log_prob, rtol=1e-5, atol=1e-6
    )
    assert evaluated["intent_loss"].item() == 0.0

    original = policy._logits(obs).detach()
    permutation = torch.tensor([0, 2, 1])
    permuted = copy.deepcopy(obs)
    for key in (
        "teammate_nodes",
        "teammate_context_nodes",
        "teammate_mask",
    ):
        permuted[key] = permuted[key][:, permutation]
    torch.testing.assert_close(
        policy._logits(permuted), original, rtol=1e-5, atol=1e-6
    )


def test_commitment_aware_actor_is_mutually_exclusive_with_intention_gate() -> None:
    try:
        HGSARActorCriticPolicy(
            n_agents=3,
            rrt_top_k=2,
            enable_commitment_aware_actor=True,
            enable_teammate_intention_coordination=True,
        )
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("experimental paths must be mutually exclusive")
