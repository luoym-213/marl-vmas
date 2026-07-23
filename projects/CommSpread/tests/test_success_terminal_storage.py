"""Task-success terminal, post-success tail, timeout, and vector-env tests."""

from __future__ import annotations

import torch
from tempfile import TemporaryDirectory
from pathlib import Path

from comm_spread.async_smdp import AsyncSMDPCollector, FirstExploreNodePolicy
from comm_spread.env_factory import make_sar_env
from comm_spread.smdp_returns import STRICT_SMDP
from comm_spread.chapter1_config import resolve_chapter1_config
from scripts.train_high_ppo_sar import load_checkpoint, select_train_active_batch


class CountingPolicy(FirstExploreNodePolicy):
    def __init__(self) -> None:
        self.actor_env_ids: list[list[int]] = []
        self.value_env_ids: list[list[int]] = []

    def __call__(self, observation):
        self.actor_env_ids.append(observation["env_id"].cpu().tolist())
        return super().__call__(observation)

    def value(self, observation):
        self.value_env_ids.append(observation["env_id"].cpu().tolist())
        return torch.full((observation["ego_node"].shape[0], 1), 5.0)


def test_vector_success_closes_all_options_and_excludes_tail() -> None:
    env = make_sar_env(num_envs=2, device="cpu", seed=19, mode="high", max_steps=100)
    policy = CountingPolicy()
    collector = AsyncSMDPCollector(
        env, high_level_policy=policy, return_mode=STRICT_SMDP, gamma=0.9,
    )
    collector.reset()
    counter = {"step": 0}

    def fake_step(_actions):
        counter["step"] += 1
        env.scenario.high_rewards.fill_(1.0)
        if counter["step"] == 3:
            env.scenario.success[0] = True
        if counter["step"] == 5:
            env.scenario.success[1] = True
        return None, None, torch.zeros(2, dtype=torch.bool), None

    env.step = fake_step
    for _ in range(8):
        collector.step()
    env0 = [t for t in collector.transitions if t.env_id == 0]
    env1 = [t for t in collector.transitions if t.env_id == 1]
    assert len(env0) == len(env1) == 3
    assert {t.decision_end_t for t in env0} == {3}
    assert {t.decision_end_t for t in env1} == {5}
    assert all(bool(t.task_success_terminal) and bool(t.terminal) for t in env0 + env1)
    assert all(float(t.next_value) == 0.0 for t in env0 + env1)
    torch.testing.assert_close(torch.stack([t.reward for t in env0]), torch.full((3,), 1 + 0.9 + 0.9**2))
    torch.testing.assert_close(torch.stack([t.reward for t in env1]), torch.full((3,), sum(0.9**i for i in range(5))))
    assert collector.train_active_mask.tolist() == [False, False]
    assert not collector._pending
    assert counter["step"] == 8
    assert len(policy.actor_env_ids) == 1  # initial decisions only; none in success tails


def test_timeout_is_truncated_and_bootstraps_horizon_value() -> None:
    env = make_sar_env(num_envs=1, device="cpu", seed=23, mode="high", max_steps=100)
    policy = CountingPolicy()
    collector = AsyncSMDPCollector(env, high_level_policy=policy, return_mode=STRICT_SMDP)
    collector.reset()

    def fake_step(_actions):
        env.scenario.high_rewards.fill_(0.0)
        return None, None, torch.tensor([True]), None

    env.step = fake_step
    collector.step()
    assert len(collector.transitions) == 3
    for transition in collector.transitions:
        assert not bool(transition.task_success_terminal)
        assert bool(transition.environment_done)
        assert bool(transition.time_limit_truncated)
        assert not bool(transition.terminal)
        assert float(transition.next_value) == 5.0


def test_post_success_rows_are_removed_before_all_losses() -> None:
    batch = {
        "action": torch.tensor([0, 1, 2]),
        "train_active_mask": torch.tensor([[1.0], [0.0], [1.0]]),
        "old_log_prob": torch.tensor([[10.0], [999.0], [30.0]]),
        "return": torch.tensor([[1.0], [999.0], [3.0]]),
        "scalar": torch.tensor(7.0),
    }
    selected = select_train_active_batch(batch)
    assert selected["action"].tolist() == [0, 2]
    assert selected["old_log_prob"].flatten().tolist() == [10.0, 30.0]
    assert selected["return"].flatten().tolist() == [1.0, 3.0]
    assert float(selected["scalar"]) == 7.0


def test_chapter1_strict_config_and_resume_guard_fail_closed() -> None:
    config = Path(__file__).resolve().parents[1] / "configs/chapter1/high_train_v1.yaml"
    resolved = resolve_chapter1_config(config)
    high = resolved.run["training"]["high_level"]
    assert high["return_mode"] == STRICT_SMDP
    assert high["training_success_terminal"] is True
    assert high["environment_continue_after_success"] is True
    assert high["exclude_post_success_steps_from_training"] is True
    assert resolved.task_config_sha256 == "1ac17a9cb5c0bc7631b0c615c5bc52d90b30573acc983f4742df508bdd0e267c"
    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "legacy.pt"
        torch.save({"config": {"high_level_return_mode": "legacy_event_step"}}, checkpoint)
        try:
            load_checkpoint(
                checkpoint, None, None, "cpu",
                expected_return_mode=STRICT_SMDP,
                expected_task_config_sha256=resolved.task_config_sha256,
            )
        except ValueError as error:
            assert "strict SMDP resume rejected" in str(error)
        else:
            raise AssertionError("legacy checkpoint was accepted for strict resume")


if __name__ == "__main__":
    test_vector_success_closes_all_options_and_excludes_tail()
    test_timeout_is_truncated_and_bootstraps_horizon_value()
    test_post_success_rows_are_removed_before_all_losses()
    test_chapter1_strict_config_and_resume_guard_fail_closed()
    print("success terminal storage tests passed")
