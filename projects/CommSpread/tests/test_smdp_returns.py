"""Hand-calculated regression tests for strict high-level SMDP returns."""

from __future__ import annotations

import torch

from comm_spread.smdp_returns import STRICT_SMDP, add_smdp_gae, discounted_option_reward


def _batch(reward, duration, value, next_value, terminal, *, truncated=None):
    n = len(reward)
    return {
        "reward": torch.tensor(reward, dtype=torch.float32).view(-1, 1),
        "duration": torch.tensor(duration, dtype=torch.float32).view(-1, 1),
        "old_value": torch.tensor(value, dtype=torch.float32).view(-1, 1),
        "next_value": torch.tensor(next_value, dtype=torch.float32).view(-1, 1),
        "terminal": torch.tensor(terminal, dtype=torch.float32).view(-1, 1),
        "time_limit_truncated": torch.tensor(truncated or [0] * n, dtype=torch.float32).view(-1, 1),
        "environment_done": torch.tensor(truncated or terminal, dtype=torch.float32).view(-1, 1),
        "train_active_mask": torch.ones(n, 1),
        "done": torch.tensor(terminal, dtype=torch.float32).view(-1, 1),
        "env_id": torch.zeros(n, dtype=torch.long),
        "agent_id": torch.zeros(n, dtype=torch.long),
        "decision_start_t": torch.tensor([0, *duration[:-1]], dtype=torch.long),
    }


def test_hand_calculated_two_option_return() -> None:
    gamma, lam = 0.9, 0.95
    r0 = float(discounted_option_reward([1, 2], gamma))
    r1 = float(discounted_option_reward([3, 4, 5], gamma))
    assert abs(r0 - 2.8) < 1e-12
    assert abs(r1 - 10.65) < 1e-12
    batch = _batch([r0, r1], [2, 3], [10, 8], [8, 0], [0, 1])
    add_smdp_gae(batch, gamma=gamma, gae_lambda=lam, return_mode=STRICT_SMDP)
    delta1 = 10.65 - 8.0
    delta0 = 2.8 + gamma**2 * 8.0 - 10.0
    advantage0 = delta0 + (gamma * lam) ** 2 * delta1
    torch.testing.assert_close(batch["advantage"].flatten(), torch.tensor([advantage0, delta1]))
    torch.testing.assert_close(batch["return"].flatten(), torch.tensor([10 + advantage0, 10.65]))


def test_duration_one_reduces_to_mdp_gae() -> None:
    gamma, lam = 0.9, 0.95
    batch = _batch([1, 2], [1, 1], [3, 4], [4, 0], [0, 1])
    add_smdp_gae(batch, gamma=gamma, gae_lambda=lam, return_mode=STRICT_SMDP)
    delta1 = 2 - 4
    delta0 = 1 + gamma * 4 - 3
    expected = torch.tensor([delta0 + gamma * lam * delta1, delta1])
    torch.testing.assert_close(batch["advantage"].flatten(), expected)


def test_success_and_timeout_have_distinct_bootstrap() -> None:
    success = _batch([1], [2], [2], [99], [1])
    add_smdp_gae(success, gamma=0.9, gae_lambda=0.95, return_mode=STRICT_SMDP)
    assert float(success["advantage"]) == -1.0
    timeout = _batch([1], [2], [2], [5], [0], truncated=[1])
    add_smdp_gae(
        timeout, gamma=0.9, gae_lambda=0.95, return_mode=STRICT_SMDP,
        time_limit_bootstrap=True,
    )
    torch.testing.assert_close(timeout["advantage"], torch.tensor([[1 + 0.9**2 * 5 - 2]]))


if __name__ == "__main__":
    test_hand_calculated_two_option_return()
    test_duration_one_reduces_to_mdp_gae()
    test_success_and_timeout_have_distinct_bootstrap()
    print("SMDP return tests passed")
