import json
from pathlib import Path

import numpy as np

from comm_spread.chapter1_retrospective import _paired_success


def window(successes):
    return {"success_by_layout": {str(i): value for i, value in enumerate(successes)}}


def test_paired_bootstrap_is_reproducible():
    baseline = window([False] * 100 + [True] * 28)
    candidate = window([True] * 20 + [False] * 80 + [True] * 28)
    first = _paired_success(
        baseline,
        candidate,
        rng=np.random.default_rng(7),
        bootstrap_samples=10_000,
    )
    second = _paired_success(
        baseline,
        candidate,
        rng=np.random.default_rng(7),
        bootstrap_samples=10_000,
    )
    assert first == second
    assert first["candidate_only"] == 20
    assert first["baseline_only"] == 0
    assert first["ci95"][0] > 0.05


def test_paired_bootstrap_rejects_weak_gain():
    baseline = window([False] * 100 + [True] * 28)
    candidate = window([True] * 6 + [False] * 94 + [True] * 28)
    result = _paired_success(
        baseline,
        candidate,
        rng=np.random.default_rng(7),
        bootstrap_samples=10_000,
    )
    assert result["difference"] < 0.05
    assert result["ci95"][0] < 0.05


if __name__ == "__main__":
    test_paired_bootstrap_is_reproducible()
    test_paired_bootstrap_rejects_weak_gain()
    print("chapter1 retrospective tests passed")
