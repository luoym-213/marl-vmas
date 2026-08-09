"""Tests for the universal Chapter 1 checkpoint finalizer."""

from pathlib import Path

from comm_spread.chapter1_finalizer import (
    _classify_health,
    _select_best_candidate,
)


def candidate(update, success, detect, reward):
    return {
        "update": update,
        "success_rate": success,
        "detect_all_rate": detect,
        "high_reward_mean": reward,
    }


def test_checkpoint_ranking_uses_declared_lexicographic_order():
    items = [
        candidate(10, 0.80, 0.90, 5.0),
        candidate(20, 0.85, 0.80, 1.0),
        candidate(30, 0.85, 0.90, 0.5),
        candidate(40, 0.85, 0.90, 1.5),
    ]
    assert _select_best_candidate(items)["update"] == 40


def test_checkpoint_ranking_prefers_earlier_update_after_ties():
    items = [
        candidate(20, 0.85, 0.90, 1.5),
        candidate(40, 0.85, 0.90, 1.5),
    ]
    assert _select_best_candidate(items)["update"] == 20


def test_completed_gate_failure_remains_finalizable():
    assert _classify_health(
        {"status": "completed_but_gate_failed"}
    ) == "gate_failed"
    assert _classify_health(
        {"status": "passed_for_final_evaluation"}
    ) == "passed"


def test_trainer_has_no_inline_finalizer_call():
    trainer = Path(__file__).resolve().parents[1] / "scripts/train_high_ppo_sar.py"
    source = trainer.read_text(encoding="utf-8")
    assert "write_derived_eval_config" not in source


if __name__ == "__main__":
    test_checkpoint_ranking_uses_declared_lexicographic_order()
    test_checkpoint_ranking_prefers_earlier_update_after_ties()
    test_completed_gate_failure_remains_finalizable()
    test_trainer_has_no_inline_finalizer_call()
    print("chapter1 finalizer tests passed")
