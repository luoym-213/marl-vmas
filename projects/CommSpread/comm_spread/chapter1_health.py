"""Pure Chapter 1 training-health decisions."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ValidationWindow:
    update: int
    success_rate: float
    high_reward_mean: float

@dataclass(frozen=True)
class TrainingDiagnostics:
    update: int
    effective_options: int
    approx_kl: float

@dataclass(frozen=True)
class HealthDecision:
    status: Literal["continue_training", "early_stop_failed", "completed_but_gate_failed", "passed_for_final_evaluation"]
    reason: str
    @property
    def continue_training(self): return self.status == "continue_training"
    @property
    def early_stop_failed(self): return self.status == "early_stop_failed"
    @property
    def completed_but_gate_failed(self): return self.status == "completed_but_gate_failed"
    @property
    def passed_for_final_evaluation(self): return self.status == "passed_for_final_evaluation"

def evaluate_training_health(
    validation_windows: list[ValidationWindow],
    diagnostics: list[TrainingDiagnostics],
    *, target_update: int = 200,
) -> HealthDecision:
    windows = _unique(validation_windows, "validation")
    diag = {item.update: item for item in _unique(diagnostics, "diagnostics")}
    if 0 not in {item.update for item in windows}:
        return HealthDecision("completed_but_gate_failed", "missing_update_0_validation")
    latest = max(item.update for item in windows)
    last = [item for item in windows if item.update > 0][-3:]
    if len(last) < 3:
        return HealthDecision("continue_training" if latest < target_update else "completed_but_gate_failed", "fewer_than_three_validation_windows")
    if latest >= 60 and _plateau(last):
        return HealthDecision("early_stop_failed", "three_window_plateau")
    if latest < target_update:
        return HealthDecision("continue_training", "target_update_not_reached")
    baseline = next(item for item in windows if item.update == 0)
    for window in last:
        current = diag.get(window.update)
        if current is None:
            return HealthDecision("completed_but_gate_failed", f"missing_diagnostics_update_{window.update}")
        if window.success_rate - baseline.success_rate + 1e-12 < 0.05:
            return HealthDecision("completed_but_gate_failed", "success_improvement_below_0.05")
        if current.effective_options <= 0:
            return HealthDecision("completed_but_gate_failed", "effective_options_not_positive")
        if current.approx_kl > 0.02:
            return HealthDecision("completed_but_gate_failed", "approx_kl_above_0.02")
    return HealthDecision("passed_for_final_evaluation", "all_health_gates_passed")

def _unique(items, label):
    ordered = sorted(items, key=lambda item: item.update)
    if len({item.update for item in ordered}) != len(ordered):
        raise ValueError(f"duplicate {label} update")
    return ordered

def _plateau(windows):
    def relative(a, b):
        denominator = max(abs(a), abs(b), 1e-12)
        return abs(b - a) / denominator
    pairs = list(zip(windows, windows[1:]))
    return all(abs(b.success_rate - a.success_rate) < 0.01 and relative(a.high_reward_mean, b.high_reward_mean) < 0.01 for a, b in pairs)
