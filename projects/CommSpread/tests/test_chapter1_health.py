from comm_spread.chapter1_health import HealthDecision, TrainingDiagnostics, ValidationWindow, evaluate_training_health

def w(update, success, reward=1.0): return ValidationWindow(update, success, reward)
def d(update, options=1, kl=0.02): return TrainingDiagnostics(update, options, kl)

def test_before_60_cannot_early_stop():
    result = evaluate_training_health([w(0, .1), w(20, .2), w(40, .2), w(59, .2)], [d(20), d(40), d(59)])
    assert result.continue_training

def test_plateau_at_60_fails():
    result = evaluate_training_health([w(0, .1), w(40, .2), w(50, .2), w(60, .2)], [d(40), d(50), d(60)])
    assert result.early_stop_failed

def test_199_cannot_pass():
    result = evaluate_training_health([w(0, .1), w(179, .2, 1), w(189, .3, 2), w(199, .4, 3)], [d(179), d(189), d(199)])
    assert result.continue_training

def test_200_passes_exact_boundaries():
    result = evaluate_training_health([w(0, .1), w(180, .15, 1), w(190, .16, 2), w(200, .17, 3)], [d(180), d(190), d(200)])
    assert result.passed_for_final_evaluation

def test_gate_failures_and_ordering():
    windows = [w(200, .2, 3), w(0, .1), w(190, .2, 2), w(180, .2, 1)]
    assert evaluate_training_health(windows, [d(180, 0), d(190), d(200)]).completed_but_gate_failed
    assert evaluate_training_health(windows, [d(180), d(190, kl=.021), d(200)]).completed_but_gate_failed
    try: evaluate_training_health([w(0, 0), w(0, 0)], [])
    except ValueError: pass
    else: raise AssertionError("duplicate update accepted")

if __name__ == "__main__":
    test_before_60_cannot_early_stop(); test_plateau_at_60_fails(); test_199_cannot_pass(); test_200_passes_exact_boundaries(); test_gate_failures_and_ordering()
    print("chapter1 health tests passed")
