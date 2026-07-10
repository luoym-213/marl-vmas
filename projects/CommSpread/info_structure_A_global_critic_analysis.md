# Global State Critic Analysis

Variant yaml: `configs/spread_mpe_global_critic.yaml`
Outputs: `outputs/info_structure/global_critic`

Training budget: 3 seeds, 3.0M frames per seed. Runs were stopped at the observed plateau rather than full 10.5M because the curves had stabilized and evaluator success stayed at/near zero.

## Evaluator Summary

| seed | success_rate | episode_len | matched_dist | episode_reward | action distribution |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 50.000 | 0.5667 | -30.207 | noop=0.076, left=0.287, right=0.271, down=0.177, up=0.189 |
| 1 | 0 | 50.000 | 0.5684 | -30.557 | noop=0.033, left=0.264, right=0.246, down=0.224, up=0.233 |
| 2 | 0 | 50.000 | 0.5606 | -30.424 | noop=0.006, left=0.222, right=0.244, down=0.257, up=0.271 |

Mean success_rate: 0 +/- 0
Mean matched distance: 0.5653 +/- 0.0041
Mean episode reward: -30.396 +/- 0.177

## Log Diagnostics

| seed | final collection reward | final collection matched | final collection success | entropy first -> final | critic loss first -> final / max | eval len final |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -30.72 | 0.6145 | 0 | 1.597 -> 1.012 | 31.54 -> 5.475 / 47.23 | 50 |
| 1 | -29.71 | 0.5944 | 1.667e-05 | 1.593 -> 1.026 | 29.67 -> 4.73 / 47.65 | 50 |
| 2 | -30.47 | 0.6095 | 0 | 1.59 -> 1.009 | 31.16 -> 4.853 / 49.58 | 50 |

## Interpretation

Explicit global state made early learning faster, but all seeds converged to the same failed region as baseline: success_rate 0, horizon locked at 50, matched distance about 0.56. This weakens the hypothesis that critic-only global information is the missing ingredient.

Action distribution did not collapse to a single action in any seed. No-op probability often decreased, but directional actions remained broad. Entropy decreased from about 1.6 toward lower values as expected; the available logs do not show value loss divergence large enough to explain the failure.
