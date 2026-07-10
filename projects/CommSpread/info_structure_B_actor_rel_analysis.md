# Actor Other-Agent Relative Positions Analysis

Variant yaml: `configs/spread_mpe_actor_rel.yaml`
Outputs: `outputs/info_structure/actor_rel`

Training budget: 3 seeds, 3.0M frames per seed. Runs were stopped at the observed plateau rather than full 10.5M because the curves had stabilized and evaluator success stayed at/near zero.

## Evaluator Summary

| seed | success_rate | episode_len | matched_dist | episode_reward | action distribution |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 50.000 | 0.5723 | -30.289 | noop=0.124, left=0.202, right=0.178, down=0.222, up=0.274 |
| 1 | 0 | 50.000 | 0.6656 | -33.259 | noop=0.274, left=0.269, right=0.113, down=0.244, up=0.100 |
| 2 | 0 | 50.000 | 0.5739 | -30.632 | noop=0.030, left=0.134, right=0.194, down=0.302, up=0.340 |

Mean success_rate: 0 +/- 0
Mean matched distance: 0.6039 +/- 0.0534
Mean episode reward: -31.393 +/- 1.624

## Log Diagnostics

| seed | final collection reward | final collection matched | final collection success | entropy first -> final | critic loss first -> final / max | eval len final |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -30.07 | 0.6017 | 1.667e-05 | 1.598 -> 1.062 | 54.6 -> 6.818 / 64.09 | 50 |
| 1 | -33.06 | 0.6609 | 1.667e-05 | 1.593 -> 1.034 | 57.25 -> 8.184 / 64.41 | 50 |
| 2 | -29.87 | 0.5976 | 1.667e-05 | 1.595 -> 1.06 | 56.02 -> 6.7 / 64.68 | 50 |

## Interpretation

Adding other-agent relative positions to the actor did not improve success and increased seed variance. One seed stayed much worse, and the two better seeds still plateaued near the baseline distance. This does not support actor-only relational information as a sufficient fix.

Action distribution did not collapse to a single action in any seed. No-op probability often decreased, but directional actions remained broad. Entropy decreased from about 1.6 toward lower values as expected; the available logs do not show value loss divergence large enough to explain the failure.
