# Actor + Critic Richer Observation Analysis

Variant yaml: `configs/spread_mpe_richer_both.yaml`
Outputs: `outputs/info_structure/richer_both`

Training budget: 3 seeds, 3.0M frames per seed. Runs were stopped at the observed plateau rather than full 10.5M because the curves had stabilized and evaluator success stayed at/near zero.

## Evaluator Summary

| seed | success_rate | episode_len | matched_dist | episode_reward | action distribution |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 50.000 | 0.5620 | -29.780 | noop=0.013, left=0.233, right=0.242, down=0.252, up=0.261 |
| 1 | 0 | 50.000 | 0.5669 | -30.186 | noop=0.025, left=0.265, right=0.260, down=0.225, up=0.225 |
| 2 | 0.001953 | 49.928 | 0.5557 | -29.975 | noop=0.082, left=0.242, right=0.254, down=0.201, up=0.221 |

Mean success_rate: 0.000651 +/- 0.001128
Mean matched distance: 0.5615 +/- 0.0056
Mean episode reward: -29.980 +/- 0.203

## Log Diagnostics

| seed | final collection reward | final collection matched | final collection success | entropy first -> final | critic loss first -> final / max | eval len final |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -29.53 | 0.5914 | 3.333e-05 | 1.598 -> 1.034 | 30.12 -> 4.941 / 45.38 | 50 |
| 1 | -29.52 | 0.5902 | 1.667e-05 | 1.593 -> 1.039 | 31.2 -> 4.56 / 51.56 | 50 |
| 2 | -29.55 | 0.5905 | 0 | 1.595 -> 1.05 | 30.62 -> 4.837 / 48.84 | 50 |

## Interpretation

Combining richer actor observation with explicit global critic gave the best collection reward and one rare success in 512 deterministic episodes for seed 2, but the mean success remained effectively zero and episode length stayed at 50. This is the strongest information-structure variant, but only marginally better.

Action distribution did not collapse to a single action in any seed. No-op probability often decreased, but directional actions remained broad. Entropy decreased from about 1.6 toward lower values as expected; the available logs do not show value loss divergence large enough to explain the failure.
