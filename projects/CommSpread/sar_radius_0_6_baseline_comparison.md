# SAR Radius 0.6 Baseline And Reward-Shaping Comparison

All heuristic baseline evaluations use `sensor_radius=0.6`, `retire_on_rescue=True`, deterministic high-level policies, 64 vector envs per seed, and 3 seeds. Learned-policy references are deterministic seed-0 evaluator runs unless otherwise noted.

## Heuristic Baselines

| baseline | success mean +/- std | success step mean +/- std | detected | visited | detected-unassigned steps | discovery->assignment | known-unvisited fail | undiscovered fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Greedy rescue | 0.224 +/- 0.05156 | 66.89 +/- 3.083 | 2.542 +/- 0.03683 | 2.047 +/- 0.0893 | 70.69 +/- 4.865 | 18.33 +/- 0.3868 | 0.4688 +/- 0.07103 | 0.3073 +/- 0.0483 |
| Search all then rescue | 0.2031 +/- 0.05846 | 70.52 +/- 2.934 | 2.714 +/- 0.03211 | 1.474 +/- 0.1085 | 136 +/- 4.041 | 30.07 +/- 0.2176 | 0.7969 +/- 0.05846 | 0 +/- 0 |

## Learned And Shaped References

| policy | train updates | train best success | train last5 success | eval success | success step | detected | visited | detected-unassigned steps | duplicate excess | discovery->assignment | known-unvisited fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HGSAR fixed baseline | 30 | 0.5625 | 0.3 | 0.3438 | 64.27 | 2.609 | 2.219 | 60.86 | 1.797 | 18.73 | 0.3281 |
| soft rescue-phase shaping | 30 | 0.4375 | 0.2625 | 0.25 | 65.81 | 2.625 | 2.125 | 71.08 | 0.2031 | 17.58 | 0.4688 |
| target assignment features | 30 | 0.375 | 0.2 | 0.2969 | 59 | 2.641 | 2.25 | 65.92 | 2.703 | 16.4 | 0.375 |
| assignment bonus/penalty shaping | 30 | 0.375 | 0.175 | 0.1406 | 56.22 | 2.594 | 2.031 | 73.38 | 6.562 | 15.15 | 0.5156 |
| latency-only duplicate-strong shaping | 20 | 0.5 | 0.1625 | 0.2188 | 71.64 | 2.625 | 2.109 | 75.81 | 3.938 | 17.73 | 0.4844 |

## Interpretation

The greedy rescue baseline matches the expected timing tradeoff: successful episodes finish relatively quickly, but success is lower because early retirement reduces search capacity. The search-all-then-rescue baseline detects more targets and almost never fails due to undiscovered targets, but it waits too long to assign rescue; under the 100-step horizon its success is not higher.

The learned fixed HGSAR baseline is currently better than both hand-written heuristics on deterministic success, but it still leaves many detected targets unassigned. This validates reward shaping as a reasonable direction, but the shaping must target assignment latency without inducing duplicate claims.

The assignment bonus/penalty variant reduced discovery-to-assignment delay but created many duplicate claims and degraded success. The latency-only duplicate-strong variant was less unstable and reached a stochastic training peak of `0.5`, but deterministic success stayed low and duplicate assignment remained above the fixed baseline.

## Practical Next Experiment

The next shaping experiment should combine reward shaping with coordinated target selection:

1. keep flat policy logits, but apply per-env sequential target masking during action sampling so simultaneous agents cannot select the same target;
2. use a small detected-unassigned penalty, e.g. `0.0025` to `0.005`;
3. avoid positive unique-assignment bonus initially, because it increased duplicate pressure;
4. compare directly to greedy rescue and search-all-then-rescue on `success_rate`, `success_step_mean`, `detected_unassigned_steps_mean`, and `duplicate_assignment_excess_mean`.
