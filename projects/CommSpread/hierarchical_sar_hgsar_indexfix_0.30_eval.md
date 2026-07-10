# Hierarchical SAR Diagnostics

This report evaluates high-level assignment quality, RRT candidate quality, low-level goal reaching, and relaxed sensor radii without training new code.

| policy | sensor | success | visited | detected | goal_hits/tr | step_hit | goal_dist | entropy_gain | rrt_util | rrt_zero | duration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hgsar_checkpoint | 0.30 | 0.031 | 1.344 | 1.562 | 0.889 | 0.0565 | 0.248 | 2736.6 | 0.783 | 0.000 | 15.7 |

## Interpretation

- `goal_hits/tr` and `step_hit` measure whether the fixed low-level controller reaches the high-level assigned waypoint.
- `visited` and `success` measure whether high-level assignment plus target handling solves the SAR task.
- Large entropy gain with low visited targets indicates exploration is happening but target-to-rescue assignment is weak.
- If larger `sensor` sharply improves visited/success, the current task is likely too sparse for validating the high-level objective early in training.
