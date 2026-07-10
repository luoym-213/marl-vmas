# Action Mapping Probe Analysis

Deterministic VMAS probe with agent_0 at a fixed position and all other agents no-op:

| action id | observed delta | meaning |
|---:|---:|---|
| 0 | (0.000, 0.000) | no-op |
| 1 | (-0.006, 0.000) | left |
| 2 | (0.006, 0.000) | right |
| 3 | (0.000, -0.006) | down |
| 4 | (0.000, 0.006) | up |

This matches MPE simple_spread discrete action semantics: 0 no-op, 1 left, 2 right, 3 down, 4 up. No action mapping fix was applied.
