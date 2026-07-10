# Information Structure Report

Baseline reference from `baseline_analysis.md`: 3 seeds all had success_rate 0, episode length 50, final matched distance roughly 0.565/0.616/0.567, and no action collapse.

## Aggregate Comparison

| experiment | actor obs | critic input | success_rate mean | matched distance mean | episode reward mean | conclusion |
|---|---|---|---:|---:|---:|---|
| baseline_mpe_parity | old 10-dim | grouped agent obs | 0 | 0.5827 | -31.084 | failed baseline |
| A_global_critic | old 10-dim | explicit global state | 0 | 0.5653 | -30.396 | marginal/no success |
| B_actor_rel | old + other rel pos | grouped richer obs | 0 | 0.6039 | -31.393 | marginal/no success |
| C_richer_both | old + other rel pos | explicit global state | 0.000651 | 0.5615 | -29.980 | marginal/no success |

## Findings

1. Reward improves early in all information-rich variants, especially A and C, but it stabilizes around the same failed region instead of continuing toward reliable coverage.
2. Success rate does not materially rise. A and B remain exactly 0 across 512-episode deterministic evaluation. C has one rare success in seed 2 during consolidated evaluation, i.e. 1/512 episodes, which is not a meaningful recovery.
3. Episode length remains effectively locked at 50. C seed 2 averaged 49.93 only because of the single rare success; otherwise all runs hit the horizon.
4. Critic-only global state is not enough. A has adequate absolute positions, velocities, landmark positions, and pairwise distances, but still converges to matched distance around 0.56.
5. Actor other-agent relative positions alone are not enough and are less stable. B seed 1 regressed to matched distance about 0.66.
6. Actor+critic richer observation is the best of this set, but only marginally: matched distance is around 0.56 and success remains essentially zero.
7. Action mapping probe passed: discrete actions map as 0 no-op, 1 left, 2 right, 3 down, 4 up. This matches MPE and does not explain the failure.

## Diagnosis

The second-stage evidence weakens the centralized-critic/observation-information hypothesis. Information structure affects learning speed and slightly affects final reward, but does not restore the paper behavior. The remaining stronger suspects are now algorithm/objective mismatch and environment dynamics/initialization details rather than simple missing global/relational information.

Most likely next checks: compare old MPE PPO/MAPPO update details against BenchMARL MAPPO defaults, verify old action force/damping/substep physics against VMAS dynamics, and compare rollout distributions under a fixed hand-coded or old checkpoint policy.
