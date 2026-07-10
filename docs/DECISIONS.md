# Decisions

Generated: 2026-07-10 UTC  
Git HEAD: `3aaf458e00e25049940996e31b906d4964abc811`  
Worktree: dirty; decisions are based on current code, reports, and output logs.

## Designs To Keep

### Keep `spread_mpe_physics_spawn_parity` as the current spread baseline

Status: verified fact / retained design.

Evidence:

- Physics/action scale mismatch was directly probed: old VMAS parity moved far slower than MPE; physics parity matches the MPE repeated-action displacement sequence.
- `physics_spawn_parity` has the best spread matched distance and rare successes among tested spread VMAS parity variants.
- `physics_objective_diagnosis_report.md` recommends keeping it as base.

Remaining uncertainty:

- It still does not recover paper performance; architecture or objective mismatch remains.

### Keep `retire_on_rescue=True` as formal SAR semantics

Status: verified project constraint / retained design.

Evidence:

- User explicitly confirmed SAR task semantics: after an agent rescues a target and reaches the rescue point, it stops working.
- No-retire counterfactual improves success but changes task into a different problem.

Remaining uncertainty:

- None about semantics. How best to learn under this constraint remains open.

### Keep SAR radius `0.6` as current debug/curriculum regime

Status: retained experimental setup.

Evidence:

- Radius 0.6 fixed PPO baseline train best `0.5625`, last5 `0.300`, eval `0.34375`.
- Radius 0.3 after PPO fix remains much weaker: train best `0.125`, last5 `0.05`.

Remaining uncertainty:

- The final target may still require radius 0.3; transfer/curriculum effectiveness is untested.

### Keep the PPO minibatch indexing fix

Status: verified bug fix.

Evidence:

- Before fix: radius 0.6 best success `0.0625`, last5 `0.025`.
- After fix: radius 0.6 best success `0.5625`, last5 `0.300`.
- Code now indexes all tensors whose first dimension matches rollout transition batch size.

Remaining uncertainty:

- Other PPO bugs may still exist, but this specific mismatch is fixed.

### Keep assignment diagnostics in evaluator

Status: retained tool.

Evidence:

- Diagnostics showed target switching is not the dominant failure (`switch_away_actions_mean=0`).
- Detected-unassigned target steps and duplicate claims explain why reward shaping variants fail.

Remaining uncertainty:

- Assignment diagnostics are heuristic; exact target assignment inferred from assigned goal proximity to targets.

### Keep target assignment features as optional, not default

Status: retained optional feature.

Evidence:

- Features reduce discovery-to-assignment delay (`18.73` to `16.40`) but do not improve success over fixed baseline and increase duplicate claims.

Remaining uncertainty:

- Features may become useful once coordinated target selection prevents duplicate claims.

## Designs Rejected Or Deprioritized

### Do not use `no_retire` as final SAR solution

Status: rejected final design, retained diagnostic.

Evidence:

- No-retire eval success `0.671875` vs retire eval `0.34375`, but violates task definition.

### Hard staged rescue masking is deprioritized

Status: rejected for now.

Evidence:

- Threshold 2 + progress: best `0.375`, last5 `0.2375`.
- Threshold 3 + progress: best `0.25`, last5 `0.15`.
- Both are worse than fixed baseline best `0.5625`, last5 `0.300`.

### Search-all-then-rescue heuristic is not a final policy

Status: rejected final design.

Evidence:

- 3-seed success `0.2031 +/- 0.05846`.
- Detects more targets (`2.714`) but detected-unassigned steps are very high (`136`), meaning rescue starts too late for 100-step horizon.

### Greedy rescue heuristic is not a final policy

Status: rejected final design.

Evidence:

- 3-seed success `0.224 +/- 0.05156`, below fixed HGSAR eval `0.34375`.
- Fails from both undiscovered targets and known-unvisited targets.

### Bonus-heavy assignment shaping is rejected

Status: rejected specific shaping.

Evidence:

- Eval success `0.140625`.
- Duplicate assignment excess `6.5625`, far above fixed baseline `1.796875`.

### LR-only fix is rejected

Status: rejected as standalone solution.

Evidence:

- `lr=1e-3` increased early KL/clip but did not improve success; best `0.375`, final `0.1875` in short diagnostic.

### Generic BenchMARL GNN full run is deprioritized until feature parity is checked

Status: not rejected, but blocked/deprioritized.

Evidence:

- GNN smoke passed after PyG dependency fix.
- First longer seed showed returns around `-38` to `-43`, worse than MLP physics baseline around `-28`.

## Conclusions Still Insufficient

- Whether coordinated per-env target masking plus small latency penalty improves SAR is not tested.
- Whether a two-stage high-level actor (`search/rescue` mode then candidate) outperforms flat logits is not tested.
- Whether old paper `entity-mp` can be replicated in current VMAS/BenchMARL stack is not tested.
- Whether radius 0.6 training can transfer to radius 0.3 is not tested.
- Exact reproducibility of all experiments is incomplete because current results are from a dirty worktree and many outputs do not record Git diff.
