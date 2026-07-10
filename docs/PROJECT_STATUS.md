# Project Status

Generated: 2026-07-10 UTC  
Project: `projects/CommSpread/`  
Git branch: `comm`  
Git HEAD: `3aaf458e00e25049940996e31b906d4964abc811`  
Worktree state: dirty; many experiment reports/outputs are untracked. Results below are from the current workspace state, not a clean committed artifact.

## Current Research Goal And Problem Definition

Verified facts:

- The project is a multi-agent reinforcement learning migration/debugging project for CommSpread/SAR tasks.
- The old successful paper implementation is under `projects/CommSpread/old_algo/` and was based on MPE.
- The current implementation uses VMAS/BenchMARL plus local high-level SAR PPO code.
- The main research question is why the VMAS reimplementation does not recover the paper-level behavior, and how to restore performance while keeping the intended task semantics.

Current active focus:

- Spread/MPE parity: keep `spread_mpe_physics_spawn_parity` as the current best VMAS parity baseline. Physics/action scale mismatch has been verified and fixed, but success remains weak.
- Hierarchical SAR: use `sensor_radius=0.6` as the current curriculum/debug regime and keep `retire_on_rescue=True`. The formal SAR semantics are that an agent retires after reaching a rescue target; `no_retire` is diagnostic only.

Known problem shape:

- Spread still has near-zero success even after physics/spawn parity; likely remaining issues are architecture/entity-message-passing mismatch or deeper objective mismatch.
- SAR high-level policy can learn after fixing PPO minibatch indexing, but deterministic success remains low and failures are dominated by detected targets staying unassigned too long.

## Current Algorithm And Training Flow

Verified facts:

- Spread training entrypoint: `projects/CommSpread/scripts/train_mappo.py`.
- Spread task variants are defined in `projects/CommSpread/comm_spread/training_config.py` and mirrored by YAML configs under `projects/CommSpread/configs/`.
- SAR high-level training entrypoint: `projects/CommSpread/scripts/train_high_ppo_sar.py`.
- SAR deterministic failure/evaluator entrypoint: `projects/CommSpread/scripts/analyze_sar_failure_modes.py`.
- SAR low-level policy checkpoint currently used by default:
  `projects/CommSpread/outputs/sar_low_full/low_safe_0.06/checkpoints/checkpoint_50040000.pt`.

SAR flow:

1. VMAS SAR scenario provides belief maps, target detection, RRT candidates, target masks, and retirement state.
2. `BenchMARLLowLevelPolicy` loads the fixed low-level checkpoint and tracks assigned high-level goals.
3. `AsyncSMDPCollector` asynchronously asks the high-level actor for RRT exploration or target rescue actions.
4. `HGSARActorCriticPolicy` uses a heterogeneous graph actor and map critic.
5. `train_high_ppo_sar.py` trains the high-level policy with standalone PPO and saves `checkpoints/latest.pt`, `scalars/train.csv`, and `texts/config.json`.
6. `analyze_sar_failure_modes.py` evaluates trained policies and heuristic baselines with assignment diagnostics.

## Implemented Modules

Verified from code/diff:

- Spread MPE parity support in `comm_spread/scenario.py`:
  - `reward_mode=mpe_parity`
  - MPE-style discrete action mapping and force scaling
  - matched-distance reward via Hungarian assignment/fallback exhaustive solver
  - global critic state emission
  - independent MPE-style spawn option
  - physics parameters for substeps, drag, dt, collision force, contact margin
- Spread BenchMARL task support in `comm_spread/benchmarl_task.py`:
  - `GlobalStateFromAgentInfo`
  - reward return normalization transform for old-PPO diagnostic
- SAR high-level features in `comm_spread/async_smdp.py` and `comm_spread/sar_scenario.py`:
  - progress features: detected/visited/active/entropy
  - target assignment features: claim count, nearest active-agent distance, detected age
  - staged rescue mask options
  - soft rescue timing and assignment shaping reward options
- SAR diagnostics in `scripts/analyze_sar_failure_modes.py`:
  - HGSAR checkpoint evaluation
  - heuristic baselines: `rescue_greedy`, `search_all_then_rescue`, `target_first`, `search_then_target`
  - assignment metrics: detected-unassigned steps, duplicate claims, discovery-to-assignment delay, assignment-to-visit delay, switch-away actions, success step
- SAR PPO fix in `scripts/train_high_ppo_sar.py`:
  - `index_batch()` now indexes every tensor whose first dimension equals the rollout transition batch size.

## Current Code Entrypoints And Key Files

Core files:

- `projects/CommSpread/comm_spread/scenario.py`: VMAS CommSpread/Spread scenario and MPE parity behavior.
- `projects/CommSpread/comm_spread/sar_scenario.py`: SAR VMAS scenario, belief/detection/reward/retirement logic.
- `projects/CommSpread/comm_spread/async_smdp.py`: asynchronous SAR high-level collector and observation construction.
- `projects/CommSpread/comm_spread/high_level_policy.py`: HGSAR actor-critic wrapper and dimensions for optional features.
- `projects/CommSpread/comm_spread/models/hgsar.py`: heterogeneous graph actor and map critic model definitions.
- `projects/CommSpread/comm_spread/env_factory.py`: VMAS environment constructors and SAR defaults.
- `projects/CommSpread/comm_spread/training_config.py`: BenchMARL task variants.
- `projects/CommSpread/scripts/train_mappo.py`: Spread/MAPPO training CLI.
- `projects/CommSpread/scripts/train_high_ppo_sar.py`: SAR high-level PPO training CLI.
- `projects/CommSpread/scripts/analyze_sar_failure_modes.py`: SAR evaluator and baseline comparator.
- `projects/CommSpread/scripts/probe_spread_mpe_physics.py`: spread physics/action mapping probe.

Important reports already in project root:

- `projects/CommSpread/information_structure_report.md`
- `projects/CommSpread/physics_objective_diagnosis_report.md`
- `projects/CommSpread/entity_message_passing_progress_report.md`
- `projects/CommSpread/sar_radius_0_6_training_audit.md`
- `projects/CommSpread/sar_radius_0_6_baseline_comparison.md`

## Most Reliable Current Commands

Spread physics probe:

```bash
PYTHONPATH=projects/CommSpread python projects/CommSpread/scripts/probe_spread_mpe_physics.py --variant spread_mpe_physics_spawn_parity --device cuda:0
```

Spread current baseline smoke/training command surface:

```bash
PYTHONPATH=projects/CommSpread python projects/CommSpread/scripts/train_mappo.py   --variant spread_mpe_physics_spawn_parity   --model mlp   --train-device cuda:0   --sampling-device cuda:0   --seed 0   --max-n-frames 3000000   --save-folder projects/CommSpread/outputs/physics_spawn_parity/seed_0   --no-render
```

Spread GNN smoke command that has passed:

```bash
PYTHONPATH=projects/CommSpread python projects/CommSpread/scripts/train_mappo.py   --variant spread_mpe_physics_spawn_parity   --model gnn   --comms-radius 1.0   --train-device cuda:0   --sampling-device cuda:0   --seed 0   --max-n-frames 1000   --save-folder projects/CommSpread/outputs/gnn_physics_smoke/seed_0   --no-render
```

SAR current fixed high-level baseline training:

```bash
PYTHONPATH=projects/CommSpread python projects/CommSpread/scripts/train_high_ppo_sar.py   --device cuda:0   --seed 0   --num-envs 16   --updates 30   --low-level-steps-per-batch 400   --minibatch-size 64   --sensor-radius 0.6   --save-folder projects/CommSpread/outputs/sar_high_hgsar_indexfix_sensor_0.60/seed_0
```

SAR fixed baseline evaluator:

```bash
PYTHONPATH=projects/CommSpread python projects/CommSpread/scripts/analyze_sar_failure_modes.py   --device cuda:0   --seed 0   --num-envs 64   --steps 300   --sensor-radius 0.6   --policy hgsar   --high-level-checkpoint projects/CommSpread/outputs/sar_high_hgsar_indexfix_sensor_0.60/seed_0/checkpoints/latest.pt   --output projects/CommSpread/outputs/hierarchical_sar_diagnostics/failure_hgsar_0.60_assignment.json   --csv-output projects/CommSpread/outputs/hierarchical_sar_diagnostics/failure_hgsar_0.60_assignment.csv   --markdown projects/CommSpread/failure_hgsar_0.60_assignment.md
```

SAR heuristic baseline comparison commands:

```bash
PYTHONPATH=projects/CommSpread python projects/CommSpread/scripts/analyze_sar_failure_modes.py   --device cuda:0 --seed 0 --num-envs 64 --steps 300 --sensor-radius 0.6   --policy rescue_greedy   --output projects/CommSpread/outputs/hierarchical_sar_diagnostics/baseline_rescue_greedy_seed0_0.60.json   --csv-output projects/CommSpread/outputs/hierarchical_sar_diagnostics/baseline_rescue_greedy_seed0_0.60.csv   --markdown projects/CommSpread/sar_baseline_rescue_greedy_seed0_0.60.md
```

```bash
PYTHONPATH=projects/CommSpread python projects/CommSpread/scripts/analyze_sar_failure_modes.py   --device cuda:0 --seed 0 --num-envs 64 --steps 300 --sensor-radius 0.6   --policy search_all_then_rescue   --output projects/CommSpread/outputs/hierarchical_sar_diagnostics/baseline_search_all_then_rescue_seed0_0.60.json   --csv-output projects/CommSpread/outputs/hierarchical_sar_diagnostics/baseline_search_all_then_rescue_seed0_0.60.csv   --markdown projects/CommSpread/sar_baseline_search_all_then_rescue_seed0_0.60.md
```

## Current Known Issues

Verified facts:

- The current worktree is not clean. Results are based on uncommitted code changes.
- Spread physics parity is necessary but not sufficient: `spread_mpe_physics_spawn_parity` gives rare success only, still below 1% in reported 3M-frame runs.
- Existing BenchMARL GNN path is technically unblocked after installing PyG dependencies, but early GNN full seed was worse than MLP physics baseline and was stopped early.
- SAR fixed baseline at `sensor_radius=0.6` improves after PPO indexing fix, but deterministic eval still only reaches `success_rate=0.34375` on 64 envs.
- SAR reward shaping variants so far often reduce discovery-to-assignment delay but increase duplicate target claims or detected-unassigned time.
- Learned policies can show high stochastic training peaks but weaker deterministic evaluator performance.

Observed results:

- Greedy rescue heuristic: success `0.224 +/- 0.05156`, success step `66.89 +/- 3.083`.
- Search-all-then-rescue heuristic: success `0.2031 +/- 0.05846`, success step `70.52 +/- 2.934`, detected `2.714`, detected-unassigned steps `136`.
- Fixed HGSAR baseline eval: success `0.34375`, success step `64.27`, detected-unassigned steps `60.86`.

Hypotheses not yet fully validated:

- Coordinated target selection during action sampling may be required before reward shaping can help.
- A two-stage high-level actor (`search` vs `rescue`, then candidate selection) may be a better structural fix than scalar reward shaping.
- Spread recovery may require closer old `entity-mp` feature/model parity rather than BenchMARL generic GNN.

## Unfinished Tasks

- Implement and test coordinated per-env target masking for simultaneous SAR high-level decisions.
- Run multi-seed SAR training for any shaping variant that first passes deterministic diagnostic criteria.
- Decide whether target assignment features should remain optional or become part of the main actor interface.
- Implement or verify a closer old paper entity/message-passing policy for spread.
- Re-run spread GNN only after checking entity feature parity.
- Clean up or commit current untracked reports/outputs; exact archival policy is UNKNOWN.
- Create reproducible experiment runner scripts for 3-seed comparisons; currently many commands are manual.
