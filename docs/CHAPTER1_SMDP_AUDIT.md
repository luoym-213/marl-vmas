# Chapter 1 Strict SMDP Audit

Date: 2026-07-23
Branch audited: `research/align-mpe-physics`
HEAD at audit start: `3bb5d1c5fd9be9bdba24a47f45ff5ce6db610581`
Frozen task hash: `1ac17a9cb5c0bc7631b0c615c5bc52d90b30573acc983f4742df508bdd0e267c`

## 1. Result and terminology

The implementation found at the start of this audit was **B: duration-aware
semi-Markov PPO approximation**. It had asynchronous per-UAV options and used
`gamma ** duration` in bootstrap, but accumulated an undiscounted option reward,
used event-step lambda decay, had no explicit end-state value, conflated timeout
with terminal through `done`, and did not make task success a training terminal.

After the changes documented below, the Chapter 1 `high_train_v1.yaml` path is
**A: Strict SMDP**. The permitted paper term is:

> Event-triggered asynchronous SMDP optimization.

Historical checkpoints must instead be described as trained with a
"duration-aware semi-Markov PPO approximation" unless their metadata proves a
different return mode.

## 2. Event-to-loss code map

| Stage | File and lines | Class/function | Fields | Current behavior |
|---|---|---|---|---|
| Environment reward and success | `projects/CommSpread/comm_spread/sar_scenario.py:439-442,704-746` | `SarScenario.reward`, `_compute_step_rewards` | `high_rewards`, `target_visited`, `success`, `world_steps` | Rescue/discovery/shaping rewards are computed before `success`; therefore the reward that completes target 3 is visible in the same returned step. |
| Environment done | `sar_scenario.py:469-473` | `done` | timeout, `success` | Frozen task has `early_done_on_success=false`; environment done is horizon timeout, so a successful environment can continue physically. |
| Reward accumulation | `comm_spread/async_smdp.py:262-277` | `AsyncSMDPCollector.step` | `_return_accumulator`, pending start time | Strict mode adds `gamma ** elapsed * high_rewards` only for pending options in train-active envs. |
| Event detection | `async_smdp.py:286-327` | `step` | success, goal, retirement, detection, assignment | Events are evaluated per env/per UAV. Assignment release uses an agent-wise change mask; finder-first returns only the finder until cascade diffusion. |
| Option closure | `async_smdp.py:328-343,713-810` | `step`, `_finalize` | terminal flags, duration, next value | Success forces every pending option in that env to close; only pending keys generate transitions. No new decision is made for a successful env. |
| Option start | `async_smdp.py:553-620` | `_decide` | action, old log-prob, start value, start state | The exact actor observation/action mask and selection-time policy outputs are stored per `(env_id, agent_id)`. |
| End-state critic | `comm_spread/high_level_policy.py:260-268`; `async_smdp.py:731-759` | `HGSARActorCriticPolicy.value`, `_finalize` | `next_value`, next critic state | Evaluates only the critic at the option boundary; it does not sample an action or update actor statistics. Terminal next value is forced to zero. |
| Flat storage/batch | `scripts/train_high_ppo_sar.py:564-631` | `transitions_to_batch` | explicit terminal/truncation/valid fields | Retains env, agent, start time, duration, reward, start/end value and boundary flags. No synchronous-agent padding is created. |
| Return and GAE | `comm_spread/smdp_returns.py:24-105`; `train_high_ppo_sar.py:634-641` | `add_smdp_gae`, `add_gae` | advantage, return | Groups independently by env and agent, orders by option start, and applies strict environment-time equations. |
| Valid-row filtering | `train_high_ppo_sar.py:645-659` | `select_train_active_batch` | `train_active_mask` | Invalid tail/padding rows are removed before normalization or any loss evaluation. |
| PPO actor/critic loss | `train_high_ppo_sar.py:661-715` | `ppo_update` | old log-prob, advantage, return, entropy | Ratio uses selection-time log-prob and action mask. Policy, critic, entropy and intention losses see only train-active option transitions. |
| Checkpoint metadata/guard | `train_high_ppo_sar.py:867-920` | `save_checkpoint`, `load_checkpoint` | return mode, success semantics, task hash | Run config is saved in checkpoints. Strict resume fails closed if mode/terminal semantics/task hash are absent or different. |

## 3. High-level transition

`HighLevelTransition` (`async_smdp.py:15-57`) contains start actor state,
action, old log-prob, start value, discounted option reward, duration, end critic
state, explicit next value, environment and agent identities, and these distinct
boundary fields:

- `task_success_terminal`: first team success in that environment;
- `environment_done`: simulator episode boundary;
- `time_limit_truncated`: horizon reached without success;
- `train_active_mask`: eligibility for high-level optimization;
- `agent_terminal`: an individual UAV retired while team success was false;
- `terminal`: task-success or individual-retirement terminal.

The current actor/critic is feed-forward. Width-zero `recurrent_state` and
`next_recurrent_state` tensors make this fact explicit; there is no recurrent
hidden state to advance or leak across events. If a recurrent high-level model
is introduced, these fields must be replaced by the real selection/end states
before the config can use a recurrent mode.

## 4. Strict mathematical definition

For option `k`, starting at environment time `tau_k`, ending at
`tau_(k+1)`, and lasting `d_k >= 1` environment steps:

```
R_k = sum_(l=0)^(d_k-1) gamma^l r_(tau_k+l)
delta_k = R_k + gamma^d_k * b_k * V(s_(tau_(k+1))) - V(s_(tau_k))
A_k = delta_k + (gamma * lambda)^d_k * c_k * A_(k+1)
return_k = A_k + V(s_(tau_k))
```

`b_k = 0` for a true terminal and 1 otherwise. With the frozen
`time_limit_bootstrap=true`, timeout is not a true terminal and uses the horizon
state value. `c_k = 0` at terminal, timeout, or environment boundary, so a
bootstrap at timeout never connects the GAE trace to the next reset episode.

The selected GAE definition is `(gamma * lambda) ** duration`: both discount
and trace eligibility are measured in environment time. When every duration is
1 it reduces exactly to ordinary MDP GAE.

## 5. Historical mathematical behavior

The explicit `legacy_event_step` mode preserves the old return interpretation:

```
R_k_legacy = sum_l r_(tau_k+l)
delta_k_legacy = R_k_legacy + gamma^d_k * (1-done_k) * V(next event) - V_k
A_k_legacy = delta_k_legacy
             + gamma^d_k * lambda * (1-done_k) * A_(k+1)
```

The old implementation inferred `V(next event)` from the next stored start
value and used zero for the last stored transition. Environment timeout was
treated as terminal/no-bootstrap. Task success was absent from the mask when
the environment continued to horizon. This mode is retained for explicit
reproduction only and is not the Chapter 1 final training mode.

## 6. Duration and asynchronous storage

Duration is the number of low-level environment steps, not the number of
high-level events. It is computed independently as `end_t - start_t`, with
minimum 1. Detection, finder-first release, an affected assignment release,
goal completion, retirement, timeout, or task success can close an option.

One UAV ending an option does not close the others. Unaffected pending options
retain their reward accumulator and start time. A team success is the exception:
it closes all keys still present for that environment. Already retired UAVs and
UAVs without a pending option cannot create duplicate or synthetic transitions.
The flat batch keeps `(env_id, agent_id, decision_start_t)` and GAE grouping uses
all three, preventing env/agent cross-talk.

The strict Chapter 1 resolver also requires
`low_level_steps_per_batch >= horizon` so `collector.reset()` cannot silently
discard a partial episode between PPO updates. The frozen values are 400 and
100 respectively.

## 7. Reward credit assignment

`scenario.high_rewards` is per-agent, not one team scalar copied to every UAV.
It consists of the detecting/rescuing UAV's discovery/rescue credit plus the
configured per-agent shaping and time term (`sar_scenario.py:718-735`). At team
success all valid pending transitions are closed simultaneously, but the final
rescue bonus remains on the actual rescuer's high reward; it is not duplicated
as a team success bonus on every transition. Every still-pending agent does
include its own reward from the environment step that caused success.

Post-success steps are multiplied out by the per-env train-active mask and no
pending option remains, so reward and duration cannot grow after success.

## 8. Success terminal and environment continuation

The first step where all three targets are visited sets
`task_success_terminal=1`. Reward accumulation happens before that check, so the
last rescue reward is included. The collector then closes every pending option
in that env, sets `terminal=1`, forces `next_value=0`, clears all pending keys,
and flips that env's persistent `train_active_mask` to zero.

The simulator can continue to horizon 100 because environment `done()` remains
timeout-based. During this physical tail the collector can still call
`env.step()` for trajectory/diagnostics, but the success env has no pending
options, accumulates no option reward, creates no transition, and is excluded
from redecision. The actor is not called for that env. A defensive batch filter
also removes any invalid row before advantage normalization, actor value,
critic value, entropy, ratio, or intention loss.

> 第一章训练中，任务首次成功时视为高层SMDP terminal。环境可为统一评估长度继续步进，但成功后的环境步不进入高层训练storage、return、GAE或PPO loss。

## 9. Timeout and truncation

At horizon without success:

```
task_success_terminal = 0
environment_done = 1
time_limit_truncated = 1
terminal = 0                 # unless the UAV itself retired
```

The frozen rule is time-limit bootstrap: the centralized critic evaluates the
horizon state, `gamma ** duration * next_value` enters the TD residual, and the
GAE trace stops at the reset boundary. This differs deliberately from both task
success (zero bootstrap) and the legacy implementation (timeout no-bootstrap).

## 10. Finder, assignment, commitment, and retirement

- Finder-first closes only the finder option while non-finders continue.
- Assignment-change handling was corrected from an environment-wide `.any()`
  interruption of all uncommitted UAVs to an agent-wise changed mask. A released
  UAV is interrupted once; unchanged UAVs are not given fake transitions.
- The existing commitment latch/cascade state prevents repeated finder
  diffusion decisions; tests cover one-shot assignment release and finder-only.
- Retirement closes that UAV's pending option as `agent_terminal=1`, zeros its
  bootstrap, and leaves other agents train-active.
- Team success closes all remaining pending options, regardless of which UAV
  completed the third rescue.

## 11. Actor, critic, recurrent state, and PPO

Old log-probability and the action mask are stored at option selection. PPO
evaluates the same action under the current policy and compares it with that
selection-time probability. Duration affects reward/return targets but does not
shift actions or log-probabilities. The critic uses the actual option-end global
map and agent nodes. Minibatches are random flat batches because the current
network is feed-forward; no claim of recurrent sequence PPO is made.

Advantage normalization, policy loss, critic loss, entropy and auxiliary
intention loss occur only after `select_train_active_batch`. Storage contains no
post-success transition in the first place; the loss mask is a second barrier.

## 12. Vector environments

`_train_active_mask`, `_time`, reward accumulators and pending keys are indexed
per environment. A success bit closes only that env. Other envs continue
collecting and deciding. Return recursion groups by env then agent. The test with
env 0 succeeding at step 3 and env 1 at step 5 verifies independent closure,
continued physical stepping, independent rewards/durations, and no post-success
actor call or storage.

## 13. Configuration and hashes

`configs/chapter1/high_train_v1.yaml:17-25` freezes:

```
execution_mode: event_triggered_async
return_mode: strict_smdp
gae_duration_mode: environment_time
training_success_terminal: true
environment_continue_after_success: true
exclude_post_success_steps_from_training: true
time_limit_bootstrap: true
recurrent_state_mode: feedforward_empty
```

`chapter1_config.validate_high_training` rejects any different value. The
task-level YAML was not changed, so the task hash remains:

`1ac17a9cb5c0bc7631b0c615c5bc52d90b30573acc983f4742df508bdd0e267c`

The high-train role hash changed from
`580f924fb2b54783e0f2833f1ffe160fb8297353f3e0b9fe72639d912009d2b9`
to `11389e199c427032f6d74cc66ac5b45b27053f3cce84e35b09534f53d3736ee3`
because training mathematics and terminal handling are role-level optimization
semantics, not task/environment semantics.

## 14. Checkpoint compatibility

New checkpoints record `high_level_return_mode`,
`training_success_terminal`, `environment_continue_after_success`,
`exclude_post_success_steps_from_training`, and `task_config_sha256` through
the serialized run arguments. Strict resume rejects missing/mismatched fields
before loading model or optimizer state.

Old high-level checkpoints remain valid for pure evaluation because evaluation
loads policy weights without asserting that they were trained with strict
returns. They cannot be resumed as strict training and cannot be cited as
evidence of strict SMDP optimization. Their historical provenance is
`legacy_event_step` with no proven success-terminal training semantics.

## 15. Tests and hand calculations

Required focused tests:

- `tests/test_smdp_returns.py`: two-option hand calculation, discounted rewards,
  `gamma ** duration`, `(gamma lambda) ** duration`, terminal, timeout and the
  duration-1 MDP reduction;
- `tests/test_async_option_storage.py`: durations 3/7/5, independent rewards,
  finder-only interruption and retirement;
- `tests/test_event_trigger_transition_boundaries.py`: finder-first and one-shot
  affected-agent assignment release;
- `tests/test_success_terminal_storage.py`: multi-UAV team closure, final reward,
  post-success tail, vector env isolation, timeout bootstrap, pre-loss valid-row
  filtering, frozen config and strict-resume rejection.

All 13 zero-argument `tests/test_*.py` files passed in the `marl-vmas-dev`
container. `py_compile` passed for every modified Python module. A real two-env,
100-step frozen-task smoke using a discrete no-op low-level policy completed
with both envs at step 100, no pending options, 6 timeout transitions, zero
invalid storage rows, and the frozen task hash unchanged. No training update or
formal evaluation was run.

For the specified manual case (`gamma=0.9`, `lambda=0.95`):

```
R0 = 1 + 0.9*2 = 2.8
R1 = 3 + 0.9*4 + 0.9^2*5 = 10.65
delta0 = 2.8 + 0.9^2*8 - 10 = -0.72
delta1 = 10.65 - 8 = 2.65
A1 = 2.65
A0 = -0.72 + (0.9*0.95)^2*2.65
```

## 16. Bugs fixed

1. Option reward was undiscounted.
2. End-state next value was implicit and the final nonterminal value became zero.
3. GAE lambda decayed once per option rather than in environment time.
4. Success did not terminate high-level training when the env continued.
5. Post-success events could enter high-level collection.
6. Timeout and success shared ambiguous `done` semantics.
7. Assignment change interrupted all uncommitted agents in an env instead of
   only an affected agent.
8. PPO had no explicit train-active filtering barrier.
9. Strict resume could silently accept a historically trained checkpoint.

## 17. Remaining limitations and non-SMDP observations

- Per the task constraint, no high-level model was trained after this semantic
  change. The implementation is test-verified, not yet learning-curve verified.
- The current high-level model is feed-forward. Recurrent-state correctness is
  therefore explicit but vacuous (width-zero state); a future recurrent actor
  requires new sequence-storage tests.
- A private `_finalize(replan, dones)` compatibility path remains for an old
  ceiling diagnostic. The training collector always supplies the explicit
  terminal/truncation tensors; the compatibility path is not used by training.
- The collector's legacy default proportional controller emits a 2-D continuous
  action and is incompatible with frozen `mpe_strict` discrete actions. Chapter
  1 training supplies the guarded MPE-strict low-level checkpoint, and the real
  smoke used an explicit discrete policy. This existing diagnostic default is
  outside the SMDP change and must not be used for a strict task run.
- `PROJECT_STATUS.md`, `EXPERIMENT_LOG.md`, and `DECISIONS.md` contain older
  historical states and do not yet describe this strict-return implementation;
  this document and the frozen Chapter 1 configs are authoritative for the
  audited path.

## 18. Training consequence

High-level retraining is required before claiming a strict-SMDP-trained Chapter
1 result. The old checkpoint can still be used to reproduce/evaluate historical
execution performance, but changing option reward, terminal targets, timeout
bootstrap and GAE changes the optimization target materially. Low-level
retraining is not implied by this audit.
