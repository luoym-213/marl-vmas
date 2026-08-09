# Chapter 1 Immediate-open FCSR 训练指南

本实验只改变 finder 拒绝后的目标开放与 option 中断语义。冻结的低层 checkpoint、环境、奖励、PPO 参数和训练预算均保持不变。

## 语义

`immediate_open` 的规则为：finder 在首次发现时独享一次高层响应机会；若接受则建立 commitment；若拒绝则目标立刻向团队开放，但不产生额外高层决策事件。其他 UAV 继续当前 option，并在下一自然决策时刻看到该目标。现有 `immediate` 仍表示下一低层步强制合格 UAV 重决策。

任务配置为 `configs/chapter1/task_immediate_open_v1.yaml`。五个训练配置为 `configs/chapter1/high_train_immediate_open_seed0_v1.yaml` 至 `seed4`。配对 finder-only 的 seed 1 至 4 配置为 `high_train_finder_only_seed1_v1.yaml` 至 `seed4`；seed 0 使用冻结的 `high_train_v1.yaml`。

## 阶段 A：seed 0

训练不要求 Git 工作区干净；工作区状态只记录为 provenance，不作为训练或 finalization 门槛。不要复用 run-id，也不要从 finder-only 高层 checkpoint 恢复。

```bash
cd /workspace/projects/CommSpread
export PYTHONPATH=/workspace/projects/CommSpread
python scripts/train_high_ppo_sar.py \
  --chapter1-config configs/chapter1/high_train_immediate_open_seed0_v1.yaml \
  --run-id chapter1-immediate-open-seed0-<UTCSTAMP>-<GITSHA>
```

训练完整达到配置的目标 update 后，使用通用 finalizer 校验 run 并选择 checkpoint：

```bash
python scripts/finalize_chapter1_run.py \
  --run-dir outputs/chapter1/high_train_immediate_open_v1/seed_0/runs/<RUN_ID>
```

finalizer 不调用 retrospective-v2，也不检查 `git_worktree_clean`。它按 success、detect-all、训练 reward、较早 update 的顺序排名；即使终态 health gate 未通过，完整 run 仍会生成结果并在 `checkpoint_selection.json` 标记 `gate_failed`。生成的 `derived_eval_final.yaml` 绑定任务 hash、选定 checkpoint 及 SHA，默认 seeds `[0, 1]` 各 500 episode。分别运行：

```bash
python scripts/analyze_sar_failure_modes_mpe_physics.py \
  --chapter1-config outputs/chapter1/high_train_immediate_open_v1/seed_0/runs/<RUN_ID>/derived_eval_final.yaml \
  --policy hgsar --seed 0 --num-envs 500 --device cuda:0 \
  --output outputs/chapter1/high_train_immediate_open_v1/seed_0/runs/<RUN_ID>/evaluation_final_1000/seed_0/failure_modes.json
```

seed 1 使用同一命令并将 `--seed` 和输出子目录改为 `1`。只有语义测试通过、训练/评估无非有限值或重复 commitment、health gate 成功，并且配对成功率相对 finder-only seed 0 不下降超过 5 个百分点时，才进入阶段 B。该门槛只判断工程健康，不作为论文显著性结论。

## 阶段 B：五训练种子

依次使用 `high_train_immediate_open_seed1_v1.yaml` 至 `seed4` 开始四个 fresh run；同时使用 `high_train_finder_only_seed1_v1.yaml` 至 `seed4` 补齐配对基线。每个 run 都采用唯一 run-id、相同训练预算、固定 validation layouts、通用 finalizer 选择规则，以及 seeds `[0, 1]` 各 500 episode 的最终评估。

只有当现有 finder-only seed 0 的任务 hash、低层 SHA、预算和随机初始化 provenance 全部匹配时才复用。统计时以五个训练种子为重复单位，报告 paired-layout 差值与置信区间；vectorized episodes 不是独立训练重复。

## 核心检查指标

除 success、detect-all 和完成时间外，至少比较 finder accept/reject、拒绝后被其他 UAV 接管比例、rejection-to-assignment delay、duplicate commitment、高层决策次数、option duration 和 option interruption。`immediate_open` 的机制验收条件是拒绝后目标可被后续自然决策选择，同时扩散事件本身不终止任何非 finder pending option。
