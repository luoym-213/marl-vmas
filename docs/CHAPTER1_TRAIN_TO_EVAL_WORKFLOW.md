# Chapter 1 单配置训练到评估完整指南

本文说明如何为一个新的 Chapter 1 SAR 实验配置完成配置创建、高层训练、checkpoint finalization 和正式评估。示例使用 `immediate_open`、观测半径 `0.3`、训练 seed `0`；其他机制、半径和训练 seed 按相同规则替换。

## 1. 目录、命名与配置创建

所有静态配置统一放在：

```text
/workspace/projects/CommSpread/configs/chapter1/
```

推荐命名组成：

```text
<role>_<mechanism>_r<radius>[_seed<seed>]_v<version>.yaml
```

半径使用三位整数编码：`0.3 → r030`、`0.4 → r040`、`0.5 → r050`、`0.6 → r060`。以本示例为例，需要三份静态配置：

| 角色 | 文件名 | 用途 |
| --- | --- | --- |
| task | `task_immediate_open_r030_v1.yaml` | 冻结环境、任务、感知和 FCSR 语义 |
| eval | `eval_immediate_open_r030_v1.yaml` | 冻结评估协议模板 |
| high_train | `high_train_immediate_open_r030_seed0_v1.yaml` | 冻结训练超参数、依赖和输出根目录 |

不要直接修改已有 `*_v1.yaml`。先复制最接近的已验证配置：

```bash
cd /workspace/projects/CommSpread
cp configs/chapter1/task_immediate_open_v1.yaml configs/chapter1/task_immediate_open_r030_v1.yaml
cp configs/chapter1/eval_immediate_open_v1.yaml configs/chapter1/eval_immediate_open_r030_v1.yaml
cp configs/chapter1/high_train_immediate_open_seed0_v1.yaml configs/chapter1/high_train_immediate_open_r030_seed0_v1.yaml
```

### 1.1 Task 配置

在 `task_immediate_open_r030_v1.yaml` 中只修改本实验计划改变的任务字段。例如观测半径实验只修改：

```yaml
perception:
  sensor_radius: 0.3
```

保持 `hierarchy.finder_first.cascade_mode: immediate_open`、物理、奖励、horizon 和低层接口不变，才能把差异解释为观测半径变化。当前解析器只支持已注册的 task version，因此复制半径变体时保留：

```yaml
task_version: chapter1_sar_immediate_open_v1
```

文件名和配置 SHA256 区分 r030/r040/r050；不要仅为文件名创建一个解析器不认识的新 `task_version`。

### 1.2 Eval 模板

在 `eval_immediate_open_r030_v1.yaml` 中修改：

```yaml
task_ref: task_immediate_open_r030_v1.yaml
task_version: chapter1_sar_immediate_open_v1
evaluation:
  output_dir: outputs/chapter1/eval_immediate_open_r030_v1
```

`evaluation.high_level_checkpoint` 仍然是占位内容。不要手工填最佳 checkpoint；finalizer 会复制模板并写入已验证的 checkpoint 路径、SHA256、选择证据和最终评估规模。低层 checkpoint 及其 SHA256 必须与训练配置保持一致。

### 1.3 High-train 配置

在 `high_train_immediate_open_r030_seed0_v1.yaml` 中修改：

```yaml
task_ref: task_immediate_open_r030_v1.yaml
task_version: chapter1_sar_immediate_open_v1
training:
  seed: 0
  base_eval_config: configs/chapter1/eval_immediate_open_r030_v1.yaml
  output_dir: outputs/chapter1/high_train_immediate_open_r030_v1/seed_0
```

其余训练预算和算法参数应保持冻结，例如 200 updates、16 environments、Strict-SMDP return、PPO 参数以及低层 checkpoint。不同训练 seed 必须复制新的 high-train 文件，并同步修改文件名、`training.seed` 和 `training.output_dir`。

### 1.4 配置一致性预检

运行训练前解析三份配置，确认 task SHA 完全相同：

```bash
cd /workspace/projects/CommSpread
export PYTHONPATH=/workspace/projects/CommSpread
python -c "from comm_spread.chapter1_config import resolve_chapter1_config as r; from pathlib import Path; p=Path('configs/chapter1'); xs=[r(p/n) for n in ['task_immediate_open_r030_v1.yaml','eval_immediate_open_r030_v1.yaml','high_train_immediate_open_r030_seed0_v1.yaml']]; print([(x.run_type,x.task_config_sha256,x.config_sha256) for x in xs]); assert len({x.task_config_sha256 for x in xs})==1"
```

预检还应确认训练与 eval 模板引用的低层 checkpoint 文件存在且 SHA256 正确。配置解析和训练入口都会再次执行约束检查。

## 2. 使用配置开展训练

训练命令必须提供：

- `--chapter1-config`：本次 high-train 静态配置。
- `--run-id`：全局唯一的运行标识；不能复用已有 run 目录。

推荐命令：

```bash
cd /workspace/projects/CommSpread
export PYTHONPATH=/workspace/projects/CommSpread
RUN_ID="chapter1-immediate-open-r030-seed0-$(date -u +%Y%m%dT%H%M%SZ)-$(git -C /workspace rev-parse --short HEAD)"
python scripts/train_high_ppo_sar.py --chapter1-config configs/chapter1/high_train_immediate_open_r030_seed0_v1.yaml --run-id "$RUN_ID"
```

训练不要求 Git 工作区干净；Git HEAD 和 diff hash 只作为 provenance 记录。不要使用 finder-only 高层 checkpoint 恢复 immediate-open 实验，也不要把旧 run-id 当成新训练目录。

训练结果保存到：

```bash
RUN_DIR="outputs/chapter1/high_train_immediate_open_r030_v1/seed_0/runs/$RUN_ID"
```

主要产物：

| 位置 | 内容 |
| --- | --- |
| `checkpoints/checkpoint_<update>.pt` | 固定 update 保存的候选 checkpoint |
| `checkpoints/latest.pt` | 最后训练状态，不等同于最佳 checkpoint |
| `validation/update_<update>/` | 固定 layout 的验证 summary、逐回合记录和 checkpoint binding |
| `scalars/train.csv` | 每 update 训练诊断和高层 reward |
| `health_decisions.jsonl` | health gate 决策 |
| `run_manifest.json` | Git、task/config、低层 checkpoint provenance |
| `resolved_config.json` | 完整解析后的训练配置 |
| `texts/run_summary.txt` | `status` 和 `final_update` 等训练终态 |

进入 finalization 前，`run_summary.txt` 必须显示 `status: complete` 且达到配置中的目标 update。early-stop、中断或 hash/binding 不一致的 run 不应进入正式评估。

## 3. Checkpoint 选择与评估准备

训练完成后运行独立 finalizer：

```bash
cd /workspace/projects/CommSpread
export PYTHONPATH=/workspace/projects/CommSpread
python scripts/finalize_chapter1_run.py --run-dir "$RUN_DIR"
```

默认最终协议是 seeds `[0,1]`、每个 seed 500 episodes。若需要显式写出同样参数，可使用：

```bash
python scripts/finalize_chapter1_run.py --run-dir "$RUN_DIR" --seeds 0 1 --episodes-per-seed 500
```

finalizer 会：

1. 校验 run 完整性、目标 update、task/config/低层/checkpoint SHA256 和 validation binding。
2. 在所有非 update-0 validation checkpoint 中按 success rate、detect-all rate、训练 high-level reward、较早 update 的字典序选择。
3. 对完整但 health gate 未通过的 run 继续选择，并记录 `health_status: gate_failed`；中断或证据损坏仍 fail closed。
4. 从 high-train 配置指定的静态 eval 模板派生最终评估配置。

生成：

```text
$RUN_DIR/checkpoint_selection.json
$RUN_DIR/derived_eval_final.yaml
```

`checkpoint_selection.json` 是 checkpoint 选择证据；`derived_eval_final.yaml` 才是正式评估入口。后者已经绑定选中 checkpoint 的路径与 SHA、正确的 task reference、评估 seeds、episodes 数和输出根目录。不要再手工编辑派生配置。

finalizer 不依赖 retrospective-v2，也不检查 `git_worktree_clean`。输出采用幂等语义：相同输入可重复调用；只有单个产物存在或内容冲突时拒绝静默覆盖。

## 4. 正式评估

Chapter 1 最终协议由两个独立进程完成：seed 0 和 seed 1 各 500 episodes。每个命令必须包含：

- `--chapter1-config "$RUN_DIR/derived_eval_final.yaml"`
- `--policy hgsar`
- `--seed 0` 或 `--seed 1`
- `--num-envs 500`，必须等于派生配置中的 `episodes_per_seed`
- `--sensor-radius 0.3`，必须与 task 配置一致
- `--device cuda:0` 或实际评估设备
- 独立的 `--output`、`--csv-output` 和 `--markdown` 路径

推荐执行：

```bash
cd /workspace/projects/CommSpread
export PYTHONPATH=/workspace/projects/CommSpread
for EVAL_SEED in 0 1; do
  EVAL_DIR="$RUN_DIR/evaluation_final_1000/seed_$EVAL_SEED"
  python scripts/analyze_sar_failure_modes_mpe_physics.py \
    --chapter1-config "$RUN_DIR/derived_eval_final.yaml" \
    --policy hgsar \
    --seed "$EVAL_SEED" \
    --num-envs 500 \
    --sensor-radius 0.3 \
    --device cuda:0 \
    --output "$EVAL_DIR/failure_modes.json" \
    --csv-output "$EVAL_DIR/failure_modes.csv" \
    --markdown "$EVAL_DIR/failure_modes.md"
done
```

每个 seed 的输出目录包含：

| 文件 | 内容 |
| --- | --- |
| `evaluation_metrics.json` | 固定五项论文指标；评估结束时自动生成 |
| `failure_modes.json` | 原有 summary、逐回合记录和 101 点全局 entropy trace |
| `failure_modes.csv` | 逐回合表格 |
| `failure_modes.md` | 人类可读诊断报告 |
| `failure_modes_assignment_events.csv` 等 | assignment/crossing 辅助诊断 |
| `resolved_config.json` | 当前 seed 的解析配置与 provenance |

不要让 seed 0 和 seed 1 共用 CSV 或 Markdown 路径，否则后运行的进程会覆盖前一个结果。`evaluation_metrics.json` 也应分别位于 `seed_0/` 和 `seed_1/`。

### 4.1 固定五项指标

`evaluation_metrics.json` 与终端固定报告：

1. **成功率**：全部回合的成功指示量均值和样本标准差 `ddof=1`。
2. **任务完成时间 `T_all`**：仅成功回合，从任务开始到全部目标救援完成的环境步；输出均值、标准差和有效样本数。
3. **平均救援响应时间 `mean_delta_t_res`**：成功回合内先对全部目标计算 `首次救援步 - 首次发现步` 的平均值，再跨成功回合统计均值和标准差。
4. **搜索效率 `eta_search(t)`**：`(H_0-H_t)/H_0`，使用团队全局 belief entropy；输出 `t=0…100` 共 101 点的跨回合均值曲线，不压缩为单一标准差。
5. **最后目标发现时间 `T_det_last`**：发现全部目标的回合中，最后一个目标首次发现的环境步；输出均值、标准差和有效样本数。

时间单位统一为 environment step；配置中的物理 `dt` 作为元数据写入。reset 时已发现的目标记为 step 0。条件指标没有有效回合时 `mean/std` 为 `null`；只有一个有效回合时 mean 有值、样本标准差为 `null`。

### 4.2 评估完成检查

每个 seed 结束后至少检查：

```bash
test -s "$RUN_DIR/evaluation_final_1000/seed_0/evaluation_metrics.json"
test -s "$RUN_DIR/evaluation_final_1000/seed_1/evaluation_metrics.json"
python -c "import json,sys; [json.load(open(p)) for p in sys.argv[1:]]; print('evaluation metric JSON files valid')" "$RUN_DIR/evaluation_final_1000/seed_0/evaluation_metrics.json" "$RUN_DIR/evaluation_final_1000/seed_1/evaluation_metrics.json"
```

同时确认终端没有 checkpoint SHA、task hash、非有限状态或配置一致性错误。当前流程按 seed 单独导出五指标，不自动把两个 seed 合并为一份统计文件；论文级跨训练 seed 汇总应作为后续独立统计步骤进行。

## 5. 单配置流程摘要

```text
创建 task/eval/high_train 三份静态配置
        ↓
解析并确认 task SHA 一致
        ↓
train_high_ppo_sar.py --chapter1-config ... --run-id ...
        ↓
完整 run + validation checkpoints
        ↓
finalize_chapter1_run.py --run-dir ...
        ↓
checkpoint_selection.json + derived_eval_final.yaml
        ↓
seed 0 × 500 episodes + seed 1 × 500 episodes
        ↓
每 seed 的 evaluation_metrics.json 与 failure-mode 诊断产物
```
