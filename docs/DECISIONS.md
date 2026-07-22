# CommSpread MARL 设计决策

更新日期：2026-07-22 UTC；当前 HEAD：`54c95bd`。本文件明确区分已验证决定、观察和未充分结论。

## 已确定保留的设计

### 1. 正式任务保持 `retire_on_rescue=True`

**已验证事实**：no-retire 虽曾得 `.671875`，但改变任务语义。
**决定**：正式结果和主表一律 retire；no-retire 仅 ceiling。

### 2. 固定 repaired MAPPO + goal fallback

**已验证事实**：固定目标三智能体低层成功 `249/256 = 97.27%`；full-information SAR `254/256 = 99.22%`。
**决定**：当前高层比较固定该低层与 fallback；不以 auto-resampled 98.4% 叙述部署能力。

### 3. 固定正式比较条件

**已验证事实**：radius .3、LR-only、多个 reward shaping 和 hard stage 未给出稳定改善。
**决定**：固定 3 UAV/3 targets/100 steps/radius .6、原 reward/网络/PPO、同一低层与成功判定；实验按成对布局单变量比较。

### 4. Finder-only 执行机制作为当前正式策略

**已验证事实**：旧 checkpoint + finder-only 相对 staggered pooled 512 `128 -> 220`，`+17.969pp`，CI `[14.063,21.875]pp`，两独立 256 集均为正。
**决定**：保留该机制和旧 checkpoint10；机制显式开启，默认路径不变。

### 5. checkpoint 选择不看训练 batch 或 latest

**已验证事实**：trained update10/20/30 固定 64 都 26/64；update30 按预注册的 baseline-only/detect-all/failure 规则选出。
**决定**：任何后续训练先固定验证集与选择规则，再正式配对；禁止 latest 自动替换。

### 6. 只允许用已发现目标做执行分配

**已验证事实**：E28 exact visible matching 只使用 globally detected、unvisited、unclaimed targets 和 active/current-decision UAV；无 invalid/nonfinite。
**决定**：任何集中/精确分配不得读取未发现目标位置。

### 7. 新消融功能默认关闭

**已验证事实**：`--enable-module-ablation`、coverage 指标均为显式开关，A identity path 已与旧评估 parity 对齐。
**决定**：保留为评估工具；不能静默改变训练或默认评估行为。

## 已否定或降优先级的设计

### 1. finder-first 重训练作为已验证提升：否定

**依据**：old+finder -> trained update30 仅 `+3/512`，CI `[-2.344,3.516]pp`。
**决定**：不把重训练 checkpoint 替换为正式策略；训练 seed-1 前不扩展该路线。

### 2. 单独精确 matching 作为下一主线：降优先级

**依据**：D−A `−.78pp`，CI 跨零；虽降低几何 regret/crossing，却没有独立 success 收益。
**决定**：不单独训练/调参分配器；仅在可靠 all-detected 终局阶段研究它。

### 3. 单独 all-detected gate 作为充分解：否定

**依据**：C−A `+1.95pp`，CI 跨零；虽将 premature retirement `69.73% -> 0`，却产生 post-detection 过探索。
**决定**：gate 只能与搜索/终局 handoff 联合考虑。

### 4. no-retire、hard sequential mask、reward shaping、LR-only：否定或降级

**依据**：分别改变语义、性能不稳、造成 duplicate/低 success，或无独立改善。
**决定**：不要与当前主线叠加。

### 5. proportional controller 进入核心主表：否定

**依据**：E28 明确固定 repaired MAPPO；proportional 仅可作低层诊断。
**决定**：不得混入公平核心比较。

## 仍不充分的结论

- **观察**：三通道搜索增加 detect-all、减少覆盖重叠，但 coverage efficiency 略降，且条件 all-detected step 更晚。这个条件均值受困难布局构成影响，不能简单判断搜索速度。
- **观察**：matching 在 all-detected context 有条件收益（H−C `+4.69pp`、E−F `+10.55pp`）。
- **推测**：最值得学习的是搜索与终局切换的联合策略，而非独立 assignment 网络。
- **UNKNOWN**：finder-only 训练 seed-1 是否复现；未来可训练版本能否同时保留三通道覆盖与可靠终局 handoff。
- **UNKNOWN**：现有 dirty 改动能否拆成无交叉、可审计的小 commits，尚未执行。

## 约束

- 不同时改 reward、网络、低层、物理参数、retire 语义、成功判定或布局。
- 新实验必须保留逐环境 JSON、配置/命令、layout hash、paired CSV 和 invalid/nonfinite 审计。
- `D_seed0.json` 是无效审计产物；禁止重新纳入统计。
