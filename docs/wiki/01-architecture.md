# 01 · 整体架构

## 1. 设计目标与硬约束

| 约束 | 含义 | 落点 |
|---|---|---|
| 无梯度 | 不用反向传播/梯度下降；全部是计数、均值、闭式最小二乘（ridge/NNLS）、isotonic、PAV | `node_runtime.py`、`substrate_value.py`、`node_features.py` |
| 纯计数、可解释 | 每个统计量都能追到"哪条规则、哪个格子、多少样本" | 特征格子、原型统计、`prediction_report()` |
| CPU 可流式 | 单进程可跑；多种子并行靠进程池/线程池 | `substrate_nodes.py` 的 `main()` |
| 任务无关 | 运行时只接收 `observation, action, valence(dopamine/pain), episode boundary`，**不读** `GAP/2`、`BIRD_X`、碰撞窗口、`terminal` 风险标签 | 守卫测试 `test_core_does_not_embed_task_geometry` |
| 确定性与可复现 | 同命令重复运行逐位相同；行为流被哈希钉死 | `test_behavior_stream_is_pinned` |

## 2. 三层世界观

```
┌──────────────────────────────────────────────────────────────────────┐
│ ① 世界层 (World)                                                       │
│   flappy.py          无延迟 Flappy；每步输出 dopamine / pain 两个通道   │
│   latent_world.py    Flappy + lag 步延迟动作；Latent(seed) 变体 lag0..3 │
│   职责：只发事实（obs、dopamine/pain、done），不解释、不打风险标签       │
└──────────────────────────────────────────────────────────────────────┘
                                │  obs / action / dopamine+pain / done
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ② 运行时层 (Runtime / 认知基底)                                        │
│   node_graph.py       统一 Node/Edge 图：概念、转移边节点、记忆、派生    │
│   node_runtime.py     NodeRuntime：在线出生、动作边、TD(λ)/MC 信用分配  │
│   node_features.py    特征规则归纳（谓词 + AND/OR）+ (feature,action)格子│
│   node_components.py  运行时组件 + 三态符号节点（差分/电容/电阻）        │
│   signal_bank.py      每节点信号算子：id/d1/d2/leak/unleak/移动平均      │
│   graph_memory.py     原型图记忆（kmeans 原型→概念/边节点/记忆）         │
│   mixed_nodes.py      合取门 + 混合疼痛记忆（连续/分层/lift 版）         │
│   职责：把观测离散成节点，从 valence 信号里"自己学出"危险与价值          │
└──────────────────────────────────────────────────────────────────────┘
                                │  读出 readout（危险率 / 价值 / 证据）
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ③ 搜索与决策层 (Search / Planning)                                     │
│   trajectory_search.py  任务无关的 exhaustive / beam 滚动时域搜索        │
│   substrate_generic.py  LinearObservationModel + 通用 MPC（对照原型）    │
│   substrate_nodes.py    节点中心规划（**活跃主线**）                     │
│   substrate_sandbox_search.py  Flappy 适配器（世界反馈价值 / 任务基线）  │
│   substrate_graph_search.py    图记忆风险驱动的搜索原型                  │
│   substrate_value.py    价值头 / 终止风险头 / 重放缓冲 / AUC 工具        │
│   职责：用学到的世界模型滚 H 步候选，按学到的读出打分选计划               │
└──────────────────────────────────────────────────────────────────────┘
```

配套的**基础验证层**（`substrate_*.py`）与**诊断层**（`trace_*.py` / `*_probe.py`）不参与生产回路，只验证机制与暴露瓶颈；`substrate_rules.py` + `rule_*.py` 构成独立的"规则层"实验线。

## 3. 两条并行管线

仓库里同时存在两套"学习→规划"路径，不要混读：

### A. 世界反馈价值路径（旧 / 对照）

```
flappy/latent_world → substrate_generic.collect / substrate_sandbox_search.practice
        → LinearObservationModel + WorldFeedbackValueHead + TerminalRiskHead
        → trajectory_search.exhaustive_search / beam_search
        → 存活帧数
```

- **特征**：显式风险头 `TerminalRiskHead`（逻辑回归 + 平方/交互项）、MC 回报价值头、类别平衡 `TransitionReplay`。
- **适配器**：`substrate_sandbox_search.py` 提供 Flappy 专用评分（默认世界反馈价值；`--task-baseline` 切换到 gap-center 评分与碰撞可行性过滤，仅作对照）。
- **图记忆变体**：`substrate_graph_search.py` 用 `graph_memory.PrototypeGraphMemory` 的风险替代/叠加全局风险头。
- 该路径有**任务专用**评分/特征工程（只作为基线），与"任务无关"纪律区分开。

### B. 节点中心路径（新 / 活跃主线）

```
flappy/latent_world → substrate_generic.collect（真实转移）
        → NodeRuntime（概念节点 + 动作边 + TD(λ) 信用分配 + 事件记忆）
        → FeatureRuleBank（痛覆盖归纳 / 高回报归纳）→ 物化为图中特征节点
        → (feature, acting) 痛格子 / 价值格子（计数 + 收缩先验）
        → NNLS 非负堆叠组合 + isotonic 后置校准（hazard_composition）
        → policy_result：世界模型滚 H 步 → 危险 + 价值 + 好奇心 → exhaustive_search
        → evaluate / DAgger 回流 → 再训练
```

- **没有** `TerminalRiskHead`、`GAP/2`、`target_gap`、碰撞窗口、near-pipe 采样。
- 标签口径默认是**点事件 pain**；前兆标签是 `--precursor-labels` 的 opt-in。
- `substrate_nodes.py` 是这条线的唯一 CLI 入口，配 `eval_protocol.py` 做 held-out 评估。

## 4. 训练循环（节点中心路径，`substrate_nodes.fit` → `_train`）

```
① 采集采集：collect(seed, steps, coverage)         真实转移 (obs/action/history/signals/done)
② 世界模型：LinearObservationModel.fit            (obs, action-history) → next_obs
③ 多 epoch：  for epoch in range(epochs):
                 NodeRuntime.learn_episode(rows)   出生/动作边/记忆/关联/格子累积
                 if epoch == 0: 归纳痛规则 → 物化为特征节点 → (feature,action) 痛格子
                 if iterative_induction: 逐代重归纳 + held-out 回拟合（劣化即回滚）
④ 结构精化：方差出生 / 效用修剪（GNG-U）
⑤ 读出拟合：fit_readout_calibration（逐(根,acting)配对的 isotonic PAV）
            fit_hazard_composition（NNLS 非负堆叠 → isotonic 钉到经验率）
⑥ 目标学习：learn_objective_weight(rows)（执行计划-结局对的秩一致性选 w；仅 DAgger 轮）
```

可选的闭环形态：`--dagger-rounds N` 在每轮训练后用当前规划器在**训练种子的独立世界**执行若干 episode，把**真实执行**的转移追加进行为数据后从零重训（沙盒反事实预测绝不入训练）。

## 5. 决策循环（`substrate_nodes.policy_result`）

```
候选动作序列（2^H 个）
   → LinearObservationModel 批量滚 H 步，得到预测状态
   → 对第 step 个预测状态：
        hazard = agent.pain_signal_batch(...)     危险份额（点事件标签）
        value  = agent.feature_value_batch(...)   终点序列级价值
        Φ      = 1/sqrt(1+evidence)               证据势（好奇心项）
        逐步成本 = hazard (+ curiosity 势差)，终点成本 = −w·value
   → trajectory_search.exhaustive_search 按列 0 字典序取最小
   → decision_action 交付选中计划；可选 --decision cellq 走单步读格子
```

- 目标列：`[标量化成本, −dopamine, 危险份额, 价值份额, 好奇心]`；恒等式 `score[0] == score[2]+score[3]+score[4]`。
- `plan_score` 与 `policy_result` 走同一目标（有单元测试保证一致）。
- 评测时好奇心权重强制为 0（`_curiosity_off`），保证"尺子不被塑造"。

## 6. 评测协议

- **fast 协议**：1500 步 / 3 epoch / 1000 节点 / H=8 / cap=400（`--fast`）。
- **benchmark 协议**：1500 帧离线腿只回放 1 遍 + 一轮 5000 帧自由控制 + 评测 cap=5000（`--benchmark`）。
- **held-out**：`--eval-seeds 120-139`（与训练种子分离）配合 `--eval-episodes`；世界 id = `eval_seed*10000 + episode`。
- **统计**：`eval_protocol.py` 输出逐 episode 行、按种子分组汇总、**按训练模型聚类**的 bootstrap 95% CI、以及两臂 `paired_delta`。空 `--eval-seeds` 会打印警告并回退旧 self-eval。
- `--report file.json` 落盘，便于后续 paired 比较。

## 7. 核心不变量

1. **任务无关**：核心模块（`node_runtime.py`、`node_graph.py`、`node_features.py`、`substrate_nodes.py`）不得出现 Flappy 术语（`BIRD_X`、`GAP`、`pipes`、`flap` 等）——由测试守卫。
2. **沙盒预测不是现实证据**：反事实 rollout 只作搜索经验，永不写入真实转移或价值/世界模型训练集。
3. **反馈确定性**：pain 是点事件，只修正记忆/风险率，不给之前的正常飞行状态贴标签（前兆标签属 opt-in 的"人造信号"，历史结论需重锚）。
4. **`lag` 语义**：动作延迟生效，真正掌管 `s_{t+1}` 的动作是 `history[lag]`（acting action）；特征/价值格子按 acting 键控。
5. **确定性**：同命令重复运行逐位相同（含 DAgger 多轮与多进程 `--jobs`）；行为流采样器被哈希钉死，任何改动必须有意更新哈希。
6. **内存可预算**：图节点有 `max_nodes` 上限；原型有方差出生 + 效用修剪；分片/进程并行避免单点爆炸（见 `TODO.md` 运行卫生）。
