# 07 · 新增实验线

本页覆盖早期 `docs/wiki` 未收录的模块。它们**全部是只读探针 / 纯数据层 / 独立库**：通过 `substrate_nodes` 的 `fit / evaluate / policy_result / _curiosity_off / _encoded_pending` 观察主线，**不改变默认训练与决策路径**（`probe_cache` 的 docstring 明确写了这条纪律）。

> 这些文件随仓库持续演进（仅 `main` 分支就领先 `origin/main` 38+ 提交），`test` 目录已从 12 个增至 30 个。本节以"实验线"组织，而非按文件顺序。

## 7.1 阶段 / 相位线（stage & phase）

试图把"世界按时间分段"变成可用的粗粒度决策单元。

| 文件 | 职责 |
|---|---|
| [stage_graph.py](file:///c:/workspace/code/algo_core/stage_graph.py) | **纯数据层**：阶段图 schema + 离线动作残差分割器 |
| [stage_support_probe.py](file:///c:/workspace/code/algo_core/stage_support_probe.py) | 可行性探针：阶段数/驻留时间、`(stage,action)` 单元支撑、`(stage,action)→next` 边支撑与未见率 |
| [stage_sweep_probe.py](file:///c:/workspace/code/algo_core/stage_sweep_probe.py) | 扫描分割惩罚，找产生"决策尺度（~8–24 帧）驻留"的粒度 |
| [stage_content_probe.py](file:///c:/workspace/code/algo_core/stage_content_probe.py) | 内容审计：边界由什么构成（帧范围、动作计数、动作效应向量、跨边界相似度间隙） |
| [phase_stage_probe.py](file:///c:/workspace/code/algo_core/phase_stage_probe.py) | 相位相对 vs 绝对状态表示的信息量对比（**最新提交改的就是这里**） |
| [state_partition_probe.py](file:///c:/workspace/code/algo_core/state_partition_probe.py) | 把阶段定义为**状态空间类**而非时间段，检查驻留时间/状态时序/结果方差 |

### 核心接口

- `stage_graph.py`
  - `@dataclass Stage(index, episode, start, end, action_means, action_effect, residual_rss, support, action_counts)`（`.frames` 属性）
  - `@dataclass StageTransition(source, action, target, frames, episodes)`
  - `@dataclass StageGraph(stages, transitions, cluster_radius, gate_r2, clusters, stream_frames, episodes)`，方法 `cross_episode_informative(min_support=5, min_episodes=3)`、`summary()`
  - `pooled_oof_r2(features, target, groups, alpha=RIDGE_ALPHA)`（留一 episode-out 池化 R²）
  - `build_stage_graph(rows, lag=3, min_length=MIN_LENGTH, penalty=1.0, radius=None, gate_min_r2=0.0, clusters=None)`
  - `sweep_vocabulary(rows, ks, ...)`、`shuffle_actions(rows, lag=3, seed=12345)`（动作打乱对照）、`evaluate_stream(rows, ...)`
  - 常量：`MIN_LENGTH=5`、`RIDGE_ALPHA=1.0`、`ACTING_FALLBACK=0`
  - CLI：`--streams`、`--sweep-k`、`--report`

### 已记录的结论

- **分割惩罚平坦**：`"The penalty sweep is flat from 0.02 to 1.0, so the accepted boundaries are not a tunable granularity: they are a few very strong discontinuities."`
- **分割器抓住的是周期性世界重置**：`state_partition_probe` 指出那个最大残差增益的不连续点"在每个 episode 的同一帧，不携带决策信息"，因此把对象反转——**阶段是状态的类，转移是状态跨类**。
- 该线的结论是**判负**：`phase-relative stages` 被否决（`fed96ea`），线性去趋势也被否决（`47832ef`，去趋势把危险信号一起扔掉）；`phase_stage_probe` 仍在做最后的表示对比。

## 7.2 死亡规则线（0/1 死亡条件学习）

把危险从"概率 hazard"改写成**可证伪的 0/1 规则**：规则只在"覆盖训练正例且不覆盖训练负例"时被接受，其余一律报 `UNKNOWN`。

| 文件 | 职责 |
|---|---|
| [rule_expressions.py](file:///c:/workspace/code/algo_core/rule_expressions.py) | 0/1 谓词的可组合表达式树（NOT/AND/OR/XOR + 数值项与阈值比较） |
| [death_rules.py](file:///c:/workspace/code/algo_core/death_rules.py) | **AQ 式分离-征服**死亡规则集：归纳合取式 DNF，输出 1/0/UNKNOWN |
| [death_rule_loop.py](file:///c:/workspace/code/algo_core/death_rule_loop.py) | 失败驱动精化循环（把遗漏终止态/误报作为反例重拟合） |
| [contextual_rules.py](file:///c:/workspace/code/algo_core/contextual_rules.py) | 按上下文分桶各拟合一套规则，稀疏桶回退全局 |
| [pattern_library.py](file:///c:/workspace/code/algo_core/pattern_library.py) | 跨世界**模式（维度签名）库**：存"用了哪些关系"而非阈值，新世界检索 schema 后局部重实例化 |
| [death_transition_quality.py](file:///c:/workspace/code/algo_core/death_transition_quality.py) | 分离"世界模型预测误差"与"死亡规则判断误差"两个来源 |
| [death_rule_probe.py](file:///c:/workspace/code/algo_core/death_rule_probe.py) / [death_failure_probe.py](file:///c:/workspace/code/algo_core/death_failure_probe.py) / [death_feature_probe.py](file:///c:/workspace/code/algo_core/death_feature_probe.py) / [death_pattern_probe.py](file:///c:/workspace/code/algo_core/death_pattern_probe.py) | 离线验证探针：规则/模式、留出世界验证、反例精化、表示对比、跨世界复用 |

### 核心接口

- `rule_expressions.py`：`Expr`（基类）/`Literal`/`Not`/`Nary(op, children)`/`Term`/`FeatureTerm`/`NumericOp`/`Compare(term, op, threshold)`/`ExpressionRule(expression, label, support, positives, negatives)`；`search_compositions(states, target, literals, label, negatives=None, max_base=24, max_pairs=256, max_rules=8, min_support=2, max_complexity=8)`。
- `death_rules.py`：`UNKNOWN=-1`、`NEAR=2`；`negate(primitive)`；`@dataclass(frozen) DeathRule(literals, label, support, positives, negatives)`（`mask/mask_margin/text/signature/complexity`）；`DeathRuleSet(max_rules=24, max_literals=4, min_support=2, max_thresholds=48, max_attempts=96, compose=False, coverage_first=False, boundary_margin=0.0, ...)` → `fit/classify/classify_death/classify_death_status/verify/coverage_report/refine/add_examples/rules_for/modes/describe_modes/describe`。
- `contextual_rules.py`：`ContextualDeathRuleSet(buckets=2, min_positive=2, fallback_global=True, thresholds=None, **rule_kwargs)` → `fit(states, labels, context)/classify/classify_death/refine/add_examples/coverage_report`。上下文是**调用者提供的标量**（残差、距离、不确定度…），模块从不解释它。
- `pattern_library.py`：`Pattern(dims, label, support, text, world=None)`、`PatternLibrary(min_support=1)` → `add_rule/add_rules/add_rule_sets/schema_dims(label=1, min_fraction=0.25)/instantiate(states, labels, label=1, **rule_kwargs)/describe`。

### 已记录的结论

- **可证伪性**：`"unknown states are a request for counterexamples, not a probability."`
- **反事实失败探针**发现：`"The actual control trajectory rarely enters a terminal state the rules miss: the planner usually dies where the rules already fire, and the candidate plans that would die are never executed."`
- `death_pattern_probe` 用合成关系任务（真实条件 `(|z0|<0.15 AND |z1|>0.4) OR (|z2|>0.9)`）比较 scratch / schema / normalise / schema_normalise 四臂。

## 7.3 上下文 / 动力学 / 记忆线

| 文件 | 职责 |
|---|---|
| [context_model.py](file:///c:/workspace/code/algo_core/context_model.py) | 任务无关潜在上下文线性动力学：共享映射 + 在线低秩上下文校正 + 残差变化检测 |
| [dynamics_context_probe.py](file:///c:/workspace/code/algo_core/dynamics_context_probe.py) | 在隐藏参数随时切换的世界里比较 global / context / context+mem 三臂 |
| [generated_prior_probe.py](file:///c:/workspace/code/algo_core/generated_prior_probe.py) | "生成先验"：预处理世界→从记忆模拟机制→选匹配上下文，比较 global / online / generated |
| [context_benchmark.py](file:///c:/workspace/code/algo_core/context_benchmark.py) | 生成先验的多种子配对基准（标准 `--benchmark` 隐藏动力学固定，无法评估动力学先验） |
| [branches.py](file:///c:/workspace/code/algo_core/branches.py) | 共享节点图上的**角色特异强度分支**：current/recent/prediction/value |
| [variable_registry.py](file:///c:/workspace/code/algo_core/variable_registry.py) | 带类型变量 + 关系注册表，可信度/学习率从关系推导 |
| [memory_tree.py](file:///c:/workspace/code/algo_core/memory_tree.py) | 任务无关的上下文向量**合并树**（B 树式，但作用于度量向量） |
| [hierarchical_memory.py](file:///c:/workspace/code/algo_core/hierarchical_memory.py) | 异构分层记忆 frames→events→phases→worlds |
| [generative_expansion.py](file:///c:/workspace/code/algo_core/generative_expansion.py) | 从父摘要关键词**生成**新的近似流（而非存储副本） |
| [learning_trace.py](file:///c:/workspace/code/algo_core/learning_trace.py) | 帧级学习轨迹探针：逐决策记录原型/特征根/记忆/风险/价值/规划器输出 |

### 核心接口

- `ContextualLinearModel(obs_dim, action_dim, ridge=1e-8, context_dim=4, context_ridge=1e-2, window=64, change_detection=True, change_patience=3, change_factor=3.0, warmup=4, segment_size=128, discovery="cluster", merge_threshold=0.5)` → `fit/predict/observe/reset_context/begin_episode/apply_prior/end_episode`；模块级 `_cluster_corrections(vectors, fraction=0.5)`。
- `branches.py`：`NodeBranch(name, graph=None, decay=0.0, learn_rate=1.0, timescale="fast")`（`activate/step/read/top/reset/compare/snapshot`）+ `CurrentBranch`（瞬时）/`RecentBranch`（指数痕迹，decay 0.8）/`PredictionBranch`/`ValueBranch`（慢速池化，learn_rate 0.05），`BranchBank(graph=None)`（`add/get/activate/step/reset/transition_pairs(action=None, top=32, threshold=1e-6)/prediction_error/snapshot`）。
- `memory_tree.py`：`@dataclass MemoryNode(...)`（`.variance()`）、`MemoryTree(dim, max_fanout=4, leaf_capacity=24, merge_radius=0.5, split_variance=1.0, min_count=0.1, decay=0.95, keyword_dim=2)` → `insert/query/expand/keyword_persistence/persistence/decay/snapshot`。
- `hierarchical_memory.py`：摘要 dataclass `FrameSummary/EventSummary/PhaseSummary/EpisodeSummary`；`HierarchicalMemory(context_dim, observation_dim, window=4, salience_z=1.5, phase_split=0.6, max_phase_gap=24, world_fanout=4, world_radius=1.5, world_capacity=24, decay=0.95, base_prior=4.0, keyword_dim=2)` → `observe/end_episode/phase_summary/expand_phase/expand_world/world_keywords/world_reference_observations/query_world/persistence/snapshot`。
- `generative_expansion.py`：`GenerativeExpander(model, memory=None, policy=None, horizon=24, phase_len=6, bound_scale=6.0)` → `pick_start/bounds/expand/summarize`；模块级 `nearest_reference_distance(generated, reference)`。
- `variable_registry.py`：`Variable(variable_id, kind, estimate, support=0.0, variance=0.0, timescale="slow", metadata={})`、`VariableRegistry(base_prior=4.0, parent_gain=1.0, fast_base=0.5, slow_scale=1.0)` → `add/relate/parents/children/siblings/credibility/learning_rate/update/register_tree/register_direction/register_summaries/snapshot`。核心公式：`credibility = min(1, own_cred * (1 + parent_gain*parent_cred))`，慢学习率 `= 1/(support + base_prior)`。

### 已记录的结论

- `"A variable is persistent when the same direction carries support across phases and across world siblings, and temporary when its support lives in one low-level node."`
- `learning_trace` 的宗旨：`"A death with no local evidence and a death with a rising hazard are different failure stories."`（只读包装器，不改学习器）。

## 7.4 动作 / 时程 / 选择诊断线

回答"死亡是视界不够、选择排名错、还是根本不可避免"。

| 文件 | 诊断的问题 |
|---|---|
| [action_effect.py](file:///c:/workspace/code/algo_core/action_effect.py) | 任务无关动作效应统计（`ActionEffect`/`ActionEffectLibrary`） |
| [action_conditioning_probe.py](file:///c:/workspace/code/algo_core/action_conditioning_probe.py) | 动作是否增加预测力（LOEO R² + 排列检验） |
| [action_conditioning_diagnostics.py](file:///c:/workspace/code/algo_core/action_conditioning_diagnostics.py) | 定位 LOEO 异常（k=3 数值爆炸 / k=2 负 R²） |
| [causal_horizon_probe.py](file:///c:/workspace/code/algo_core/causal_horizon_probe.py) | 死亡前多少帧任意动作仍能逃脱（因果视界） |
| [doom_horizon_probe.py](file:///c:/workspace/code/algo_core/doom_horizon_probe.py) | 沿死亡轨迹回溯，视界内是否还存在存活计划 |
| [oracle_selection_probe.py](file:///c:/workspace/code/algo_core/oracle_selection_probe.py) | 固定视界下用真实存活替换选择规则 → 差距在视界还是排序 |
| [coarse_search_probe.py](file:///c:/workspace/code/algo_core/coarse_search_probe.py) | 宏动作长程粗搜索能否看到 8 帧细搜索看不到的东西 |
| [score_column_probe.py](file:///c:/workspace/code/algo_core/score_column_probe.py) | 逐列消融：去掉 risk/value/curiosity/q 哪列能命中 oracle |
| [decision_b0_probe.py](file:///c:/workspace/code/algo_core/decision_b0_probe.py) | B0 只读诊断：seed 101 miss 的首动作 gap、oracle-best、局部动作支持、端点 OOD 与 plan/suffix 判定 |
| [plan_score_probe.py](file:///c:/workspace/code/algo_core/plan_score_probe.py) | B0.5：把 miss 的 regret 分解为 action/suffix，逐列与子集消融 risk/value/curiosity/q |
| [risk_readout_probe.py](file:///c:/workspace/code/algo_core/risk_readout_probe.py) | B0.6：预测轨迹 vs 真实轨迹的风险读数对照，区分 step/terminal 与 world 漂移 |
| [risk_contrast_probe.py](file:///c:/workspace/code/algo_core/risk_contrast_probe.py) | 风险列实际能改变多少决策（有/无风险列的首步动作翻转率） |
| [metric_probe.py](file:///c:/workspace/code/algo_core/metric_probe.py) | 几何接近是否意味着结果接近（检验原型高斯核假设） |
| [n_tie_probe.py](file:///c:/workspace/code/algo_core/n_tie_probe.py) | 好奇心端点势 Φ 在平局时是否真能改变决策 |
| [outcome_coarse_probe.py](file:///c:/workspace/code/algo_core/outcome_coarse_probe.py) | 以"距终端时间桶"作粗状态是否可估计且能区分动作 |
| [event_lead_probe.py](file:///c:/workspace/code/algo_core/event_lead_probe.py) | 最后一个阶段边界/特征变化距死亡多少帧（有无领先） |
| [warning_lead_probe.py](file:///c:/workspace/code/algo_core/warning_lead_probe.py) | 每个危险通道在死亡前多少帧开始上升 |
| [planner_rollout_quality.py](file:///c:/workspace/code/algo_core/planner_rollout_quality.py) | 在规划器实际评估的计划上测死亡侧信号质量（含可复用解析/统计工具） |

### 关键工具与结论

- `action_effect.py`：`ActionEffect(action, count, mean_delta, mean_abs_delta, scale)`（`change/normalized_change`）；`ActionEffectLibrary`（`fit/effect/change/normalized_change/preferred/ranked/describe`）。声明："库不声称动作效应能跨世界迁移。"
- `planner_rollout_quality.py` 是被最多探针复用的工具模块：`status_callback/near_status_callback/oracle_plan/flatten_batches/plan_death_counts/action_best/action_rank/action_score_gap/plan_score_gap/oracle_best_details/summarize_margins/probe_config`。
- **视界不是瓶颈的证据**：`doom_horizon_probe`——`"The planner searches 8 frames ahead and, at almost every state, finds a plan that survives the whole horizon. Yet the agent dies."`；`coarse_search_probe`——`"the death is sealed about ten frames ahead, so all remaining headroom is in a long-range, coarse layer."`
- `metric_probe` 的核心警告：`"survival hazard rises with time in an episode, and observations encode that time, so a geometric neighbour can look predictive only because it sits at the same phase of another episode."`

## 7.5 共享工具

| 文件 | 职责 |
|---|---|
| [probe_cache.py](file:///c:/workspace/code/algo_core/probe_cache.py) | `load_or_fit(config, path=None, refit=False, fit_fn=None) -> (world, agent, loaded)`：各探针臂间复用已拟合模型，配置不匹配/损坏则回退重拟合 |
| [wiring_audit.py](file:///c:/workspace/code/algo_core/wiring_audit.py) | 接线审计：检查机制链每个环节（观测/尺度、原型、归纳特征、特征单元、危险合成、价值单元、记忆、好奇心证据、规划器得分列、世界模型滚动误差）是否真到达决策 |

`wiring_audit` 的宗旨：`"the point is to expose links that are computed but never consumed, or consumed with empty statistics."`——与主 wiki [02 主要模块职责 §2.10](./02-modules.md) 提到的"死机制清单"（详见 `TODO.md` 附录）互为印证。

## 7.6 与主线的接法

```
substrate_nodes.fit()  ──→  (world, agent)
        │                        │
        │                        ├─ evaluate(record=True) → 真实转移流
        │                        ├─ policy_result(...)     → 候选计划的分数/风险/价值
        │                        └─ _curiosity_off(agent)  → 评测时关好奇心
        ▼
   各探针脚本（只读）：stage_* / death_* / *_probe / learning_trace / wiring_audit
        │
        └─ 共享：probe_cache（复用拟合）、eval_protocol（种子解析/统计）、planner_rollout_quality（计划质量工具）
```

纪律：**探针不改变默认训练与决策路径**；沙盒反事实预测从不进入训练数据。
