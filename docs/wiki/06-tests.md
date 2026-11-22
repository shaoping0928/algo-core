# 06 · 测试与回归

## 6.1 运行

```bash
python -m unittest discover -s tests
```

测试基于标准库 `unittest`（无 pytest 依赖）。测试目录 `tests/`，共 30 个文件（12 个覆盖主线，18 个覆盖 [07 新增实验线](./07-experiment-lines.md)）。

## 6.2 覆盖总览

**主线（12 个）**

| 测试文件 | 目标模块 | 覆盖重点 | 用例数（约） |
|---|---|---|---|
| `test_node_runtime.py` | `node_runtime.py`、`substrate_nodes.py`、`node_features.py`、`node_graph.py`、`latent_world.py`、`substrate_generic.py` | 运行时全链路（见 6.3） | ~52 |
| `test_substrate_value.py` | `substrate_value.py` | 价值头/风险头/重放/AUC/优先级 | 17 |
| `test_node_graph.py` | `node_graph.py`、`graph_memory.py` | 边类型语义、剪枝级联、序列化往返、图记忆确定性 | 14 |
| `test_node_components.py` | `node_components.py`、`node_graph.py` | 组件运行时性、三态、滞回、序列化 | 10 |
| `test_node_features.py` | `node_features.py` 等 | 谓词/组合、覆盖率门槛、共享节点、序列化 | 9 |
| `test_generic.py` | `substrate_generic.py`、`mixed_nodes.py`、`graph_memory.py`、`trace_signal_world.py` | 世界模型、行为流 pinning、混合记忆 | 10 |
| `test_eval_protocol.py` | `eval_protocol.py` | 种子解析、bootstrap、聚类汇总、paired delta | 9 |
| `test_trajectory_search.py` | `trajectory_search.py` | 穷举/beam 正确性、多目标字典序、max 聚合 | 6 |
| `test_flappy.py` | `flappy.py`、`latent_world.py` | cap 语义、反馈通道、import 无副作用 | 6 |
| `test_sandbox_regressions.py` | `substrate_sandbox_search.py`、`substrate_value.py` | target_gap 跟随 horizon、risk_action_dim 校验、探针 next-state | 3 |
| `test_substrate_rules.py` | `substrate_rules.py` | 不变量变化要求、决定性覆盖率 | 2 |
| `test_recall_split.py` | `recall_world.py` | CLI split 臂 | 1 |

**新增实验线（18 个）**

| 测试文件 | 目标模块 | 覆盖重点 | 用例数（约） |
|---|---|---|---|
| `test_stage_graph.py` | `stage_graph.py` | 恒定流单阶段、动作效应切换成边界、打乱对照无额外结构、不可预测流门控弃权、`_kmeans` 分离两种动作效应、词汇表扫描 | 7 |
| `test_stage_support_probe.py` | `stage_support_probe.py` | BIC 分段找切换点、恒定流单段、分段代价、单链聚类 | 4 |
| `test_death_rules.py` | `death_rules.py` | 两文字/析取规则恢复、XOR 报 UNKNOWN、反例精化补变量、`coverage_first`、边界裕度返回 UNKNOWN/NEAR、冗余文字移除、`modes` 发现两种死亡模式 | 10 |
| `test_death_rule_loop.py` | `death_rule_loop.py` | 分离遗漏终止态/误报、精化补第二模式、可跳过误报、无失败即停 | 4 |
| `test_death_failure_probe.py` | `death_failure_probe.py` | 死亡侧报告口径、随机/全终止添加的数量与标签 | 4 |
| `test_rule_expressions.py` | `rule_expressions.py` | AND/OR/XOR/NOT 嵌套掩码、数值项比较、组合搜索恢复 XOR/NOT、无法分离时保持可证伪 | 5 |
| `test_contextual_rules.py` | `contextual_rules.py` | 分桶条件优于全局、稀疏桶回退、精化加桶样本、`classify_death` 从不判 safe | 4 |
| `test_pattern_library.py` | `pattern_library.py` | `feature_dims` 限制规则语言、跨世界提取共享维度并重实例化 | 2 |
| `test_context_model.py` | `context_model.py` | 快速上下文适配未见机制、变化检测标记中段切换、关闭检测则不报切换、无标签下发现 basis、`apply_prior` 保留 warmup | 5 |
| `test_branches.py` | `branches.py`、`node_runtime.py` | current 瞬时/ recent 衰减累加、逐节点预测误差、transition_pairs 权重、value 慢移、共享 id 独立强度 + 5 个运行时集成用例 | 14 |
| `test_hierarchical_memory.py` | `memory_tree.py`、`hierarchical_memory.py`、`variable_registry.py` | 聚类/查询/分裂、持久 vs 临时方向、父摘要有损可展开、注册关键词；变量关系塑造可信度/学习率；各层级构建与阶段切换 | 12 |
| `test_generative_expansion.py` | `generative_expansion.py`、`context_model.py`、`hierarchical_memory.py` | 上下文提示改变生成流、从记忆取起点/边界、参考距离偏好匹配流 | 3 |
| `test_action_effect.py` | `action_effect.py` | 有符号/绝对效应、按维度选动作、归一化尺度、未知动作/空数据报错 | 5 |
| `test_action_conditioning_probe.py` | `action_conditioning_probe.py` | 岭回归恢复线性映射、动作增量 > 0.5 且胜过打乱、无动作信号时增量小 | 3 |
| `test_counterfactual_failure_probe.py` | `counterfactual_failure_probe.py` | 采样含尾部、去重限幅、终端报告用死亡侧、加正例限幅打标 | 4 |
| `test_n_tie_probe.py` | `n_tie_probe.py` | 机器精度平局 + Φ 不同则翻转、Φ 恒定不翻转、分数不同无平局、空决策 | 4 |
| `test_planner_rollout_quality.py` | `planner_rollout_quality.py` | 配置保留、计划死亡计数、`action_best`/`action_rank` 用可行存活、三态回调、分数/计划差距、裕度分层、oracle_plan 不改环境 | 10 |
| `test_probe_cache.py` | `probe_cache.py` | 同配置复用、异配置重拟合、`refit` 强制、损坏缓存回退、无 path 每次拟合 | 5 |

## 6.3 `test_node_runtime.py` 覆盖主题（最核心）

- **动作选择/概率**：`action_probability` 在平坦边际下的行为、两动作采样、动作边选择低痛动作。
- **信用分配与价值**：pain 信用回传到更早节点、episode 回报、k 步 bootstrap 目标算术、`future_pain_labels` 窗口与截断、`future_return_labels` 与 MC 反向 pass 一致、价值格子学折扣回报、格子跟踪 episode 多样性。
- **节点出生/增长**：新观测出生、方差出生切分混合格子、效用修剪合并进邻居。
- **记忆系统**：记忆强化与按成员召回、低支撑遗忘、pain 触发记忆、`memory_credit` 修正更早记忆/链接上下文、记忆风险读"活体率"。
- **特征/规则**：pain 前兆标签、`(feature, acting)` 配对、风险依赖 acting 动作、`evidence_batch` 计数、点事件标签只用进入状态的 pain、每节点窗口、校准用更新前读出、系统性错标标记、反事实向导（威胁定向/相对饥饿/均衡不动）、价值通道独立安装读取、子图分支分离与继承根窗口、`predicate_readouts` 丢弃原型通道、pain 对比居中于动作均值、`feature_value_batch` 报告支撑、分级威胁/可控超额拆分、读出校准到经验率、分级读含带标签短窗根、pain 锚点与价值基线是累积均值、无根时支撑为 0、校准逐读出粒度、IS 加权缩放证据、价值格子从 dopamine 通道 bootstrap、卸载只撤注册、回拟合损失反事实、代日志记录。
- **OOD/信任**：`ood_trust` 向基线收缩、`obs_trust` 降权观测通道。
- **组件**：绑定到运行时节点、按 valence 方差出生、gating 用签名。
- **好奇心**：是势差、评测时禁用并恢复。
- **关联/键控**：association 通道报"率"不是包络、pain 由关联学习、动作边按 governing 动作键控。
- **NNLS/组合**：`_nnls` 解正确、`hazard_composition` 替代 max、未拟合时保留历史 max。
- **架构中立性**：`test_core_does_not_embed_task_geometry`。
- **编码/离线对齐**：离线 pending 与运行时编码一致、合并 episode 保持反向引用存活。
- **端到端 smoke**：Flappy 信号跑通、规划器轨迹记为真实转移、采集遵守帧预算、`cellq` 无需搜索决策、`fit` 接受 value_horizon/dagger 轮、采集应用反事实预算与 propensity、`learn_objective_weight` 跟随实际存活、ledger 报告标记盲分量、规划器行带决策账本、`value_dispersion_labels` 覆盖两端。
- **终端语义**：`test_runtime_does_not_use_terminal_as_a_risk_label`。

`PlanScoreTests`：`plan_score` 与 `policy_result` 选择一致（含好奇心份额、`step_stride/stride_includes_lag` 下与穷举一致）。

## 6.4 回归 / pinning 测试（改动前必读）

| 测试 | 保护什么 |
|---|---|
| `test_generic.py::test_behavior_stream_is_pinned` | 用 MD5 钉死 `collect` 前 101 步转移流（七坐标流 `6bb2a643…`；旧六坐标流 `223efe…`）。**任何改变行为流的改动都必须有意更新哈希。** |
| `test_node_runtime.py::test_core_does_not_embed_task_geometry` | 断言核心模块源码不含 Flappy 术语——架构边界守卫 |
| `test_flappy.py::test_import_does_not_parse_cli_args_or_run_experiment` | `import flappy` 不解析 CLI、不输出（子进程验证） |
| `test_flappy.py::test_latent_runner_respects_its_frame_cap` | `latent_world.run` 遵守 cap |
| `test_node_graph.py::test_prototype_graph_memory_fit_is_repeatable` | 图记忆 fit 确定性 |
| `test_node_graph.py::test_graph_state_round_trip_preserves_all_edge_kinds` | 序列化往返保留全部边类型与属性 |
| `test_node_components.py::test_state_round_trip_preserves_graph_and_components` | 组件 + 图状态往返 |
| `test_node_features.py::test_feature_node_predicate_survives_serialization` | 特征谓词跨序列化存活 |
| `test_node_runtime.py::test_offline_pending_matches_the_runtime_encoding` | 离线读出与运行时编码对齐（防一行错位回归） |
| `test_node_runtime.py::test_merged_episode_keeps_backward_references_alive` | DAgger 多轮 id 重叠的历史 bug 回归 |
| `test_generic.py::test_mixed_memories_can_be_refitted_without_accumulating_state` | 三种混合记忆可重复 fit |

## 6.5 已修复缺陷对应的回归测试（背景）

- **一行错位（R4）**：离线校准/残差扫描曾用 `history[:lag][::-1]`（下一行编码）→ `test_offline_pending_matches_the_runtime_encoding`。
- **DAgger 轮次 id 重叠 → KeyError**：偏移量按 `planner_episodes` 步进 + 中途 reset 恢复剪枝保护 → `test_merged_episode_keeps_backward_references_alive`。
- **行为流分叉回归**：rng 抽取顺序被改 → `test_behavior_stream_is_pinned`。
- **import 顺序改变物理**：`flappy` 默认物理改为 latent 基准值 → `test_flappy` + `test_generic` 的 pinning。

## 6.6 提交前检查清单

1. `python -m unittest discover -s tests` 全绿；
2. 若改了 `collect` 行为流 → 有意更新 `test_behavior_stream_is_pinned` 的哈希并在 commit message 说明；
3. 若改了核心模块 → 确认 `test_core_does_not_embed_task_geometry` 仍通过（未引入 Flappy 术语）；
4. 若改了序列化 schema → 对应 round-trip 测试须覆盖新字段；
5. 机制类改动 → 用 `--fast` + held-out 做同批**配对基线** A/B（不要与历史数字直接比）。
