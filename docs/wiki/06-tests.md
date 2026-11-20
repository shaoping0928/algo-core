# 06 · 测试与回归

## 6.1 运行

```bash
python -m unittest discover -s tests
```

测试基于标准库 `unittest`（无 pytest 依赖）。测试目录 `tests/`，共 12 个文件。

## 6.2 覆盖总览

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
