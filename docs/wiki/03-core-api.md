# 03 · 关键类与函数说明

本文只列**对外/对决策有影响**的接口。签名以源码为准，省略 `self`。

---

## 3.1 `node_graph.py`

### 常量

```
节点类型： CONCEPT="concept"  TRANSITION="transition"  MEMORY="memory"  DERIVED="derived"  EDGE="edge"
边类型：   ASSOCIATION="association"  TRANSITION_EDGE="transition"  MEMBERSHIP="membership"  DERIVATION="derivation"
```

### `@dataclass Node`

| 字段 | 说明 |
|---|---|
| `id, kind, beta=0.2` | 标识、类型、痕迹衰减率 |
| `activation, trace` | 瞬时激活与泄漏痕迹 |
| `members` | 仅记忆节点使用 |
| `source, depth, edge_key` | 派生/边节点元数据 |
| `value, dopamine, count` | 学到的 valence 与计数 |
| `eligibility_pain, eligibility_dopamine` | 资格迹（记忆回溯修正用） |
| `immediate_pain, immediate_dopamine` | 本步立即价 |
| `role, component_id, sign, predicate, pain_horizon` | 特征/组件/标签窗口元数据 |

方法：`activate(value)`（更新激活与泄漏痕迹）、`decay(factor)`。

### `@dataclass Edge` / `@dataclass ActionEdge`

- `Edge(source, target, kind, weight, eligibility, edge_node, order)`；`key()` 返回 `(source, target, kind)`。
- `ActionEdge(source, target, action, count, q_pain, q_dopamine, prediction_error)`——动作条件转移，`key()` 为 `(source, target, action)`，是节点级 `Q(s,a)` 的基底。

### `class NodeGraph`

| 方法 | 作用 |
|---|---|
| `add_node(kind, node_id=None, **kwargs) -> str` | 建节点；`node_id` 省略时按 `kind_<counter>` 命名；重复 id 抛错 |
| `add_association(first, second, weight=0.0)` | 对称关联边（**不给边节点**），禁止自关联 |
| `add_transition(source, target, weight=0.0, create_edge_node=True)` | 有向转移边，可拥有 `edge:<src>-><tgt>` 边节点 |
| `add_action_edge(source, action, target) -> ActionEdge` | 动作条件边；`action_edges_from/action_targets` 查询 |
| `add_memory(members, node_id=None) -> str` | 记忆节点 + membership 边 |
| `add_membership(memory_id, member_id)` | 追加成员 |
| `add_derived(source, kind, max_depth=2, ...)` / `add_derivation(derived, source, weight=1.0)` | 旧派生根 / 复合特征指向子节点 |
| `activate(ids, values=None)` | 批量激活 |
| `observe_association(active, rate=0.05)` | 共激活塑形关联权重与资格迹 |
| `observe_transition(source, target, rate=0.1)` | 塑形转移边并激活边节点 |
| `observe_memory(memory_id, rate=0.1)` | 记忆激活 = 成员痕迹均值 |
| `propagate(steps=1, gain=0.5, decay=0.9)` / `recall(cues, steps=2)` | 激活传播 / 线索回忆 |
| `decay(edge_factor=0.999, trace_factor=0.9)` | 全局遗忘 |
| `prune(min_association=1e-3, min_edge_trace=1e-4)` | 剪低权关联/低痕迹边节点（含级联清理） |
| `_remove_node(node_id)` | 删除节点并级联清理所有引用它的边、子边节点、members |
| `snapshot()` | 返回节点/边/动作边/边节点/记忆计数 |
| `to_state()` / `from_state(state)` | 全量序列化/反序列化（含动作边与所有属性） |

---

## 3.2 `node_runtime.py`

### 模块级函数

| 函数 | 说明 |
|---|---|
| `future_pain_labels(rows, window) -> List[float]` | 前兆标签 `P_k(t)=max pain over 下 k 个状态`（截断于 done/换 episode）；`window=1` 即点事件 |
| `future_return_labels(rows, gamma) -> List[float]` | 每步之后的 MC 折扣 dopamine 回报，与 `learn_episode` 反向 passe 逐位一致 |
| `_isotonic_map(reads, targets)` | 经典 isotonic（PAV）单调映射，按块质心插值到经验率 |
| `_nnls(design, target, ...)` | Lawson-Hanson 主动集非负最小二乘（Tikhonov ridge 增广） |
| `_expand(solution, passive, ...)` | NNLS 内层解展开辅助 |

### `class NodeRuntime`

构造签名（关键参数）：

```python
NodeRuntime(obs_dim, action_space=(0,1), lag=0, gamma=0.99, lam=0.9,
            alpha_v=0.05, alpha_q=0.1, alpha_now=0.2, alpha_mc=0.2,
            birth_threshold=0.45, max_nodes=5000, exploration=0.1,
            uncertainty_weight=0.05, unknown_penalty=0.5,
            soft_neighbors=5, soft_sigma=0.75,
            component_kinds=(), component_min_count=8, max_components=128,
            component_threshold=0.2, component_smoothing=0.5,
            component_weight=0.5, component_gating=False,
            graph_rate=0.1, memory_decay=0.9, memory_min_strength=0.02,
            memory_weight=0.1, association_rate=0.0, valence_rate=0.2,
            feature_rate=0.05, feature_action_prior=2.0, association_prior=2.0,
            importance_weighting=False, variance_births=True, utility_pruning=True,
            pain_horizon=None, point_labels=False, memory_credit=False,
            rule_min_coverage=0.0, rule_min_primitive_coverage=0.0,
            pairwise_features=True, voi_target=3, voi_cell_cap=16,
            voi_min_prior=0.5, voi_threat_min=0.25,
            feature_value_prior=4.0, feature_value_weight=0.02,
            subgraph_cells=False, predicate_readouts=False, event_memory_risk=True,
            contrast_objective=False, value_trust_samples=150.0,
            control_risk_weight=0.0, curiosity_weight=0.0, value_bank_only=False,
            value_horizon=None, ood_trust_scale=0.0, obs_trust=1.0,
            action_temperature=0.1, action_uncertainty=0.5, seed=0, scales=None)
```

构造期做**大量参数合法性校验**（维度、`gamma∈[0,1)`、`lam∈[0,1]`、`alpha_*`、`birth_threshold>0`、各 prior ≥0 等）。

方法按主题分组（行号为源码定位）：

**A. 节点出生与指派**

| 方法 | 作用 |
|---|---|
| `encode(observation)` | 观测 → 运行态向量（叠加 pending 动作窗口） |
| `_zero_pending()` | 清空 pending |
| `_nearest(state)` | 最近原型与平方距离 |
| `_create_node(state)` / `_assign(state, create=True, ...)` | 建节点 / 指派（含出生判定） |
| `_refresh_variance_bar()` / `_variance_split(node_id, distance)` | 人口相对方差门 + 方差出生（refinement） |
| `_prune_prototypes(exclude=())` | GNG-U 效用修剪，最低效用原型合并进最近邻居 |

**B. 记忆与关联**

| 方法 | 作用 |
|---|---|
| `_new_memory()` / `_interact(...)` | 建记忆 / 单步交互（关联、转移、动作边、pain 记账） |
| `memories_for(node_ids)` / `_reinforce_memories(node_ids)` | 检索 / 强化记忆 trace 与 membership |
| `_propagate_association_values()` | （可选）关联价值传播 |
| `_forget_memories()` / `_prune_components()` | 记忆遗忘 / 组件修剪 |
| `_note_valence(node_ids, pain)` / `_note_valence_weighted(node_ids, pain, weight)` | 记 valence（含加权/资格迹版） |
| `_advance_memory_traces(active, memories, created_memory, ...)` | 每步推进记忆痕迹、membership 衰减、事件记忆 |
| `_learn_pain_association(pain, ...)` / `_pain_association(node_id)` / `_association_base()` | 转移级 pain 率统计（洁净转移也是样本） |

**C. 危险读出与组合**

| 方法 | 作用 |
|---|---|
| `pain_signal_from_weights(weights)` / `pain_signal_state(state)` | 单状态危险读出（路径入口） |
| `_hazard_channels(states, ...)` | 抽离四通道（原型/特征/记忆/rules）为同一组设计列 |
| `pain_signal_batch(states, ...)` | **决策主读出**：已拟合组合则 NNLS 加权和 + isotonic，未拟合则历史 max 路径 |
| `fit_hazard_composition(states, pending, labels)` | 对标签闭式拟合非负堆叠权重 `w≥0` |
| `fit_readout_calibration(states, ...)` | 逐 `(根, acting)` 配对的 isotonic 校准 |
| `_calibrate_threat(threat)` | 威胁校准 |
| `pain_graded_batch(states, ...)` | 分级读出：`threat=min_a P(pain|f,a)` 与 `excess=逐动作多冒的可控部分` |
| `pain_contrast_batch(states, ...)` | 动作对比读出 `P(pain|f,a) − mean_a' P(pain|f,a')` |

**D. 特征规则物化与格子**

| 方法 | 作用 |
|---|---|
| `induce_rules(train_states, train_pain, ...)` | 归纳并（可选）安装痛规则 |
| `_materialize_feature(feature, role)` | 把谓词物化为图节点（共享子节点去重） |
| `install_feature_rules(rules=None, channel=...)` / `uninstall_feature_rules(node_ids)` | 安装/卸载（可逆，供 held-out 回拟合） |
| `_node_horizon(node_id)` / `_active_feature_ids(state)` / `_active_feature_roots(state)` / `_all_value_roots()` | 标签窗口与激活根查询 |
| `_feature_mask_table(obs)` / `_table_entry(...)` | 特征掩码批量计算 |
| `_update_feature_pain(ids, pain, weight)` / `_update_feature_action(ids, action, weight)` / `_update_feature_value(ids, acting, ...)` | 格子累积 |
| `_feature_cell_read(node_id, acting)` | 单格子读（含收缩先验） |
| `_feature_risk_batch(states, ...)` / `_feature_memory_risk_batch(states, ...)` / `feature_risk_batch(states)` | 风险批量读出（特征 / 事件记忆） |
| `feature_value_batch(states, ...)` | **价值批量读出**，`return_support=True` 时返回 `(值, 支撑)` |
| `_k_step_value_target(steps, position, snapshot)` | k 步 bootstrap 价值目标 |

**E. 预测误差回路**

| 方法 | 作用 |
|---|---|
| `_record_prediction(root_ids, acting, actual, ...)` | 更新前记录读出预测（与实际严格分离） |
| `prediction_report()` | 按格子的 pred/actual 均值、Brier、系统性错标标记 |

**F. 决策与动作**

| 方法 | 作用 |
|---|---|
| `_signature(node_id)` / `_update_gated(node_id, signature, ...)` | 组件三态签名 / gated 动作边更新 |
| `_neighbor_weights(state)` / `_soft_weights_batch(states)` | 软邻域权重 |
| `_score_actions(node_id, ...)` / `decision(observation)` | 动作打分 / 决策 |
| `_action_margin(scores)` / `_action_probability(scores)` / `_set_action_nodes(margin)` / `act(observation, explore=True)` | 动作概率化与采样 |
| `_update_transition_value` / `_update_action_edges` / `_update(current, action, next_node, ...)` | TD/MC 更新 |
| `observe(observation, action, valence, ...)` / `reset()` / `learn_transition(...)` | 旧式单步接口 |
| `learn_episode(rows)` | **核心训练入口**：前向交互 + 反向 MC/TD(λ) 信用分配 + 格子/记忆更新 |
| `evidence_batch(states)` | 好奇心用的证据计数（`(特征,动作)` 格子观测数） |
| `counterfactual_action(observation)` | VOI 向导：威胁定向 + 相对饥饿门 |

**G. 组件与持久化**

| 方法 | 作用 |
|---|---|
| `bind_component(base_node, kind, ...)` / `_component_step(...)` / `_maybe_bind_components(node_id)` | 组件绑定与步进 |
| `immediate_risk_batch / value_state / dopamine_state / node_value / node_dopamine` | 单点读出 |
| `prototype_pain_baseline()` / `prototype_valence_baselines()` / `_shrink_to_baseline(values, nearest_min, baseline)` | OOD 信任收缩 |
| `valence_state_batch(states)` / `_memory_values_batch(...)` / `_memory_value_from_weights(...)` | 原型/记忆价值批量读出 |
| `top_nodes(limit)` / `top_edges(limit)` / `snapshot()` | 诊断快照 |

> 完整方法清单可用 `Grep "^    def " node_runtime.py` 枚举（共约 97 个定义）。

---

## 3.3 `node_features.py`

| 符号 | 说明 |
|---|---|
| `QUANTILES` | 阈值候选分位集合（0.01–0.99 线性 49 点 + 尾部极值） |
| `Primitive(dim, magnitude, op, threshold)` | 单坐标阈值谓词；`.mask(x)` / `.label()` / `.complexity()` |
| `Composite(op, children)` | `op∈{"and","or"}`；`.mask(x)` / `.label()` / `.complexity()` |
| `FeatureRule(feature, support, hits, precision, test_support, test_hits, test_precision, confidence)` | 一条规则及其统计 |
| `expand_observation(x)` | 追加全部成对坐标差 `x_i - x_j`（纯代数） |
| `FeatureRuleBank(max_rules=8, min_support=5, max_per_group=8, min_z=2.0, min_lift=2.0, min_coverage=0.0, min_primitive_coverage=0.0, pairwise_features=True, max_candidates=240)` | 规则银行 |
| `FeatureRuleBank.induce(train_x, train_y, test_x=None, test_y=None, coverage_x=None)` | 生成原语候选 → AND 对 → OR 对 → `AND(prim,OR)`；覆盖率门槛；**覆盖驱动选择**（优先解释未被解释的 pain） |
| `FeatureRuleBank.activate(x)` | 逐状态返回 `max(confidence·mask)` |
| `FeatureRuleBank.describe()` / `state_dict()` / `load_state_dict(state)` | 描述与持久化 |
| `feature_dims` / `feature_primitives` / `feature_state` / `feature_from_state` | 结构工具 |
| `_stats(mask, target)` | support/hits/precision/z |

---

## 3.4 `node_components.py`

| 符号 | 说明 |
|---|---|
| `@dataclass(frozen=True) ComponentSpec(id, base_node, kind, params=(), threshold=0.2, hysteresis=0.0, smoothing=0.5)` | 组件规格（可持久化） |
| `RuntimeComponent(kind, params=())`，`kind ∈ {difference, capacitor, resistor}` | `update(x) -> float`：差分 `x - prev`；电容 `state=(1-β)state+βx`；电阻 = `x - state` |
| `ComponentBank(graph, rate=0.2, threshold=0.2, hysteresis=0.0, smoothing=0.5)` | 组件银行 |
| `bind(base_node, kind="difference", params=(), component_id=None, threshold=None, hysteresis=None, smoothing=None) -> str` | 绑组件并建三个符号概念节点 `component:<id>:+ / 0 / -` |
| `step(activations) -> Dict[str,int]` | 更新所有组件并激活符号节点（建关联/转移） |
| `active_sign_nodes(base_node=None)` / `add_memory(component_ids=None)` | 活动符号节点 / 建记忆 |
| `observe_valence(pain, dopamine=0.0, rate=None, active=None)` | 从世界价更新符号节点 valence |
| `utility(component_id)` / `observations(component_id)` / `prune(min_observations=20, quantile=0.25)` / `remove(component_id)` | 效用评估与修剪 |
| `state_dict()` / `load_state_dict(state)` | 持久化 |
| `classify(value, previous, threshold, hysteresis) -> int` | 三态分类（含滞回） |

---

## 3.5 `trajectory_search.py`

```python
exhaustive_search(initial_state, action_space, horizon,
                  transition_batch, evaluate_step, batch_size=4096,
                  evaluate_terminal=None, score_combine="sum",
                  evaluate_death=None, evaluate_death_status=None,
                  unknown_penalty=0.0, candidate_callback=None) -> SearchResult
beam_search(initial_state, action_space, horizon,
            transition_batch, evaluate_step, evaluate_terminal=None,
            beam_width=256, score_combine="sum") -> SearchResult
```

- `transition_batch(states, sequences, step)`：批量一步预测；**完整候选序列可见**（供延迟动作读前缀）。
- `evaluate_step(states, sequences, step) -> (score_increment, feasible)`：增量可为标量或向量（向量按字典序最小化）；`score_combine ∈ {"sum","max"}`。
- `SearchResult(actions, score, feasible, feasible_count, candidate_count, first_state, final_state, first_death, first_unknown, unknown_count, safe_count)`：后四个字段默认 `-1/0/0`，只有提供三态回调时才有意义。
- `evaluate_death_status(states, sequences, step) -> 1/0/-1`：`1`=死亡、`0`=安全、`-1`=未知；与 `evaluate_death` 互斥。未知不单独否决计划，`unknown_penalty` 只给主目标按未知步数加罚（返回的 `score` 仍是未加罚目标）。
- `candidate_callback(batch)`：诊断/探针钩子，接收每批候选的序列、分数、可行性、首死/首未知/未知步数与预测状态；搜索本身不依赖它。
- 无可行解 → 返回最低分候选 + `feasible=False`；核心不保存完整轨迹（只留 current/first/final）。`evaluate_death=None` 且 `unknown_penalty=0` 时选择路径逐位不变。

---

## 3.6 `substrate_nodes.py`

| 函数 | 说明 |
|---|---|
| `value_dispersion_labels(returns, quantile)` | 取回报分布**两端**分位（对齐的顶部分位 + 死亡的底部分位）作价值归纳标签 |
| `_graded_holdout_loss(agent, enc, labels, mask, exclude=())` | held-out 上分级威胁读出的 MAE；`exclude` 逆转属性交换做精确反事实（F5 回拟合） |
| `_encoded_pending(rows, lag)` | 行对应的 pending 窗口 = `history[1:lag+1][::-1]`（**本行**编码，R4 修复） |
| `_train(transitions, seed, epochs, lag, ...)` | 离线腿 + 每个 DAgger 轮的核心训练序列 |
| `collect_planner(seed, world, agent, episodes=2, cap=1200, horizon=10, ...)` | 用规划器自采**真实执行**轨迹（可带帧预算） |
| `_rankdata` / `_spearman(first, second)` | 秩统计工具 |
| `learn_objective_weight(rows, min_decisions=8)` | P4：执行计划-结局对上的秩一致性选 w；候选格从数据尺度导出 |
| `ledger_survival_report(rows, weight, ...)` | 逐账本分量对实际剩余存活的 Spearman（失配即标记"不预测存活的目标分量"） |
| `fit(seed, steps=5000, coverage=0.1, max_nodes=5000, ...)` | 单种子端到端训练（含可选 DAgger） |
| `_curiosity_potential(support)` / `_curiosity_term(phi_start, phi_end, weight, discount, horizon)` / `_curiosity_off(agent)` | 好奇心势差 `Φ=1/sqrt(1+evidence)` 与评测归零 |
| `policy_result(env, world, agent, prefix, horizon, lag=LAG, ...)` | **决策主函数**：世界模型滚 H 步 → 危险/价值/好奇心 → 选计划 → 返回评分与账本 |
| `policy_action(...)` / `cell_q_action(env, world, agent, prefix, lag=LAG)` | 搜索动作 / 单步 `argmin_a[hazard − w·V_f]`（cellq 消融） |
| `decision_action(env, world, agent, prefix, horizon, ...)` | 交付选中计划（可 `ledger_out` 写决策账本） |
| `plan_score(initial_state, world, agent, prefix, plan, horizon, ...)` | 对**固定计划**按同一目标打分（有与搜索一致性单测） |
| `evaluate(seed, world, agent, cap=6000, horizon=10, lag=LAG, record=False)` | 评测；`record=True` 返回规划器**实际执行**的真实转移（schema 同 `collect`） |
| `run_seed(task)` / `main()` | 任务封装 / CLI（`--jobs` 进程或线程池 + `--report`） |

CLI 旗标分组（默认值）：

- **规模**：`--seed 100 --seeds 1 --jobs 8 --steps 5000 --coverage 0.1 --cap 6000 --horizon 10 --max-nodes 5000 --birth-threshold 0.45 --epochs 5 --lag 3`
- **组件/记忆**：`--components "" --component-min-count 8 --max-components 128 --component-weight 0.5 --component-gating --memory-weight 0.1 --association-rate 0.0`
- **标签口径**：`--precursor-labels --pain-horizon 0 --instant-labels(兼容别名，无效果)`
- **特征**：`--no-features --rule-min-coverage 0 --rule-min-primitive-coverage 0 --rule-validation-seed 0 --rule-validation-frames 600 --no-pairwise-features --iterative-induction`
- **读出/目标**：`--feature-value-weight 0.02 --value-quantile 0.8 --no-value-features --value-dispersion-labels --value-bank-only --value-trust-samples 150 --value-horizon 0 --contrast-objective --control-risk-weight 0 --predicate-readouts --subgraph-cells --no-event-memory-risk --no-compose-readout`
- **可信度**：`--ood-trust-scale 0 --obs-trust 1.0 --feature-action-prior 2.0 --feature-value-prior 4.0`
- **探索/闭环**：`--curiosity-weight 0.5 --voi-fraction 0 --voi-target 3 --voi-threat-min 0.25 --importance-weighting --memory-credit --dagger-rounds 0 --dagger-episodes 2 --dagger-cap 1200 --dagger-offset 50000 --dagger-horizon 0 --dagger-frames 0`
- **决策/评测**：`--decision {search,cellq} --step-stride 2 --stride-includes-lag --eval-seeds "" --eval-episodes 1 --no-learn-objective --report ""`
- **预设**：`--fast`（1500/3/1000/cap400/H8）、`--benchmark`（1500×1 + 5000 在线，cap5000）

---

## 3.7 `substrate_value.py`

| 符号 | 说明 |
|---|---|
| `InputNovelty(feature_dim, precision=6, max_patterns=100000)` | 观测计数式在线新颖度；`score()` / `observe()` |
| `WorldFeedbackValueHead(feature_dim, hidden_dim=32, learning_rate=0.001, discount=0.99, seed=0)` | 两层 tanh 价值头；`value/update/update_return/update_batch` |
| `TerminalRiskHead(feature_dim, learning_rate=0.05, l2=1e-4, ...)` | 逻辑回归风险头（含平方/交互项 + 先验截距校正） |
| `TransitionReplay(capacity=20000, ...)` | 类别平衡重放；`for_stream(count)` 保留每个 outcome 类的全部转移 |
| `ReturnReplay(capacity=20000, seed=0)` | MC 回报重放（支持优先级） |
| `substrate_features(model, events=None, novelty_score=0.0)` / `planned_substrate_features(model, sequences, novelty_model=None)` | 模型健康 / 规划特征编码 |
| `fit_value_head / fit_value_head_returns / fit_terminal_risk(head, replay, updates, batch_size=128, ...)` | 训练入口 |
| `discounted_returns(transitions, discount)` | MC 折扣回报 |
| `binary_auc(scores, labels)` / `value_probe_metrics(records, head, risk_head=None, gap=None)` | AUC 与探针指标（pain_auc / risk_auc / near_pipe / Brier） |

---

## 3.8 `substrate_generic.py`

| 符号 | 说明 |
|---|---|
| `LinearObservationModel(obs_dim, action_dim, ridge=1e-8)` | `fit(observations, histories, next_observations, holdout=0.1)`；`predict(...)`；记录 `train_error/holdout_error`（确定性尾部 holdout） |
| `history_features(sequences, steps, prefix, action_dim=ACTION_DIM)` | 动作历史特征（**新动作在前**，早期步用 prefix 填充） |
| `collect(seed, steps=PRACTICE, coverage=0.0, lag=LAG, guide=None, budget=0, episode_start=0, propensities=None)` | 真实转移采集；`guide` 反事实覆盖**叠加**在采样器上；`propensities` 记录每步发出的动作概率 |
| `fit(transitions, seed)` / `policy_action(env, world, value_head, risk_head, prefix)` / `evaluate(seed, world, value_head, risk_head, cap=6000)` / `main()` | 训练/决策/评测/CLI |

常量：`LAG=3, H=14, PRACTICE=5000, ACTION_DIM=LAG+1, VALUE_UPDATES=3000, RISK_UPDATES=6000, TERMINAL_FRACTION=0.15, PRIORITY_ALPHA=1.0`。

---

## 3.9 `substrate_sandbox_search.py`

| 函数 | 说明 |
|---|---|
| `action_history(sequences, steps)` / `nearest_pipe(env)` / `observe_transition(...)` | 适配器工具 |
| `practice(seed, steps=PRACTICE, value_updates=..., ...)` | 采集 + 训练世界模型/价值头/风险头 |
| `target_gap(env, horizon=H)` | 按预判距离选管子的缺口中心（**任务基线专用**） |
| `rollout_candidates(...)` | 把 `predict_batch` / `feedback_features` / `risk_features` / `evaluate_step` / `evaluate_terminal` 接进 `trajectory_search` |
| `run(seed, value_mode="feedback", engine="exhaustive", horizon=H, ...)` / `report(rows, value_mode)` / `main(argv=None)` | 运行 / 汇总 / CLI |

CLI：`--task-baseline --engine {exhaustive,beam} --horizon --beam-width --jobs --seeds --practice-steps --value-updates --risk-updates --terminal-fraction --near-pipe-fraction --risk-interaction-dim --risk-action-dim --risk-aggregate {terminal,sum,max,crossing}`。

---

## 3.10 `graph_memory.py` / `mixed_nodes.py`

**`PrototypeGraphMemory(clusters=48, seed=0, rate=0.2, memory_weight=0.25)`**

| 方法 | 作用 |
|---|---|
| `fit(rows)` | kmeans 原型 → 概念节点；连续原型 → 转移边 + 边节点；episode → memory 节点；更新 pain/dopamine 值 |
| `prototype(observations)` / `_rebuild_tables()` | 最近原型 / 缓存评分表 |
| `_update_value(node_id, pain, dopamine)` / `_state_values(labels, previous_labels, attribute)` | 值更新 / 核心评分（概念值 + 边节点值 50/50 + memory 加权） |
| `risk(rows) / risk_labels / value_labels / risk_states / value_states / snapshot` | 读出与快照 |
| `kmeans(observations, clusters, seed=0, iterations=25)` | 模块级 k-means |

**`mixed_nodes.py`**

| 类 | 说明 |
|---|---|
| `PrimitiveGate(channel, direction, threshold, temperature)` | 单通道 sigmoid 阈值门；`activation()` / `describe()` |
| `MixedNode(first, second, valence, support, pain_rate, lift)` | 两个门的乘积（合取）节点 |
| `ContinuousMixedMemory(quantiles=(0.1,0.25,0.5,0.75,0.9), min_support=8.0, capacity=64, prior=2.0, forgetting_overlap=0.92)` | 覆盖驱动出生、疼痛仅定价、冗余遗忘；`fit/risk/describe` |
| `LayeredMixedMemory(common_capacity=48, rare_capacity=16, rare_max_support=160.0, merge_overlap=0.96, ...)` | 分层：常见节点覆盖驱动 + 稀有节点新颖性驱动，稀有受保护不被合并 |
| `MixedNodeRiskMemory(..., min_lift=0.02, min_pain=2.0, max_nodes=64, learning_rate=0.05)` | lift 筛候选 + 局部迭代疼痛价更新 |

风险评分统一为"基线率 + 活动合取节点的价偏差"，裁剪到 `[0,1]`。

---

## 3.11 规则层与自监视

`substrate_rules.py`：

| 符号 | 说明 |
|---|---|
| `Invariant`（dataclass，`predict(row)`）、`Determinant`（dataclass） | 不变量 / 决定性节点 |
| `RuleBook(tol=1e-9, min_group=5, min_coverage=0.8)` | `add_invariance/add_determinacy/observe/scan/holding/all_invariants_hold/exact_determinants/explain/roll(state, steps)` |

`live_core.py`：`ridge(H, Y)`、`eff_rank(H)`、`class Live(rng, gain=0.2, every=1, monitor=True, rule_specs=None, rule_thr=1e-6, rule_min=None, rule_budget=3)`（`step(tr, te, row)` / `measure` / `_act` / `rule_signal` / `score`）、`make(rng, ctxs, keep=0.6)`、`trial(seed, monitor)`。

---

## 3.12 环境与工具

`flappy.py`：`class Flappy(seed, max_t=None)`（`reset/_gap/obs/step/feedback`）；`run(policy_fn, seed, cap=None)`；策略 `zero_policy/scripted/scripted_lookahead(o, horizon=14)/linear_policy(w)`；`fitness(w, seeds)`；`es(gens, lam=16, sigma=0.4, seeds=(1,2,3))`；`main(argv=None)`。

`latent_world.py`：`class Latent(Env)`（`lag` 类属性、`pending()`、`obs()` 追加存活坐标）；`latent = {0..3}`；`recover(lag)`；`make_policy(r, lag, hor, delta=0.0)`；`class Decoder(ws, cs)` + `fit_decoder(acts, lag)`；`run(lag, seed, pol, source, dec=None, eps=0.0, flap=True, cap=None, rng=None)`；`class Schedule(lag, mag)`；常量 `CAP=6000, BETAS, SEEDS=range(100,120)`。

`predict_layer.py`：`class LayerPair(k_max=8, eta0=0.05, tau=2e4, learn=True)`（`predict/observe/lag_profile/identified`）。

`eval_protocol.py`：`parse_seed_spec(spec)`、`bootstrap_mean_ci(values, samples=2000, confidence=0.95, seed=0)`、`summarize(values)`、`group_values/group_summaries(rows, value="frames", group="train_seed")`、`cluster_summary(rows, value="frames", cluster="train_seed", samples=2000, seed=0)`、`paired_delta(left, right, key="train_seed", value="frames")`。
