# 02 · 主要模块职责

按角色分组。路径均相对 `algo_core/`。

## 2.1 速查表

| 分组 | 文件 | 一句话职责 | 类型 |
|---|---|---|---|
| 环境 | `flappy.py` | 无延迟 Flappy 世界；输出 `dopamine/pain` 反馈通道 | 库 + CLI |
| 环境 | `latent_world.py` | 延迟 Flappy（`lag=0..3`）+ 测量工具（`recover/make_policy/Decoder/Schedule`） | 库 |
| 图基底 | `node_graph.py` | 统一 Node/Edge 图：概念/转移边节点/记忆/派生 + 四类边 + 动作边 + 序列化 | 库 |
| 运行时 | `node_runtime.py` | 节点中心运行时：在线出生、动作边、TD(λ)/MC 信用分配、事件记忆、特征格子、组合读出 | 库（最大） |
| 特征 | `node_features.py` | 任务无关的组合特征规则归纳（阈值谓词 + AND/OR）与激活 | 库 |
| 组件 | `node_components.py` | 运行时数值组件（差分/电容/电阻）+ 三态符号概念节点 | 库 |
| 信号 | `signal_bank.py` | 每节点输入轴算子 `id/d1/d2/leak/unleak/移动平均` | 库 |
| 记忆 | `graph_memory.py` | 观测原型 → 概念/转移边节点/事件记忆的**原型图记忆** | 库 |
| 记忆 | `mixed_nodes.py` | 合取门 + 混合疼痛记忆（`ContinuousMixedMemory`/`LayeredMixedMemory`/`MixedNodeRiskMemory`） | 库 |
| 搜索 | `trajectory_search.py` | 任务无关 exhaustive / beam 滚动时域搜索（多目标字典序） | 库 |
| 搜索 | `substrate_generic.py` | 任务无关原型：`LinearObservationModel` + 通用 MPC（对照路径） | 库 + CLI |
| 搜索 | `substrate_nodes.py` | **节点中心规划主线**：训练/决策/评测/CLI（约 50+ 旗标） | 库 + CLI |
| 搜索 | `substrate_sandbox_search.py` | Flappy 适配器：世界反馈价值模式 + `--task-baseline` 对照 | 库 + CLI |
| 搜索 | `substrate_graph_search.py` | 图记忆风险驱动的搜索原型（按 seed 进程并行） | 库 + CLI |
| 价值 | `substrate_value.py` | `WorldFeedbackValueHead` / `TerminalRiskHead` / 重放缓冲 / AUC 与探针指标 | 库 |
| 价值 | `predict_layer.py` | 预测层-真实层 `LayerPair`：误差是唯一教师，在线学延迟与幅度 | 库 |
| 规则 | `substrate_rules.py` | 规则层：不变量节点 `Invariant`、决定性节点 `Determinant`、`RuleBook` | 库 |
| 规则 | `rule_loop.py` | 把规则层作为第六信号/第六动作并入 substrate 循环 | 脚本 |
| 规则 | `rule_mainline.py` | 在 substrate 轨迹上验证规则层（恢复不变量/拒绝控制变量/验证决定性） | 脚本 |
| 规则 | `rule_policy.py` | 从恢复的规则**推导策略**（系数自动掉出来，无手工设定） | 脚本 |
| 规则 | `rule_trigger.py` | 标定规则层触发参数（首次扫描行数、残差阈值、扫描预算）的分辨率与代价 | 脚本 |
| 自监视 | `live_core.py` | 把监视器做成 substrate 成员：任务循环只调 `step()` | 库 + 脚本 |
| 回忆 | `recall_world.py` | 延迟回忆任务：比较多条记忆臂的容量 `d_max` | 脚本 |
| 数学 | `math_memory_search.py` | 算术事实记忆 + 通用关系搜索（`AssociativeMemory` + `greedy_determine`） | 库 + CLI |
| 数学 | `raw_math_experiment.py` | 为 nanoGPT 生成原始算术语料（无中间步骤）并逐位评估 | CLI |
| 评估 | `eval_protocol.py` | held-out 种子解析、中位数/IQR、聚类 bootstrap CI、paired delta | 库 |
| 探针 | `memory_credit_probe.py` | 记忆回溯修正的机制探针（修正量/召回率/风险率） | CLI |
| 探针 | `feature_generalization_probe.py` | 特征根 held-out 激活率与阈值脆弱性诊断 | CLI |
| 探针 | `training_introspection_probe.py` | 挂在 `fit` 真实路径的只读包装器，多时间点内省学习动力学 | CLI |
| 诊断 | `trace_*.py`（10 个） | 训练/决策/灵敏度/信号/记忆/规则等多角度追踪（见 2.8） | 脚本 |
| 验证 | `substrate_*.py`（约 30 个） | 逐机制的最小验证脚本（见 2.9） | 脚本 |
| 测试 | `tests/*.py`（12 个） | unittest 回归（见 [06 测试与回归](./06-tests.md)） | 测试 |
| 文档 | `README.md` / `TODO.md` / `docs/substrate.md` | 结构 / 会话与实验记录 / 基座机制规格 | 文档 |

## 2.2 环境层

- [flappy.py](file:///c:/workspace/code/algo_core/flappy.py)：世界本体。观测 `[y-0.5, vy*10, px-BIRD_X, gap-y, next_dx, next_dy]`；动作是二元 flap；奖励每帧 +1。物理常量从 `FL_*` 环境变量读取（默认即基准值），**默认物理改成 latent 基准值以消除 import 顺序地雷**。`step()` 末尾设置 `_feedback = {dopamine, pain}`，`feedback()` 暴露它。附带 `zero_policy` / `scripted` / `scripted_lookahead` / `linear_policy` 参考策略与 ES 搜索 `es`。
- [latent_world.py](file:///c:/workspace/code/algo_core/latent_world.py)：把 flap 延迟 `lag` 步执行（内部 `deque` 队列）。`Latent` 额外暴露**累计存活 dopamine** 作为第 7 个观测坐标。`latent = {lag: Latent0..Latent3}`。工具：`recover(lag)` 用 `greedy_determine` 从数据恢复转移系数；`make_policy` 滚 H 步比较两动作；`Decoder`/`fit_decoder` 从 `u` bank 线性解码滞后动作；`run` 用 `source ∈ {exact,none,decoded,noisy}` 触发不同信息源；`Schedule` 用"到期后果"替代移位寄存器。

## 2.3 图基底与运行时（核心）

- [node_graph.py](file:///c:/workspace/code/algo_core/node_graph.py)：把节点/边语义落到数据层。节点字段含 `activation/trace/value/dopamine/count/eligibility_*/immediate_*/role/component_id/sign/predicate/pain_horizon`。边四类；只有转移边可拥有 `edge_node`；关联边不给边节点。提供 `activate/observe_association/observe_transition/observe_memory/propagate/recall/decay/prune`、`_remove_node`（级联清理）与 `to_state/from_state` 全量持久化。
- [node_runtime.py](file:///c:/workspace/code/algo_core/node_runtime.py)：**仓库最大、最核心的模块**。承载：在线概念出生、动作条件边、TD(λ) 资格迹与 MC 反向信用分配、事件记忆生命周期、`(feature, acting)` 痛/价值格子、特征规则物化与卸载、分级威胁读出、NNLS 堆叠组合、isotonic 校准、好奇心势差、OOD 信任收缩、组件绑定、决策（`act`/`policy_result` 的读出侧）。细节见 [03 关键类与函数](./03-core-api.md#noderuntime)。
- [node_features.py](file:///c:/workspace/code/algo_core/node_features.py)：`Primitive`（单坐标阈值测试，支持 `|x|` 幅值形式）与 `Composite`（AND/OR 组合）。`FeatureRuleBank.induce()` 从训练状态+pain 归纳规则并在 held-out 上复核，支持覆盖率门槛与成对坐标差扩展。`expand_observation()` 追加全部 `x_i - x_j` 差分坐标（纯代数，无语义）。
- [node_components.py](file:///c:/workspace/code/algo_core/node_components.py)：`RuntimeComponent` 只做数值计算（差分/电容/电阻），**不进图**；`ComponentBank` 把它的三态符号（+1/0/-1）表示为普通 `CONCEPT` 节点（`role=component_state`），参与关联/转移/记忆/valence 与持久化；`classify()` 带滞回防抖。
- [signal_bank.py](file:///c:/workspace/code/algo_core/signal_bank.py)：`bank(X, betas, means)` 对每个输入列生成 `id/d1/d2/leakβ/unleakβ/ma_w` 算子输出——**列内运算，不混节点**。`unleak` 是 `leak` 的精确逆；`noise_gain(beta)` 给出反演对白噪声的放大倍数。

## 2.4 搜索与决策层

- [trajectory_search.py](file:///c:/workspace/code/algo_core/trajectory_search.py)：搜索核心只知"初始状态 + 动作空间 + 批量一步转移预测 + 批量启发式评估"。`exhaustive_search` 枚举 `2^H` 并支持多目标字典序；`beam_search` 是长时域的近似。返回 `SearchResult`（含 `feasible/candidate_count/first_state/final_state`）。无可行解时返回最低分候选并标记 `feasible=False`。
- [substrate_generic.py](file:///c:/workspace/code/algo_core/substrate_generic.py)：任务无关原型。`LinearObservationModel` = 岭回归 `(obs, action-history) → next_obs`；`collect()` 只按 `obs/action/feedback` 采集（支持 `guide` 反事实覆盖与 `propensities` 记录）；`fit/policy_action/evaluate/main` 串起完整 MPC。
- [substrate_nodes.py](file:///c:/workspace/code/algo_core/substrate_nodes.py)：**主线入口**。`_train`（离线腿+DAgger 轮）、`fit`、`policy_result`（危险+价值+好奇心目标）、`cell_q_action`（单步读格子消融）、`decision_action`、`plan_score`、`evaluate(record=)`、`run_seed`、`main`（进程/线程池并行 + `--report`）。预设 `--fast` / `--benchmark`。
- [substrate_sandbox_search.py](file:///c:/workspace/code/algo_core/substrate_sandbox_search.py)：Flappy 适配器。`practice()` 采集并训练世界模型+价值头+风险头；`rollout_candidates()` 把线性模型、价值头、风险头接进 `trajectory_search`；`target_gap()` 按实际 horizon 选管（任务基线）。CLI 含 `--task-baseline/--engine/--risk-aggregate/--near-pipe-fraction/--risk-interaction-dim/--risk-action-dim` 等。
- [substrate_graph_search.py](file:///c:/workspace/code/algo_core/substrate_graph_search.py)：把 `graph_memory` 的概念/边/记忆价值并入搜索评分；`--seeds/--jobs` 进程级并行。

## 2.5 价值层

- [substrate_value.py](file:///c:/workspace/code/algo_core/substrate_value.py)：`InputNovelty`（观测计数式新颖度）、`WorldFeedbackValueHead`（两层 tanh，学 dopamine/pain 两路未来值）、`TerminalRiskHead`（逻辑回归 + 平方/交互项 + 先验截距校正）、`TransitionReplay`（按类别平衡采样，`for_stream()` 保留全部转移）、`ReturnReplay`（MC 回报 + 优先级）、`substrate_features/planned_substrate_features`（模型健康与规划特征）、`binary_auc`、`value_probe_metrics`、`fit_value_head*`、`fit_terminal_risk`、`discounted_returns`。
- [predict_layer.py](file:///c:/workspace/code/algo_core/predict_layer.py)：`LayerPair`——`d_hat = b + Σ h_k·e(t-k)`，误差在线训练 `b` 和 `h`；`lag_profile()` / `identified()` 用于识别真实延迟。

## 2.6 规则层（独立实验线）

- [substrate_rules.py](file:///c:/workspace/code/algo_core/substrate_rules.py)：`Invariant`（不变量节点，`predict(row)`）、`Determinant`（决定性节点）、`RuleBook(tol, min_group, min_coverage)`，方法 `add_invariance/add_determinacy/observe/scan/holding/exact_determinants/explain/roll`。核心思想：**规则是检验而不是投影**——不变量要么在机器精度成立要么不成立，成立时诱导的依赖可外推。
- [rule_loop.py](file:///c:/workspace/code/algo_core/rule_loop.py)：把规则层作为 substrate 的"第六信号（残差）+ 第六动作（scan）"并入 `live_core.Live` 循环。
- [rule_mainline.py](file:///c:/workspace/code/algo_core/rule_mainline.py)：在 substrate 自身轨迹上恢复不变量（`dy/dv`）、拒绝控制变量、验证决定性规则。
- [rule_policy.py](file:///c:/workspace/code/algo_core/rule_policy.py)：从恢复的仿射规则滚 H 步导出线性阈值策略——**系数是从数据里掉出来的，不是手设的**。
- [rule_trigger.py](file:///c:/workspace/code/algo_core/rule_trigger.py)：测量"多少行后真不变量成立/假候选是否永不成立"与"触发设置的扫描代价"，用于标定超参。

## 2.7 自监视、回忆与数学

- [live_core.py](file:///c:/workspace/code/algo_core/live_core.py)：`Live` 把监视器做成 substrate 成员，任务循环只调用 `step(tr, te, row)`；监视器按 `rank/held-gap/marg` 三个阈值决定 `gain/route/prune`。`ridge` / `eff_rank` 是内部工具。
- [recall_world.py](file:///c:/workspace/code/algo_core/recall_world.py)：延迟回忆任务（线索出现一次→静默 d 步→报告符号），对照 `u` 痕迹、每边残留 `res`、联想记忆 `M`、类型化节点、节点分裂等多条记忆臂，报告精度 vs 延迟。
- [math_memory_search.py](file:///c:/workspace/code/algo_core/math_memory_search.py)：`AssociativeMemory` 精确键值记忆 + `substrate_search.greedy_determine` 通用关系搜索；恢复 `a+b`/`a*b` 并在无记忆命中的多位数上逐题预测。
- [raw_math_experiment.py](file:///c:/workspace/code/algo_core/raw_math_experiment.py)：为 nanoGPT 生成"只有题目和最终答案"的原始算术语料（`prepare`），并提供逐位精确匹配评估（`eval`）。

## 2.8 诊断层（`trace_*.py`）

| 文件 | 追踪什么 |
|---|---|
| `trace_node_runtime.py` | 部署路径拆 8 节：世界模型误差 / 节点价值 AUC / OOD 与风险激活 / 动作边双动作支撑 / 前兆 AUC / 滞后槽位 / 方向准确率 / 预测校准与 surprise 缓冲 |
| `trace_node_training.py` | 训练过程：概念诞生、边增长、双动作支撑、值百分位、信用分配样本 |
| `trace_memory_change.py` | 记忆节点学习前后变化（created/updated/forgotten、value/trace/members） |
| `trace_generic_training.py` | 任务无关原型的训练波次（覆盖、世界误差、价值/风险 AUC/Brier） |
| `trace_signal_world.py` | raw / 差分 / 缓冲 / 全部信号对世界预测的离线消融 |
| `trace_sensitivity.py` | 世界/价值/风险模型的逐维置换灵敏度 |
| `trace_mixed_nodes.py` | 全局风险头 vs 各混合节点变体的 AUC/Brier/置换重要度 |
| `trace_graph_memory.py` | 原型图记忆在 held-out 真实转移上的风险与灵敏度 |
| `trace_feedback.py` | 管道附近动作/缺口方向与风险预测（方向准确率） |
| `trace_rule_induction.py` | 从转移中归纳单谓词与合取对并做 held-out 验证（独立于 `node_features`） |

三个 `*_probe.py` 是更聚焦的机制探针：`memory_credit_probe.py`（记忆回溯修正是否真的被读出消费）、`feature_generalization_probe.py`（逐根的 train/行为/held-out 激活率与瓶颈合取项）、`training_introspection_probe.py`（挂在真实 `fit` 路径的只读快照，量化每阶段结构/读出/预测账本）。

## 2.9 验证层（`substrate_*.py`）

每个脚本验证一个最小机制假设并记录结论，构成自底向上的证据链：

- **算子/信号**：`substrate_signals.py`、`substrate_signals_mean.py`、`substrate_signals_noise.py`、`substrate_capacitance.py`、`substrate_capacitor.py`、`substrate_timescale.py`、`substrate_noise_adapt.py`、`substrate_noise_kind.py`、`substrate_modes.py`。
- **延迟/动力学**：`substrate_delay.py`、`substrate_latency.py`、`substrate_margin.py`、`substrate_layers.py`、`substrate_depth.py`、`substrate_events.py`、`substrate_predict.py`、`substrate_selfdelay.py`、`substrate_rollout_check.py`。
- **状态/规则**：`substrate_state.py`、`substrate_rules.py`、`substrate_constants.py`、`substrate_conjunction.py`、`substrate_recognition.py`、`substrate_bigram.py`、`substrate_edges.py`、`substrate_basis.py`、`substrate_search.py`。
- **端到端**：`substrate_generic.py`、`substrate_sandbox_search.py`、`substrate_graph_search.py`、`substrate_nodes.py`、`substrate_value.py`。

## 2.10 文档层

- `README.md`：结构与运行入口，含"机制三栏表（实现/训练/被控制器读取）"、基准协议、当前测量与瓶颈清单。
- `TODO.md`：会话状态 + 四子代理解剖附录（A 信息流 / B 收敛数学 / C 常数审计 / D 文献方法）+ 死机制清单。
- `docs/substrate.md`：基座规格（0–15 节：原语、执行顺序、学习规则、时间尺度、节点分裂…）+ 附录 A.1–A.28（规则层实验记录）。

## 2.11 新增实验线的模块清单（详见 [07 新增实验线](./07-experiment-lines.md)）

仓库在 01–06 章定稿后继续演进，新增了下列模块。它们**不改变默认训练/决策路径**：要么是纯数据层/库，要么是只读探针。

| 线 | 文件 | 一句话职责 |
|---|---|---|
| 阶段/相位 | `stage_graph.py` | 阶段图 schema + 离线动作残差分割器（纯数据层） |
| 阶段/相位 | `milestone_graph.py` | 任务无关的相位复位分段（慢变量跳变高分量 + NMS）+ 里程碑级转移记忆（分位桶身份、时长分布） |
| 阶段/相位 | `stage_support_probe.py` / `stage_sweep_probe.py` / `stage_content_probe.py` | 可行性 / 惩罚扫描 / 边界内容审计 |
| 阶段/相位 | `phase_stage_probe.py` / `state_partition_probe.py` | 相位相对 vs 绝对表示；阶段作为**状态空间类** |
| 死亡规则 | `rule_expressions.py` | 0/1 谓词可组合表达式树 |
| 死亡规则 | `death_rules.py` / `death_rule_loop.py` | AQ 式 0/1 死亡规则集 + 失败驱动精化循环 |
| 死亡规则 | `contextual_rules.py` / `pattern_library.py` | 上下文分桶规则 / 跨世界维度签名模式库 |
| 死亡规则 | `death_rule_probe.py` / `death_failure_probe.py` / `death_feature_probe.py` / `death_pattern_probe.py` / `death_transition_quality.py` | 离线验证 / 失败选择 / 表示对比 / 跨世界复用 / 误差源分离 |
| 上下文/动力学 | `context_model.py` | 任务无关潜在上下文线性动力学（共享映射 + 低秩上下文 + 变化检测） |
| 上下文/动力学 | `dynamics_context_probe.py` / `generated_prior_probe.py` / `context_benchmark.py` | 动力学切换三臂 / 生成先验三臂 / 种子配对基准 |
| 分支/变量 | `branches.py` / `variable_registry.py` | 角色特异强度分支 / 带类型变量与关系注册表 |
| 记忆/生成 | `memory_tree.py` / `hierarchical_memory.py` / `generative_expansion.py` | 合并树 / frames→events→phases→worlds 分层记忆 / 从摘要生成近似流 |
| 学习轨迹 | `learning_trace.py` | 帧级学习轨迹探针（逐决策记录全部学习面） |
| 动作/时程/选择 | `action_effect.py` | 任务无关动作效应统计库 |
| 动作/时程/选择 | `action_conditioning_probe.py` / `action_conditioning_diagnostics.py` | 动作预测力 LOEO + 排列检验 / 其异常定位 |
| 动作/时程/选择 | `causal_horizon_probe.py` / `doom_horizon_probe.py` / `oracle_selection_probe.py` | 因果视界 / 不可避免性 / 视界 vs 排序 |
| 动作/时程/选择 | `coarse_search_probe.py` / `outcome_coarse_probe.py` | 宏动作长程粗搜索 / 结果桶粗状态 |
| 动作/时程/选择 | `score_column_probe.py` / `risk_contrast_probe.py` / `metric_probe.py` / `n_tie_probe.py` | 分数列消融 / 风险列决策影响 / 几何邻域假设 / 好奇势平局 |
| 动作/时程/选择 | `event_lead_probe.py` / `warning_lead_probe.py` / `planner_rollout_quality.py` | 事件领先 / 通道警告领先 / 计划级死亡侧质量（含可复用工具） |
| 工具 | `probe_cache.py` / `wiring_audit.py` | 探针模型缓存 / 机制接线审计 |
