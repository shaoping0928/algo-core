# Algo Core：节点信号基建与 Flappy Bird 实验

这是从主工作区 `algo_lab/` 中整理出的独立实验目录，聚焦离散节点基座、节点强度信号算子、规则/关系发现，以及在 Flappy Bird 延迟世界中的价值学习与策略搜索验证。源文件从主工作区复制而来；本目录可独立配置 Git，不包含其他工作区项目或临时实验。

## 当前状态与默认栈（2026-11-23）

当前默认栈（截至 2026-11-23，未变）：**点事件 pain 标签、帧级 MPC、NNLS 堆叠组合、好奇心 0.5 / 评测 0、成对特征关、Q 读关**。多分支强度是行为中性的实现重构（决策逐位不变），预测通道是诊断读出。以下机制均为 opt-in、默认关闭：前兆标签、成对特征、覆盖率门槛、记忆回溯、VOI、生成先验、动作边 Q、死亡规则硬约束、milestone/value 阶段项、DAgger。

当前基准（默认栈，训练 100–102 × held-out 120–124，3 个种子 × 5 个世界）：**191.8 / 178.4 / 88.4 帧，均值 152.9，中位数 152**。这是 2026-11-21 之后各 A/B 共用的 control 基线；2026-11-23 的收敛结论是默认栈保持不变（point-event、帧级 MPC、milestone/value opt-in）。

已完成/已关闭的实验、逐批验证与旧口径数字已按 UTC0 日期移入 `docs/archive/`；本文件只保留当前概览、默认栈、瓶颈与归档索引。`TODO.md` 继续记录当前待办与新方向。历史记录中 2026-11-20 之前（以及部分 11-20 当日）的数字多为前兆标签 `P(pain within k)` 口径，不能与点事件基准直接比较；详见 2026-11-20 归档。

## 当前瓶颈（2026-11-23）

- **决策尺度的分辨力**：动力学模型与短窗危险读数达标，但第一动作 gap 中位仅 0.008–0.026，4–33% 的决策 <1e-3；顶端并列大多共享第一动作，真正改写行为的是少数近并列决策。
- **危险读出没有覆盖死亡封闭的尺度（≥10 帧）**：8 帧处危险 AUC 已降到 0.55–0.59。显式前兆窗口可以把 h12 判别力做到 0.92–0.94，但作为逐帧罚项或端点否决都不超过默认栈。
- **阶段/图策略不能替代模型**：阶段级信号在 8 帧帧网格上无法被消费；单步转移图与图上值迭代因覆盖不足，不能替代可外推的世界模型。
- **seed 101 的 10 个 miss 已定位到 plan/suffix 排序**：B0 只读诊断（85 决策）显示 8/10 在选中/最佳首动作的各自最佳后缀下真实存活并列，只有 2/10 存在正的首动作对比；动作级 coarse router 不是当前主杠杆。详见 `TODO.md` 实验 B0。
- **plan-level miss 的 score 实际只有 risk 一列活跃**：B0.5 显示 8 个 plan-level miss 中 risk 差值为正且占 100%，value/curiosity/q 全为 0；去掉任何列都不能恢复 plan-level miss。B0.6 进一步显示 risk 的 step 部分全为 0、几乎全是 endpoint terminal hazard；预测 endpoint 把 selected 排得更安全（8/8），真实 terminal 在 7/8 判对。主因是 **world endpoint 漂移**，不是 risk 权重；下一步是 plan-level endpoint 校正/重排，而不是首动作 router。
- **B1-plan endpoint 读数：seed 101 有 ranking 信号，但不迁移**：predicted endpoint（仅 7 维）在 seed 101 把 same-action top1 从 61/69 提到 **68/69**、B0 miss 从 2/10 提到 **9/10**，same-AUC **0.789**；但 3 种子门失败：seed 100 从 **62/62 退化到 60/62**（worsened 2, improved 0），seed 102 的 48 个可用决策 **terminal base rate 0.000**、模型退化；calibration 在所有 seed 上失败（Brier skill −0.09…−0.36）。OOD 距离/fine score 作为特征会打掉 within-decision 信号。**不接线**；只剩 OOD/UNKNOWN 门后的 rank-only 复测或 world endpoint 改善。
- 8 帧帧级线（B0–B1-plan）已关闭；全局 span 读数排序可用但阈值不迁移、决策内不跨种子。当前机制重设计为**动作对齐的经验转移测度 + 多视窗查询**，而不是独立 span hazard：训练只用真实状态转移；预测时用候选动作序列递推 successor/feedback/terminal 分布，缺少支持则 UNKNOWN。第一版 fine frame kernel 的 h8/h16 支撑仅 0.055–0.064/0.034–0.048；折内重建表示后，嵌套 16→4→1 backoff 在 seed100/101 五折的 h8/h16 支撑约 0.79–0.83，fine-only 约 0.05–0.08，但 mixed-resolution top1 不能单独证明预测力，逐层跨种子审计是当前门槛。默认 planner 尚未接入、默认栈不变；下一步及实验数字见 `TODO.md`「P0：多视窗状态转移机制重设计」和 [2026-11-23 归档](docs/archive/README-2026-11-23.md)。
- **通用阶段记忆基座（2026-11-24，默认关闭、planner 未接线）**：目标从「更长的帧窗」改为通用决策近视基座——只用世界交互（观测、实际动作、反馈、终止）自监督地发现信号/阶段与非帧转移。已落地 `stage_outcome_memory.py`（变长阶段、变阶上下文回退、仅真实有向兄弟结果）、`stage_demand_heuristic.py`（需求只影响中间 frontier 排序，不改最终价值）、`stage_experience.py` 自动管线（真实 rows → episode → 信号 → 阶段 → pointwise demand 样本；`--stage-memory` 默认关，特征源可选稀疏 `branch` 或稠密 `prototype`）与 `stage_state_resolver.py`（runtime prototype → 实际 stage key，UNKNOWN 带原因）。fast smoke（seed 100、1500 帧）：38/38 局、367 stages（9.66/局）、duration 中位 4（q90 6）、keyed 0.90、1257 demand samples；但**内部 feature 根覆盖率仅 0.03（44/1500 行）**，demand 关联当前不能当主信号。resolver 的 episode-held-out 审计（fold-refit 表示、5 折、被动 5000 帧/128 局）：**coverage 0.468、supported accuracy 0.740、同子集常量 key 0.435 → +0.305**（5/5 折同向）；UNKNOWN 主因是 OOD 半径门（1400），其次支撑/歧义。预测等价阶段树（`predictive_stage_tree.py`）已落地：叶为真实 stage key，父节点只在 validation episodes 的 joint 似然不劣于分开模型时合并；3 种子 held-out 审计里阶段档案本身比全阶段 base 好 1.5–2.6 nats（top1 0.57–0.60 vs 0.37–0.38），但 **merge 回退只在低支撑门（hi10）显形（joint Δ +0.02/+0.14/+0.08，3/3 正）**，默认门几乎不触发。内部需求选择器（`stage_demand_selector.py`）也用真实阶段反馈训练：折外 memory + 岭回归预测“之后的净反馈和”，在支持的 sibling 中选非终局目标；同时修正了 stage outcome 统计取错阶段（源→目标）的问题。需求关联审计（`stage_demand_association_probe.py`）显示：稀疏 feature-root 覆盖仅 0.3%；稠密 runtime prototype 在 exact context 下 top1 低于 chance；**stage-tree root 池化 + 因果 action prefix** 在 3/3 seeds 把 top1 做到 0.84–0.86（chance 0.82–0.84、majority 0.64–0.70），但 coverage 仅 0.18–0.24（放宽支撑 0.41）。在线 provider（`stage_online.py`）已接通并通过固定预算 gate 首轮：默认支撑下 rank hint 基本空转（0–2/3078 候选有证据，frames 不变）；放宽到 1/1 根级后 hint 真正激活（1608/2322 supported），但 **frames 25.0 → 18.3**（seed 100、3 worlds、beam 4）；**5 worlds / beam 16 复核为 73.8 → 32.2（−56%）**；target probability ≥0.6 / support ≥4 的置信门仍有 72/161 决策产出 demand，frames 仍为 32.2；近并列 `rank_mode=tie`（margin 0.0/0.05）仍为 32.4/32.2。质量门未通过，**不接线**；下一步转向可验证 value target / 完整 frontier recall，而不是继续调 priority。非帧转移的决策侧第一层已落地：`stage_option_search.py` 完整枚举阶段级 option 路径（无 beam/top-k；每条边 outcome 概率和校验为 1；terminal/censored 终止；深度截断记 `open`；无支持节点记 `unresolved` 且不重归一化；拒绝 root 池化编造跨源阶段边）。停时不再是均值：`StageOutcomeMemory` 为每个 outcome 报告真实 duration 直方图（`duration.distribution`），`StageOptionSearch` 用离散卷积合成路径 elapsed，并提供 `distribution_quantile` / `survival_curve`（P(T > t)）；缺失分布直接抛错，卷积支持集超限抛错而不静默截断。seed 100、depth 2 探针：elapsed mean **8.0 帧**、q90 **14.6 帧**、**P(elapsed ≤ 8) = 0.703**（online 40 决策：7.8 / 14.35 / 0.712），卷积支持集最大 18 个时间点。→ depth-2 阶段计划约 **30%** 的时长质量超出 8 帧视界。seed 100、depth 2 探针：71 个 option、1059 条支持边；离线 118 次搜索 terminal 可达 **106/118**、单次最多 166 条路径；在线 40 个决策 terminal 可达 **38/40**；每个阶段上下文平均 **8.8–9.9** 个 option 受支持（词表 71）；2 次阶段转移期望 elapsed ≈ **8 帧**（单阶段中位 4）。option 等价类（`predictive_option_quotient.py`）也已落地：同一源阶段下按后继/时长/反馈的可预测等价合并 trace，并在 held-out episodes 上接受/拒绝整个商；`StageOptionSearch(option_quotient=...)` 可切到 class 模式。seed 100、depth 2：Gaussian 反馈模型下 86 traces → 63–65 classes，但 tolerance 0.0–0.05 全部被拒（Δ −0.126…−0.235），通道分解显示拒绝几乎全部来自反馈（−0.128），categorical 反而 +0.015；**改为 per-dimension quantile histogram 后门通过**——tolerance 0.0 → 86→67 classes、Δ +0.015；tolerance 0.05 → **86→59 classes（−31%）、Δ +0.024**。class 模式首次跑通（71 次验证搜索）：leaves **102→205/搜索**、plans **83→181**、terminal 可达 **62→64**、unresolved **25→0**、within-8 **0.720→0.571**；覆盖变好但分支翻倍、计划变长。仍只做离线搜索、未接 planner。阶段定义扩充：**静态扩展是内生的、非时间性的状态转移** `s → E(s)`（用已学特征/节点关联整合当前状态），与外延的时间转移 `(s,a) → s'` 相反。`static_transition.py` 只沿 association/membership/derivation 边做闭包，**绝不跟随 transition 边**；`stage_change_bundles.py` 消费 `E(s)` 序列做共变 bundle、边界与区间。seed 100、1500 步：仅 seed 时 7.51 节点/状态、76 个 bundle（29 个 size-3）、4.95 边界/局；depth 1 闭包后 11.31 节点/状态（+50.6%）、122 个 bundle（44 个 size-3）、7.13 边界/局；depth 2/3 与 depth 1 完全相同 → 当前图上**一跳饱和**，且边计数只有 derivation（association/membership 未参与）。二值 prototype 快照源的旁证是 57 个 size-2 prototype 交接（3.6 边界/局）。→ 机制成立、整合后 bundle 更丰富，但当前图的非时间结构偏薄。连续强度（`static_graph_strengths.py`，只读）：从共激活学 association Jaccard 强度（209 条、均值 0.24）与 63 条 membership 结构绑定；原始强度会**过度整合**（d2 覆盖 24.6/30 节点、bundles 反降到 87），**度归一化**后节点/状态 12.6→13.0、闭包到 d2–d3，bundles **128**、size-3 **50**（graph d1 为 122/44）→ 可用变体是 normalized。阶段类型（`stage_type_clusters.py`）：对 normalized `E(s)` 做关键节点加权余弦 complete-linkage 聚类，类型数不预设，并在 held-out episodes 上以 coverage / margin / 预测增益过门。seed 100、1500 步、38 局（295 train / 71 validation）：所有相似度阈值都过门；**0.70 为全覆盖 + 最高增益的平衡点**（43 types、coverage 1.000、gain **+0.232**），0.90 增益更高（+0.314）但 coverage 降到 0.732，0.95 仅 0.521。这是 1 seed 的方向性结果，类型数偏多（43/295），下一步需要层级合并/更多样本控制碎片化。分层（C，`stage_type_layers.py`）：对 `E(s)` 建 complete-linkage 合并树，按 held-out 前沿选择；**所有设置都过门，tolerance 控制层数**——0.05 → 36 层、coverage 1.0、gain +0.235；0.10 → 17 层、1.0、+0.202；0.20 → 3 层、+0.111。门只能验证“可达且有用”，层数仍需跨 seed/折稳定性或下游目标决定。跨折稳定性（`stage_layer_stability.py`，5 折 × 3 seeds）：层数在折间从 **2 到 137** 剧烈变化、agreement 仅 **0.20–0.40**，无稳定候选（mean gain 虽仍为 0.13–0.39）→ **层数不可识别，C 不接线**；需要更多 episode 或更强约束。B 的决策侧（`stage_type_value.py`）：类型值作为**软终点项**加入 terminal cost，同预算 gate（beam 4、3 worlds、cap 400）下 weight 0/0.1/0.5 的 frames 完全一致（**25.0**）；单次决策内终点候选 penalty 的 distinct 平均仅 **1.29/4**，预测终态几乎都塌到同一 tree root/type → 终点项无杠杆，判定 `stage_type_endpoint_value_inert`，不接线。Stage-level value + executor（`stage_executor.py`，仅探针）：阶段边界选 option 并执行其动作 trace，非边界回退帧级 planner；同预算（3 worlds、cap 400、horizon 8）baseline **101.3** → full trace **23.7** / one action **33.3** / class mode **34.7**，passes 全 0。选项在 stage memory 里 terminal mass=0，但执行期间仍会死 → 阶段级安全 ≠ 帧级安全，判定 `stage_executor_unsafe_without_frame_safety`，不接线；下一步需要帧级安全包络。视野诊断（`lookahead_diagnosis.py`）：下一根柱子在 **~80 帧**外（spacing/speed），world model 的 next_dy MAE 在 1/8/32/64 帧为 **0.001/0.009/0.046/0.101**（gap 的 13%/30%）；阶段类型对 next_dx/next_dy 的 eta² = **0.672/0.885**（raw R²≈1）→ 信息没丢，但 8 帧 planner 与 depth-2 阶段搜索（≈8 帧）只覆盖柱间距的 ~10%，exhaustive 深度 20 需 9^20 不可行。判定 `lookahead_insufficient`：分层没有延长视野，只降低了决策频率。默认决策路径仍逐位不变。

### 事件/阶段层后续（Batch 8 follow-up，2026-11-23 晚）

- **P0-F 尸检（`h6_autopsy_probe.py`，3 种子）**：沿 h6 真实评测轨迹
  75 决策/种子、全候选 oracle、h8 同状态对照。三种子一致：**窗内排序
  基本没错**（in-window misrank 仅 1/3/3，只有 near-tie/challenge；
  临终 15 决策 window-blind 0）；**失败是死亡进窗太晚**——63–66% 的
  采样决策处于动作对称的死亡走廊（两动作都在 ≤6 步内死），h8 的逃逸
  全部落在 h6 的 post-window 状态（12/12、15/15、12/13）。判定
  `h6_horizon_visibility_not_ranking`：短视界是视野问题，不是候选
  排序问题；帧级打分无法修复。
- **事件视窗跨度（`event_span_probe.py`，3 种子）**：12–16 帧 span
  hazard（状态+事件历史）迁移到 planner 流 AUC 0.70–0.91（2/3 种子
  0.85–0.91，近死带 0.78–0.82）；但**离散先导事件不是 8–12 帧
  触发器**（首次 lead ≥8 仅覆盖 6–40% 死亡；规则 h12 precision
  0.21–0.48）；被动 tau 迁移 precision 0.23–0.38。→ 信号在连续
  readout 里，触发形态错了。
- **E1 粗层 span 状态树（`tree_transition_probe.py`，3 种子）**：
  两级 state-only 表在 planner 消费分布内 AUC 0.80–0.96、
  precision@tau 0.39–0.51（首次接近 0.5）；事件历史作为表格维度
  3/3 变差（K=32 切碎支撑），兄弟池化在当前规模惰性；被动→planner
  迁移不稳。
- **E2 消费分布读数（`span_consume_probe.py`，3 种子）**：h6 DAgger
  训练 + 折外校准，在 h6 评测流（120–124）上 span12 3/3 未达门；
  span16 的 state-only 表在 5 世界口径下可用（AUC 0.82–0.88、
  lift 2.0–2.4×、召回 0.76–0.94、precision 0.34–0.46）；被动训练
  退化为基率检测器；support 可作 abstention（AUC 0.75–0.92）。
- **E2.1 部署配置流（`span_consume_probe.py --deploy-*`，3 种子）**：
  `curiosity off` + 同 horizon/decision 再采 5000 帧。部署流局长
  **重现评测流的 72 帧地板**（DAgger ~81）→ **采集配置位移实锤**。
  deploy 增益 2/3 种子成立（span12 precision +0.21/+0.20；span16
  0.475–0.611），101 反向（table 对 50 局过拟合；参数化臂在 5%
  coverage 下 precision 0.667）。**世界数量/数据量不是瓶颈**：被动
  128 局、DAgger 62 局，每局都是新环境实例；评测 120–124 只是同
  分布另 5 个抽样。触发级仍未稳定达标（span12 全种子 <0.5 或
  coverage <0.05；span16 2/3 达 0.475–0.611）。
- **口径修正（2026-11-24，E2.3）**：以上 E2/E2.1 数字均为 **5 世界**
  （120–124）估计；同一读数在 20 世界口径下移动约 0.1（101
  0.581→0.463、104 0.433→0.531）。稳健的是**排序**（20 世界固定
  top5/top10 告警预算 6/6）与定性机制（部署流重现 72 帧地板、
  被动训练退化为基率）；**阈值/触发级**已被 6 种子否决
  （E2.2 3/6、E2.4 决策内不跨种子）。
- **E2.2 形态融合（`span_consume_probe.py`，6 种子完成，确认失败）**：
  部署流上 `0.5·rank(table_deploy) + 0.5·rank(logistic_deploy)`，
  span16 在 100/101/102 过 3 种子门（precision 0.590/0.581/0.508），
  但 6 种子只有 **3/6 ≥0.5**（104 0.433、105 0.477、103 0.210）。
  固定融合被更强的一臂拖下水（104 的 logistic 0.682 → 0.433；
  105 的 0.600/0.591 → 0.477）；每种子最优臂也只有 4/6，102/103
  的读数本身不够。101 部署流加到 12k 帧后 table16 仍 0.250、
  fuse16 0.541 → 瓶颈是形态/种子，不是数据量。
- **E2.4 决策内诊断（`span_decision_probe.py`，6 种子完成）**：把
  span 表读数打到 B0 候选端点上，按 h8 真值算 within-decision AUC。
  deploy8 只在 100（0.790，n=2）/101（0.713，n=11）为正，
  103（0.387）/104（0.390）/105（0.347）反向；dagger8 只在 100 为正；
  102 没有混合决策（同动作候选结果全同）。**没有稳定消费者，E3 关闭。**
- **当前读法（span/事件线，2026-11-23 收口）**：全局状态排序可用
  （20 世界固定 top10 告警预算 6/6），但**阈值不跨流、决策内不跨种子**；
  事件/阶段跨度读数保持只读诊断，不接线。默认栈与 8 步 MPC 不变，
  6 步 executor 继续关闭。

## 当前结构

```text
flappy.py                 Flappy Bird 环境，并输出 dopamine / pain 反馈通道
live_core.py              节点、稀疏边、状态与动作循环
latent_world.py           延迟 Flappy 世界和测量辅助函数
recall_world.py           节点/稀疏边网络的延迟回忆实验
substrate_rules.py        不变量、决定性与规则层
substrate_search.py       基函数、滞后列和确定性搜索
predict_layer.py          预测层 / 真实层；残差驱动的在线更新
trajectory_search.py      通用批量 rollout、可行性筛选与词典序方案选择
substrate_sandbox_search.py Flappy 适配器：学习世界反馈；可选 gap 基线
substrate_value.py        从世界反馈通道学习 dopamine / pain 价值与终止风险
substrate_generic.py      任务无关原型：完整观测世界模型 + 通用搜索
trace_feedback.py         追踪管道附近的动作/缺口方向与风险预测
trace_generic_training.py 抽样追踪世界模型、价值和风险训练波次
trace_signal_world.py     对照 raw / 差分 / 缓冲观测信号的世界预测
trace_sensitivity.py      扰动各输入维度，检查世界/价值/风险是否真的使用它
mixed_nodes.py            通用原始 gate + 混合节点 + 连续记忆/遗忘原型
trace_mixed_nodes.py      比较全局风险头与混合节点风险记忆
math_memory_search.py     算术事实记忆 + 关系搜索实验
raw_math_experiment.py    nanoGPT 原始等式数据生成与实验
node_graph.py             统一 Node/Edge 图：概念、转移边节点、记忆、派生
graph_memory.py           观测原型 -> 概念/转移边节点/事件记忆的原型图记忆
trace_graph_memory.py     在真实 Flappy 转移上评估图记忆风险
substrate_graph_search.py 用学习世界模型 + 图记忆风险做搜索的原型
substrate_nodes.py        节点中心原型：在线节点价值 + 沙盒轨迹搜索
node_runtime.py           节点运行时：动作条件边、MC 信用分配、事件记忆
memory_credit_probe.py    记忆回溯修正的机制探针（修正量/召回率）
feature_generalization_probe.py 特征根的 held-out 激活率与阈值脆弱性诊断
context_model.py          通用潜在上下文线性动力学（残差基 + 在线推断）
memory_tree.py            有界扇出分层记忆：合并/分裂/关键词概括/展开
variable_registry.py      变量属性与关系：support/credibility/learning_rate
hierarchical_memory.py    帧→事件→阶段→世界→根的分层记忆
stage_outcome_memory.py   通用阶段实例、变阶上下文与真实有向结果统计（纯数据层）
stage_experience.py       自监督经验管线：真实 rows → 信号 → 阶段 → demand 样本
stage_demand_heuristic.py 需求导向 feature 排序：只改中间 frontier，默认关闭
stage_state_resolver.py   连续状态→历史阶段类型（prototype 关联 + UNKNOWN 门）
stage_resolver_probe.py    episode-held-out resolver 审计（fold-refit 表示）
predictive_stage_tree.py  预测等价阶段树：held-out 似然门决定 merge/回退
stage_tree_probe.py        episode-held-out 阶段树审计（tree/leaf/base 对照）
stage_online.py            在线阶段 context + option-pooled 需求 provider
stage_option_search.py     非帧 option 搜索：完整枚举阶段级路径，无剪枝
stage_option_search_probe.py 阶段级路径/可达性/未解析质量诊断
predictive_option_quotient.py option 等价类：按后继/时长/反馈等价合并 trace
option_quotient_probe.py  option 等价类的 held-out 门与 trace/class 对照
stage_change_bundles.py    共变 bundle：特征节点强度协同变化的阶段边界
stage_change_bundle_probe.py 共变 bundle 的支撑/边界/区间诊断
static_transition.py       内生静态转移：非时间边的状态整合闭包
static_graph_strengths.py  从共激活学习 association/membership 连续强度（只读）
stage_type_clusters.py     相似度阶段类型：加权余弦聚类 + held-out 门
stage_type_layers.py       相似度分层：complete-linkage 树 + held-out 前沿选择
stage_type_cluster_probe.py 阶段类型的覆盖/间隔/预测增益诊断
stage_type_layer_probe.py  阶段类型分层的层数/增益/覆盖诊断
static_transition_probe.py 静态转移的深度/整合规模/bundle 丰富度诊断
stage_rank_gate_probe.py  固定预算 rank-only 质量门（baseline vs hint）
stage_demand_selector.py   真实反馈 value readout + 支持 sibling 的需求选择
stage_demand_association_probe.py feature→next-stage 关联的 episode-held-out 审计
multiview_transitions.py  动作对齐经验转移核 + 任意窗口查询（planner 未接入）
generative_expansion.py    关键词提示 → 节点策略 + 上下文动力学 rollout
branches.py                共享图上的多角色稀疏强度：current/recent/prediction/value
generated_prior_probe.py   生成先验 vs 在线 vs 零先验的存活配对
dynamics_context_probe.py 跨世界动力学变化的通用上下文探针
eval_protocol.py          held-out 种子解析、中位数/IQR、聚类 bootstrap 区间
signal_bank.py            每节点信号源：差分、积分/泄漏、逆泄漏、均值
substrate_*.py            各算子、记忆、噪声、延迟和搜索的验证脚本
tests/                    unittest 回归测试
docs/substrate.md         A.1–A.28 机制与实验记录
docs/archive/            README 历史实验记录（按 UTC0 日期）
```

## 快速运行

需要 Python 3 和 NumPy。本目录的唯一第三方运行依赖列在 `requirements.txt` 中：

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

快速检查（使用 Python 自带的 unittest）：

```bash
python -m unittest discover -s tests
```

然后在本目录中运行：

```bash
# 世界反馈价值默认模式；按种子并行
python substrate_sandbox_search.py --jobs 8

# 任务 gap 评分基线，仅用于对照
python substrate_sandbox_search.py --task-baseline --jobs 8

# 可选：长时域 beam 搜索对照
python substrate_sandbox_search.py --engine beam --jobs 8

# 任务无关原型：只通过 obs/step/feedback 交互
python substrate_generic.py --seed 100

# 抽样查看训练波次
python trace_generic_training.py --seed 100 --coverage 0.1

# 比较 raw / 差分 / 缓冲世界模型
python trace_signal_world.py --seed 100 --coverage 0.1

# 检查各输入维度的重要度
python trace_sensitivity.py --seed 100 --coverage 0.1

# 比较混合节点风险记忆
python trace_mixed_nodes.py --seed 100 --coverage 0.1

# 评估原型图记忆
python trace_graph_memory.py --seed 100 --coverage 0.1

# 图记忆风险驱动的搜索原型（按种子并行；完整 cap=6000 成本很高）
python substrate_graph_search.py --seed 100 --seeds 4 --jobs 4 --coverage 0.1 --cap 1200

# 追踪一个种子的管道附近动作/风险
python trace_feedback.py 100

python substrate_selfdelay.py
python substrate_signals.py
python substrate_signals_noise.py
python recall_world.py 200 600 0
python math_memory_search.py
```

脚本是独立的实验/验证器；它们不是一个已集成的生产运行时，也不会自动训练出通用策略。

并行执行说明（保留自原 README）：

脚本会检测 Python 是否启用 GIL：free-threaded 构建使用 `ThreadPoolExecutor`，普通构建使用
`ProcessPoolExecutor`。当前 `.venv` 是 GIL 构建；本机的 `python3.15t` 存在，但 free-threaded
numpy wheel 的 `_multiarray_umath` DLL 被应用控制策略阻止，暂时无法在该解释器加载 numpy。

## 价值与策略搜索

目标是让系统从世界提供的反馈中学习价值，而不是在搜索器里写任务专用的评分公式。当前实现把这一责任拆成三层：

1. **世界反馈**：`flappy.py` 每步输出 `dopamine` / `pain` 两个通道。它们只是环境输入，代码不根据缺口中心或管道位置推导价值。
2. **价值学习**：`substrate_value.py` 的 `WorldFeedbackValueHead` 从真实交互的 Monte-Carlo 回报学习 dopamine/pain 两路未来值；`TerminalRiskHead` 从同一 `pain` 标签学习终止风险。风险特征和标签都取 **next state**；真实转移进入 replay，终止样本按比例采样，采样后的先验偏移在推理时做截距校正。反事实 rollout 不参与价值或风险训练。
3. **方案选择**：`trajectory_search.py` 提供任务无关的 exhaustive 和 beam 搜索，支持多目标词典序比较。Flappy 适配器默认先最小化预测疼痛/终止风险，再用 pain 回报和 dopamine 排序；`--task-baseline` 才切换回 gap-center 评分与碰撞可行性过滤。

默认搜索是 `exhaustive H=14`；`--engine beam` 是搜索方式，不改变评分目标。beam 在默认的
`terminal` 聚合下没有中间风险信号，长时域对照应显式加 `--risk-aggregate max` 或 `sum`，
并保持两个引擎使用同一聚合方式。

这不是“价值函数天然等于存活”：反馈通道的语义仍由世界提供。当前实现也没有把新颖度直接当作奖励，它只是价值模型的一个输入特征。

## 节点运行时：只给信号，不标注风险

`substrate_nodes.py` / `node_runtime.py` 是节点中心路径。人类只规定
`obs`、`action`、`dopamine/pain`；`done` 只作为 episode 边界，不作为
风险标签。这条路径不读取 `GAP/2`、`target_gap`、`BIRD_X`、碰撞窗口
或 near-pipe 区域，也没有 `TerminalRiskHead`。

该路径的标签口径、机制三栏表、当前测量与收敛链已分别移入 [2026-11-20](docs/archive/README-2026-11-20.md) 与 [2026-11-21](docs/archive/README-2026-11-21.md) 归档；当前默认口径见上文。

## 历史归档 / Archive index

已完成/已关闭的实验记录按 UTC0 日期存放；每个文件保留原 README 的章节标题与正文，按原相对顺序排列。日期取章节标题中的显式日期；只有月份或没有日期的章节取引入该章节的 commit 日期（UTC0）。跨日期的整节按规则保留在最早日期并注明跨度。归档保留原文的相对位置，因此正文中的「见下文 / 见上节 / 文末」等交叉引用可能指向相邻日期的文件；请用本表定位。

| 日期 (UTC) | 归档文件 | 内容 |
|---|---|---|
| 2026-11-17 | [README-2026-11-17.md](docs/archive/README-2026-11-17.md) | 早期测量概况（跨 11-17–11-19）与加法/乘法原始输入实验 |
| 2026-11-18 | [README-2026-11-18.md](docs/archive/README-2026-11-18.md) | 任务无关世界模型原型及其测量 |
| 2026-11-19 | [README-2026-11-19.md](docs/archive/README-2026-11-19.md) | held-out/DAgger 协议、泛化 A/B、20 世界复验、cell-Q/价值目标、三态组件、运行时追踪、E1/E2 与旧特征层 |
| 2026-11-20 | [README-2026-11-20.md](docs/archive/README-2026-11-20.md) | 点事件标签切换、当前测量/收敛链、好奇心、记忆回溯、特征根、训练内省、基准协议、价值银行 |
| 2026-11-21 | [README-2026-11-21.md](docs/archive/README-2026-11-21.md) | 机制三栏表、上下文/分层记忆、生成式展开/先验、多分支、帧级失败追踪、0/1 死亡规则全链、已关闭实验、旧归档索引 |
| 2026-11-22 | [README-2026-11-22.md](docs/archive/README-2026-11-22.md) | 分层决策与表示层诊断：细层饱和、死亡视界、预警 lead、阶段单位、容量/相位否决、里程碑数据前提、阶段判据与决策质量诊断 |
| 2026-11-23 | [README-2026-11-23.md](docs/archive/README-2026-11-23.md) | 转移单位图策略、五种长期机制的统一失效面、长窗口/否决 A/B 与收敛结论（续 11-22） |

原 README 的「实验归档索引（2026-11-21 整理）」作为历史索引保存在 [2026-11-21 归档](docs/archive/README-2026-11-21.md) 末尾。

## 边界与纪律

1. 动力学系数和延迟可以从真实观测学习；任务目标（例如缺口中心）、候选动作表示和搜索预算仍是外部提供的。
2. 沙盒预测不是现实证据。必须与后续实际转移分开存储、比较和学习。
3. 合成、无噪声实验的机器精度结果不代表真实感知输入下的性能。
4. 本目录的结论以 `docs/substrate.md` 和各验证脚本为准；负结果及其适用条件也保留。

主工作区和本目录是两份文件副本。后续修改应明确选择维护哪一份，再同步需要的变更。
