# TODO

> **维护约定（2026-11-21 整理）**：已完成/已关闭的实验记录归档到
> `README.md`（见文末「实验归档索引」）；本文件只保留**当前状态、
> 未完成事项、新方向设计**。更早的审计与解剖材料保留在文末
> 「历史附录（只读）」，不再维护。

## 方向校正（2026-11-23，用户）

- **8 步帧级组合是默认栈的基线诊断，不是设计轴。** 它的 endpoint drift
  是预期现象；不要用 B0.6/B1-plan 的帧级结论判断或调优事件/阶段级模型。
- **目标架构**：fine executor 缩到 **6 步**；粗层是**事件/阶段级 outcome
  transition**；跨度是停时分布，不是固定 6/8 帧；粗层只做方向/约束/
  早期预警，fine-first。
- **只依赖帧不能解决近视**：6 步只会让事件/阶段层更必要；不能把
  terminal 判断继续压在帧级 endpoint 上。
- **B0/B0.5/B0.6/B1-plan 的作用**：解释当前 frame-level 默认栈为何在
  seed 101 失败；**不构成对事件/阶段级设计的否定**。事件/阶段级模型
  尚未按新口径测试。
- **P0-E（帧级 OOD-gated rank-only）删除，不做**；8 帧调试线
  （B0–B1-plan）就此关闭，不再加帧级探针或帧级补丁。

## 最近更新（2026-11-23 晚）

- **B0（`729aa24`）**：seed 101、85 决策、10 个 relevant miss；
  6 个首动作不同、4 个同动作；**8/10 是 plan/suffix 问题**，
  只有 2/10 有真实首动作对比。首动作 router 只有 2/10 理论空间，
  近并列可修 0/10。
- **B0.5（`0d3fe73`）**：8 个 plan-level miss 上只有 **risk 一列活跃**
  （8/8 非零、positive share 1.00）；value/curiosity/q 全为 0；
  没有任何单列或非退化列组合能恢复 plan-level miss（0/8）。
- **B0.6（`0d3fe73`）**：risk 的 step 部分全为 0，risk 列几乎全是
  **endpoint terminal hazard**；预测 terminal 在 8/8 把 selected 排得
  更安全，真实 terminal 在 7/8 判对；state error 0.111。
  判定 `step_readout_flat_endpoint_drift`。
- **B1-plan（`bc9bf17`）**：predicted endpoint 的 leave-one-world-out
  读数 same-action top1 **68/69**、same-AUC **0.789**、B0 miss **9/10**
  （fine 61/69、2/10；shuffle placebo 65/69、6/10）。
  **ranking gate 通过，calibration gate 失败**（Brier skill −0.092、
  within-Brier skill −0.793）；OOD/fine 作为特征会打掉 within-decision
  信号。未接线。
- **B1-plan-3seed（100/101/102，已完成）**：
  - seed 101：A1 68/69、B0 miss 9/10、improved 7 / worsened 1；
  - seed 100：A0 fine 62/62，A1 60/62（improved 0 / worsened 2），
    flip precision 0.00；
  - seed 102：候选 terminal base rate **0.000**，48 个决策全是
    非终止标签；模型退化，无法评估；
  - calibration 在所有 seed 上失败；候选 endpoint OOD 率
    101/100/102 = 28.7% / 68.9% / 63.0%。
  → **3 种子非退化门失败**：seed 101 的增益不迁移，seed 100 退化，
  seed 102 退化。不接线；该线已关闭（见「方向校正」）。
- **Batch 8 事件/阶段 outcome audit（被动流 3 种子，已完成）**：
  6 个事件级 level 候选跨种子稳定通过五测（terminal、x0_sign、
  x1_high、x0_high、x5_low、x3_low）；动作回声类（impulse/apex）
  与状态协变量类被五测分开；**阶段/世界周期边界在被动流里 0-1 次/
  5000 帧，support 不可学**。详见 Batch 8 结果。
- **P1-B 事件级预警（被动流 3 种子，全部 pass）**：4 个稳定先导
  （x1_high/x0_high/x5_low/x3_low，lift 2.4–3.8，中位 lead 3–5 帧）；
  事件历史特征使 terminal hazard ΔAUC **+0.026/+0.082/+0.113**
  （≥+0.02，placebo 过线 +0.03~+0.11）；next-event top-1 baseline
  **0.46–0.48**（≈2.8× chance）。事件层带转移信息，不只是可读标记。
- **P1-C 阶段层（planner 流 3 种子，已完成）**：过柱重置
  **7 / 22 / 0 次**每 ~5000 帧（100/101/102）；seed 101 首次达
  support → 五测全过但**动作对比 ≈0 → boundary**（外生边界，
  阶段协变量，不是决策层）；seed 102 的流里连重置都没有。
- **P0-F 6 步 executor 基线（benchmark 协议，已完成）**：h6 vs h8 配对
  （15 个世界）：seed 100 **−119.8**（全部塌到 72 帧地板，
  拍翅率不变但时机错）、101 **−68.6**、102 **+15.8**；
  合并 h6 ≈95.3 vs h8 152.9（**−57.6**，1/3 种子改善）。
  → **短视界不能单独切换**：6 步 executor 必须由事件层补偿，
  这正是 Batch 8 的动机，也量化了“只依赖帧的近视”代价。
- **P1-D.0 触发验证（已完成，3/3 fail）**：事件预警模型在行为流上
  拟合后，tau 在 planner 评测分布上失准（precision 0.50 → 0.12–0.49）；
  改在 planner/DAgger 流上拟合后 AUC 反而下降（0.81→0.61）。
  **行为→planner 分布位移是 P1-D 的瓶颈，不是事件层本身**。
  → P1-D 暂停，不再做触发式 deepening；
  **6 步 executor 在事件层找到可靠的消费者之前保持关闭**。
- **P0-F 尸检（`h6_autopsy_probe.py`，已完成）**：沿 h6 真实评测轨迹
  采样 75 个决策点/种子，打开搜索取全部候选 + 真实 oracle，h8 同状态
  对照。三种子一致：
  - **窗内排序基本没错**：in-window misrank 仅 1/3/3 每 75（只有
    near-tie/challenge，无 large-gap）；临终 15 个决策 window-blind 0
    （分数看得见窗内死亡）。
  - **失败是"死亡进窗太晚"**：死亡走廊（`in_window_doomed` 27–29 + 
    `post_window_death` 20 每 75，≈63–66%）且**动作对称**——两动作都在
    ≤6 步内死，可比决策每种子仅 1–4 个；走廊里任何排序都无济于事。
  - **h8 的逃逸全部落在 h6 的 `post_window_death` 状态**（12/12、
    15/15、12/13）：多出的 2 步恰好覆盖"死亡在窗后 1–2 帧"的最后
    逃生窗口；同状态上 h8 也大多无解（both-die 35–38/75）→ h8 赢在
    **更早的分岔点不进走廊**，不是在走廊里更会逃。
  - **risk 量级校准过弱**：rem 4–6（死亡已在窗内）时 selected risk
    仅 0.16–0.95（seed 102 几乎平坦 0.16），rem ≤3 才跳变；但即使校准
    正确，走廊内也无候选可救——**窗外信息源是唯一出路**。
  - → 判定 `h6_horizon_visibility_not_ranking`：h6 的失败是视野不足
    导致的早期承诺错误，不是候选预测/排序错误；帧级打分无法修复，
    与 P1-D 结论闭环（事件层预警有信号、缺可靠消费者）。
- **事件视窗跨度（`event_span_probe.py`，已完成，3 种子）**：
  12–16 帧 span hazard（状态+事件历史）迁移到 planner 流 AUC
  **0.70–0.91**（2/3 种子 0.85–0.91，近死带 0.78–0.82）；事件历史
  在 101 把 state-only 从 chance（0.534）拉到 0.701（Δ+0.167）。
  但离散先导事件首次 lead ≥ 8 只覆盖 6–40% 的死亡、规则 h=12
  precision 0.21–0.48 → **触发形态错了，信号在连续 readout 里**；
  被动 tau 迁移 precision 0.23–0.38 仍失败（校准需目标分布重做）。
- **E1 粗层 span 状态树（`tree_transition_probe.py`，已完成，3 种子）**：
  两级树（fine k-means + coarse 父节点）+ 每节点停止时间直方图 +
  trust 收缩。三种子：
  - **决策分布内（planner 5 折）树对象立得住**：state+history AUC
    0.888/0.757/0.902、band 0.735/0.616/0.797；**state-only 更强
    0.958/0.797/0.941**、band 0.866/0.660/0.853；
    precision@tau 0.435/0.323/0.532（state+hist）→ 校准在消费分布
    上第一次接近/达到 0.5。
  - **被动→planner 迁移不稳**：state+hist 0.692/0.732/0.619 vs
    logistic 0.885/0.701/0.911；state-only 0.876/0.668/0.653——
    分段常数的覆盖对被动聚类落点敏感（102 崩）。
  - **事件历史作为表格维度是毒药**：3/3 within-distribution 都是
    state-only 更好；历史信息真实（logistic 增量 +0.11~+0.17），但
    K=32 下 15 维流相对特征把节点切碎。→ 正确形态是
    **state-only 非参数树 + 参数化低维历史修正**。
  - **兄弟池化当前规模惰性**：tree ≈ fine-only（0.888/0.888、
    0.757/0.753、0.902/0.904）；应在低支撑/跨流池化/更深层级才显值。
  → 预注册门 1/3 pass，但表对象在决策分布内成立；下一步 E2：
    DAgger 插桩 (预测, 实现) 配对 + state 树 + 参数化历史修正，
    在 h6 消费分布上验校准。
- **E2 消费分布 span 读数（`span_consume_probe.py`，已完成，3 种子）**：
  h6 DAgger 训练 + 训练流折外校准，在 h6 评测流（held-out 世界
  120–124）上读：
  - **span 12 全线 fail**：table_dagger precision 0.25/0.33/0.25
    （目标 0.5）、band 0.60/0.65/0.51；param 臂跨世界不稳
    （0.46–0.83）。
  - **span 16 的 state-only 表可用但未校准**：AUC **0.84/0.88/0.82**、
    band 0.67/0.75/0.59、precision 0.46/0.35/0.34（lift 2.0/2.4/2.2×）、
    coverage 0.46/0.36/0.35、recall 0.94/0.85/0.76。
  - **消费分布训练在 span 16 明确加分**：被动训练 precision
    0.23/0.17/0.16、coverage 0.99/0.89/0.98 → 退化为基率检测器。
  - **支撑度是可靠弃权信号**（support AUC 0.75/0.83/0.92）；
    distance 较弱（0.50/0.30/0.44）。
  - **剩余瓶颈不是世界数量/数据量**：每局都是新环境实例（被动 128、
    DAgger 62 个 gap 序列），评测 120–124 只是同分布另 5 个抽样。
    真正的位移是**策略配置**：DAgger 采集时 curiosity=0.5 开着，
    评测是 `_curiosity_off`——「消费分布」并未真正对齐部署配置。
    次要因素：评测仅 5 局、span16 约 100 正样本（precision CI ±0.09）。
  → 判定 `span16_usable_not_calibrated`：跨世界读数可用（lift ~2×、
  召回 0.76–0.94），校准差 0.04–0.16；支撑度可作 abstention 门。
  → 下一步不是多世界，是**部署配置再采一条流**（curiosity off）。

> **范围说明**：以上 B0–B1-plan 全部是 **frame-level 默认栈** 的诊断
> （8 步帧级 rollout / 帧级 endpoint risk）。它们回答“为什么不能继续
> 调 8 步帧级 score”，**不是**对事件/阶段级 outcome transition 的测试，
> 也不能用来否定 Batch 8。

## 里程碑 / 阶段 / 长程：2026-11-23 总结

> 以下是当日较早的总结；最新 B0–B1-plan-3seed 结论见上方
> 「最近更新」。默认栈与基准未变。

**一句话**：预测侧（动力学、短窗危险、长窗危险、静态联想、状态级价值、
结果分布身份）**全部达标**；失效集中在**候选之间的分辨力**——第一动作 gap
中位 0.008–0.026，4–33% 的决策 <1e-3，任何 0.01–0.1 的信号都会改写赢家。
默认栈保持帧级 MPC + point-event 标签，其余全部 opt-in。

### 已判定（按证据）

1. **动力学**：h=1 R² 0.99、h=8 训练 0.95 / held-out 0.59–0.70；RMSE 始终
   低于 persistence。
2. **短窗危险**：h=1 AUC 0.84；h=8 只有 0.55–0.59，因为默认标签窗口是
   `lag+1=4`。
3. **长窗危险可以做出**：precursor 标签、窗口 12 → h8 0.958 / h12 0.920
   （默认 0.586/0.560），isotonic 不再压平。但作为逐帧罚项 0/3（75.7）、
   作为端点否决仍低于默认（121.7 vs 152.9）。**AUC ≠ 决策质量。**
4. **静态联想**：prototype 通道 h8 0.925（训练）/0.79（held-out）；feature
   中；memory 恒 0.5（结构性死亡）。
5. **状态级价值**：关联价值（后继质量按共现计数加权）组内秩相关
   +0.34~+0.62、AUC 0.18–0.47；计划级选对率 0.82/0.86（vs 0.73/0.79）；
   局级 1/3。
6. **结果分布身份**：阶段按“每帧反馈 + 是否以死亡结束”聚类 → 坏段基率
   0.37–0.69（原 0.00）、转移 top-1 0.53–0.76 vs 机会 0.30–0.50；但每局
   约 1 条转移、option 单元仅 2 个。
7. **图策略不能替代模型**：单步选边 40/25/59、图上值迭代 25/27/56
   （帧级 192/178/88）；原因是覆盖——图外状态映射回已访问节点，世界模型
   才能外推。
8. **两层结构已实现但不达标**：`track`（相位不变全观测目标）均值 153.7 vs
   152.9（1/3）；位置目标（identity 投影）0/3。
9. **门控与方向性 tie-break 否决**：`margin_gate` 0/3（1e-2 时 78.6），因为
   90% 的动作是“不拍翅”，保持上一步等于持续下坠。
10. **决策诊断**：候选 cost 跨度 0.3–0.6，但顶端 margin 中位 1e-3、第一动作
    gap 中位 0.008–0.026；`oracle@8 = policy@8`（给定成本选择最优）。

### 当前默认（未改）

`decision=search`（8 帧穷举）、point-event pain（`precursor_labels=False`）、
`pain_horizon=lag+1`、`hazard_readout="calibrated"`、`value_readout="cells"`、
`milestone_mode="off"`、`margin_eps=0`、`veto_fraction=0`。
基准：1500 步 / epochs 1 / max_nodes 1000 / cap 5000 / horizon 8 / eval
120–124；3 种子 191.8 / 178.4 / 88.4，均值 152.9。

### 未决 / 下一步

- [x] **P0 表示层：把时序放进状态**（2026-11-23 已判定）。状态加入
      “距上次拍翅的帧数 / 相位变化率”：
      - full-state 首批负向被维度控制混淆（常数额外维也 312→192）；
      - conditioned world 的 episode-held-out gate 判零
        （validation gain **-0.314**，residual_scale 0），行为与
        baseline 逐位相同 **312/6 → 312/6**；
      - phase period 12/14/14 vs ~80，当前流不可辨识。
      → **中性/null，不采纳、不跑 3 种子**。per-target、age-only
      残差也已判零（无目标通过；candidate gap/regret 不变）。
      详见下方「P0 表示层：time-in-state」。
- [x] **P0-B 查询驱动图重写 / 最小规则库**（Batch 6，2026-11-23）：
      探针、schema、每目标稀疏规则、共享 applicability、事件谓词、
      关键节点读数、stage/interval 目标、组件 free-run、查询图世界模型
      已落地；剩余 planner/NodeRuntime 接线与决策侧质量门。默认栈不变。
- [x] **P0-C 候选分辨力（Batch 7）诊断阶段**（2026-11-23）：
      B0/B0.5/B0.6 已完成，判定 seed 101 的主瓶颈是
      **plan-level endpoint drift**，不是首动作排序、列权重或覆盖；
      B1-plan seed 101 ranking 通过、calibration 失败；
      3seed 已完成并判定非退化失败。详见下方 Batch 7。
- [x] **P0-D B1-plan-3seed / 校准**（2026-11-23）：seed 100/101/102
      已完成；seed 101 有 ranking 增益但 calibration 失败；seed 100
      从 62/62 退化到 60/62（worsened 2, improved 0）；seed 102
      terminal base rate 0.000，模型退化。**非退化门失败**，不接线。
- [x] **帧级 8 帧调试线（B0–B1-plan）关闭**（2026-11-23）：只保留两条
      基线证据（B0 决策分类、8 步组合终点不可靠）；不再加帧级探针、
      不做帧级 endpoint re-ranker（原 P0-E 删除）。
- [x] **P0-F 6 步 executor 基线 A/B**（2026-11-23）：h6 合并
      **−57.6 帧**（100 塌到 72 帧地板、101 −68.6、102 +15.8）；
      拍翅率不变、时机错 → **6 步不能单独切换**，需事件层补偿。
- [x] **P1 事件/阶段结果转移审计**（2026-11-23，被动流 3 种子）：
      6 个事件级 level 候选跨种子稳定通过五测；动作回声/状态协变量
      被分开；阶段/世界周期边界在被动流 0–1 次/5000 帧，不可学。
      详见 Batch 8。
- [x] **P1-B 事件级转移 / 早期预警模型**（2026-11-23，3/3 pass）：
      4 个强先导（lift 2.4–3.8，lead 3–5 帧）；事件历史特征
      ΔAUC +0.026/+0.082/+0.113（placebo 过线）；next-event top-1
      ≈2.1–2.8× chance。事件层带转移信息。
- [x] **P1-C 阶段层的数据审计**（2026-11-23，planner 流）：
      过柱重置 7/22/0 次；seed 101 首次达 support → boundary
      （动作对比 ≈0，外生协变量）；102 连重置都没有。
- [x] **P1-D 6 步 executor + 事件级预警组合（P1-D.0 触发验证）**
      （2026-11-23，3/3 fail）：行为流 tau 在 planner 评测分布上
      precision 0.50→0.12–0.49；planner 流拟合 AUC 反降 0.81→0.61。
      **P1-D 暂停**；6 步 executor 不切，8 步保持默认。
- [ ] **P1 阶段数据规模**：结果分布身份已通过，但每局约 1 条转移；需要
      采集规模（~100 局/option）才能填 `(stage, option)` 单元。
      P0 time-in-state 已判零，不能靠 clock/phase 补这个规模。
- [ ] **P1 阶段数据规模**：结果分布身份已通过，但每局约 1 条转移；需要
      采集规模（~100 局/option）才能填 `(stage, option)` 单元。
      P0 time-in-state 已判零，不能靠 clock/phase 补这个规模。
- [ ] **P1 value 保持 opt-in**：状态级有效、计划级不差、局级 1/3；
      P0 表示层已判零；等 endpoint/world 读数修复后再重测。
- [ ] 可复现探针：`milestone_retest_probe.py`、`stage_diag_probe.py`、
      `stage_reach_probe.py`、`milestone_plan_ab.py`、`decision_diag_probe.py`、
      `decision_margin_probe.py`、`hazard_readout_ab.py`、`hazard_lane_ab.py`、
      `value_readout_probe.py`、`association_value_probe.py`、`value_lane_ab.py`、
      `value_harm_probe.py`、`stage_option_probe.py`、
      `transition_policy_probe.py`、`hierarchy_ab.py`、`hierarchy_track_ab.py`、
      `margin_gate_ab.py`、`pain_horizon_probe.py`、`precursor_ab.py`、
      `veto_ab.py`、`plan_level_probe.py`、`graph_rewrite_scale_probe.py`、
      `query_world_probe.py`、`event_predicate_probe.py`、
      `component_free_run_probe.py`、`decision_event_probe.py`、
      `time_in_state_probe.py`、`time_world_probe.py`、
      `time_wiring_probe.py`、`age_residual_probe.py`。


## Batch 6：查询驱动图重写与最小规则库（2026-11-23）

> **状态（2026-11-23）**：结构性部分已完成并保留作参考；默认栈未变。
> 剩余 planner/NodeRuntime 接线是独立后置项，不与当前 B1-plan 的
> plan-level endpoint 校正混在一起。

> 结构性部分已完成（2026-11-23）；默认栈不变，所有新机制 opt-in。目标
> 不是再加一层预测，而是把“转移的类型 / 适用条件 / 消费方式”从硬编码
> 改成**查询驱动 + 结构匹配**。剩余是 planner/NodeRuntime 的决策侧接入
> 与 3 种子质量门。

**一句话结论**：查询图能复现线性世界模型（最大差 3×10⁻⁸），但慢
18.7×；事件谓词应包现有 death rules；关键节点查询是 O(m) 且能补全，
但 horizontal 仍被过柱事件卡住。因此它是**结构载体**，不是默认预测器
或默认决策路径的替代品。

### 动机

- 预测侧已达标（动力学 h1 R² 0.99、短窗/长窗危险、静态联想、结果分布
  身份）；失效在候选分辨力与阶段/事件的单位。
- 阶段图第一轮：动作条件化真实（LOEO ΔR² 0.18–0.21），但同 K 下
  real ≈ shuffle、身份跨 episode 不共享、主断点常是世界时钟；
  **不能把 context 簇直接当概念图**。
- 图策略失败在覆盖（图外状态被最近原型别名），不是算术；**图应做评估，
  控制仍由世界模型/帧级搜索承担**。
- 因此下一步是“查询驱动图重写”：源图 + 目标模板 + 规则检索/应用 +
  读侧；不是给转移硬编码 type，也不是把阶段项塞进逐帧 cost。

### 量级实测（`graph_rewrite_scale_probe.py`，seed 100）

| 项 | 1500 帧 | 5000 帧 |
|---|---:|---:|
| 每目标 fan-in | 2.0 | 2.3 |
| 动作进入的目标 | 垂直 0/1/3/5 | 同 |
| 动作槽选择 | `history[3]` | 同 |
| 全局 held-out R² | 1.000 | 0.998 |
| k=16 规则数 | 112 | 112 |
| k=16 共享搜索 | 16 | 16 |
| k=16 独立搜索 | 2.7×10⁸ | 2.7×10⁸ |
| 关键节点 m=2 / m=3 独立 | ~256 / ~4.1k | 同 |
| 近第二 regime 行冲突率 | 4.1% | 7.9% |
| 原型图出度 p50 / p90 | 1 / 2 | 1 / 2 |

读法：

1. **帧级动力学近似 regime 不变**：每目标一条稀疏规则已 held-out
   R² ≈ 1.0；按 2/4/8/16 个 regime 拆开没有增益（k 越大还略降）。
   → 默认不建 `k × 7` 规则库。
2. **规则库本身线性增长**；爆炸来自“每目标独立选 regime”：共享是
   O(k)，独立是 O(k^m)。k=16、m=7 时是 2.7×10⁸，不可接受。
3. **动作只进入垂直目标**，水平/时间目标不读动作；动作槽自动选中
   `history[3]` → 动作窗口 + 稀疏 scope 可以发现延迟，不必硬编码 lag。
4. **关键节点 m=2–3 时独立搜索也只有几百到几千**；全查 7 个才到
   1×10⁸。查询只填读侧需要的节点。
5. **小数据下 scope 不可辨识**：1500 帧水平目标选中累计时间而不是
   水平坐标；5000 帧才选中水平坐标。晋升规则前必须有跨 episode /
   跨数据量的稳定性门。

### 最小规则库（默认形态）

- **每目标一条稀疏规则**，不是每目标每 regime 一条。
- `Rule`：`pattern_in` / `pattern_out` / `dt` / `rule` / `support` /
  `confidence` / `applicability`。
- **共享 applicability**：一个当前 regime/context 节点，搜索 O(k) 而
  不是 O(k^m)；只有某个目标证明按 regime 拆分有 held-out 增益时，
  才分裂那一条规则。
- **事件谓词**：碰撞/过柱/终止是读侧规则，读组合状态、写
  event/terminal；不计算状态，不进入帧 cost。
- **关键节点查询**：读侧声明 `required_slots`，只填需要的槽位，其余
  `UNKNOWN`；m=2–3 足以覆盖近期决策，不要求补全整个状态。
- **不硬编码 type**：type 由 `pattern_out + dt + 查询目标` 涌现；
  代码层只声明图/时间代数、统一器、未知传播、预算。
- **查询驱动流程**：源图 → 目标模板 → 反向检索 → 正向应用 → 读侧；
  目标图是锚点，解决“类型/适用性循环定义”。

### 派生概念节点：训练期标注，推理期转移

- 训练/观测：代码层 `difference` / `capacitor` / `resistor` 作为
  **labeler**，产生 sign 标签；component sign 节点仍是普通 `CONCEPT`
  节点。
- 推理/预测：**不调用 `ComponentBank.step`**；概念节点由学到的
  `DerivationRule` 填；无规则/冲突 → `UNKNOWN`，不能当安全。
- 先做 `difference`（无隐藏状态）：如 `sign(dy)` from `vy` / `y`
  history；再做 `capacitor` / `resistor`（隐藏 leaky 状态）——需要
  latent / 窗口 / observer 三选一。
- **当前前提**：`_component_step` 传的是节点激活 `{node_id: 1.0}`，
  不是原始数值；要做“`vy > 0 → dy > 0`”这种数值概念，先对齐 base
  变量为 value-bearing 槽，或把规则直接建在原始观测上。
- 验证必须区分 **teacher-forced** 与 **free-running**；下游转移要用
  scheduled sampling / 自由运行验证，避免概念分布漂移。

### 探针扩展（已实现）

- 在 `graph_rewrite_scale_probe.py` 增加“实际搜索”而不只算组合上界：
  - 共享 applicability vs 每目标独立；
  - 全查 7 个目标 vs 关键节点 m=1..3；
  - 预算内实际访问节点数、被 support/confidence 剪掉的规则数、
    `UNKNOWN` 槽位数、规则冲突数。
- 读侧指标：帧预测 per-variable 误差/R²；事件谓词 0/1/UNKNOWN 的
  precision/recall/coverage；派生概念自由运行准确率/UNKNOWN
  （先 difference）。
- 数据：seed 100、1500/5000 帧先跑；稳定后再 3 种子。

### 本轮进度（2026-11-23）

已落地（默认关/只读探针，不改默认栈）：

- `query_graph.py`：`Slot` / `Pattern` / `DtSpec` / `Applicability` /
  `Rule` / `Readout` / `Query` / `RuleBank`。精确查询只接受精确 `dt`
  的规则；变长 `dt` 规则结构上填不进 `+1` 槽位。无输入、无规则、
  同置信度冲突均返回 `UNKNOWN`。
- `tests/test_query_graph.py`（19 条）：精确/变长 `dt`、窗口目标、
  共享门、独立门、关键节点、递归解析（含 3 步）、冲突、事件谓词、
  readout。
- `sparse_rules.py`：每目标一条稀疏规则（BIC 前向选择 + raw OLS），
  动作窗口只作候选；`fit_event_rule` 单槽阈值谓词（双向、balanced
  accuracy、support 门）。
- `tests/test_sparse_rules.py`（6 条）：稀疏 scope 恢复、规则查询、
  事件高/低侧、纯标签 abstain。
- `component_derivation.py`：`ThresholdSignRule` / `DifferenceSignRule` /
  `ComponentDerivationBank`；缺输入/缺规则 → `UNKNOWN`。
- `tests/test_component_derivation.py`（6 条）。
- `event_rules.py`：把现有 `DeathRuleSet` 包成 query-graph 的
  `EVENT` 规则；缺槽位/规则集 `UNKNOWN` → 图 `UNKNOWN`，不当安全。
- `tests/test_event_rules.py`（2 条）。
- `tests/test_query_world.py`（2 条）：held-out 与线性模型一致、
  无规则 → NaN 不回落。
- `event_predicate_probe.py`（seed 100 / 1500 帧，held-out 210 行）：
  单槽阈值 `x0 < -0.38`：precision 0.30 / recall 0.50 / balanced 0.733；
  `DeathRuleSet` 包装：precision 1.00 / recall 1.00 / balanced 1.00，
  3/210 UNKNOWN。**结论：最小事件谓词应包现有 death rules，不另造
  单槽死亡语义。**
- `key_node_readout_probe.py`（seed 100）：
  full 7 / fixed m=1/2/3 / vertical / horizontal 的 completion 均为
  1.000，visited 均为目标数（每目标一条规则）；但标准化误差差异很大：
  vertical 0.009（1500/5000 帧都稳），horizontal 0.31–0.38，
  full 0.55。**关键节点查询本身是 O(m) 且能补全；但“补全”不等于
  “正确”**——horizontal 目标在过柱/换管处仍有未建模事件，需要
  `EVENT` 规则，而不是再加线性规则。
- `query_graph.TimeWindow` + `Query.target_windows`：
  - 精确目标仍只接受精确 `dt`；
  - 变长 `dt` 规则只进入窗口目标，输出标记 `coarse=True`、
    `times=None`；
  - 同槽位同时有精确/粗规则时，精确优先；
  - 变量规则约定 `pattern_out` 相对时间 0，查询从当前时刻起算；
  - 3 条窗口测试。
- `query_world.py` + `query_world_probe.py`：查询图世界模型与线性
  世界模型在同一批 held-out 行上**逐目标一致**（teacher-forced 最大差
  3e-8；8 步 rollout RMSE 0.000000）；但 Python 图查询比向量化线性
  模型慢 **18.7×**（0.045 vs 0.0024 ms/row）。→ 查询图是结构载体，
  不是默认世界模型的性能替代；planner 接入应是选择性/门控，不是整体
  替换。
- `ComponentBank.free_run_step(derivation, state)`：不更新数值
  `RuntimeComponent`、不写计数/联想，只按学习规则激活/停用三个 sign
  节点；无规则/无输入 → 三个 sign 全停用并返回 `UNKNOWN`。默认路径
  不变；`NodeRuntime` 的预测路径尚未调用它。
- `component_free_run_probe.py`：seed 100 / 1500 帧，
  `vy` 阈值规则 teacher 0.946 / free 0.987；difference teacher 1.000 /
  free 0.927；说明“代码教、转移算”闭环可跑，但 free-run 会漂。
  5000 帧复跑：`vy` teacher 0.929 / free 0.985；difference teacher
  1.000 / free 0.912（漂移随数据量缩小）。
  3 种子（100/101/102，1500 帧）：`vy` free 0.987/0.985/0.996，
  difference free 0.927/0.926/0.951。
- `graph_rewrite_scale_probe.py` 新增实际搜索表（1500 帧、k=16）：
  `global-shared` 与 `cluster-shared` 的 visited = 1/2/3/7（m=1/2/3/7）；
  `independent` = 16/32/48/112，pruned = 15/30/45/105；无 UNKNOWN/
  conflict。**共享是 O(m)，独立是 O(k·m)。** 5000 帧与种子 101/102
  同样成立。

### 决策侧探针：端点事件否决（2026-11-23）

`decision_event_probe.py`：seed 100/101/102，eval 120–124，每个 55
decisions；`branches=8 + tail=4`、horizon 8、无 hard death veto；事件
谓词 = 现有 `DeathRuleSet` 的 query-graph 包装，作用在搜索的
`final_states`（预测端点）。只做 opt-in，不改默认。

| seed | arm | oracle-best | survives | relevant-miss | mean regret |
|---|---|---:|---:|---:|---:|
| 100 | baseline | 54/55 | 34/55 | 2 | 0.15 |
| 100 | veto-death | 55/55 | 35/55 | 1 | 0.04 |
| 100 | veto-death-or-unknown | 55/55 | 36/55 | **0** | **0.00** |
| 101 | baseline | 52/55 | 35/55 | 3 | 0.33 |
| 101 | veto-death | 53/55 | 36/55 | 2 | 0.22 |
| 101 | veto-death-or-unknown | 53/55 | 36/55 | 2 | 0.22 |
| 102 | baseline | 54/55 | 34/55 | 2 | 0.16 |
| 102 | veto-death | 55/55 | 36/55 | **0** | **0.00** |

端点事件质量：

| seed | precision | recall | UNKNOWN |
|---|---:|---:|---:|
| 100 | 0.980 | 0.849 | 347/14080 |
| 101 | **0.998** | **0.379** | 243/14080 |
| 102 | 0.811 | 0.645 | 208/14080 |

读法：

1. **弱正向、无恶化**：veto 在 100/101/102 上 improved 1/1/2、
   worsened 0/0/0；但 active 决策只有 45/51/45，收益是 1–2 个决策。
2. **seed 101 仍是硬案例**：event recall 只有 0.379，relevant-miss
   3→2、mean regret 0.33→0.22；precision 0.998 说明“报死基本对”，
   但“该死没报”仍是主因。
3. **fallback 频繁**：100/101/102 分别 10/4/10 个决策所有候选都被
   veto，回退 baseline；veto 在这些点实际不生效。
4. **oracle-best-vetoed 3/4/0**：多为“所有计划都死/并列最优”的
   不可避免状态，regret 未恶化；不能只看这个计数。
5. **协议差异**：这里的 baseline relevant-miss 是 2–3，不同于早先
   10+8 协议的 10；不能直接跨协议比较。
6. **结论**：端点事件否决是小幅、非恶化的信号，但不足以进默认栈；
   若继续，下一步是真实 3 种子配对基准（opt-in），不是默认接入。
   当前瓶颈仍是候选分辨力与事件覆盖，不是“再加一个 veto”。

仍未做：

- `ComponentDerivationBank` 尚未接进 `NodeRuntime` 的预测路径；当前是
  独立模块 + 探针 + `ComponentBank.free_run_step`，默认不生效。
- 关键节点查询尚未接 planner；`required_slots` 仍由探针手动传入。
- 端点事件否决只做了 55 决策/种子的探针；真实 3 种子配对基准尚未做。
- 查询图世界模型与线性模型等价但慢 18.7×，因此 planner 接入只能是
  选择性/门控，不能整体替换；需要先定义接入点与质量门。

### 实施清单

- [x] schema：`Rule` / `Pattern` / `Query` / `Readout` /
      `Applicability` 纯数据类 + 单测；不写 `FRAME/STAGE` 枚举，
      type 由 `pattern_out + dt` 派生。
- [x] 探针：shared/independent × full/key-node 的实际搜索节点数；
      记录预算、剪枝、`UNKNOWN`、冲突。
- [x] 拟合每目标一条稀疏规则（1500/5000），报告 scope 稳定性；
      动作窗口只作为候选总线，scope 选 lag。
- [x] 共享 applicability + 关键节点查询：engine 支持，探针给出
      O(m) vs O(k·m) 的实际 visited/pruned。
- [x] 事件谓词：`fit_event_rule` 与现有 `death_rules` 在真实流上对照
      precision/recall/UNKNOWN；结论是单槽阈值不足，生产事件谓词应包
      `DeathRuleSet`，不新增第二套死亡语义。
- [x] 关键节点查询接读侧：`key_node_readout_probe.py` 已比较 full /
      fixed m=1..3 / vertical / horizontal；completion 均 1.0、visited
      = m，但 horizontal 误差 0.31–0.38，说明过柱事件仍未建模。
- [ ] 关键节点查询接 planner：把 readout 的 required_slots 接到决策
      路径，比较 full vs m=2–3 的 UNKNOWN/决策覆盖；当前仍是探针。
- [x] 决策侧端点事件探针：`decision_event_probe.py` 把 death rules 的
      query-graph 包装作用到候选端点；seed 101 relevant-miss 3→2、
      mean regret 0.33→0.22，improved 1 / worsened 0；弱正向。
- [ ] 决策侧质量门：veto 端点事件只做 opt-in，需真实 3 种子配对
      基准确认；当前 55 决策/种子，不能作为默认依据。
- [x] 查询图世界模型：`query_world.py` 与线性模型逐目标一致
      （最大差 3e-8，8 步 RMSE 0.000000），但慢 18.7×。
- [ ] planner 接入：不能整体替换线性世界模型；只考虑选择性/门控
      （阶段入口、关键节点、事件谓词），并先过 3 种子质量门。
- [x] 派生概念：difference 规则 + label/free-run 探针；默认关。
- [x] 派生概念接 `ComponentBank`：`free_run_step` 不跑数值过滤器，
      无规则/输入 → UNKNOWN；默认路径不变。
- [ ] 派生概念接 `NodeRuntime` 预测路径：需要把 derivation bank 与
      component id 绑定并在 rollout 中调用；当前只有 bank 方法。
- [x] 测试：pattern/time 不匹配拒绝；UNKNOWN 传播；共享 O(k)；
      深度 3 递归；精确 `dt` 不匹配；stage/interval 窗口；关闭时逐位
      不变（新模块独立；**402 测试全过**）。
- [x] 3 种子诊断复跑：组件 free-run、key-node readout、搜索表在
      100/101/102 上同向；这不是决策侧质量门。
- [x] stage/interval 目标：`TimeWindow` + `target_windows` 已实现；
      变长 `dt` 只进窗口目标、结果标 coarse，精确目标仍拒绝；3 条测试。

### 本轮提交（2026-11-23）

- `40d9ca9` 查询图内核 + 每目标稀疏规则 + 组件派生
- `e541d30` death rules 事件包装 + 对照探针
- `cad7989` key-node 读数探针 + `ComponentBank.free_run_step`
- `3166215` 3 种子诊断 + 深递归测试
- `18a6152` Batch 6 状态整理
- `1fcbe30` stage/interval 粗查询目标
- `b91f12e` 查询图世界模型适配器 + 线性模型对照

### 下一步（Batch 7 候选）

1. **P0 表示层（推荐，直接针对决策）**：状态加入“距上次拍翅帧数 /
   相位变化率”，验证阶段转移坏段基率、图策略退化和长窗读数是否转化
   为决策改善。判据：3 种子方向 + 6 种子确认（≥5/6 同向）。
2. **决策侧端点事件（探针已完成，待质量门）**：seed 101
   relevant-miss 3→2、mean regret 0.33→0.22；improved 1 / worsened 0。
   弱正向；若继续，跑真实 3 种子配对，不先改默认。
3. **NodeRuntime 组件接线**：把 derivation bank 绑到 runtime 的 slot
   表示，做 opt-in free-run；价值偏概念验证。

**不建议**：整体用查询图替换默认世界模型（等价但慢 18.7×）；在未过
3 种子质量门前把 stage/事件项塞进逐帧 cost。

### 质量门

- 关闭时决策路径逐位不变；**402 测试全过**。
- 帧规则：held-out R² 不低于全局线性模型；scope 跨 1500/5000 与
  3 种子稳定；动作 lag 选择一致。
- 搜索：共享 applicability 的实际访问节点数 << 独立；key-node
  m=2–3 覆盖读侧且 `UNKNOWN` 率可控。
- 事件谓词：held-out precision/recall/`UNKNOWN` 达标；无任务专用
  槽位/几何。
- 派生概念：free-run sign 准确率/`UNKNOWN` 可接受，下游转移不落到
  support 外；否则保持 teacher-forced 或 opt-in。
- 决策侧仍只由 3 种子配对基准准入；fast 屏只筛 bug。

### 开放问题 / 非目标

- 关键节点由读侧声明还是学习 relevance？先声明，后学习。
- 共享 applicability 用 regime 簇还是阶段图？先 regime；阶段图需先
  解决身份/时钟，不要再用 context 距离单独合并。
- 同槽位多规则冲突：优先级 = 时间/分辨率精确 > support/confidence；
  仍冲突 → `UNKNOWN`，不平均。
- 隐藏状态：capacitor/resistor 的 `u` 是 latent / 窗口 / observer？
  先 difference。
- Scope 可辨识性：时钟/滚动共线；用跨 episode/跨数据量/干预。
- 非目标：默认不建 `k × 7` 规则库；不硬编码转移 type；不把代码
  component 当运行时 oracle；不做任务专用阶段词/几何；死亡仍
  0/1/UNKNOWN，不用概率/梯度。


## P0 表示层：time-in-state（2026-11-23）

> **状态：已关闭（中性/null）**；保留作参考。不要把 time-in-state /
> clock / phase 重新引入 hazard 或 world。

**目标**：把“距上次拍翅帧数 / 相位变化率”放进状态，直接针对候选
分辨力；不改默认栈，先做 opt-in A/B。

### 已实现

- `time_in_state.py`：
  - `TimeInStateEnv` 追加每个动作的 age（帧数，capped）作为状态；
  - 可选 phase：`sin/cos(2πt/period)` + `1/period`；
  - `make_env_factory(period, include_actions, include_phase)`；
  - `augment_observations`（离线流版本）；
  - `estimate_period`（逐 episode 频谱，任务无关）。
- `env_factory` opt-in：`substrate_generic.collect`、
  `substrate_nodes.fit/evaluate/collect_planner`；默认 `None`，默认路径
  不变。
- `time_in_state_probe.py`：同一 fit 配置、同一 held-out 世界，比较
  baseline / time；`--features action|phase|both`。

### 首批读数（seed 100，1500 帧，eval 120–121）

| variant | frames | passes | action rate |
|---|---:|---:|---:|
| baseline | 312.0 | 6 | 0.0787 |
| both (age+phase, max_age 64) | 192.0 | 3 | 0.0865 |
| action, all ages, 64 | 169.0 | 3 | 0.1036 |
| action, all ages, 16 | 140.5 | 2 | 0.1101 |
| action 1 only, 16 | 189.0 | 3 | 0.0848 |
| **control: one constant extra dim** | **192.0** | **3** | **0.0783** |

读法：

1. **维度控制是决定性的**：只加一个常数维度（无信息）就把 312→192，
   与 both/action 的损失同量级 → 本轮负向主要是**固定原型容量/分辨率
   被额外维度稀释**，不能归因于“时间特征无用”。
2. **action-only 没有超过 control**：169/140.5/189 vs control 192；
   没有正向信号，不进入 3 种子。
3. **phase 不可辨识**：period 估计 1500/5000/15000 帧分别是 12/14/14，
   不是世界周期 ~80；行为流 episode 太短、没有完整相位。phase 要等
   有相位的流或更可靠的 reset 检测。
4. **公平复测的必要条件**：保持有效原型密度（按维度缩放 `max_nodes`），
   或把时间特征只接世界模型/读侧，不改 agent 的原型状态。当前实现
   是“全状态追加”，不是公平 A/B。

**首批结论：full-state 的负向被维度控制混淆，不可归因于特征；
后续 conditioned world 的 held-out gate 判零（gain -0.314），
最终是中性/null。不采纳、不跑 3 种子。**

### 第二轮：world-model-only（agent 保持 raw）

- baseline：raw world + raw agent + raw env；
- time-world：world 在 augmented 观测上拟合，agent 仍 raw，env 返回
  augmented；`policy_result` 只把 raw 前缀给 agent。
- seed 100，1500 帧，eval 120–121，action 1 age / max_age 16：
  baseline **312 帧 / 6 过柱** → time-world **117 帧 / 1 过柱**。
- 但 constant-extra-dim 的 world 控制也是 **120 帧 / 1 过柱**，同样
  大幅下降；常数维与 intercept 共线，`LinearObservationModel` 的
  ridge 1e-8 可能数值不稳。
- 非恒定 control（`sin(0.7t)`，无信息）仍是 **152 帧 / 2 过柱**，
  远低于 baseline → world 多一维/未标准化本身就会破坏搜索，不能
  直接比较。正在用 **per-column 标准化 world**（baseline/time/control
  都标准化）复核；若 control 回到 baseline 附近，才谈时间特征的净效应。
- 结论暂不成立：需要标准化 world 或容量匹配，才能区分“时间特征
  有害”与“world 多一维/分裂接口有问题”。

### 第三轮：conditioned world（最终判据）

- `ConditionedTimeWorld`：raw world 保持 baseline 不变；时间特征只作为
  **输入残差模型**，不进入输出维度；辅助 age 由确定性更新规则推进；
  agent 仍 raw。
- 这是目前最干净的 world-model-only 形态：base raw 预测与 baseline
  完全相同，时间特征只能通过残差修正影响候选。
- 标准化 control 已说明：baseline/control 都标准化时是 152/2 vs
  152/2；而标准化本身就把未标准化的 312/6 降到 152/2，所以
  “标准化 world”不能当无偏对照。
- `ConditionedTimeWorld` 新增 `residual_scale`：control 设 0，残差
  完全不生效，base raw 预测与 baseline 逐位相同；这是接口 control。
- **conditioned control（constant age 99, residual_scale=0）已回到
  baseline 逐位相同：312/6 → 312/6，delta 0/0。** split 接口
  通过。
- conditioned time 未加 gate：**234/4** vs baseline **312/6**
  （eval 120: 472→316；eval 121: 152→152）→ 残差修正会改变行为并
  在一个世界上变差。
- **gated conditioned time（episode-held-out）**：
  validation gain **-0.314** → `residual_scale=0.000`；
  行为回到 baseline 逐位相同：**312/6 → 312/6，delta 0/0**。
- 判定：**action-1 age 对 raw next-state 没有可迁移增益**；
  gate 正确判零。P0 在本数据上是**中性/null**，不是“age 有害”，
  也没有达到 3 种子门槛。
- phase 仍不可辨识（period 12/14/14 vs ~80），暂不测。
- 若继续 P0，只剩 **readout-only hazard conditioning**（age 只进
  危险读出，不进 world）这一种形态；否则回到候选分辨力的其他机制。
  **不采纳、不跑 3 种子、不改默认。**

### 接线诊断（`time_wiring_probe.py`，seed 100 / 1500 帧）

- **wiring 校验通过**：online `TimeInStateEnv` 与 offline
  `augment_observations` 的 raw、age、actions 全部逐位一致。
- age 统计：mean 0.552、std 0.382、17 个离散值；`corr(age, alive) =
  -0.120`；`corr(alive, step) = 1.000` → age 不是时钟的副本。
- **raw next-state**：base held-out R² 已经 1.0000；逐目标 base MSE：
  - 垂直目标 0/1/3/5：3.2e-5 / 3.2e-3 / 3.2e-5 / 3.2e-5；
  - 水平/时间目标 2/4/6：1.6e-13 / 1.0e-12 / 4.5e-10。
  → 水平/时间上的“巨大相对增益”是机器精度噪声，不能当证据。
- age 残差对垂直目标在 **`StandardizedRidge` 代理模型**上有
  MSE gain ≈ +0.15、`corr(age, residual)` y/vy **-0.398**、
  gap **+0.398**；但这是**代理模型**的结果，不是生产世界模型。
- **同一流上换成 `LinearObservationModel`（生产同款，raw + ridge
  1e-8）后，垂直目标 gain ≈ 0.000**；用 `fit` 出来的 DAgger 世界
  做 per-target gate 时更是 `0 -0.0000 / 1 +0.0000 /
  3 -0.2627 / 5 -0.2332` → **没有目标通过**。
- per-target、age-only 残差（`age_residual_probe.py`，targets
  0/1/3/5，episode-held-out，min_gain 1e-3）：
  - enabled targets = `[]`；
  - baseline / age / control 全部 **312 帧 / 6 过柱**；
  - candidate 10 decisions：gap median 0.02314、oracle-best 10/10、
    regret 0；changed actions **0**。
- 结论修正：**此前“world model 漏掉 flap age”的正信号是标准化
  代理模型的产物**；在生产线性世界模型上，age 没有可迁移的线性
  残差增益，也不改变候选选择。P0 仍是中性/null。
- **hazard 读出不应加 age**：
  - raw：point-pain AUC 0.785 / pain8 0.855；
  - raw+age：0.758 / 0.813（变差）；
  - age only：0.448 / 0.484（无信号）；
  - clock only：0.870 / 0.880，但这是**时间-死亡混淆**；
    raw+clock 与 raw 相同（clock 已由 alive 携带）。
- **质量不足的位置不是特征计算，而是“诊断代理 vs 生产模型”的错配**：
  1. `StandardizedRidge` 代理放松了 raw 精度，放大了可被 age 解释的
     残差；生产 `LinearObservationModel` 已经把这些方差吃掉，
     per-target age gain ≈ 0，gate 不启用任何目标；
  2. 因此“per-target age-only 残差只修 0/1/3/5”的提议在真实流上
     **没有可修的量**；candidate gap/regret 也不变；
  3. age 对 hazard 无益，clock 是冗余/混淆项；
  4. 真正的质量缺口仍是**候选分辨力**：第一动作 gap 中位 0.023，
     P0 的 age 修正不改变任何候选选择。

**下一步**：P0 关闭。per-target、age-only 残差已经证明没有可迁移
增益，候选 gap/regret 不变；readout-only hazard conditioning 也
不推荐（age 对 hazard AUC 是负增益）。回到候选分辨力的其他机制。
**不要加 current-frame clock；不要用 age 进 hazard readout。**

### 判据（未变）

- 3 种子方向 + 6 种子确认（≥5/6 同向）；
- 先看 frames/passes，再看候选第一动作 gap / decision regret；
- 默认关，未过质量门不接默认。

## Batch 7：候选分辨力（2026-11-23）

**一句话**：候选分辨力不是“再加一个分数项”，而是四个不同问题：
**存在性、可识别性、稳定性、实现性**。不同 seed / 状态落在不同问题
上，不能用一个全局 cost 同时解决。**H=8→H=12 只是把帧窗口放大，
不是多视界。B0 后已确认：seed 101 的主问题是 plan/suffix 的
predicted endpoint drift，首动作 router 不是主杠杆。**

### 四个子问题

| 子问题 | 问题 | 当前证据 |
|---|---|---|
| **存在性** | 当前状态附近，H 帧内有没有更好的候选？ | `oracle@8 = policy@8`；H=8/10 内所有候选都死；H=12 有逃生，但默认策略走到的状态 12 帧内宏/随机方案存活率 0% |
| **可识别性** | 有更好候选时，score/readout 能不能排上来？ | seed 101：oracle-best **10/10 feasible**，但 **10/10 被 score 排后**；B0：**8/10 是 plan/suffix**，只有 2/10 有真实首动作对比；B0.5：只有 risk 活跃，0/8 单列修复；B0.6：risk 是 endpoint terminal hazard，预测 endpoint 反向 8/8 |
| **稳定性** | 顶端候选差距是否小于噪声/新项？ | 第一动作 gap 中位 **0.008–0.026**；**4–33%** 决策 <1e-3；顶端 margin p50 **1e-3** |
| **实现性** | 第一动作选对后，后续重规划是否吃掉优势？ | 计划级 value 选对率 0.82/0.86 vs 0.73/0.79，几乎不分歧；伤害是分叉后的累积 |

### 证据排序

1. **seed 101：plan-level endpoint drift（最高优先，B0–B1-plan 已定位）**
   - 10 个 decision-relevant miss；oracle-best 10/10 feasible；
     10/10 被 score 排后；death veto 0；
   - B0：8/10 是 plan/suffix（4 个同动作 + 4 个零首动作对比），
     只有 2/10 有正的首动作对比；近并列动作级可修 0/10；
   - B0.5：8 个 plan-level miss 上只有 risk 活跃；value/curiosity/q
     全 0；0/8 能被单列/列组合修复；
   - B0.6：risk 的 step 部分为 0，几乎全是 endpoint terminal hazard；
     预测 terminal 反向 8/8，真实 terminal 7/8 正确；
     state error 0.111；
   - B1-plan：predicted endpoint 读数 same-AUC 0.789、B0 miss 9/10；
     ranking 通过，calibration 失败（Brier skill −0.092）；
   - B1-plan-3seed：seed 101 +7/−1，但 seed 100 从 62/62 退化到
     60/62（improved 0 / worsened 2），seed 102 terminal base rate
     0.000、模型退化；**非退化门失败**。
   → 不是没候选、不是死亡规则没抓、不是列权重；seed 101 的
   plan-level endpoint 读数不迁移，不能直接接线。下一步只剩
   OOD/UNKNOWN 门后的 rank-only 复测或 world endpoint 改善。

2. **seed 100/102：存在性不足（晚期状态）**
   - H=8 内所有候选都死；
   - 再好的分类/score 也救不了；
   - `oracle@8 = policy@8`。
   → 需要更早行动 / 更长有效视界 / 不同状态分布，而不是更好的选择。

3. **所有 seed：稳定性不足**
   - 第一动作 gap 0.008–0.026，4–33% <1e-3；
   - 价值项均值 0.022，是顶端 gap 的 10–60 倍，从第 3–11 帧就改写动作；
   - 任何 0.01–0.1 的新项都会翻掉一部分近并列决策。
   → 需要 **margin-aware 仲裁**，不是全局调权重。

4. **实现性：分叉后的累积**
   - 局部第一动作并不差；
   - 伤害来自分叉后落在不同状态，再叠加同样的低 margin。
   → 需要 commitment / 事件边界 / option，但当前数据/身份不足，后置。

5. **探索/覆盖**
   - 行为策略拍翅率 8–11%，强自相关；
   - 世界模型在行为分布上 h1 R² 0.99，但 planner 候选是 off-policy；
   - 同状态两个动作的真实结局样本很少（plan-level 仅 11–14 个可判定决策）。
   → 候选评分在危险附近动作对比上数据不足。

### 修正：不是 H=8→H=12，而是多视界结果转移 + router

固定帧视界是近视的：H=8 看不到死亡封闭，H=12 又太贵，而且默认策略
走到的状态在 12 帧内也封死。**继续加 H 只是扩大窗口，不是改变决策
单位。**

需要的结构：

```text
多视界结果转移库：
  层级 = 帧 / 事件 / 阶段 / 回合（由目标结果决定，不由固定帧数决定）
  每条边 = entry_condition + action/option
           + target_outcome / next_level
           + duration_distribution
           + support / calibration / validity
Router：
  request(target, required_resolution, level, margin)
  → 细层模型优先；粗层只做早期预警/剪枝/排序
  → 入口不满足或支持不足 → UNKNOWN
  → 粗层不能覆盖有效的细层判断
```

**B0 前的最小可行版本（已被 B0 修正，保留作历史）**：

- 学 **horizon-indexed、action-conditioned 结果模型**：
  `P(terminal within h | state, a)` 或 `E[survival | state, a]`，
  h = 1/4/8/12/16；
- 用现有 precursor 标签；episode-held-out 校准；
- 同时报告 **action contrast**：`risk(a=1) - risk(a=0)`；
- router 只在 baseline 第一动作 gap 小、或细层看不到 terminal 时
  启用粗层；
- coarse 只改候选排序/剪枝，不进入逐帧 cost。

为什么这不是之前失败的阶段图：

- 不学 context 簇/阶段身份，不依赖 clock/phase；
- 目标是**结果**（terminal/survival），不是全状态；
- 只在决策边际小的时候消费，避免改写强信号；
- H=12 搜索不是默认，只由粗层风险触发、按状态局部扩展。

### B0 后的修正（2026-11-23，**仅 frame-level 基线**）

B0/B0.5/B0.6/B1-plan 把**当前 8 步帧级默认栈**在 seed 101 的失败从
“首动作/分数列”缩到 **frame-level predicted endpoint**：

- 8/10 miss 的 selected 与 oracle-best 在各自最佳后缀下首动作等价，
  首动作 router 即使完美也修不了；
- 8 个 plan-level miss 只有 risk 一列活跃，value/curiosity/q 为 0；
- risk 的 step 部分为 0，几乎全是 endpoint terminal hazard；
  预测 terminal 反向 8/8，真实 terminal 7/8 正确；
- frame-level endpoint 读数在 seed 101 有 ranking 信号，但 3seed
  非退化失败、calibration 失败。

**这些是 frame-level 基线诊断，不是设计轴。** 不要用它们来：

- 判断事件/阶段级 outcome transition 是否可行；
- 调 8 步帧级 endpoint / risk 权重；
- 把首动作 router 或帧级 endpoint re-ranker 接进默认栈。

当前设计方向改为 **Batch 8：6 步 fine executor + 事件/阶段级
outcome transition**。帧级 endpoint 只作为基线和局部执行器，不承担
长程 terminal 判断。

### 两份设计备忘录的合并结论（2026-11-23）

**训练侧**：

- 层 = 结果谓词 `(O, A_O)`，跨度 = 停时分布 `P(T | entry, option)`；
- 粗层只学 outcome/terminal/survival，不学全状态端点；
- 一个 distributed-lag cause-specific hazard，一次训练、任意窗口查询；
- 层要过五测：事件支持、动作条件性、非时钟性、事件后 regime 改变、
  跨 episode 校准；
- 默认单段；只有 held-out 结果似然失败才分段；merge 按 outcome
  散度，不按 context 距离；
- 被动数据可学事件/停时/入口/行为结果/逐 lag 动作剖面；长跨度
  option 对比需要定向配对采集（序列 propensity 乘积会塌）。

**推理侧**：

- fine-first router；粗层只在近并列 / 细层失明 / OOD 时查询；
- 三种模式：SUBSTITUTE（细层 UNKNOWN/全死）、ASSIST（近并列
  band 内排序）、CHALLENGE（粗层反对时触发局部细层重搜）；
- 区间/共形下界仲裁；`L_coarse > U_fine + δ` 才改动作；
- 不做全局逐帧 cost、不做全局 veto；UNKNOWN 不当安全；
- 规则 router 先行，不学习 router；默认关，关闭时逐位不变；
- 接线在 `policy_result` 外部后选择，不改 6 列 score 账本。

**两份备忘录的分歧/风险**：

1. 推理侧先提“候选端点 `P(terminal within 4 | final_state)`”，
   训练侧警告全状态端点会漂移；必须先测 **endpoint OOD 率**。
   若 OOD 高，改用 action-conditioned state hazard，而不是端点模型。
2. 长跨度 action contrast 可能不可识别；B1/B2 必须先过
   contrast 校准，否则 router 只是换一种噪声。
3. 1500 帧只够筛查；3 种子方向 + 6 种子确认，安慰剂/打乱臂必需。

**B1-plan 后的更新（2026-11-23）**：

- endpoint 读数在 seed 101 ranking 通过（same-AUC 0.789、B0 miss
  9/10），但 3seed 非退化失败：seed 100 62/62 → 60/62，
  seed 102 terminal base rate 0.000；calibration 全失败；
- 候选 endpoint OOD 率 101/100/102 = **28.7% / 68.9% / 63.0%**；
  OOD 高的 seed 正是失败的 seed；
- 因此不再把 action-conditioned state hazard 当 fallback（seed 101
  本身没有首动作对比），也不直接接 endpoint 读数；下一步只做
  OOD/UNKNOWN 门后的 rank-only 复测，或 world endpoint 改善。

### 实验 A：候选存在性审计（只读，先做）

> **状态：已并入 B0/B0.5/B0.6**（2026-11-23）。B0 已完成候选存在性/
> 可识别性/plan-level 分类；本节保留作定义参考，不再单独执行。

在每个真实决策点，把决策分类：

```text
A. H=8 内无逃生            → 存在性（需要更早/更长视界）
B. H=8 内有更好候选，但 score 排后 → 可识别性
C. 已选中 oracle-best      → 正常
```

同时记录：

- H=8 / 10 / 12 的 escape 数；
- 第一动作 gap；
- oracle-best 是否 feasible；
- 是否存在「H=8 全死、H=12 有逃生」。

复用：
`doom_horizon_probe.py`、`causal_horizon_probe.py`、
`planner_rollout_quality.py`、`decision_margin_probe.py`。

**判据**：

- seed 101 若主要是 B → 先修可识别性；
- seed 100/102 晚期若主要是 A → 先修存在性/视界/状态分布；
- 若 A 在所有 seed 占主导 → 不要再加 score 项，直接做更早行动/视界；
- 只读，不改默认。

### 实验 B0：seed 101 miss 诊断门（B2 前置，只读）

对 seed 101 的 10 个 decision-relevant miss，逐条记录：

- `first_action_gap` / `plan_margin`；
- `oracle_best_feasible` / `oracle_best_score_gap` / rank；
- selected vs oracle-best 的第一动作；
- 粗层对该两个动作的 support / calibration / admissible；
- 候选端点 OOD 率。

判定树：

```text
>=7/10 miss 的 gap <= 0.03 且两动作粗层可准入 → 跑 coarse-rank
>=5/10 miss 的 gap > 0.1 → 契约不允许直接改；跑 coarse-challenge
                                        或先修 fine score
>=5/10 miss 的 true_action_contrast = 0 → plan/suffix 排序问题；
                                         先修 fine score / suffix
粗层对 oracle-best 动作不可准入 >=5/10 → 覆盖失败；先做定向采集
```

**B0 先决定这是排序问题、覆盖问题，还是细层分数问题；**
**不过 B0，不进入 B1/B2。**

**B0 结果（2026-11-23，seed 101，120–124，85 决策）**：

- 复现 10 个 decision-relevant miss：6 个首动作不同，4 个首动作相同。
- **只有 2/10 有正的真实首动作对比**（120@67、124@68）；
  其余 8/10 在选中/最佳首动作的各自最佳后缀下真实存活并列
  （4 个同动作 + 4 个不同动作但 `true_action_contrast=0`），
  属于 **plan/suffix 排序**，首动作 router 即使完美也无法修复。
- 动作级可修 2/10；其中局部 support 可达 1/10（124@68，
  gap 0.0878，intermediate），**近并列可修 0/10**。
- flap 局部支持极薄：6 个首动作不同的 miss 中，最近 64 个训练状态里
  flap 行为 0–3 次（仅 124@68 为 7 次）；pending 上下文覆盖
  （57/58/60）不等于动作对比覆盖。
- selected / oracle-best 端点 OOD **0/10**；但全部候选端点 OOD
  q95 **37%**、q99 **19%**，中位 percentile 0.87 → 全局 endpoint
  ranker 必须带 OOD 门；relevant pair 本身不是 OOD 问题。
- **B0 判定：`plan_level`，不是 `ranking`，也不是单纯 `coverage`。**
  先修 fine score / 后缀排序；首动作 coarse router/B1/B2 暂缓。
  仅对 2 个正对比状态考虑定向配对采集。

### 实验 B0.5：plan/suffix score 诊断（B0 后，只读）

- 对 8 个 plan-level miss 逐列消融 risk / value / curiosity / q，
  定位把 oracle-best 后缀排后的列/段；
- 记录 selected 与 oracle-best 的完整 score 向量、各列排序贡献、
  以及同一首动作下最佳后缀的差距；
- 若没有任何单列能修复，问题在 world 预测漂移或列间组合，
  再考虑 plan-level endpoint 重排（带 OOD 门），而不是首动作 router。

**B0.5 结果（2026-11-23，seed 101）**：

- 10 个 miss：8 plan-level、2 action-level；8 个 plan-level 的
  suffix regret 为 2–5 帧，same-action-best 的 score rank 中位 7、
  均值 9.5。
- **默认 score 在这些决策上实际只有一个活跃列：risk**。
  plan-level 的 same-best-minus-selected 风险差 +0.042…+0.072，
  value / curiosity / q 差值全部为 0（8/8 非零列只有 risk；
  positive share = 1.00）。
- **没有任何单列或列组合消融能恢复 plan-level miss**：
  plan-level global 0/8、same-action 0/8；全部 miss 的 global 也是 0/10。
  去掉 risk 后成本近似常数，之前的“fix”是并列下的词典序偶然。
- 结论：不是“某一列权重太大”，而是 **risk 列本身的排序错了**：
  它给真实更差的后缀更低的风险。下一步必须区分
  **world 预测漂移**与 **risk readout/calibration 错误**。

### 实验 B0.6：risk readout 真/预测轨迹对照（只读）

对 8 个 plan-level miss 的 selected 与 same-action-best 后缀：

- 用同一个 `world.predict` 复现 planner 的预测轨迹，按 `step_stride`
  累加 risk；用真实环境复现同一序列，在真实状态上按相同步长累加 risk；
- 记录 `risk_pred_selected/best`、`risk_true_selected/best`、
  `state_error`、真实首死步；
- 判据：
  - `risk_true(best) < risk_true(selected)` 在多数 miss 成立 →
    risk readout 在真实状态上是对的，主因是 **world 预测漂移**；
  - 若在真实状态上也反了 → risk readout/calibration 本身错误，
    先修 readout，不要做 plan-level endpoint 重排；
  - 若两者都不成立 → 检查 pain 标签/风险合成本身。
- 只在真实转移上计算，不训练新模型；输出逐 miss 对照与聚合。

**B0.6 结果（2026-11-23，seed 101）**：

- 复现校验 **8/8**：预测轨迹的 risk 与 planner 的 risk 列逐位一致。
- **预测 risk 的 step 部分全为 0**（pred step margin 0.0000）；
  整个 risk 列是 endpoint 的 **terminal hazard**。
- 预测 terminal risk 在 8/8 把 selected 排得更安全
  （pred terminal margin -0.053），但真实 selected 在 4–7 帧死亡、
  best 后缀全部存活。
- 真实 terminal risk 在 **7/8** 把 best 排得更安全
  （true terminal margin +0.706）；真实 step risk 只有 2/8 正确，
  first-step 0/8，step margin 仅 +0.008。
- state error mean **0.111**；predictive endpoint 漂移。
- 例外 `122@67`：真实 terminal 也给 best 1.0（readout 认为 best 也
  终态危险），true total 反而偏向 selected；这是唯一 readout 侧反例。
- **判定：`step_readout_flat_endpoint_drift`。**
  - 不是 risk 权重/列组合问题：step 风险几乎全 0；
  - 真实 endpoint 的 terminal 读数基本正确（7/8），
    错在 **world 预测的 endpoint**；
  - 逐帧罚项无法修复（step 信号为 0）；首动作 router 也无关。
- 下一步：**plan-level endpoint 校正/重排**，用 planner 的 predicted
  endpoint + 真实 terminal 标签训练/校准，带 OOD 门；或先修 world
  endpoint 漂移。改 risk 权重/加逐帧项均不在候选内。

### 实验 B1-plan：predicted endpoint 校正（只读）

用 B0 的 69 个有效决策（16 个无可行 same-action 候选被跳过；
10 个 relevant miss 全在）和 16102 个候选，按 `eval_seed` 做
leave-one-world-out；标签是候选真实 terminal（within H=8）。
对比：

- A0：fine score，同动作内选最低；
- A1：predicted endpoint（7 维）；
- A2：endpoint + OOD 距离；
- A3：endpoint + OOD + fine score 列；
- A5：fine score 列（校准器）；
- A4 / A4b：label-shuffle / feature-shuffle placebo。

**B1-plan 结果（2026-11-23，seed 101）**：

| arm | same-oracle | same-regret | relevant-miss | within dec-AUC | within same-AUC | top1 |
|---|---:|---:|---:|---:|---:|---:|
| A0 fine | 61/69 | 0.362 | 10 | — | — | — |
| **A1 endpoint** | **68/69** | **0.058** | **3** | **0.760** | **0.789** | **68/69** |
| A2 endpoint+ood | 67/69 | 0.101 | 4 | 0.543 | 0.465 | 67/69 |
| A3 endpoint+fine | 65/69 | 0.203 | 6 | 0.618 | 0.554 | 65/69 |
| A5 fine-only | 62/69 | 0.333 | 9 | 0.182 | 0.189 | 62/69 |
| A4 label-shuffle | 65/69 | 0.188 | 6 | 0.543 | 0.545 | 65/69 |
| A4b feature-shuffle | 65/69 | 0.188 | 6 | 0.545 | 0.550 | 65/69 |

- B0 miss 子集：A0 same-oracle **2/10**，A1 **9/10**
  （improved 7 / worsened 1），A4 6/10。
- **A1 endpoint-only 是唯一有清晰 within-decision 信号的 arm**：
  same-AUC 0.789 vs placebo 0.545–0.550；fine-only 0.189（反向）。
- **A2/A3 把信号打掉**：OOD 距离和 fine score 作为特征会主导
  within-decision 排序；OOD 应做 **排序后的门控**，不是特征。
- **校准失败**：A1 pooled Brier skill **−0.092**、ECE 0.097、
  slope 0.037；within-Brier skill **−0.793**；预测 p-std 仅 0.0265。
  它是 ranker，不是可用的概率读数。
- 例外 `122@67` 仍是 A1 唯一 worsened 的 miss，与 B0.6 的
  readout-side 反例一致。
- **判定：ranking gate 通过，calibration gate 失败；不接线。**
  下一步 B1-plan-3seed：seed 100/102 非退化 + 温度/isotonic/嵌套
  校准 + 多种子 shuffle placebo；只有 rank-only near-tie assist
  可用后才考虑 B2-plan。

**B1-plan-3seed 结果（2026-11-23，100/101/102）**：

| seed | decisions | A0 same-oracle | A1 same-oracle | A1 improved/worsened | A1 dec/same AUC | terminal base rate | endpoint OOD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 69 | 61/69 | **68/69** | +7 / −1 | 0.760 / 0.789 | 0.110 | 0.287 |
| 100 | 62 | **62/62** | 60/62 | 0 / −2 | 0.352 / 0.250* | 0.158 | 0.689 |
| 102 | 48 | 48/48 | 48/48 | 0 / 0 | None (0) | **0.000** | 0.630 |

\* seed 100 只有 2 个决策同时含两类候选，AUC 只作描述，不作证据。

- seed 101：B0 miss 2/10 → 9/10，ranking 有信号；但 calibration
  失败（Brier skill −0.092、within −0.793）。
- seed 100：fine 已经 62/62；A1 反而 **60/62**（worsened 2），
  flip precision 0；endpoint OOD 68.9%，模型不在训练密度内。
- seed 102：48 个可用决策里 **没有任何 terminal 候选**，
  base rate 0.000；模型只剩常数/并列，AUC 不可算，无法评估。
- placebo（A4/A4b）与 A1 在 seed 100 同为 60/62；seed 101 上
  A4 6/10 vs A1 9/10。增益确实是 seed-specific。
- **判定：3 种子非退化门失败，B1-plan 不接线。**
  - 不是“再校准一下”的问题：seed 100 的失败是泛化/覆盖，
    seed 102 是标签退化；
  - 只有 OOD/UNKNOWN 门后的 rank-only 复测还有一点空间：
    当 endpoint OOD 或 fine 已明确不并列时保持 fine；
  - 若 OOD 门仍不能让 100/102 不退化，停止 plan-level endpoint
    correction，转回 world endpoint 改善 / 更早行动。
- 不做 B2-plan；不默认开；不调 risk 权重；不加逐帧项。

### 实验 B：多视界结果转移 + router（opt-in，B0 后暂缓）

**前置状态（B0/B1-plan 后）**：首动作 router 只有 2/10 的理论空间，
其中近并列可修 0/10。plan-level endpoint 读数在 seed 101 有 ranking
信号，但 3seed 非退化门失败（seed 100 退化、seed 102 无 terminal
标签）。B1/B2 的 action-level 设计继续保留；当前不启动，先做
OOD/UNKNOWN 门后的 rank-only 复测或 world endpoint 改善。

**B1：结果转移库**

- 学 **horizon-indexed、action-conditioned** 结果：
  `P(terminal within h | state, a)` / `E[survival | state, a]`，
  h = 1/4/8/12/16；
- 用现有 precursor 标签；episode-held-out 校准；
- 同时报告 **action contrast**：`risk(a=1) - risk(a=0)`；
- 目标只是结果，不是全状态；不学阶段身份、不依赖 clock/phase；
- 支持/校准不足 → 该 horizon 不启用。

**B2：router 探针**

- fine-first：细层有效且 gap > ε 时，不查询粗层；
- 只在 `first_action_gap < ε`、细层失明、OOD 或端点歧义时查询；
- 三种模式：SUBSTITUTE（细层 UNKNOWN/全死）、ASSIST（近并列
  band 内排序）、CHALLENGE（粗层反对时触发局部细层重搜）；
- 区间/共形下界仲裁：`L_coarse > U_fine + δ` 才改动作；区间重叠
  → abstain，保留细层；不用“保持上一步”当 abstain；
- 不做全局逐帧 cost、不做全局 veto；UNKNOWN 不当安全；
- 端点模型必须报 **endpoint OOD 率**；OOD 高则改用
  action-conditioned state hazard，不用端点；
- 探针臂：
  - A0 baseline；A1 placebo-random（same band，flip 数匹配）；
    A2 placebo-shuffle（决策内打乱粗值）；
    A3 coarse-rank；A4 coarse-prune；A5 coarse-challenge；
- 指标：seed 101 decision-relevant miss / mean regret / improved-
  worsened；seed 100/102 selected survives；changed rate；
  coarse admissible / UNKNOWN rate；成本。

**B2 通过标准（建议）**：

- seed 101：mean regret 或 relevant-miss 相对 baseline 下降
  ≥20%，且优于 A1/A2；`improved >= worsened + 3`；
- seed 100/102：selected survives 不低于 baseline−1，mean regret
  不高于 baseline+0.05；changed rate ≤10%；
- 粗层在 seed 101 miss 状态上 admissible ≥30%；整体 UNKNOWN
  ≤50%；action contrast accuracy ≥0.55；
- 成本 ≤1.5× baseline；不默认跑 H=12；
- 3 种子方向 + 6 种子确认（≥5/6 同向）。

**B3：自适应搜索（后做）**

- 只有当粗层风险高、且 H=8 内全死时，才对**该状态**局部扩展到
  H=12；不是默认 H=12；
- 记录成本/收益：新增搜索时间、找到的 escape、决策改变、regret。
- 若成本不可接受或只改变近并列噪声，关闭。

质量门：

- B1：h=12 action contrast 有可用校准/支持；不是只报 AUC；
- B2：seed 101 miss/regret 下降；seed 100/102 survives 不下降；
- 3 种子配对；6 种子确认（≥5/6 同向）；
- 默认关；未过门不接默认。

### 实验 C：定向反事实覆盖（可选，后做）

> **状态**：B0 后只对 2 个真实首动作对比状态（120@67、124@68）
> 有直接价值；当前不启动。若 OOD-gated rank-only 也失败，再考虑
> 用配对采集补 endpoint/首动作标签。

只在**高不确定/高危险候选端点**上尝试另一个动作，收集 paired
transition，重训 world/hazard；指标是候选端点 R² 和 plan-level 选对率，
不是全局覆盖率。现有全局 coverage / VOI 在 6 种子是 null，不再重复。

### 非目标

- 不再加 time-in-state / current-frame clock / phase（P0 已判零）；
- 不把 H=12 当默认或唯一解；H=12 只是粗层触发的局部扩展；
- 不用全状态长程预测替代结果转移（endpoint drift 已有证据）；
- 不做全局逐帧长窗罚项 / 全局端点 veto（已有 0/3、121.7 vs 152.9）；
- 不在没有阶段数据/身份时继续 stage/option 消费；
- 不在 3seed 非退化门和校准门未过时接 plan-level endpoint 重排；
- 不用小 beam 长视界做贪心剪枝（w64 长视界 −22）；
- 不把“再加一个 0.01–0.1 的项”当成候选分辨力的解。

## Batch 8：6 步 fine executor + 事件/阶段结果转移（2026-11-23，用户方向）

**一句话**：帧级 8 步只作为基线诊断；目标是把 fine executor 缩到
**6 步**，由**事件/阶段级 outcome transition** 承担长程；跨度是
**停时分布**，不是固定 6/8 帧；粗层只做方向/约束/早期预警，fine-first。

### 为什么

- 8 步帧级组合的 endpoint drift 是预期现象，**不作为设计轴**；
- 6 步会降低 compounding error，但近视更严重；只依赖帧不能解决
  长程 terminal 判断，必须由事件/阶段层补上；
- 旧 stage graph 的失败点是 context 聚类身份不跨 episode、主断点常是
  世界时钟；新口径的层 = **outcome predicate**，不是上下文簇。

### 设计

- fine：6 步帧级 MPC，只做局部执行；不在自身视界外做 terminal 判断；
- 事件/阶段层：
  - 层 = `(结果谓词 O, 结果字母表 A_O)`；
  - 跨度 = `P(T | entry, option)` 的停时分布，持续时间是输出；
  - 入口条件、action/option、target outcome、support、calibration、
    validity；不足 → UNKNOWN；
- router：fine-first；粗层只在 fine 近并列 / 失明 / OOD 时查询；
  粗层不能覆盖有效 fine；默认关；UNKNOWN 不当安全；
- 数据：被动数据可学事件/停时/入口/行为结果；长跨度 option 对比
  需要定向配对采集（序列 propensity 乘积会塌）；
- 评估：事件支持、动作特异性、非时钟性、事件后 regime 改变、
  跨 episode 校准；**不是** 8 步 state MSE 或 frame endpoint drift。

### 事件/阶段 outcome audit 结果（2026-11-23，被动流 3 种子，已完成）

协议：`collect` 被动流 5000 帧（coverage 0.1），122–132 局，局长
p50 **34.5** / p90 **72**；候选谓词 = terminal + 各维分位穿越/过零/
极值/跳变（时钟计数维排除，排除决策带 std/jump 比率入报告，
`--keep-dims` 可做敏感性）；每谓词一个 distributed-lag
cause-specific hazard（person-period，stop-time 口径），五测；
leave-episode-fold 校准。

**跨种子稳定通过五测的 level 候选（100/101/102 全部）**：

| 谓词 | 解读（仅分析语） | events | AUC8 | skill8 | Δ8 |
|---|---|---:|---:|---:|---:|
| terminal | 死亡 | 121–131 | 0.75–0.80 | +0.34~+0.40 | +0.14~+0.19 |
| x0_sign | 高度越过中线 | 123–129 | 0.69–0.72 | +0.07~+0.09 | +0.16~+0.19 |
| x1_high | 快速上升（vy>q90） | 102–111 | 0.90–0.91 | +0.35~+0.38 | +0.39~+0.41 |
| x0_high | 接近上边界（y>q90） | 94–99 | 0.98 | +0.63~+0.65 | +0.27~+0.28 |
| x5_low | 远低于下一缺口中心 | 60–81 | 0.96–0.97 | +0.48~+0.53 | +0.26~+0.31 |
| x3_low | 远低于当前缺口中心 | 67–72 | 0.96 | +0.41~+0.45 | +0.27~+0.29 |

另有一个“大单帧位移”跳变事件家族（x0/x3/x5 jump，同族，每种子
经去重存活一个）通过五测：AUC 0.95–0.98、skill +0.25~+0.50。

**三类非 level 结论**：

- **动作回声**（x1_jump 拍翅脉冲、x1_extr 顶点/谷，×3 seeds）：
  状态可预测（AUC 0.79–0.82）但过去动作对比 ≈0（+0.01~+0.05）
  → 它们是决策的后果，不是状态结果；属于 option/策略层，
  不是 outcome 谓词。
- **状态协变量**（x1_low、x0_low、x1_sign、x0_extr 等，regime fail）：
  无 regime 改变的 band 穿越；可作入口条件，不是转移层。
- **阶段/世界周期边界**：结构重置（2 号维跳变）在被动流中
  **0–1 次/5000 帧**（101/102 为 0，100 为 1）→ support 拒绝；
  **阶段层在当前被动数据上不可学**，需要更长寿命的数据流
  （planner/DAgger 流或定向采集），不是帧级问题。

**转移结构已可算**（事件→事件，seed 101 例）：`x3_low → terminal`
中位 ~2 帧（n=13）；`x1_high → x0_high` ~1 帧（n=18）；
`x0_sign → x1_extr` ~3 帧（n=48）。这正是事件级早期预警需要的链。

**协议修正记录（全部无任务词汇）**：

- 时钟规则：时钟 = 单调计数器（增量近恒定且无跳变）；锯齿+重置
  坐标不是时钟；排除决策带 std/jump 比率入报告。
- 跳变阈值 = `(q50+max)/2`：两值增量分布下尾分位退化，
  中点规则分开两个众数。
- regime 窗口 = 4：上下文窗口必须小于典型局长（p50 34.5）。

### 首批实验

1. **事件/阶段 outcome audit（只读，第一优先）——已完成**：
   从 feedback/terminal/pain/reset 与观测分位生成候选结果谓词；拟合
   distributed-lag cause-specific hazard / stopping-time 模型；
   过五测（支持、动作条件性、非时钟、事件后 regime 改变、
   跨 episode 校准）；报告 event AUC/Brier、duration 分位数、
   action contrast、support/UNKNOWN。
   **口径自足**：B0 的“2/10 action contrast”是当前帧级候选集的性质，
   不用来 gate 事件/阶段层；事件层的对比按自己的 option/事件口径测。
2. **事件级转移 / 早期预警模型（P1-B）——已完成，三种子 pass**：

   **对象**：audit 稳定的 6 个 level（terminal + x0_sign / x1_high /
   x0_high / x5_low / x3_low）；位移跳变家族暂不入模（种子间去重存活的
   维不同，跨种子不可比）。设计：lead 分析（对事件密度期望覆盖率的
   lift）+ terminal hazard A0(state-only)/A1(+因果事件历史特征：
   since/count16/recent4)/A1p(局内打乱 placebo)，leave-episode-fold；
   next-event top-1（modal-successor 基线 vs state-only hazard
   argmax，同一 fold 划分）。

   **结果（被动流 100/101/102）**：

   - **强先导 4 个**（lift = 实测覆盖率/事件密度期望覆盖率）：
     `x1_high` lift 2.37–2.45（中位 lead 4–5）、`x0_high` lift
     3.69–3.76（中位 3，coverage 0.65–0.71）、`x5_low` lift 2.45–2.55
     （中位 4）、`x3_low` lift 2.37–2.57（中位 3–4）；
     `x0_sign` 是反先导（lift 0.27–0.41：恢复性事件，发生在脱险后）。
   - **事件历史特征带来增量信息**：terminal hazard
     A0(state-only) AUC 0.769/0.811/0.778 →
     A1(+事件历史) 0.850/0.837/0.891，
     **ΔAUC +0.082/+0.026/+0.113**（≥+0.02）；
     局内特征打乱 placebo A1p 0.764/0.809/0.783，
     **A1−placebo +0.086/+0.029/+0.108**（≥+0.01）；
     A1 skill +0.394/+0.431/+0.429 > 0。
   - **next-event top-1**：modal-successor 基线
     0.475/0.473/0.463（n=400–410，≈2.8× chance 0.167）；
     state-only hazard argmax 0.355/0.388/0.410（n≈1200，2.1–2.5×）。
   - **判定：3/3 pass。** 事件层带转移信息（不只是可读标记），
     4 个先导有 3–5 帧 lead，可作为早期预警/方向信号。
     失败判读未触发（ΔAUC≈0 的“状态已足够”假设被否定）。

   **非目标**：不进逐帧 cost；不做全局 veto；不改 8 帧默认栈；
   动作回声类（impulse/apex）与位移跳变家族不建 outcome 模型。
3. **阶段层数据审计（P1-C）——已完成**：planner/DAgger 流重跑同一
   audit（`--source planner`，episode ≥50000 的行）：

   - seed 100：63 局/5036 行（局长 ~80），过柱重置 **7 次** →
     support 拒绝；
   - seed 101：46 局/5148 行，过柱重置 **22 次/16 局** → 首次达
     support → 五测全过但**动作对比 ≈0（−0.001）→ boundary**；
     `x2_sign`（管道对准鸟）49 次/20 局 → 同为 boundary
     （AUC 0.910、skill +0.246）；
   - seed 102：68 局/5072 行，重置 **0 次**（dims 2/4 在该流仍是
     counter，连候选都没生成）。
   - **判定**：阶段/世界周期边界的可辨识性由**数据寿命**决定
     （~1 次/80 帧 × 存活超过 80 帧的局）；即便可学，动作对比 ≈0
     证明它是**外生 boundary**——只能当阶段协变量/入口条件，
     不是决策层。事件级 level 在两种流上都稳定。

4. **6 步 fine executor 基线 A/B（P0-F）——已完成**：benchmark 协议
   （每种子 × 每 horizon 独立 fit：1500 帧 + 5000 帧 DAgger，
   held-out 120–124，decision=search，death-rules 默认关）：

   | seed | h6 均值 | h8 均值 | Δframes | Δpasses | 方向 |
   |---|---:|---:|---:|---:|---|
   | 100 | 72.0 | 191.8 | **−119.8** | −1.40 | 0/5 改善（全部塌到 72 帧地板） |
   | 101 | 109.8 | 178.4 | **−68.6** | −1.00 | 1/3 改善 |
   | 102 | 104.2 | 88.4 | **+15.8** | +0.20 | 1/0 改善 |
   | 合并 | 95.3 | 152.9 | **−57.6** | −0.73 | 1/3 种子 |

   拍翅率几乎不变（100: 0.086 vs 0.081）——塌陷不是“不拍”，
   是**拍翅时机错了**：6 帧内看不到的死亡不可见。
   - **判定：6 步 executor 不能单独切换**。短视界的代价被量化
     （合并 −57.6 帧）；必须由事件级预警/方向层补偿后再评估组合，
     这正是 Batch 8 的动机。

5. **下一步 P1-D（后置设计）**：6 步 executor + 事件级预警的组合
   ——预警作为 opt-in 方向/约束信号（不进逐帧 cost），在 h6 上
   重跑本 A/B；只有组合 ≥ h8 基线才谈切换。router 仍后置。

### 实验 P1-D.0：触发验证（已完成，3/3 fail）

**设计**：用 deployable 预警模型（P1-B 的 A1 事件历史特征）在每个
6 步 executor 的评测决策点上评分（causal 在线事件状态 + 因果
动作 lags），标签 = `dies_within_8`（评测局在 8 帧内结束），
即 6 步视界看不到的死亡。

**两种来源**：

| 预警拟合源 | seed | AUC imminent | tau | trigger rate | precision@tau | caught | gate |
|---|---|---:|---:|---:|---:|---|---|
| 行为流 | 100 | 0.705 | 0.199 | 0.487 | 0.120 | 35/40 | fail |
| 行为流 | 101 | 0.677 | 0.164 | 0.062 | 0.485 | 16/40 | fail |
| 行为流 | 102 | 0.843 | 0.193 | 0.198 | 0.200 | 31/40 | fail |
| planner 流 | 100 | 0.812 | 0.317 | 0.042 | 0.400 | 10/40 | fail |
| planner 流 | 101 | 0.612 | 0.583 | 0.108 | 0.158 | 9/40 | fail |
| planner 流 | 102 | 0.607 | 0.349 | 0.057 | 0.133 | 6/40 | fail |

**判定：3/3 fail，两种拟合源均 fail。**
- 行为流：tau 失准（precision 0.50 → 0.12–0.49），触发率 5–49%
  不可控；AUC 0.68–0.84（部分种子过线，但 coverage fail）；
- planner 流：AUC 反而下降（0.81→0.61），precision@tau 也失败；
- **根因：行为→planner 分布位移**（B1-plan 同因）——预警模型
  在行为流上校准的 tau / 事件密度在 planner 自身的评测状态分布上
  不迁移；planner 流的评测 episode 是 DAgger 自己的（policy=eval
  policy），覆盖率/事件密度也不同。
- **结论：P1-D 暂停**。事件层预警有 within-distribution 信号
  （P1-B pass），但找不到可靠的消费者；**在事件层找到能通过
  planner 分布的触发/消费机制之前，6 步 executor 不切**。

### 实验：事件视窗跨度（`event_span_probe.py`，3 种子，已完成）

**问题**：P1-B 的 lead 只在 gap ≤ 8 内统计末次实例（构造上中位 3–5），
无法回答「事件层能否在帧级逃生窗口（rem 7–12）关闭之前发出信号」。
本探针换成首次触发 lead 分布 + l_max 8/12/16 的 span hazard 迁移 +
走廊入口可读性。

**三种子结果（planner DAgger 流，A1 = 状态 + 事件历史）**：

| 指标 | seed 100 | seed 101 | seed 102 |
|---|---:|---:|---:|
| span 8 AUC（A0 → A1） | 0.815 → 0.928 | 0.549 → 0.789 | 0.925 → 0.940 |
| span 12 AUC（A0 → A1） | 0.763 → 0.885 | 0.534 → 0.701 | 0.888 → 0.911 |
| span 16 AUC（A0 → A1） | 0.720 → 0.850 | 0.517 → 0.663 | 0.839 → 0.882 |
| 近死带 AUC span 12 | 0.804 | 0.676 | 0.818 |
| 首次 lead ≥ 8 覆盖死亡 | 0.06–0.17 | 0.09–0.20 | 0.00–0.40 |
| 规则 h=12 P / R / lift | 0.24 / 0.33 / 1.6 | 0.21 / 0.39 / 1.9 | 0.48 / 0.56 / 3.0 |
| 被动 tau → planner precision | 0.309 | 0.262 | 0.375 |
| 入口可读性（最佳单特征 AUC） | x2/x4 0.12 | ≤0.60 | x1 0.88 |

**结论**：
- **连续 span hazard（状态+事件历史，12–16 帧）在 2/3 种子达到可用
  精度**（AUC 0.85–0.91，近死带 0.78–0.82，skill 正），16 帧跨度仍有
  0.85–0.88；101 是弱种子（0.70/0.68，与 P1-D.0/B1-plan 同）。
- **事件历史是迁移稳定器**：101 的 state-only 在同流 span 12 掉到
  chance（0.534），+历史后 0.701（Δ+0.167）；100 Δ+0.123；102 状态
  已足够（Δ+0.023）。
- **离散先导事件不是 8–12 帧触发器**：首次 lead ≥ 8 只覆盖 6–40% 的
  死亡（x1_high 最好），规则预警 h=12 precision 0.21–0.48。
  → P1-D.1 的「事件触发→深搜」形态与证据不符。
- **校准仍不可迁移**：被动流 tau → planner precision 0.23–0.38
  （目标 0.5），与 P1-D.0 同因（基率位移）。
- **判定**：`span_model_works_trigger_form_wrong`——高层级预测的正确
  形态是**连续 span readout**（粗层 hazard，span 12–16，状态+事件
  历史），不是等某个事件实例触发；校准需在目标分布上重做。

### 实验：粗层 span 状态树（E1，`tree_transition_probe.py`，已完成）

**设计**：把 span probe 的外部 logistic 换成学习系统会拥有的对象——
两级状态树（fine k-means 节点 + coarse 父节点 = fine 中心的聚类）+
每节点停止时间直方图（step 1..16 / survive）+ 父级 trust 收缩
（α=4）。每次观测同时更新叶节点与祖先池化统计（兄弟信用）。只读
验证四个对照：外部 logistic 参照、被动→planner 迁移、planner 流
5 折（消费分布）、消融（state_only / fine_only / K 扫描）。

| seed | 参照 logistic AUC/band | 表·迁移 state+hist | 表·迁移 state_only | 表·planner 5 折 state+hist | 表·planner 5 折 state_only |
|---|---:|---:|---:|---:|---:|
| 100 | 0.885 / 0.804 | 0.692 / 0.541 | 0.876 / 0.671 | 0.888 / 0.735 | **0.958 / 0.866** |
| 101 | 0.701 / 0.676 | 0.732 / 0.631 | 0.668 / 0.578 | 0.757 / 0.616 | 0.797 / 0.660 |
| 102 | 0.911 / 0.818 | 0.619 / 0.468 | 0.653 / 0.508 | 0.902 / 0.797 | **0.941 / 0.853** |

**结论**：
- **消费分布内（planner 5 折）表对象成立**：AUC 0.76–0.96，band
  0.62–0.87，skill 全正；state+hist 的 precision@tau
  0.435/0.323/**0.532**，state-only 0.505/0.391/0.428——第一次
  接近/达到 0.5 校准目标（对照 P1-D.0 的离线 tau 迁移失败）。
- **迁移不稳**：分段常数节点的覆盖取决于被动聚类落在 planner 状态
  分布的哪里；102 崩溃（0.62–0.65 vs 参照 0.911），K 敏感。
- **事件历史作为表格维度是毒药**：3/3 都是 state-only 更好
  （100：0.958 vs 0.888；101：0.797 vs 0.757；102：0.941 vs 0.902）。
  历史的信息是真的（logistic 增量 +0.11~+0.17），但在 K=32 下
  15 维流相对特征把支撑度切碎。→ **state-only 非参数树 + 参数化
  低维历史修正**才是正确形态。
- **兄弟池化在当前规模惰性**：tree ≈ fine-only（0.888/0.888、
  0.757/0.753、0.902/0.904）。它应在低支撑/跨流池化/更深层级
  出现价值，E1 未覆盖该 regime。
- **判定**：预注册门 1/3 pass（state+hist 迁移未达 logistic），但
  「粗层作为树/表」的可行性在消费分布上成立；剩下的问题是 eval
  分布（h6）与跨流覆盖。

### 实验：消费分布 span 读数（E2，`span_consume_probe.py`，已完成）

**设计**：h6 模型（1500+5000，dagger_horizon=6）→ 取 DAgger 行
（episode ≥ 50000）建两级 state-only 表（K=32/Kc=4，停止时间直方图
16）+ 参数化 state+lags+事件历史 hazard；tau 一律来自**训练流的
episode 折外预测**（不碰评测标签），在 h6 评测流（世界 120–124，
逐帧 entry）上测 span 8/12/16。四臂：table/logistic × dagger/passive。

**span 12（预注册门）**：

| arm | 100 AUC/band/prec | 101 | 102 |
|---|---|---|---|
| table_dagger | 0.745 / 0.604 / 0.250 | 0.847 / 0.645 / 0.331 | 0.771 / 0.510 / 0.251 |
| table_passive | 0.414 / 0.251 / 0.228 | 0.868 / 0.645 / 0.173 | 0.772 / 0.615 / 0.137 |
| logistic_dagger | 0.457 / 0.549 / 0.000 | 0.826 / 0.698 / 0.317 | 0.538 / 0.429 / 0.227 |
| logistic_passive | 0.575 / 0.495 / 0.213 | 0.843 / 0.640 / 0.210 | 0.813 / 0.776 / 0.135 |

**span 16（可用工作点）**：

| arm | 100 | 101 | 102 |
|---|---|---|---|
| table_dagger AUC/band | 0.840 / 0.665 | 0.879 / 0.745 | 0.821 / 0.586 |
| table_dagger prec/cover/recall | 0.463 / 0.463 / 0.938 | 0.352 / 0.358 / 0.850 | 0.337 / 0.354 / 0.762 |
| table_passive prec/cover | 0.231 / 0.989 | 0.167 / 0.889 | 0.160 / 0.978 |

**结论**：
- 预注册门（span 12：precision 0.5、band 0.70）**3/3 fail**。
- **span 16 是第一个跨世界可用的读数**：state-only 表 AUC 0.82–0.88、
  lift 2.0–2.4×、召回 0.76–0.94；precision 差目标 0.04–0.16。
- **消费分布训练明确加分**：span 16 上 dagger 训练 precision
  0.34–0.46 vs 被动 0.16–0.23；被动臂 coverage 0.89–0.99 退化为
  基率检测器（101/102 band 高但 flags 几乎全部 entry）。
- **param 臂跨世界不稳**（logistic_dagger 0.45–0.83，seed 100 tau
  一个死亡都不命中）→ 非参数表是更稳的形态（与 E1 一致）。
- **支撑度是可靠弃权信号**（support AUC 0.75/0.83/0.92）；distance
  较弱。低支撑占比 0（support ≤5 未出现）→ 实用门应是百分位门。
- **瓶颈不是世界数量/数据量**：被动流 128 局、DAgger 62 局，**每局
  都是新环境实例**（`seed*10000 + episode`，gap 序列不同）；评测
  世界 120–124 只是同分布另 5 个抽样。真正的位移是**采集策略配置**：
  `collect_planner` 调 `evaluate` 时未关 curiosity（训练权重 0.5），
  评测一律 `_curiosity_off` → E2 的「消费分布」是探索策略，不是
  部署配置。次要因素：评测仅 5 局、span 16 约 100 正样本
  （precision 的 95% CI 约 ±0.09），门 0.5 是边界而非决定性失败。
- 可选修法（按代价排序）：① **部署配置再采一条流**（curiosity off、
  同 horizon/decision，5000 帧）重训读数——最小且直接检验配置位移
  假设；② 更细/更结构化的节点（K/特征），解决多世界池化稀释；
  ③ 在线重校准（决策侧，需基准）；④ 世界无关特征（研究性）。

### 非 P1-D 的下一步方向

- **P1-D 关闭后，Batch 8 的残余价值**：事件级 level（6 个）
  和转移结构（`x3_low → terminal ~2 帧` 等）仍然真实；
  问题在消费端，不在信号端。
- 可能的后续方向（不承诺）：
  1. 在线自校准：预警模型在评测过程中逐步重校准
     （online recalibration），不做离线 tau；
  2. 把预警当作 route 的替代信号而非触发器——但这是 router
     领域，已后置；
  3. 保持 8 步默认栈不变，等采集更长的 planner 流后再评；
  4. **连续 span readout（`event_span_probe` 证据支持）**：
     12–16 帧 span hazard（状态+事件历史）在 2/3 种子迁移 AUC
     0.85–0.91、近死带 0.78–0.82，但离散事件触发太晚（lead≥8
     覆盖 6–40%）。若接线，形态应是**连续 readout + 目标分布校准**，
     不是 P1-D.1 的「事件触发→深搜」；决策侧改动仍需 3 种子配对
     基准。
  5. **E2 粗层接线（已完成读数验证）**：h6 DAgger 训练 + 折外校准，
     在 h6 评测流上 span 12 未达门（precision 0.25–0.33），但
     **span 16 的 state-only 表跨世界可用**（AUC 0.82–0.88、lift
     2.0–2.4×、召回 0.76–0.94、precision 0.34–0.46），且消费分布
     训练明确优于被动（被动退化为基率检测器）。支撑度可作
     abstention 门（AUC 0.75–0.92）。世界数量不是瓶颈（每局都是
     新环境实例）；真正的位移是 DAgger 采集 curiosity-on vs 评测
     curiosity-off。下一步：按部署配置再采一条 5000 帧流重训读数，
     再决定是否接线。

### 非目标

- 8 帧调试线已关闭：不再加帧级探针/补丁，不做帧级 endpoint
  re-ranker（原 P0-E）；
- 不继续调 8 步帧级 endpoint / risk 权重；
- 不把 8 步帧级 endpoint drift 当作事件/阶段级设计的反证；
- 不把 6 步或 8 步当“层”的定义；层由 outcome predicate 定义；
- 不用 context 簇/stage 身份替代 outcome predicate；
- 粗层不得覆盖有效 fine，不做全局逐帧 cost/veto。



## 历史状态（2026-11-21，只读）

> 本节是 2026-11-21 的默认栈与瓶颈快照，**已被顶部 2026-11-23
> 总结和 Batch 7 取代**；保留作对照，不再更新。当前基准与默认栈
> 以文首和 README 为准。

**默认栈（未变）**：点事件 pain；默认目标（NNLS 组合）；好奇 0.5
（评测 0）；成对特征关；动作边 Q 读关；death-rules /
branch-transitions / memory-credit / VOI / 生成先验均默认关。
基准 = 1500 帧 ×1 遍 + 5000 帧在线自采 + 重训，评测 cap 5000，
held-out 世界 120–124；基线 129.1 / 147（3 种子）。

**机制裁决速查**

| 机制 | 状态 | 关键读数 |
|---|---|---|
| 成对坐标差 `--pairwise-features` | 默认关 | 6 种子 +3.7，4/6，null |
| 覆盖率 / 跨世界门槛 | 默认关 | 6 种子 null；5000 帧复测 null |
| 记忆回溯 `--memory-credit` | 默认关 | −6.5（首轮） |
| VOI v1 / v2 | 方向关闭 | 双负 |
| 生成先验 | 不宣称 | 单流 8/8 → 3 种子 +6.9（11/24） |
| 动作边 Q（平均 / 目标条件） | 默认权重 0 | −53.3 / −35.3（0/3） |
| 0/1 死亡硬否决 `--death-rules` | opt-in | −13.2（2/3），seed 101 回退 |
| 多分支 / 预测通道 | 行为中性 / 诊断 | 决策逐位不变 |
| 好奇 0.5 | 默认开，但现役口径**结构性惰性** | `epochs=1` → 格子空 → Φ≡1 → 项恒 0 |
| 记忆层 | 只增不减、检索 4% | 见 README「训练内省」 |

**当前瓶颈（按证据排序）**

1. 抽象层会预测危险（两臂 rho 0.76–0.87），瓶颈在**决策侧**；
2. 死亡侧在候选计划上 precision 96%，**召回 / 首死步**是瓶颈；
3. 决策相关性 **seed-dependent**：seed 100/102
   `decision_relevant_misses = 0`；**seed 101 是 score misrank**
   （oracle-best 计划 10/10 feasible，却被 hazard/value score 排后）；
4. 剩余 gap 主要是**模型预测状态漂移**，不是规则覆盖不足；
5. `epochs=1` 下特征动作格 / 价值格 / 逐行账本全空（三条通路断开）。

## 待办

> **里程碑方向的两条 P0 待办（存活 / 相位数据）见文首「里程碑方向：当前状态与
> 待办」**；本节及以下为**主线（默认栈）**的既有待办。

### A. Batch 4 尾项：决策侧 score

- [x] **score component 诊断**：在 seed 101 的 10 个 decision-relevant
      miss 上逐列消融（risk / value / curiosity），定位把存活计划排后的
      那一列；B0.5：8 个 plan-level miss 只有 risk 活跃，
      value/curiosity/q 全为 0；0/8 能被单列或列组合修复。
      B0.6：risk 的 step 部分为 0，主要是 endpoint terminal hazard；
      预测 endpoint 反向（8/8），真实 terminal 7/8 正确 → world
      endpoint 漂移。逐列/权重方向关闭；
- [ ] **score calibration / 保守下界**：只在分数边际小或 support 低时
      改变排序，避免全局重标定；
- [ ] **plan-level endpoint 校正/重排**：用 planner 的 predicted endpoint
      + 真实 terminal 标签训练/校准一个小读数，episode-held-out +
      共形区间 + OOD 门；只在 plan/suffix 层面重排，不改首动作，
      不加逐帧项；B1-plan 首轮（seed 101）：A1 endpoint-only
      same-AUC 0.789、B0 miss 9/10，ranking 通过；但 Brier skill
      −0.092、within-Brier skill −0.793，calibration 失败。3seed
      （100/101/102）：seed 101 68/69，但 seed 100 从 62/62 退化到
      60/62，seed 102 terminal base rate 0.000、模型退化；
      **非退化门失败，不接线**。只剩 OOD/UNKNOWN 门后的 rank-only
      复测或 world endpoint 改善；OOD 只做排序后门控，不做特征；
- [ ] **早期预警 / 时域 / 候选集**：seed 100/102 的晚期状态在 H=8 内
      所有候选都死，需要更早行动或更长时域/更好的候选生成；
- [ ] 决策侧改动准入：真实基准 3 种子配对（fast 屏不作为证据）。

### B. 新方向：记忆树 / 阶段图 / 多尺度预测 / 粗到细决策（2026-11-21 用户方向）

> **状态：历史设计稿**。当前实现与结论以 Batch 7、B0–B1-plan 为准；
> 其中 stage/context 聚类、固定层次和 action-level coarse router
> 已被 B0/B0.5/B0.6 的数据否定或降级，保留作设计参考。

**目标**：让系统自己发现事件的层级与关键事件，把"阶段"作为记忆树的一层，
在相邻阶段间学习转移；决策时先预测长期转移（深度由系统判断），确定方向
后再在约束下细化到原子单步；并加入"转移结果状态导向的吸引力"作为启发式。

**与现有代码的接线**

| 能力 | 现有 | 缺 |
|---|---|---|
| 事件 / 阶段 | `hierarchical_memory.py`：L0 帧 → L1 事件（salience）→ L2 阶段 → L3 世界 | 每阶段局部转移统计 `θ_i`（按动作 Δ / 残差 / support）、阶段转移边 |
| 层级合并 | `memory_tree.py`：context 距离 + 关键词/残差概括 | 合并准则加入转移结果一致性 / 预测误差 |
| 上下文 | `context_model.py`：regime 聚类 + 在线推断 | 作为阶段入口条件 |
| 转移统计 | `branches.py`：current / recent / prediction | 阶段级 sufficient statistics + residual monitor |
| 搜索 | `trajectory_search.exhaustive_search`（death / unknown 钩子、candidate trace） | 粗阶段搜索器 + fine 阶段的目标吸引力列 |
| 决策 | `policy_result` 5 列目标 | 第 6 列（吸引力）或 coarse 剪枝 + fine |

**核心算法**

- **阶段** = 一段能被同一个局部转移模型解释的连续转移；边界 = **关键事件**
  （不是固定时长）。Split：`err(单模型) − err(左右两模型) > MDL/BIC 惩罚`
  且边界有 support/salience；Merge：相邻 θ 置信区间不可区分且结果分布
  一致；**层级 = 误差下降最显著的模型层**（L1 动力学 / L2 阶段 / L3 regime）。
- **阶段转移图**：`(stage, action) → (next stage | outcome, duration_dist)`，
  带 probability / support / calibration_error / applicability（entry/exit）。
  数据只来自实际执行过的转移。
- **预测契约**：`PredictionRequest(target, concept_level,
  required_resolution, required_confidence, current_stage)`；
  `PredictionRecord(target, duration, uncertainty, support,
  calibration_error, validity)`。入口条件不满足 → `UNKNOWN`/abstain；
  **粗预测不覆盖细预测**。
- **粗到细搜索**：粗层在阶段图上有界搜索（深度由 support/不确定性决定），
  评分 = 粗结局（dead-end 概率 / 期望回报 / 阶段危险）→ 目标阶段序列；
  细层在目标阶段区域约束下跑现有 H 步原子搜索。
- **吸引力（任务无关、PBRS 势差）**：

  ```text
  Φ_target(s) = −d(s, 目标阶段区域)       目标区域 = 阶段记忆叶的均值/散布
  attraction  = λ · (γ Φ_target(s') − Φ_target(s))
  ```

  只在 fine 阶段进入目标走廊后生效；support 低时 λ→0。
  势差形式保证与未塑形目标同一最优策略。

**分片与验收（默认关，逐片 A/B）**

- [x] **片 1**：`stage_graph.py` schema（纯数据类）+ 离线分段器
      （split/merge，合并用 next-outcome 分布 ΔBIC + 期望计数门，
      从 1 阶段起步）+ 合成切换动力学测试；不接 planner。
      **结果见下“片 1 实测”**：分段器跑通，但真实流上几乎不产生
      action-relevant 边界，片 2 保持关闭。
- [ ] **片 2**：阶段转移图 + 预测契约 + 离线 router 探针
      （fine-only / coarse-only / coarse-to-fine）。
- [ ] **片 3**：粗到细搜索 + 吸引力列（`--stage-graph` opt-in；
      关闭时逐位不变）。
- [ ] **片 4**：质量门通过后接 `policy_result`，真实基准 3 种子配对 A/B。

**门槛 0 探针读数（2026-11-21，`stage_support_probe.py`，只读）**

数据：真实 `fit` 路径捕获的两条流——1500 帧 / 38 episodes，6536 帧 /
101 episodes（含 5000 帧 DAgger 实际执行行）。分段 = episode 内贪心 BIC
（每段每动作一个 mean-Δ 模型，min_length=5；动作取 **history[lag]**，
即真正掌管该转移的 acting 动作）；阶段簇 = 段均值向量的单链路聚类
（半径 = 中位两两距离）。所有计数单位为**帧**（`count_unit`）。

| 指标 | 1500 帧 | 6536 帧 |
|---|---:|---:|
| 段数 / 每 episode 阶段数（均值, max） | 115 / 3.03, 8 | 269 / 2.66, 8 |
| 阶段长度 p50 / p90 / max | 8 / 27 / 59 | 9 / 67 / 119 |
| split 次数 / BIC gain p50 / min | 77 / 9.81 / 0.226 | 168 / 14.74 / 0.226 |
| 阶段簇数 | **1** | **6** |
| (stage, action) 格 / support 中位 / ≥5 | 2 / 750 / 2 | 12 / 4 / 4 |
| (stage, action, next) 边 / 中位 / ≥5 | 4 / 274 / 4 | 24 / 9 / 15 |
| **informative 边 ≥5（非自环非 END）/ 跨≥3 ep** | **0 / 0** | **11 / 3** |
| top-4 边占帧比 | 100% | 88% |

**置换对照**（同一流，只改边界生成或动作标签）：

| 臂 | 1500：簇/边/≥5 informative/top4 | 6536：簇/边/≥5 informative/top4 |
|---|---|---|
| 真实 BIC | 1 / 4 / 0 / 100% | 6 / 24 / 11 / 88% |
| 动作打乱（acting 标签） | 2 / 8 / 1 / 99% | 7 / 23 / 9 / 98% |
| 随机边界（保留段数） | 1 / 4 / 0 / 100% | 1 / 4 / 0 / 100% |

读法（修正上一版“退化”结论）：

1. **真实分段优于随机边界**：6536 帧上真实流有 6 簇、11 条 informative 边
   （3 条跨≥3 episode），随机边界只有 1 簇、0 条 → BIC 边界确实抓到了
   结构，不是估计器构造；
2. **但动作条件化很弱**：动作打乱后仍得到 7 簇 / 9 条 informative 边，
   与真实几乎持平 → 当前一帧 Δ 的局部模型**主要解释状态漂移，而不是
   动作效应**。若要做阶段转移，局部模型应改为 **acting 动作 + 多步
   Δ/轨迹段**，而不是一帧差分；
3. **1500 帧不可用**：1 簇、0 条 informative 边 → 离线腿不支持阶段图；
   6536 帧只有 3 条跨 episode 的 informative 边，**必须把“跨≥3 episode
   的 informative 边数”作为第一门槛**；
4. **集中度是结构性的**：所有臂 top-4 都 88–100%，只是真实流略低
   （88%）→ “主阶段自环 + END”主要是短 episode + 少量簇的结果，不能
   单独当作任务无阶段结构的证据；
5. 支持度仍偏斜：6536 帧 4/12 格、15/24 边 ≥5；长尾必须 abstain。

局限：探针判据只是门槛 0（贪心 BIC + Δ 均值单链路），不是最终设计；
阶段身份仍需转移等价合并；动作条件化弱是本轮最重要的负面证据。

**动作条件化 LOEO 判据（2026-11-21，`action_conditioning_probe.py`，只读）**

簇/边计数不是检验动作效应的正确统计量。改用留一 episode 的预测增益：
A = 仅状态；B = 状态 + acting 动作 one-hot（窗口 k 步）；ΔR² = R²(B) − R²(A)，
零假设 = episode 内打乱 acting 标签（200 次）。可检测门槛 ≈ 4/N_ep
（1500 帧 0.105；6536 帧 0.040）。

| k | 1500 帧 ΔR²（null 均值, p） | 6536 帧 ΔR²（null 均值, p） | R² 状态→状态+动作（1500 / 6536） |
|---|---:|---:|---:|
| 1 | **0.179**（0.000, 0.005） | **0.203**（0.019, 0.005） | 0.53→0.71 / 0.34→0.54 |
| 2 | **0.188**（−0.020, 0.005） | **0.205**（0.040, 0.005） | 0.49→0.68 / −0.63→−0.43 |
| 3 | 0.142（数值退化） | −0.646（p 0.40） | 0.39→0.53 / −3.4→−4.1 |

判定：

1. **动作效应是可检测的**：k=1/2 的 ΔR² 0.18–0.21，远高于零假设与可检测
   门槛（p=0.005）。上一版“动作条件化很弱”是**簇/边统计量错误**，不是数据
   没有动作信息；
2. **原始拟合的 k=2/3 异常是数值问题，不是数据**（见下条诊断）：未标准化
   的 one-hot + 状态使条件数达 1e16–1e17，alpha=1e-3 下个别折外推爆炸；
   稳健复算后所有 k 的 pooled R² 都为正且动作增益更大；
3. **分段的失败在判据而不在模型**：贪心 BIC 最大化的原始 Δ 方差主要由
   漂移解释，所以边界落在漂移上。要做动作相关的阶段，分段目标应改为
   **动作条件残差的 ΔR² gain**（或对残差做变点检测），而不是原始 Δ 的 BIC；
4. 结论：局部模型用 **k=1 acting 对齐 + 逐折标准化/alpha=1.0**；阶段图
   仍受“跨 episode informative 边只有 3 条”的限制，不能直接接 planner。

**数值诊断与稳健复算（2026-11-21，`action_conditioning_diagnostics.py`）**

原始拟合的异常定位：设计矩阵条件数 1e8–1e17（one-hot 动作与 alive 积分
共线）、alpha=1e-3、目标含零方差维；个别折的 OOF 预测有限但量级爆炸
（每折 R² 最低 −1e11），把 pooled 均值拖成负数。

稳健复算（逐折标准化特征/目标、丢弃零方差维、alpha=1.0）：

| k | 1500：pooled R² 状态→状态+动作（ΔR²） | 6536：pooled R² 状态→状态+动作（ΔR²） |
|---|---:|---:|
| 1 | 0.459 → 0.668（**+0.209**） | 0.392 → 0.609（**+0.217**） |
| 2 | 0.441 → 0.669（**+0.228**） | 0.404 → 0.638（**+0.234**） |
| 3 | 0.426 → 0.669（**+0.243**） | 0.389 → 0.652（**+0.263**） |

判定修正：

- **k=2 的负绝对 R²、k=3 的“退化”都是实现问题**（条件数 + 正则化），
  不是“多步窗口不可用”；标准化后 k=3 的 pooled R² 也有 0.65；
- “绝对 R² > 0”的验收门必须建立在**标准化拟合**上，否则会把数值问题
  误判为机制问题；
- 逐折 R² 在短 episode / 低方差折上不可用（分母趋零），验收只用
  **pooled R²**（必要时加自举区间），不用逐折均值/中位；
- k=1 仍作为默认（最简单、ΔR² 与 k=2/3 只差 0.01–0.05），k=2 作敏感性。

**吸引力静态判死（2026-11-21，`n_tie_probe.py`，只读）**

在真实 search 的 candidate trace 上，按同一套证据格重算端点
`Φ(s_H)`，统计“分数并列但 Φ 不同”的对数与可翻转决策数（吸引力偏好
更高 Φ；翻转 = 该偏好在并列集内会选到另一个计划）：

| 臂 | 决策 | Φ 不同值/决策 | 精确并列对 / 不同 Φ / 翻转 | 近似并列（±0.01）/ 不同 Φ / 翻转 |
|---|---:|---:|---:|---:|
| epochs=1（现役） | 959 | 1.00 | 319,051 / **0** / **0** | 406,839 / **0** / **0** |
| epochs=2（回拟合） | 363 | 4.06 | 257,240 / 187,905 / **26** | 445,303 / 321,308 / **37** |

判定：

- **现役口径下吸引力确定性惰性**（Φ≡1 → 0 可区分对、0 翻转），不需要
  A/B；并列本身很多（平均 ~333 精确并列对/决策），所以不是“没有并列
  可破”，而是 Φ 没有信息；
- **只有回拟合臂让吸引力有信号**：7–10% 的决策存在会被翻转的并列
  （26/363 精确、37/363 近似），Φ 每决策 4.06 个值、均值范围 0.386；
- 结论：**顺序应是先接回证据通道（特征/价值格），再谈吸引力**；
  吸引力本身可用“搜索后独立 tie-break”实现，不碰 6 列 score。

**接口约束（已核实）**：`policy_result` 的 score 被测试钉死为 6 列且
`score[0] == score[2]+score[3]+score[4]+score[5]`
（`tests/test_node_runtime.py:2163-2165`, `2188-2190`），另有硬编码
`np.zeros((len(states), 6))`（`substrate_nodes.py:1083`, `1166`）；
账本只读 `score[2]/score[3]`（`:1391-1395`）。→ 吸引力**不能直接加第
7 列**；候选落法：折进 cost（需同步账本恒等式与测试）/ 搜索后独立
tie-break / 只在粗层阶段评分中使用。

**探针结论对设计的修正**

- **阶段图不能在离线腿上建**：1500 帧只有 1 簇、0 条 informative 边；
  6536 帧才出现 6 簇 / 11 条 informative 边，其中跨≥3 episode 的只有
  3 条。第一门槛 = **跨≥3 episode 的 informative 边数**；
- **动作条件化是真实可检测的**（LOEO ΔR² 0.18–0.21，p=0.005，远高于
  可检测门槛）；此前的“弱”来自簇/边统计量，统计量本身是错的；
- **分段的失败在判据**：贪心 BIC 的原始 Δ 由漂移主导，边界落在漂移上。
  做动作相关的阶段，分段目标应改为**动作条件残差的 ΔR² gain**（或残差
  变点），而不是原始 Δ 的 BIC；
- **局部模型用 k=1（acting 对齐）**；k=2 只在关联意义可用，k=3 数值退化；
- **阶段图暂不接 planner**：跨 episode 的 informative 边只有 3 条，
  第一门槛仍是“跨≥3 episode 的 informative 边数”；
- **真实分段优于随机边界**（6 簇/11 边 vs 1 簇/0 边）→ BIC 结构是真的，
  问题在动作分辨力与跨 episode 复现，不是“估计器纯构造”；
- **粗层定义**：优先“转移结果导向”（dead-end / 存活段 / 阶段结果）；
  若坚持阶段簇，先过“动作打乱对照”与“跨≥3 episode”两道门；
- **合并判据**：转移等价（next-outcome 分布 ΔBIC + 期望计数门），
  从 1 阶段起步，全合并为 1 是合法输出；
- **吸引力排在“接回证据通道”之后**，用搜索后独立 tie-break 实现。

**圆桌结论（2026-11-21，5 轮：theorist / architect / verifier）**

共识：

1. **动作条件化真实可检测**：LOEO k=1 的 ΔR² 0.179 / 0.203（null 0.000 /
   0.019，p=0.005），门槛 0.105 / 0.040；此前的“弱”来自簇/边统计量，
   统计量本身错误。
2. **暂不建阶段转移图、不接 planner**：新分段上的“跨≥3 episode 的
   informative 边数”是唯一开闸数（当前 6536 帧仅 3 条）。
3. **局部模型 k=1（acting 对齐）为主**（绝对 R² 0.53→0.71 / 0.34→0.54）；
   k=2 作敏感性；k=3 在原始拟合上数值退化，但稳健复算后 pooled R² 0.65
   ——“退化”是实现问题（见“数值诊断”），不是多步不可用。
4. **分段判据改为动作条件残差 / ΔR² gain**（保留 BIC 型 split 惩罚）；
   阶段身份用转移等价合并；**单阶段必须是合法输出**。
5. **吸引力**：现役 e1 惰性（N_tie 0 翻转）；先接回证据通道，再以
   “搜索后独立 tie-break”实现；N_tie 只判死 curiosity 的 Φ，Φ_target
   依赖未可辨识的阶段叶，尚未被覆盖。
6. **粗层优先结果导向**（dead-end / 存活段），复用 first-death /
   `action_best`；不新增第二套死亡语义。

分歧（点名）：

- **验收门**：architect 要求 `ΔR² > 4/N_ep` **且绝对 R²_B > 0**（否则
  6536 帧 k=2 这种“两边都比均值差”的增益会被放行）；verifier 同意
  R²≥0，但门槛取 0 而非正下限，并指出 4/N 是经验值而非推导值。
- **片 1 范围**：architect 主张只替换分段代价函数（不把 stage 写进
  MemoryTree）；verifier 主张 k=1 分段器可做，但片 2 保持关闭——
  ΔR² 只证关联，不证残差可分段。
- **theorist**：簇数是自适应半径的函数，不能作为裁定依据；LOEO 的
  null 还应打乱**窗口内动作顺序**（两种零假设不同）。

待确认（开工前）：

- k=3 的“数值退化”形态（NaN / inf / 极端负值？该折是否被剔除）；
- 6536 帧 k=2 绝对 R² 为负是尺度/窗口实现问题还是真实不泛化；
- 新分段器在 6536 帧的 cross3 边数（片 2 唯一开闸数）；
- k=1 下每个 (stage, action) 的 episode 级 support 是否 ≥3；
- 绝对 OOF R² 门槛取 0 还是正下限（如 0.1）。

**片 1 最小范围（圆桌修订版）**

- [ ] `stage_graph.py` schema（纯数据类，不接 planner）；
- [ ] 动作残差分段器：k=1、acting 对齐、ΔR² gain 代价 + BIC 型惩罚；
- [ ] 硬门：常数流必须返回 1 阶段；绝对 OOF R² ≥ 0 才允许残差分段；
      输出 cross3（跨≥3 episode 的 informative 边）计数；
- [ ] 合成测试：分段切换、常数流、动作打乱对照；
- [ ] **不接 planner、不写入 MemoryTree、不改默认路径**。

**片 1 实测（2026-11-21，`stage_graph.py`，纯数据层）**

实现：`Stage` / `StageTransition` / `StageGraph` schema；离线分段器
（k=1、acting 对齐、状态+动作 ridge 残差、BIC 型惩罚）；拟合门 =
逐折标准化 + alpha=1.0 + 丢零方差维的 pooled OOF R²（<0 则 abstain，
每 episode 单阶段）；阶段身份 = 动作效应向量单链路聚类（半径 = 正距离
中位数的 0.5）；输出 `cross_episode_informative`（非自环非 END、
frames≥5、episodes≥3）。动作打乱对照内置。282 测试通过。

真实流结果（缓存流，seed 100；`rounds=3` 四条流）：

| 流 | 门 R² | 阶段 | 簇 | cross3 real | cross3 shuffle |
|---|---:|---:|---:|---:|---:|
| 1500 帧 / 38 ep | 0.751 | 38 | 3 | **0** | 0 |
| 6536 帧 / 101 ep | 0.609 | 108 | 12 | 12 | 21 |
| 11645 帧 / 139 ep | 0.543 | 179 | 32 | **58** | **0** |
| 16673 帧 / 162 ep | 0.482 | 224 | 43 | **0** | 0 |

轻量逐动作模型的同期读数（informative≥5：real / shuffle / random）：
0/1/0 · 11/9/0 · 22/17/8 · 19/16/14。

判定：

1. **1500 帧无 BIC 支持的 action-relevant 边界**：状态+动作模型解释了
   大部分方差（70 个参数的 BIC 惩罚 ≈ 260/episode），残差里没有可分
   结构 → 每 episode 单阶段，cross3=0；
2. **6536 帧的 12 条 cross3 过不了打乱对照**（21 条）；
3. **11645 帧出现强信号**：real cross3=58 vs shuffle=0——这是首个
   “真实远超打乱”的读数，说明动作特异、跨 episode 复发的结构确实存在；
4. **16673 帧双方归零**：簇数从 32 涨到 43，边被切碎，达不到
   frames≥5 / episodes≥3 的门 → **cross3 对阶段词表规模不可比**；
5. 结论：**瓶颈是阶段词表控制**（固定 K 或转移等价合并 + support 下限），
   不是数据量或动作信号。片 2 在词表受控前保持关闭；受控后若
   real > shuffle 稳定，再开。

**词表受控后的判定（2026-11-21，`--sweep-k`，k-means 固定 K）**

同 K 下 real vs shuffle 的 cross3（每条流 × K ∈ {2,3,4,6,8,12}）：

| 流 | K=2 | K=4 | K=8 | K=12 |
|---|---|---|---|---|
| 1500 | 0 / 1 | 0 / 2 | 0 / 2 | 0 / 3 |
| 6536 | 0 / 3 | 2 / 9 | 4 / 16 | 4 / 18 |
| 11645 | 3 / 3 | 7 / 6 | 16 / 17 | 18 / 18 |
| 16673 | 3 / 3 | 9 / 9 | 19 / 18 | 22 / 22 |

**判定：同 K 下 real ≈ shuffle（±2），每条流、每个 K 都成立。**
之前 single-link 的 58 vs 0 是**词表粒度伪影**（real 32 簇 vs shuffle 23 簇），
不是动作特异结构。阶段图在现役机制下**关闭**（不开片 2）；LOEO 的动作
信号仍然成立，但它没有转化成跨 episode 的转移结构。

**结果导向粗层探针（2026-11-21，`outcome_coarse_probe.py`，只读）**

定义：粗状态 = time-to-terminal 桶（0 终端 / 1: 1–5 帧 / 2: 6–15 /
3: 16+），只用 episode boundary；对照 = 动作打乱、随机桶。

6536 帧结果：

| 臂 | (bucket,action) 格 ≥5 | 转移 | informative | cross3 |
|---|---:|---:|---:|---:|
| 真实 | 8/8 | 14 | 6 | 6 |
| 动作打乱 | 8/8 | 14 | 6 | **6（完全相同）** |
| 随机桶 | 8/8 | 37 | 24 | 20 |

桶内动作判别：bucket 1（最后 5 帧）a1 比 a0 多 **+0.47 帧**；bucket 2
+0.32；bucket 3（16+）**−2.37 帧**（受状态混杂：拍翅多发生在危险时）。

判定：**时间/结果桶可估但不可判别动作**（打乱后完全不变）——它是时钟，
不是状态抽象；随机桶对照的 cross3 更高，再次说明 cross3 不能单独作门槛。
结果导向粗层的有效部分退化为现有 hazard / death-side 机制，不构成新的
方向。

**证据通道 A/B（2026-11-21，epochs=2 × 3 种子，pairwise 关）**

| 臂 | 100 | 101 | 102 | 均值 | 中位 |
|---|---:|---:|---:|---:|---:|
| 基线（epochs=1，no-pairwise） | 191.8 | 178.4 | 88.4 | **152.9** | 152 |
| epochs=2（回拟合） | 72.6 | 121.0 | 132.6 | **108.7** | 72 |
| paired Δ | −119.2 | −57.4 | +44.2 | **−44.1** | 1/3 正向 |

判定：**不采用 epochs=2**。结合内省读数（价值项 −0.36、好奇心 ±0.5 vs
累加危险 ≈0.12），问题在**目标权重失衡**，不是“通路没接”。下一步应做
价值项/好奇心项的份额上限或 P4 纳入价值列的 A/B。

**权重隔离第一刀（2026-11-21，epochs=2 + value_weight=0，3 种子）**

| 臂 | 100 | 101 | 102 | 均值 | 中位 |
|---|---:|---:|---:|---:|---:|
| epochs=2 基线 | 72.6 | 121.0 | 132.6 | 108.7 | 72 |
| + value_weight=0 | 101.8 | 122.2 | 105.0 | **109.7** | 72 |
| paired Δ | +29.2 | +1.2 | −27.6 | **+0.9** | 1/3 |

**判定：杀掉价值项没有恢复生存 → 价值项不是 e2 崩塌的主因。**
下一刀：`curiosity=0`（同样 epochs=2、value_weight=0），隔离好奇心项；
若仍不恢复，则指向“第二遍原型 MC 学习 / 数据分布”而不是目标列。

**权重隔离第二刀（2026-11-21，epochs=2 + value_weight=0 + curiosity=0）**

| 臂 | 100 | 101 | 102 | 均值 | 中位 |
|---|---:|---:|---:|---:|---:|
| epochs=2 基线 | 72.6 | 121.0 | 132.6 | 108.7 | 72 |
| + value_weight=0 | 101.8 | 122.2 | 105.0 | 109.7 | 72 |
| + value_weight=0 + curiosity=0 | 85.2 | 124.2 | 62.4 | **90.6** | 72 |
| paired（vs e2 基线） | +12.6 | +3.2 | −70.2 | **−18.1** | 1/3 |

**判定：价值项与好奇心项都被排除**（关掉后不恢复，反而更差）。剩余嫌疑
收敛到**第二遍原型 MC 学习 / 采集分布**。

**特征回拟合 pass（2026-11-21，已实现）**

- `NodeRuntime.replay_features(rows)`：只更新特征动作/痛格、价值格与逐行
  预测账本；**原型、动作边、记忆与其计数保持单遍统计**；
- `_train(feature_backfit=True)`：epoch 0 正常学习 + 归纳，epoch ≥1 改走
  `replay_features`；CLI `--feature-backfit`；
- 285 测试通过（新增 `test_replay_features_updates_cells_without_prototypes`）；
- A/B：epochs=2 + `--feature-backfit`（默认权重），3 种子已完成
  （见下）——若恢复到基线 152.9 附近，则“e2 崩塌”归因于原型层双遍，
  特征通道本身可用。

**特征回拟合 A/B 第一波（2026-11-21，seed 100–102，3×5 held-out）**

| 臂 | 100 | 101 | 102 | 均值 | 中位 |
|---|---:|---:|---:|---:|---:|
| e1 基线（现役） | 191.8 | 178.4 | 88.4 | **152.9** | 152 |
| e2 全量第二遍 | 72.6 | 121.0 | 132.6 | 108.7 | 72 |
| e2 + 特征回拟合 | 98.0 | 168.6 | 110.2 | **125.6** | **135** |
| paired（vs e2） | +25.4 | +47.6 | −22.4 | **+16.9** | 2/3 |
| paired（vs e1） | −93.8 | −9.8 | +21.8 | **−27.3** | 1/3 |

判定（第一波）：

1. **原型层双遍确实是 e2 崩塌的主要贡献之一**：改回特征-only 后中位
   72→135、均值 +16.9；2/3 正向；
2. **但尚未回到 e1 基线**（−27.3，1/3）；3 种子 SE ≈ 33，方向性而非显著；
3. 需要第二波（seed 103–105）确认；对照用远端 6 种子表的 no-pairwise
   第二波 `[120.0, 120.2, 107.4]`（均值 115.9）。

**第二波与 6 种子裁定（2026-11-21）**

第二波（103–105）：特征回拟合 **168.2 / 72.0 / 88.0**（均值 109.4，
中位 72）；对照 no-pairwise 第二波 **120.0 / 120.2 / 107.4**（115.9）。

6 种子合并：

| 臂 | 100 | 101 | 102 | 103 | 104 | 105 | 均值 | 中位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| e1 基线 | 191.8 | 178.4 | 88.4 | 120.0 | 120.2 | 107.4 | **134.4** | 120.1 |
| e2 + 特征回拟合 | 98.0 | 168.6 | 110.2 | 168.2 | 72.0 | 88.0 | **117.5** | 104.1 |
| paired | −93.8 | −9.8 | +21.8 | +48.2 | −48.2 | −19.4 | **−16.9** | 2/6 |

**判定：第一波 +16.9 没有复现；6 种子 −16.9（2/6）→ null。**
保持 `--feature-backfit` 为 opt-in（默认关）。结论仍成立的部分：原型层
双遍是 e2 崩塌的一部分（第一波方向），但特征回拟合本身在 6 种子下不
带来生存收益——与“每条机制 A/B 在 6 种子下回归 null”的模式一致。

**决策侧 score 逐列消融（2026-11-21，`score_column_probe.py`）**

默认栈、H=8 oracle、每世界 4 个均匀点 + 尾部 12 点：

| 种子 | 决策 | 可行集 miss | score misrank | oracle 被模型判死 | 被单列消融修复 |
|---|---:|---:|---:|---:|---:|
| 101 | 75 | 0 | 0 | 0 | — |
| 102 | 75 | 8（11%） | 8 | 0 | **drop risk 4/8**；value/curiosity/Q 各 0 |

读法：

- seed 101 在 H=8 内**已经是 oracle-optimal**（列权重无关）；
- seed 102 的 8 个 miss 全部是**可行集内的 score 排错**，其中一半由
  “去掉 risk 列”修复 → **risk 列在部分状态过度惩罚**；
- 但 H=8 会把“活满 8 步”的计划都记成 tie，真正长时域失败被截断；
  全 episode 延续版探针尚未做。

**失败（2026-11-21）：`--risk-weight 0.5` 与基线逐位相同**

原因：现役口径（epochs=1）下目标是**纯 risk**（value=0、curiosity=0、Q=0），
缩放 risk 是单调变换，argmin 不变 → risk-weight 结构性无效。也意味着
探针里“去掉 risk 修复 4/8”是**把 cost 变成全 0 后按枚举顺序乱选**，
不是“risk 过度惩罚”的证据。正确的问题变成 **risk 的分辨率/饱和**：
若大量决策的最优 cost 并列，规划器实际上是在按枚举顺序选。

**风险分辨率探针（2026-11-21，seed 102）**

`score_column_probe` 新增每个决策的 tie 统计：`tie_share`（与最优 cost
相差 ≤1e-9 的可行候选占比）、`near_ties`（≤0.01）、`cost_spread`、
`distinct_costs`。

结果：**mean tie_share 0.01；0/75 决策有大面积并列；cost spread 0.856；
256 个候选里平均 186 个不同 cost**。→ **不是分辨率/并列问题**，规划器
没有按枚举顺序乱选。

因此 seed 102 的 8 次排错是**真实的 risk 排序错误**。最后一个待测点是
“模型是否把被选计划读成更安全”：已给探针加 `chosen_predicted_death` vs
`oracle_predicted_death`，重跑 seed 102；若多数 miss 满足
`chosen_pred > oracle_pred`，则结论是**预测状态上的 risk 读出错误**
（世界模型/预测问题），不是分数权重。

**诊断闭合（2026-11-21，seed 102）**

| 量 | 读数 |
|---|---|
| 第一步预测状态误差（缩放单位） | 0.028（小） |
| 第一步**真实**状态的 risk 读出把被选计划读成更危险 | 0/8 |
| **整条实际轨迹**的 risk 总和把被选计划读成更危险 | **6/8** |

→ **模型在真实状态上算得对，但在预测轨迹上累积后把排序翻反了**：
多步预测漂移是主因，不是权重、并列或读出本身。与远端 TODO 的
“剩余 gap 是预测状态漂移”一致。

**风险步衰减（2026-11-21，已实现）**

`--risk-decay d`：第 t 步的 hazard 贡献乘 `d^t`（终端乘 `d^H`），默认
1.0 逐位不变；286 测试（含 risk_decay=0 的 identity 回归）。A/B：
`--risk-decay 0.8` × 3 种子已完成（见下）——若正向，说明“少信后段预测”
有效；若负向，则需要真正提升多步预测（context model 接 planner 或更长时域）。

**A/B 结果（2026-11-21）：负向**

| 臂 | 100 | 101 | 102 | 均值 | 中位 |
|---|---:|---:|---:|---:|---:|
| 基线 | 191.8 | 178.4 | 88.4 | **152.9** | 152 |
| risk-decay 0.8 | 210.6 | 113.0 | 77.0 | **133.5** | 99 |
| paired | +18.8 | −65.4 | −11.4 | **−19.3** | 1/3 |

**判定：默认 1.0 保持；打折后段预测不成立——前视价值 > 漂移代价。**
诊断仍成立（失败来自多步预测），但修法必须是**提高预测精度**，不是
降低对远端的信任。

**风险列缩放 A/B（2026-11-21，已实现）**

新增 `--risk-weight w`（只缩放 hazard 列；默认 1.0 逐位不变；286 测试，
含 identity/risk_weight=0 回归）。A/B `--risk-weight 0.5` × 3 种子：
**191.8 / 178.4 / 88.4，与基线逐位相同**——见上条“失败”分析（纯 risk
目标下缩放是单调变换）。该开关保留为诊断（只有在价值/好奇/Q 激活时
才有意义）。

**链路接线审计（2026-11-21，`wiring_audit.py`，1500 帧短训，只读）**

| 环节 | 状态 | 证据 |
|---|---|---|
| 观测/尺度 | ok | 7 维，无常量维，尺度 0.169–16.9 |
| 原型覆盖 | ok | 734 个，最近距离 p50=0.000、p90=0.315，>birth 0% |
| 特征归纳 | ok | 8 根、43 节点，训练流 any-root **2.9%** |
| **特征动作/价值格** | **dead** | action cells=0、value cells=0、ledger=0（`epochs=1` 归纳在唯一一遍之后） |
| 危险组合 | ok | composed_mae 0.005 vs maximum 0.026；composed_rate 0.0264 ≈ label_rate 0.0247 |
| 危险读出 | ok | mean 0.026、饱和 2%、**spearman 0.97**（同分布内） |
| **价值通道** | **dead** | cells=0，read 0.000，support 0 |
| 记忆 | sparse | 21 条，**检索 6%**，membership 权重均值 0.015 |
| **好奇心** | **inert** | evidence 0、Φ 恒 1.000 |
| **规划器列** | **single-signal** | 仅 risk 与 −dopamine 非零；value/curiosity/q 全 0 |
| **世界模型** | **drift** | 滚动误差 k=1 **0.0000**、k=2 **0.0000**、k=4 **0.5204**、k=8 **0.7322** |
| 学习标签 | ok | 37 个痛事件/1500 行（24.7‰），acting=history[3] |

**接线结论**：决策路径实际只接通了 **“原型 hazard + dopamine 平局项”**。
特征格、价值、好奇心、记忆（6% 检索）几乎不参与；世界模型在 k≥4 处
失效（离散事件/管口重置未被建模），而规划器用的是 H=8——**rollout 的
后半段基本是外推噪声**。

**修复优先级（按“接线不良”程度）**

1. **特征格 / 价值 / 好奇心的时序**：`--feature-backfit`（已实现）可让
   格子训练而原型保持单遍；先用 `--feature-backfit --feature-value-weight 0
   --curiosity-weight 0` 验证“特征格接进决策但不引入价值/好奇项”；
   检查点：cells>0、composition 权重变化、规划器列出现 risk+特征分量。
2. **世界模型的 k≥4 失效**：k≤2 精确、k=4 起爆（0.52）说明离散事件
   （管口通过/坐标重置）未被建模；先在 planner 里用**逐步预测误差**
   给后段 risk 加权（而不是全局 decay），或做事件感知的转移模型。
3. **记忆检索 6%**：membership 权重 0.015，检索只走最近原型一跳；
   需要“检索半径/多跳”或让记忆按特征激活检索。

**接线验证结果（2026-11-21，`--feature-backfit --feature-value-weight 0
--curiosity-weight 0`，1500 帧审计）**

| 环节 | 之前 | backfit 后 |
|---|---|---|
| 特征动作格 | 0 | **85 cells / 10,474 obs** |
| 价值格 | 0 | **25 cells，read 26.2，support 13.2，rho 0.46** |
| 好奇心 | Φ≡1 | **evidence 117.95，Φ mean 0.591，54 distinct** |
| 危险组合 | composed_mae 0.005 | composed_mae **0.0**，rate=label |
| 规划器列 | 仅 risk | 仍仅 risk（本次 value/curiosity 权重为 0） |

→ **三个 dead 环节已接活**；A/B（3 种子）验证“特征格改进 hazard 读出”
是否转成生存收益。

**第一步 A/B（2026-11-21，backfit + value_weight=0 + curiosity=0）**

| 臂 | 100 | 101 | 102 | 均值 | 中位 |
|---|---:|---:|---:|---:|---:|
| 基线 | 191.8 | 178.4 | 88.4 | **152.9** | 152 |
| backfit + vw0 + cw0 | 148.0 | 126.0 | 104.6 | **126.2** | 124 |
| paired | −43.8 | −52.4 | +16.2 | **−26.7** | 1/3 |

**判定：接线修好但没有转成收益 → 保留 opt-in。** 特征格接进 hazard 读出
本身不改变决策质量；“接线不良”是必要条件，不是充分条件。按顺序进入
第二步（预测误差加权）。

**第二步：按实测逐步误差给 risk 加权（2026-11-21，已实现，已判死）**

- `_rollout_error_curve`：在训练流上按已知动作序列滚动世界模型，测
  每一步的误差（观测标准差单位）；
- **修正（审计 bug）**：原审计按平坦行序跨 episode 取窗，制造了
  “k≥4 漂移 0.52”的伪影。按 episode 分组重测后，训练世界与 held-out
  120 的 8 步滚动误差都是 **0.0000**——世界模型是精确的。
- 因此 `--risk-error-weight` 的曲线恒 0、A/B 逐位相同（必然）；
  开关保留为诊断，默认关。

**第三步诊断：危险通道的真实信号（2026-11-21，1500 帧）**

| 通道 | 均值 | 与 pain 的 rank 相关 | 非零占比 |
|---|---:|---:|---:|
| prototype | 0.032 | **−0.020** | 100% |
| feature | 0.058 | **0.688** | 7% |
| **memory** | **0.000** | — | **0%** |

- **memory 通道恒 0**：记忆风险通过“特征根 → 记忆”的 membership 读取，
  而 `epochs=1` 下记忆创建时特征还没安装，之后也没有回填；backfit 的
  `replay_features` 也不建立这条边 → 记忆风险读数是结构性死线。
- prototype 通道在同分布流上 rank 相关 ≈ 0；composed 读出的 0.97
  主要来自 feature 通道 + 同分布校准（in-sample）。
- 下一步（第三步修法）：**记忆风险改从最近原型的 membership 读取**
  （或归纳后回填 memory→feature 边），再重拟合组合并检查 memory 权重。

**第三步结果（2026-11-21，`--memory-feature-backfill`）**

- 回填后 memory 通道：mean 0.0247、**rho 0.679**、nonzero 7%（之前恒 0）；
  membership 权重 0.015 → **0.641**；
- 但 NNLS 组合权重：**feature 0.905 / memory 0.000 / prototype 0.093**
  → 记忆通道与特征通道**冗余**，即使接活也不被读取；
- 附带发现：prototype 通道在同分布流上 rho ≈ **−0.02**（100% 非零但
  无排序信号）；composed 读出的 0.97 实际由 **feature 通道**驱动。

**判定：记忆风险通道关闭（冗余），不作为杠杆**；`--memory-feature-backfill`
保留 opt-in。第三步结束。

**接线审计的最终结论**

- 决策路径 = **特征先验 hazard + 校准**（+ dopamine 平局项）；
- 原型 MC 值、价值、好奇心、记忆都不进入有效决策；
- 世界模型精确、风险分辨率充分；
- 因此“快速适应”目前只能靠**归纳出的特征先验**，没有在线学习项参与
  决策——这是机制层最根本的接线缺口。

### 第二层诊断：度量与决策对比（2026-11-21）

**更正**：上面的 “prototype rho −0.02” 是**对着“下一帧是否疼痛”**测的。
换成规划真正需要的标签（**8 步内是否疼痛**）后：

| 通道 | rho(下一帧 pain) | rho(8 步内 pain) | rho(time-to-pain) |
|---|---:|---:|---:|
| prototype | −0.020 | **0.593** | −0.608 |
| feature | 0.688 | 0.351 | −0.419 |
| composed | 0.970* | 0.351 | −0.419 |

\* 0.97 是同分布校准下的 composed 水平，不是排序。

→ 原型通道并非“无信号”，是**之前用错了标签**。两个通道各自擅长不同
目标：原型更像“离死还有多远”，特征更像“下一步会不会疼”。组合器按
下一帧标签拟合，把原型压到 0.093，等于**按错误的标签选通道**。

**相似度诊断（`metric_probe.py`，seed 100，1500 帧）**

| 检验 | 结果 | 含义 |
|---|---|---|
| 跨 episode 最近邻结局一致率 | 0.896（随机 0.683，多数类 0.803） | 几何**有**结局信息 |
| kNN 预测 8 步疼痛 AUC | 0.918（只用 episode 进度 0.766） | 不是纯时间混淆 |
| 时间匹配（±1 步）后 AUC | 0.898 | 几何仍有效 |
| 近邻（d≤p1）同动作 vs 异动作结局一致 | 0.987 vs 0.983 | **动作几乎不改变 8 步结局** |
| 时间匹配近邻结局翻转率 | 1.4% | 同上 |

→ **几何度量不是瓶颈**；“同一状态不同动作 → 不同结局”在观测数据里
几乎不存在。原因待判：要么 8 步窗口太短（动作效应在窗口外），要么
行为策略的覆盖让同状态只有一种动作（action rate 0.11）。

**决策对比诊断（`risk_contrast_probe.py`，seed 100）**

- 默认堆栈下 cost ≡ risk（value/curiosity 权重为 0）→ 去掉 risk 后
  argmin 只能选第一个候选，所以“drop risk”不是有效消融；
- 尾部采样（`--tail 12`）的 30 个决策点里 **27 个 survival span = 0**
  （所有候选同一帧死亡）→ 这些点本来就没有可选差异；
- 仅 3 个 span>0 的点上，**summed risk 的 rho = −0.76（方向正确）**，
  但 first-step risk 只有 −0.14；
- 规划器 **0% 的候选有预测死亡**（`death_rule_hard` 关时 `death_batch`
  根本不构造）→ 规划器对“这条轨迹会死”是**盲的**，只看到 hazard 率。

→ 之前“risk 排序接近随机”的结论受**尾部采样偏差**影响（27/30 是
无差异点）。需要用均匀采样（`--tail 0`）重测后再下结论。

**均匀采样重测（`--branches 10 --tail 0`，seed 100，120-122）**

| 指标 | 值 | 备注 |
|---|---:|---|
| 决策点 | 30 | 均匀分布在 episode 内 |
| 所有候选同存活帧（span=0） | 21/30 | 这些点**本来无可选差异** |
| span>0 的点 | 9/30 | 真正需要决策 |
| span>0 的 risk rho | **−0.714** | 100% 正确（≤−0.2），0% 反向 |
| span>0 时选中方案的存活 | 9.0（= 活满视界） | 9/9 选中最优 |
| first-step risk rho | −0.014 | 只看第一步没有信息 |
| 候选有预测死亡 | **0%** | `death_rule_hard` 关时规划器**死亡盲** |

**更正 2**：`risk` 在真正有差异的决策点上**排序是正确的**，而且选中了
最优方案。之前“risk 接近随机”是尾部采样造成的假象（尾部点全是
“无论怎么选都在同一帧死”）。**决策读出不是当前瓶颈。**

**死亡预告期（`doom_horizon_probe.py`，seed 100，horizon 8，120-122）**

沿真实死亡轨迹回退 k 帧，问“8 步搜索里还有没有能活满视界的方案”：

| k（死亡前帧数） | 1–8 | 9–12 |
|---|---:|---:|
| 存在存活方案的比例 | **0%** | 100% |
| 最优方案存活 | 恰好 = k | 9（=视界外） |

- 死亡前 8 帧，**256 个动作序列里没有一个能避免**（最优存活恰好
  = k，说明它和真实轨迹同一帧死）；
- 因此**死亡在 8 帧之前就已经封死**，而规划器正是只搜 8 帧，且看不
  见死亡（`death_batch=None`）→ 它在“已经输掉的区域”里继续按
  hazard 选方案；
- 9 帧以外“100% 有存活方案”只是视界看不到死亡，不代表可救。

**下一步（待 horizon 10 结果）**：若 10 帧前存在存活方案，则瓶颈是
**搜索视界短于因果视界**；否则死亡封得更早，需要长时程价值项而不是
更长的搜索。

**horizon 10 结果（seed 100，120-121）**：死亡前 **1–10 帧全部 0%**，
最优存活恰好等于真实死亡帧（0 起始）。→ 8 帧不是唯一限制，10 帧也
不够。

**horizon 12 结果（seed 100，120-121，k=10/11/12）——分水岭**

| 死亡前 k 帧 | 存在存活方案 | 最优存活 | **规划器选中** |
|---:|---:|---:|---:|
| 10 | **100%** | 13（活满视界） | 9（= 实际死亡帧） |
| 11 | **100%** | 13 | 10（= 实际） |
| 12 | **100%** | 13 | 11（= 实际） |

**两个独立结论**：

1. **视界 8/10 太短，12 足够**：12 帧搜索能在 10–12 帧前找到逃生方案；
2. **但规划器不选它**：即使逃生方案在同一候选集里，hazard 求和仍然
   选中“和真实轨迹同帧死”的那条。→ **排序目标错了，不只是视界短**。

原因可从代码确认：长时程量有三处偏差——

- `node.value`（原型 MC 痛回报）用 `gamma=0.99`，有效视界 ≈ **100 帧**，
  比因果视界（~10 帧）长 10 倍 → 在 10 帧窗口内几乎是常数（实测
  0.41–0.43 平直）；
- 特征级 `feature_value` 格子学的是 **dopamine 回报，不是 pain**
  （`_update_feature_value` 的 target 是 `return_dopamine`）→ 规划器
  端点项 `-value_weight * value` 实际在奖励“活着/前进”，不是在惩罚
  长时程死亡；
- 原型值本身数据饥饿：734 个原型 / 1500 帧 ≈ 每个 2 次访问。

**下一步实验（决定性）**：`oracle_selection_probe.py` —— 保持 8 帧
视界与候选集不变，只把“选择”换成 oracle（选真实存活最久的候选）。
若 oracle@8 显著长于 policy@8，则缺口是**选择/价值**；若持平，则缺口
是**视界/状态抽象**。

**oracle@8 结果（seed 100，120-122）**：policy 与 oracle **逐 episode
完全相同**（472/472、152/152、183/183）。→ 8 帧候选集里**本来就没有
更好的方案**，完美选择也救不了。**视界是硬约束，排序不是**（在 8 帧
内）。

**但 horizon 12 不可行**：一次 horizon-12 搜索约 **10 分钟**
（4096 候选 × 12 步 × 每步 hazard 读出），全 episode 需要 ~25 小时。
→ “加深搜索”在算力上不成立。必须用**廉价的视界外推**：端点价值
（8 步端点上的 `pain` 值，若其有效视界 ≈10 帧，就等价于看到 8–18 帧）。
这正是 `pain_gamma` 的动机。

**pain_gamma（新增，默认 = gamma，行为不变）**

- 痛回报原来用 `gamma=0.99` → 有效视界 ≈100 帧，在 10 帧窗口内基本
  是常数（实测 0.41–0.43 平直）；
- `--pain-gamma` 只改痛通道的折扣（dopamine 仍用 gamma），给痛价值
  一个与因果视界匹配的折扣；
- 代码：`NodeRuntime.pain_gamma`（默认 = gamma）、`fit(pain_gamma=None)`、
  `learn_episode`/`observe`/`learn_transition` 的痛目标；单测
  `test_pain_gamma_sets_the_pain_value_horizon`（gamma 0.9 vs 0.99 的
  两帧前读数）。
- 探针：`warning_lead_probe.py --pain-gamma 0.9` 检查提前量是否从 2 帧
  扩到 8–10 帧；若是，再跑 3 种子 A/B。

**警告提前量（`warning_lead_probe.py`，seed 100，5 个死亡 episode）**

| k（死亡前帧） | composed | prototype | value | mc |
|---:|---:|---:|---:|---:|
| 0 | 0.800 | 0.095 | 0.000 | 0.421 |
| 1 | 0.192 | 0.079 | 0.000 | 0.422 |
| 2 | 0.079 | 0.083 | 0.000 | 0.423 |
| 4 | 0.000 | 0.078 | 0.000 | 0.431 |
| 8 | 0.000 | 0.064 | 0.000 | 0.425 |
| 12 | 0.000 | 0.041 | 0.000 | 0.412 |

- `composed`/`feature` 的提前量只有 **~2 帧**（点事件标签的必然结果，
  与 README 的历史结论一致）；
- prototype 率有微弱衰减（0.095 → 0.04），但幅度接近噪声；
- `value` 在 `epochs=1` 下恒 0；backfit 后**活但饱和**（≈34.8，
  死亡前 2 帧才掉到 22）→ 提前量仍 ~2 帧；
- `mc`（原型 MC 痛回报）平直 ≈0.42，**没有距离信息**。

→ **没有任何通道能在 8 帧以外看到死亡，而死亡封死在 10 帧以外。**
这不是读出问题，是“长时程信号”结构性缺失。注意本表每格只有 5 个
样本，后续需加 episode 数复核。

**粗层可行性（`coarse_search_probe.py`，seed 100）**

- 宏动作 = k 帧固定动作模式；3 步 = 12 帧；4 种 → 64 叶，8 种 → 512 叶；
- 成本 = 沿预测轨迹累加的 hazard（和细层同一个世界模型）。

| 宏数 | rho(成本, 存活) | 粗层最优 vs 实际 | ceiling vs 实际 |
|---:|---:|---:|---:|
| 4 | −0.29 ~ −0.94 | 0（k≤12） | 0（k≤12） |
| 8 | −0.31 ~ −0.91 | 0（k≤12） | 0（k≤12） |

- **排序有效**（k≥10 时 rho ≤ −0.6，76–83% 正确）；
- **但粗层没有任何方案比实际轨迹活得久**——而 12 步穷举在同一状态能
  找到逃生（k=12 时 200/4096 与 2496/4096 个方案活过 12 帧）。

→ 不是“粗层不可行”，是**宏字典有偏**：8 种全是突发型模式
（全拍/先拍后滑/单拍），逃生需要交替型序列。下一步加随机对照：
同一状态采 128 个均匀随机 12 步动作，比较“宏方案存活率”与“随机方案
存活率”——若随机远高，则修字典（多样化/从数据学）；若相当，则这一
粒度上区域本身封死，必须换成价值外推。

**随机对照结果（seed 100，120-122，128 随机 12 步方案）**

| k | 宏方案存活 | 随机方案存活 |
|---:|---:|---:|
| 9–12 | 0% | **0%** |
| 13 | 46% | 28% |
| 14 | 46% | 25% |

→ **k≤12 时宏方案和随机方案都是 0%**，不是字典偏置问题。k=13/14 的
“存活”只是死亡帧刚好落到 12 帧跨度之外，不是逃生。

**重要更正**：之前“horizon 12 能找到逃生（200–2496/4096）”是在
**horizon-12 策略**下测的；那个策略的状态分布更宽松。**默认
（horizon-8）策略走到的状态封得更死**，12 帧内宏/随机都救不了。

**用户裁定（2026-11-22）：粗度不是帧，是事件**

> “粗度根本不是以帧为粗度，而是一些关键的事件，预示着进入下一个阶段
> 的事件。是一些关键的状态或特征的变化。”

→ **按 k 帧切宏动作的路线作废**（`coarse_search_probe.py` 保留为诊断，
不再作为机制）。粗层的单位应为**阶段**：

- **事件** = 动作条件局部模型不再解释后续观测的那一刻（残差的结构性
  变化），即 `stage_graph.py` 的 action-residual segmenter；
- **粗状态** = 阶段（同一动作效应 = 同一类阶段）；
- **粗转移** = 阶段 → 阶段，边由真实轨迹统计；
- **粗决策** = 在阶段入口选择“进入哪一类下一阶段”，不是选帧级动作；
- **细决策** = 阶段内部仍用现有 8 帧搜索，只负责执行/微调。

待验证的最小问题：**最后一个阶段边界离死亡多远**（`event_lead_probe.py`）。
若只有 2–3 帧，事件没有决策空间；若有 10+ 帧，阶段入口就是天然的粗
决策点。

**事件粒度测量（seed 100，120-122）**

| 事件源 | 每 episode 数量 | 间隔中位数 | 最后事件离死亡 |
|---|---:|---:|---:|
| 动作残差分段（penalty 1.0） | 2–3 段 | 88.5 帧（p50） | 27 / 58 / 352 帧 |
| 特征激活集变化 | 3–4 次 | 40–73 帧 | **0 帧** |

- 分段器标的是“大段动态同质区”，太粗（p50 88 帧），但**粒度由 penalty
  控制**，可调；
- 特征激活集**结构性太稀疏**：整局只翻转 3–4 次，且最后一次正好在死亡
  帧（提前量 0）。8 根根特征无法把状态切成“阶段”。

→ **事件源应该是动力学分段，不是特征翻转**；下一步给分段器加一个
**目标驻留时间**（如 10–20 帧，通用超参，不是任务常数），再在该粒度上
重测 real vs shuffle 与“最后边界提前量”。

**阶段级决策（2026-11-22 实装 + 验收）**

用户动机：无关紧要的漂浮阶段不必关注每一步的具体操作，系统只要知道
**这个阶段结束后能达到什么效果**——本质上是一种剪枝/去重，不关注过程、
只关注结果，从而收束多余的搜索。

**纪律事件（任务特异性，已回退）**：首版候选把 Flappy 的动作语义嵌进了
基建（`zeros`="不动"、`ones`="拍翅"），违反"不做任何任务相关逻辑到基建中，
只允许任务提供输入反馈"这一唯一学习路径原则。已删除。现候选为
**任务无关的块常数日程** `substrate_nodes.stage_schedules`：候选 = `|A|**blocks`，
每个动作保持 `block_frames` 帧，动作字母表取自 `agent.action_space`
（runtime 属性，正确的任务无关来源）。CLI
`--decision stage --stage-blocks B --stage-block-frames K`。评分沿用现有
hazard/value 读出，六列账本恒等式
`score[0] = score[2]+score[3]+score[4]+score[5]` 保持；执行最优日程的第一个动作。
291 测试全过，关闭时逐位不变。

A/B 1（**旧的任务特异稀疏日程，已删除**，仅存记录；24 训练种子 × 4 held-out
世界，h6，steps2000/epochs3/max_nodes1500/cap600，评估种子 200–203）：

| 臂 | mean | median | paired Δ vs search h6 | 95% CI | 候选×步 |
|---|---|---|---|---|---|
| search h6（基线） | 97.71 | 77.00 | — | — | 64×6 |
| search h8（同成本对照） | 144.53 | 104.50 | +46.82 | [+28.18, +68.39] | 256×8 |
| stage span16 | 163.89 | 135.50 | +66.18 | [+42.62, +95.26] | 138×16 |
| stage span24 | 181.69 | 152.00 | +83.98 | [+54.85, +116.60] | 303×24 |

（该版候选把"不动/拍翅"语义写进基建，已按原则删除；下表是替换后的任务无关版。）

A/B 2（**任务无关块常数日程**，同一配置）：

| 臂 | mean | median | paired Δ vs search h6 | 95% CI | 正向种子 |
|---|---|---|---|---|---|
| search h6（基线） | 97.71 | 77.00 | — | — | — |
| stage b4k4 span16 | 95.65 | 72.00 | −2.06 | [−12.18, +9.96] | 6/24 |
| stage b6k4 span24 | 74.44 | 72.00 | −23.27 | [−31.75, −15.47] | 2/24 |
| stage b8k4 span32 | 37.89 | 35.00 | −59.82 | [−66.85, −53.47] | 0/24 |

判定 1：**失败且单调变差**。据此**推翻** A/B 1 的"买到的是视界"解释——
真正起作用的是**候选集的时间分辨率**（能否在任意单帧拍翅），不是视界长度：
块常数日程丢掉了逐帧分辨率，视界越长丢得越多。

A/B 3（**beam search**：任务无关 + 逐帧分辨率 + 有界候选，唯一超参=宽度）：

| 臂 | mean | median | paired Δ vs search h6 | 95% CI |
|---|---|---|---|---|
| search h6（基线） | 97.71 | 77.00 | — | — |
| beam h6 w64 | 97.71 | 77.00 | **+0.00** | **[+0.00, +0.00]** |
| beam h12 w64 | 75.88 | 72.00 | −21.83 | [−30.38, −14.09] |
| beam h24 w64 | 76.25 | 72.00 | −21.46 | [−29.51, −13.86] |
| beam h12 w1024（≈穷举 h12） | 88.38 | 72.00 | −9.33 | [−22.51, +4.64] |

判定 2：
1. `beam h6 w64`（宽度 ≥ 全部 2^6 序列）与 `search h6` **逐位一致** → beam
   分支忠实复现穷举，接线正确。
2. **加长帧视界无增益**：h12 在 w1024（几乎不剪枝）之下仍是 −9.33，CI 跨 0。
3. **小宽度长视界有害**（−22）：用帧级近视目标做贪心剪枝会剪掉"短期分不高
   但能长期存活"的路径（w64 ≈ w256 都差，w1024 才明显回升）。
→ **结论：帧视界不是正确的轴**。beam 保留为工具，默认 0（关闭时逐位不变）。

**评估协议修正（2026-11-22）**

- 指标从"帧数"补充为"**进度里程碑数**"：world 侧 `env.passed` 在翻越一根
  柱子时 +1（`flappy.py`）。世界只把它作为**评测事实**报告，**不进观测、
  不进学习路径**（`substrate_nodes.evaluate(metrics_out=...)` 旁路取出）。
- 报告新增 `passes` 列 + `overall_passes` / `per_train_seed_passes`，并打印
  "progress milestones overall"。
- 量程校验（20 held-out 世界，`FL_SPACING=0.8`）：`zero` = 0.00；
  `scripted` = 0.30；**`recovered hor=10` = 34.40**（2806 帧）→ 指标有量程。
- 基线 `search h6` 同配置：frames mean 97.71 / median 77 / max 375
  （< cap600，**未被截断**）；**milestones mean 0.29 / median 0 /
  24-of-96 episode ≥1 / max 4**，cluster CI [0.20, 0.40]。
  → 当前评测**主要落在第一根柱子之前的区间**（中位数 0 里程碑），但**并非
  全部**：25% 的 episode 能到 1–4 根。因此帧数差异主要反映近段能力，
  里程碑级能力被压缩在少数 episode 里。

**间距（时间尺度）可配（2026-11-22 实装 + 验证）**

- `Flappy(spacing=...)` / `Latent(spacing=...)`（`flappy.py` / `latent_world.py`）
  ——世界的时间尺度参数，默认 `FL_SPACING=0.80`（= 柱间 80 帧）。
- `substrate_nodes.evaluate(..., spacing=...)` + CLI `--eval-spacing`
  （默认 0 = 世界默认；**只影响评测世界，不影响训练世界**）+ report 字段。
- 单测：`tests/test_flappy.py` 新增 4 条（放大间距拉伸通行间隔；同种子下
  **第一根柱子的几何完全不变**、只有"再下一根"的距离变大——这正是"通行是
  时间无关里程碑"的可测形式；非正间距被拒；`env.passed` 计数正确）。
- 端到端验证工具：`spacing_invariance_probe.py`（新）。

**间距不变性验收（`spacing_invariance_probe.py`，2026-11-22）**

原始帧数与原始里程碑数都是**时间量**，必然随间距移动：同一帧预算在更大的
间距下包含更少的通行。所以判据必须用**无时间量纲的统计量**：
"已达成通行 / 世界实际提供的通行" = `passes · spacing / (frames · speed)`。

| 策略 | 0.80 | 1.60 | 2.40 | paired Δ vs 0.80 |
|---|---|---|---|---|
| reference（`recovered` 前瞻，20 世界，cap6000） | 0.975 | 1.000 | 1.039 | +0.024 [+0.010, +0.042] / +0.063 [+0.021, +0.138] |
| learned `search h6`（24 种子，cap1200） | 0.132 | 0.122 | 0.137 | −0.010 [−0.100, +0.076] / +0.005 [−0.082, +0.095] |

判定：

1. **该统计量对时间尺度不变**：reference 策略逐通行成功率 ≈ 0.98–1.04，
   拉长间距不退化（略升是因为初始两根柱子在更少的"提供通行"上摊薄）。
   原始帧数在 1.6/2.4 下触 cap（中位数 = 6000），说明大间距下**世界本身更
   容易存活**（单位帧内的危险更稀疏）——再次说明帧数不是能力量。
2. learned `search h6` 的该统计量同样**对间距不敏感**（0.13→0.12→0.14，
   CI 全跨 0）。**但绝对值只有 reference 的 1/7.5**：它失败 87% 的应通行。
3. → **本验收没有证明"帧视界"是这里的瓶颈**：当前规划器的缺陷是**普遍的
   决策质量问题**，而不是时间尺度问题。里程碑指标本身表现良好且时间不变，
   这一点已被验证。若要证明帧视界受限，需要一个在该统计量上已经较强的
   策略，再拉长间距观察退化——目前基线太弱（median 0 里程碑），功效不足。

**决策路径验证测试（2026-11-22）**

- `tests/test_node_runtime.py` 新增 `DecisionPathTests`（6 条）：
  `stage_schedules` 任务无关（任意 |A|、块常数结构）；`stage_result` 六列
  账本恒等式；`decision_action(decision="stage")` 分派 + 账本记录；
  `policy_result` **用 `agent.action_space` 枚举候选**（三动作 agent →
  3²=9 个候选，证明候选集不是硬编码 `(0,1)`）；beam 宽 ≥ 2^horizon 时与
  穷举**逐位一致**；beam 遇 death 回调时显式报错。
- 顺带修一处不一致：`stage_result` 原先算了六列账本却只返回标量，
  docstring 却声明"账本契约成立"。现返回六元组，`decision_action` 的
  stage 分支据此记账（`score[2], score[3]`），与 search 分支一致。
- 291 → **301** 测试全过。

**里程碑边界（2026-11-22：先误诊 → 再诊断 → 已实装 + 完美验收）**

- 首版检测器（capacitor + **因果滑窗中位数**做尺度 + 无 NMS）失败：k=10 爆震
  （≈3×通行数）、k=40 只在 2.4 间距上恰好 1 次/通行、0.8/1.6 全静默。
- **误诊（已推翻）**：曾判定"拍翅瞬态污染了稳健尺度、k 的可行区间为空集"。
  实测反驳：拍翅帧 |Δs| 中位数 0.0130 vs 迹线帧 0.0111，仅 1.2×，**从不接近
  复位幅度**（复位在尾部 0.089–0.47）。
- **真因（实测）**：
  1. **尺度取错统计量**。复位帧的 |Δs| *中位数*（0.010–0.013）与迹线帧几乎
     相同——复位的能量全在 **q90–max 尾部**。用中位数当尺度，阈值恰好落在
     复位自身衰减尾巴的中间。
  2. **缺非极大抑制**。capacitor 把一次复位抹成 1/β = 5 帧的指数尾巴，每帧都
     在阈值之上 → 每次复位触发 3–5 次（复位只占 1% 的帧，却贡献 ≈3% 的触发）。
  3. **因果滑窗中位数让阈值随工况漂移**（同一策略下 0.8 静默、2.4 恰好）。
- **修正规则**：尺度改用**离线高分量** `3 × q90(慢变量跳变)` + **NMS(8 帧)**。
  离线是刻意的（与 `stage_graph` 同族的训练侧分段），尺度由流自估，故跨时间
  尺度自洽。
- **验收（3 种子 × 3 间距，`recovered hor=10` 控制器）**：

| spacing | boundaries | recycles | recall | precision |
|---|---|---|---|---|
| 0.8 | 222 | 222 | 1.00 | 1.00 |
| 1.6 | 111 | 111 | 1.00 | 1.00 |
| 2.4 | 69 | 69 | 1.00 | 1.00 |

  → **恰好 1 次/通行、零假阳性、跨时间尺度成立**。阈值本身几乎不变
  （0.089/0.081/0.079，普通帧动力学与间距无关），复位幅度随间距缩放——
  这正是要求的时间无关性。
- **内在尺度约束**（已写入模块 docstring）：稳态漂移 `d`/帧经 capacitor 滞后
  `d/β`，贡献跳变 ≈ `d`；复位幅度 `A` 贡献 ≈ `β·A`；可分离条件
  `β·A > k·d`。小于这个比例的"复位"按定义不是相位边界——是预期行为，不是失败。
- 实装：`milestone_graph.py`（`slow_baseline_jumps` / `segment_boundaries` /
  `milestone_statistics`）。**任务无关**，已加入 guard 白名单检查；单测覆盖
  合成锯齿的周期不变性、动作式瞬态不产生边界、时长/反馈记账。

**里程碑级转移记忆（2026-11-22 实装 + 身份诊断）**

每个里程碑（边界之间的片段）用**边界的分位桶**标识（`boundary_keys`，桶切点
取自流自身的边际分位数）；记忆按 `key → key` 累计**次数 / 时长分布 / 反馈**，
`next_milestone()` 给出"高层向下预测"（最频繁后继 + 预期时长与反馈）。

- **身份选择做过实测对比**（真实流，3 种子）：
  - 仓库阶段规则（按**动作效应**定身份）：**完美时间不变但退化**——动作效应是
    世界的常数（逐维跨里程碑 std ≈ 0.001，均值 0.2），所有里程碑塌成 **1 个 key**。
  - 分位桶 3/4 级：有区分度但 key 集跨时间尺度不再一致（Jaccard 0.44 / 0.26）。
  - **分位桶 2 级 = 可用工作点**：0.8 上 16 个 key（225 里程碑），1.6 上同样 16 个，
    key 集 **Jaccard = 1.00**。dx1/dx2 因边界处总落在同一桶而自动贡献 0 bit。
- **验收（0.8 vs 1.6）**：

| spacing | milestones | passes / milestone | distinct keys | duration median | **key Jaccard** |
|---|---|---|---|---|---|
| 0.8 | 225 | 0.99 | 16 | 80.0 | — |
| 1.6 | 114 | 0.97 | 16 | 160.0 | **1.00** |

  → **1 个里程碑 ≈ 1 次通行；时长精确 ×2；身份集跨时间尺度完全相同。**
  转移集 Jaccard 只有 0.26——这是**样本量**限制（114 vs 225，慢尺度下很多转移
  没被看过），不是结构不一致；key 集一致才是结构不变量。
- 反馈均值 == 时长（80 / 157.9），因为该世界唯一的反馈就是"每帧存活"。
  下一步用更有信息量的结果量（如在里程碑内是否被逼到危险区）才谈得上"阶段的
  效果"，这还是待办。
- 单测 6 条（monotone 缩放不变性、转移计数与时长、反馈记账、bins 校验、
  `next_milestone` 预测）。**317 测试全过。**

> **⚠ 更正（2026-11-22 第二轮，详见文首「里程碑方向：当前状态与待办」）**：本节
> 的 **"key 集 Jaccard = 1.00"** 与 **"1 个里程碑 ≈ 1 次通行"** 是在一条
> **几乎不含相位**的流上得到的，不能作为身份可用性的证据。修好检测器后在有相位的
> 流上重测：逐坐标 2 级桶给出 **23–25 个身份 / 33 个里程碑**（指纹），跨尺度
> key 集 Jaccard 仅 **0.43–0.56**。且**验收指标本身有问题**：对 2 级方案"key 集
> Jaccard"是平凡的（标签集恒为 `{0,1}`），正确指标是"同一里程碑在两个时标下是否
> 同名"。**317 → 现状 333 测试**。

**接入 planner/runtime 的技术障碍（2026-11-22 认定）**

里程碑层的任何量在**一个 6 帧 rollout 内是常数**（里程碑 ≈ 80 帧），所以它
**无法进入逐帧目标函数**——PBRS 势差会在 rollout 内归零。因此接入必须是
**epoch 级**（在边界处决策、在里程碑内执行），而不是逐帧 shaping。
可选形态（**均已实装并验收，两个都不予采纳**，见下两节）：
1. **目标原型距离塑形**（`--milestone-mode goal`）：单调退化、无正区间（失败）。
2. **epoch 级动作承诺**（`--milestone-mode rate`）：符号随评估世界集翻转（不予采纳）。
3. **低危险剪枝**：学到 hazard 低且远离边界时用廉价控制器，把搜索深度留给
   边界附近——即用户最初的"无关紧要的漂浮阶段不关注每一步"。**未实装**。
   注意：以上两个形态都是在**空层**上测的（训练流无相位），所以它们既不是"形态
   错"也不是"形态对"的定论；**先解决相位数据**（见文首）。

**选项 1 实装 + 验收：目标原型距离塑形（2026-11-22，失败）**

实装（默认关，`--milestone-weight 0` 时逐位不变）：`fit` 从 agent 自己的训练流
分段取里程碑，把每个里程碑的**起始观测**作为候选目标，按**频次**（该世界反馈
== 时长，退化，故用频次）选出模态边界状态作为目标；`policy_result` 在
`evaluate_terminal` 里加 `− w·min_g ‖(endpoint−goal)/scales‖`，记进现有
第 4 列 shaping 槽，六列账本恒等式保持。

A/B（24 种子 × 4 世界，h6，steps2000/epochs3/mn1500/cap600，eval 200–203）：

| 臂 | mean | median | Δ vs search h6 | 95% CI | 正向种子 |
|---|---|---|---|---|---|
| search h6（基线） | 97.71 | 77.00 | — | — | — |
| ms w0.005 | 96.47 | 77.00 | −1.24 | [−4.74, +2.85] | 5/24 |
| ms w0.02 | 87.66 | 72.00 | −10.05 | [−20.33, −1.04] | 8/24 |
| ms w0.05 | 54.35 | 57.50 | −43.35 | [−58.41, −28.81] | 2/24 |

→ **单调退化、没有正区间**（w=0.005 已是 null，再大显著变差，0.05 崩）。

**诊断（为什么是负结果）**：目标是一个**绝对观测向量**（模态边界状态 ≈
`y≈0.12`（贴近底部）、`dx1≈0.38`、`dy1≈−0.11`）。所以这个项本质是
**静态位置吸引子**，把 rollout 端点往"通过柱子后那一瞬的姿态"拖——它与任务
本身竞争：想让 y 停在 0.12 与"保持平稳飞行"直接冲突，dx1/dx2 又随间距缩放。
值得注意的是：里程碑**身份**的设计恰恰刻意避开了绝对值（用分位桶），而
**目标空间**没有——这是同一个陷阱的另一面。因此修复不是重新标定权重，而是
改变**高层传给低层的东西**：应当是"方向/约束"，不是一个点。
（另：第一版还有一处 double-normalization bug——`encoded` 已除以 scales，
我又除了一次，w=0.5 就崩到 11 帧；已修。）

待办（本方向未完成的部分）：

- [ ] **选项 2 或 3**，或"里程碑条件的价值读出"：让高层传**方向**而非目标点
      （如 epoch 级动作承诺 / 把里程碑身份作为价值读出的条件变量）。
      选项 1 的点目标形态已证伪，不建议重试同类。
- [ ] **更有信息量的里程碑结果量**（当前反馈 == 时长，退化）。
- [ ] **更大训练尺度稳健性**——仓库历史多次出现 fast 尺度增益在大尺度消失；
- [ ] logger 的 `--decision stage` 与 DAgger/collect 路径未接（DAgger 默认关）。


**里程碑层根因（第二轮诊断，2026-11-22）：训练流里几乎没有完整相位**

这一轮把里程碑层从头挖到底，结论是：**在当前 setup 下，里程碑层根本学不到东西。**
三层证据，从症状到根因：

**（1）检测器退化（真 bug，已修）。** `slow_baseline_jumps` 取的是**跨维 max**，
而观测里有一个近似**线性斜坡的坐标**（每帧 ~1），它的 leaky-baseline 步长是**常数**，
于是每帧的统计量恒为 **1.0000**（实测 q50=q90=q99=max=1.0），阈值 = 3×q90 = 3.0
**永远不可能被超过** → **复位检出 0 个**。修法：按**逐坐标归一化**（每个坐标除以
自己 |step| 的 q90，零尺度安全），即 `boundary_scale` / `boundary_scores` /
`pooled_boundary_model`。验收（对已知柱子回收帧）：

| 流 | recall | precision |
|---|---|---|
| reference 控制器 600 帧（7 次回收） | 0.86 | **1.00** |
| reference 控制器 152 帧（1 次回收） | **1.00** | **1.00** |

同时修掉**池化阈值**：episode 接缝不是相位复位，拼接会把 q90 抬高（实测
3.0 → 6.68），所以统计量只从**流内**帧池化（`pooled_boundary_model`）。

**（2）训练流饥饿（根因）。** `collect` 的 episode **平均 46.5 帧、中位 38 帧**，
而 **1 个通道 = SPACING/SPEED = 80 帧**。2000 步的流共 43 个 episode，**只有 2 个**
跨过 1 个通道（5000 步时 5/108）。于是**整条训练流里被检出的相位复位 = 0 个**
（`milestone3_train.py`）。换多少训练步都没用——episode 长度由**策略**决定，
不由步数决定（46.5 vs 46.3）。

**（3）因此 fit 学出来的里程碑层是空的。** 每个 episode 恰好 1 个"里程碑"、
4 个身份只是 **episode 起始状态的分位桶**、`key_rate` = 整集平均动作率、
`goals_by_key` **为空**（0/4）、边界数 0。身份"稳"只是因为它几乎不随帧变化。

**这解释了此前的一切，并撤回两条早先的结论**：

- **撤回**"覆盖 0.250 → 1.000 证明身份稳定"：没有边界时覆盖是**平凡**的
  （每个 episode 一个 key，4 个 key 覆盖几乎一切；本次实测 0.955）。
  之前的 0.25 是"每 episode 各自切点"造成的，那个 bug 是真的，但它**不是**
  身份迁移性的证据。
- **重新解释**选项 2 的"世界相关符号翻转"：既然 `key_rate` ≈ 整集平均动作率、
  身份几乎不变，那个 term 实际是**全局拍翅率平移**（把 planner 的率从 ≈0.09
  推向 ≈0.13），在世界难度上后果不同；**与里程碑语义无关**。

**修正后仍然成立的**：两个真 bug（检测器逐坐标归一化、共享身份帧/池化阈值）
已修，并在**有相位的流**上通过 recall/precision 验收；检测器现在**能用**。

**下一步（前置，必须先做）**：让训练流包含**完整相位**——当前策略活不过一个通道，
任何消费接口都无从谈起。方案：用能飞过一个通道的策略产生 phase-scale 数据
（或先解决存活/探索 curricula），再**重新验收**里程碑层（边界数、身份集、
时长分布、`key→key` 转移表是否非空且跨尺度稳定），最后才谈把它接进决策。


**打通相位数据（B）：死亡点解剖与最小可行路径（2026-11-22）**

把"为什么流里没有相位"挖到底，结论是**策略从未接触到柱子**：

| 事实 | 数值 |
|---|---|
| 第一个柱子到达鸟的帧数 | `(1.0 − BIRD_X)/SPEED` = **78 帧** |
| 完成一次"通过"（柱子出屏） | `(1.0 − (−0.2))/SPEED` = **~120 帧** |
| 行为策略（`collect`）平均存活 | **46.5 帧** —— 死于**下坠**（y 出界） |
| planner（h6）平均存活 | **~72 帧** —— 也是下坠 |

**决定性证据**：把 `spacing` 从 0.80 扫到 0.10（通道从 80 帧缩到 10 帧），
episode 长度**完全不变**（都是 46.5 帧，0 次回收）；改 `speed` 也**完全不变**。
因为 `reset()` 把首柱钉在 `x=1.0`，**首柱到达时刻与 spacing 无关**，而两种策略都
在首柱到达前就坠地死了。**死因与柱子无关**。

**机制**：行为策略的拍翅率 ≈ 8–11%，而悬停阈值 = `GRAVITY/FLAP` =
0.0016/0.020 = **8%** → 它**刚好卡在悬停线以下**，于是缓慢下坠。世界参数扫描：

| 世界 | 平均存活 | 有回收的 episode |
|---|---|---|
| benchmark | 46.5 | 0/43 |
| gravity=0.0008 / 0.0004 / 0.0002 | 50.0 / 58.8 / 66.7 | 1/40 / 1/34 / 0/30 |
| gravity=0.0 | 83.3 | **6/24** |
| speed=0.005 / 0.0025 | 52.6 / 52.6（**完全相同**） | 0 / 0 |

**最小可行路径（已实测）**：把**世界参数**作为**任务输入**交给 `collect`
（`spacing` / `gravity` / `flap` / `speed`，默认全部不变 —— 只透传，不含任何逻辑）。
在 `gravity=0, steps=6000` 下，**学习者自己的流**给出：**80 个里程碑 / 67 个
episode**、身份 **2** 个、`key→key` 转移表与方向键**非空**（2 条）、rate 表 2 项
（benchmark 下全部为 0/1）。→ 里程碑层**首次在自己的流上有内容可学**。

**但这是一个需要你定的决策**：无重力世界是**退化**的（控制问题减弱），而且它同时
改变**整个 agent** 的训练分布（value/risk 读出）。是"只用它喂里程碑层"还是"整体
在它上面训练"，是任务/curriculum 层面的选择，不应由我单方面决定。



**存活（B 选定路径）：根因是采集策略"开环"（2026-11-22）**

用户选定"先提升存活、不换世界"。继续挖，根因落到**采集策略没有任何状态反馈**：

| 测量（`collect(100, steps=4000)`） | 值 |
|---|---|
| 动作率 | 0.0777 |
| `corr(action, y−0.5)` | **+0.002** |
| `corr(action, vy)` | **−0.014** |
| 动作率（y 低 / 中 / 高三个带） | **0.077 / 0.079 / 0.076** |

即：发出的动作与鸟的位置、速度**无关**；`center = −b/gain = GRAVITY/FLAP = 0.078`
是一个**常数**（均值正确），所以速度做**随机游走**，鸟在 ~46 帧扩散到地面
——在首柱（78 帧）之前。

`LayerPair` 只回归"动作对速度的冲激"（`Δvy = b + Σ h_k·e(t−k)`），模型里**没有
位置/速度到动作的通路**，因此这个采集器结构上无法闭环。

**这意味着"越过 8% 悬停线"的说法要更正**：它早已在 8%（且均值正确）；缺的是
**反馈**，不是**频率**。三种闭环方式，附各自的约束代价：

- **(a) 手写阻尼控制律**（按观测到的 vy 阻尼）：立刻有效，但它是**手写控制律**，
  与项目硬约束"只允许任务反馈作为唯一学习路径"**直接冲突**，原则上排除。
- **(b) 用学出的读出驱动采集**（先短程开环热身 → 用 runtime 学到的 pain/value
  读数选动作，即把现有 DAgger 思路前移到**采集端**）：完全由反馈驱动，符合约束；
  工作量中等。注意现有 DAgger 用的 planner 本身只活 ~72 帧，**差 6 帧就到首柱**。
- **(c) 直接提升 planner 存活**（主线）：同上，缺口很小（72 vs 78），但那是主线本身。



**里程碑身份：从"指纹"到"概念"（2026-11-22，未收敛 + 一处自纠）**

在有相位的流上（reference 控制器，3 世界 × 2 时标）第一次真正验收这一层：

| 项 | 结果 |
|---|---|
| 边界（检测器） | fast **36** 个里程碑 / 35 次真实回收；slow 27 / 24 → **可用** |
| 时长分布 | fast 中位 **80** 帧、slow 中位 **160** 帧 → **精确 ×2**，时标不变性成立 |
| 身份（逐坐标 2 级桶） | **23–25 个身份 / 33 个里程碑**（几乎一个里程碑一个标签 = **指纹**）；跨尺度 key 集 Jaccard **0.43–0.56**（早先"1.00"不能复现） |

改法：把边界状态投影到它们的**主方向**、对**标量**分桶（`boundary_axis` /
`frame_project` / `frame_keys`），得 **2 个身份 / 33 个里程碑**。

**自纠（重要）**：我一度用"跨尺度 key 集 Jaccard = 1.00"当作新方案的验收，但
**这个指标对 2 级方案是平凡的**——2 个标签的集合永远是 `{0,1}`，所以那个 1.00
**不构成任何证据**。正确的检验是"**同一个里程碑**在两个时标下是否拿到同一个
标签"，尚未做。早先对逐坐标桶用的 Jaccard 之所以有信息，只是因为它是稀疏的
（128 格里只实现了一部分）。

**还发现（同一类病）**：主方向在**未标准化**时 **100% 由 dim6 承担**，而 dim6 与
时间正相关（+0.46 / +0.33）→ **"主方向"就是时钟**。已加**逐坐标标准化**（`sigma`）
以消除"单位大就主宰方向"，但标准化后投影与时间的相关仍是 +0.25 / −0.23，**没有
清干净**（去趋势也没有：+0.38）。

**统一根因**：观测里存在**单调 / 无界坐标**，它连续破坏了三处基于"数值水平"的
统计量——(1) 检测器的跨坐标 max 被它钉成常数（复位检出 0）；(2) episode 接缝池化
把阈值抬高（3.0 → 6.68）；(3) 身份的主方向被它占据成时钟。**凡是用"数值水平"的
地方，都要先对单调成分免疫。**

**状态**：身份层**未收敛**。而它是整个里程碑图能否成为"概念图"的前提——23 个标签
会让 `key→key` 转移表几乎全是 `count=1`，那就不叫概念图。下一步必须先定身份，
再谈消费。


**选项 2 实装 + 验收：epoch 级动作承诺（2026-11-22）**

实装（默认关）：`fit` 从 agent 自己的训练流里，按**里程碑身份**学一个
**时长加权动作率** `agent.milestone_rate[key] ∈ [0,1]`（无语义、无位置、无时间
尺度）；`--milestone-mode rate` 时 `evaluate` 用 `MilestoneTracker` 逐帧给出当前
身份，把该身份的率作为 `milestone_rate` 传给 planner，`policy_result` 的终端项加
`w · mean_t |a_t − rate|`（记进第 4 列 shaping 槽，六列账本恒等式保持）。
`--milestone-mode goal` 即选项 1 的目标形态；默认 `off` 时**逐位不变**（已核：
`search h6` 的 frames/action_rate 与更早 phase 的报告逐行一致）。

**实测数据如下，但解读见上文根因——当时里程碑层是空的，所以这些数字衡量的是
"全局拍翅率平移"而不是里程碑承诺。**

A/B（24 种子 × 4 世界，h6，steps2000/epochs3/mn1500/cap600）：

| 臂 | 评估世界 | 基线 mean | mean | Δ | 95% CI | 正向种子 |
|---|---|---|---|---|---|---|
| rate w0.05 | 200–203（基线 97.7） | 97.71 | 105.59 | **+7.89** | [+3.33, +12.84] | **14/24** |
| rate w0.16 | 200–203 | 97.71 | 107.74 | **+10.03** | [+2.72, +19.24] | **14/24** |
| rate w0.5 | 200–203 | 97.71 | 96.48 | −1.23 | [−14.71, +12.78] | 11/24 |
| rate w0.10 | 300–303（基线 123.9） | 123.85 | 118.42 | −5.44 | [−16.21, +4.26] | 11/24 |
| rate w0.16 | 300–303 | 123.85 | 110.39 | **−13.47** | [−24.41, −3.53] | 9/24 |
| rate w0.25 | 300–303 | 123.85 | 103.08 | **−20.77** | [−39.67, −4.48] | 8/24 |

分世界看 Δ（w0.16）：200–203 是 **{200:+13.1, 201:+4.0, 202:+1.1, 203:+21.8}**
（4/4 正）；300–303 是 **{300:−40.7, 301:−0.7, 302:−2.9, 303:−9.6}**（4/4 负）。

→ **符号随世界集翻转**：在 planner 本来差的世界上（基线 97.7）把动作率往
≈0.13 拉，**4/4 世界都变好**；在本来好的世界上（基线 123.9）同一件事 **4/4 变坏**。
效应很大（±8…±40 帧）但方向由世界决定 ⇒ **不予采纳**。

**其余两条断点（与根因独立，仍是设计问题）：**

**断点 1：率不含行为策略的优势——优势在时机，不在频率。**

| 策略 | 动作率 | 通过柱数 | 存活帧 |
|---|---|---|---|
| planner (h6) | 0.083–0.111 | **0** | 72–75 |
| reference 控制器 | 0.079–0.080 | **1–7** | 152–600 |

两者**拍翅频率几乎相同（≈0.08）**，而 reference 的（无时间量纲）通行率是 planner
的数十倍。行为策略靠的是**什么时候拍**，不是**拍多少**；`key_rate` 只编码频率。

**断点 2：接口粒度与尺度都不对。** 传的是一个**标量**，在一个 rollout 内近似
常数，无法表达"轨迹偏好"；唯一非恒定的部分是计划自身的动作统计，而它已被 hazard
决定（最优/次优候选分差 **0.0006–0.033**）。同时 1 个里程碑 ≈ 1 个通道 ≈ **80 帧**，
而 h6 ≈ **6 帧**——里程碑结构在 planner 目标**看不见的尺度**上。

结论：接口上，真正新的对象是**通道尺度的 `key→key` 转移图 + 时长分布**，而逐帧
标量代价**没有地方放一张图**；但在那之前，**先要让这条流包含相位**（见上）——否则
再好的接口也没有东西可传。


**阶段内容审查（`stage_content_probe.py`，seed 100，120-122）**

| 断点 | 出现在 | dim 变化 | within | cross | cross/within |
|---|---:|---|---:|---:|---:|
| frame 119 | **3/3 episode** | dim 2, 2.27 | 0.22 | 3.59 | **16.4** |
| frame 124 | 2/3 episode | dim 0/1 | 0.28 | 0.76 | 2.1–2.7 |

- **机制是对的**：最大的“相似度大幅变化”确实被检出（16 倍落差）；
- **但它学错了事件**：主断点固定在第 119 帧、三个 episode 完全一致 →
  这是**周期性世界重置/传送**（dim 2 一帧跳 2.27），不是状态相关事件。
  penalty 扫描平坦的原因也在此：这个断点 gain ≫ 1，其余全部 < 0.02；
- 次断点 frame 124 只出现在 2/3 episode、落差 2–3 倍 → 这才是状态相关
  的候选事件；
- 断点处最近原型距离 ≈ **3.0**（出生阈值 0.45）→ 原型层没有覆盖事件
  附近的状态；
- **转移表不自洽**：同一 `source + action` 映射到 target 1/2/3，且计数
  完全相同（110 帧 / 2 episode）→ 阶段标签跨 episode 不共享，图不是函数。

→ 结论：**不是“不能学”，是（a）判据把世界时钟当成事件，（b）阶段身份
跨 episode 不一致，（c）状态抽象没覆盖事件状态。** 三步修法：按
“跨 episode 时序方差”过滤时钟事件；用动作效应/状态聚类统一阶段身份；
让原型覆盖断点状态。

**状态划分（`state_partition_probe.py`，5 episode，y8 基础率 0.042）**

| k | dwell-p50 | in-window | clock-like | eta2(y8) | 去趋势后 dwell | 去趋势 eta2 | 去趋势 clock |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 48 | 25% | 1 | 0.017 | 25.5 | 0.024 | 0 |
| 8 | 31 | 25% | 3 | 0.101 | 7.0 | 0.029 | 0 |
| 16 | **13** | **57%** | 6 | 0.175 | 3.0 | 0.099 | 0 |
| 32 | 6 | 34% | 13 | 0.247 | 1.0 | 0.153 | 2 |

- k=16 本身就在决策尺度（13 帧、57% 落窗）；
- 但有 6/16 是“时钟类”（每局同一帧进入）；
- **去趋势（对进度做线性回归取残差）虽然把 clock-like 清零，却把
  dwell 压到 1–7 帧、eta² 从 0.175 掉到 0.099** → 它削掉的不只是时钟，
  还有**与危险同源的结构**。

→ “时钟在状态里”成立，但**简单去趋势不是解**：时间和决定危险的坐标
是纠缠的。而且 `clock-like` 指标也会把**共享的起始状态**误判成时钟类。
下一步若继续，只能是**相位条件化**（估计周期相位，在相位内取相对状态），
而不是减线性趋势。

**当前待验证的两条（2026-11-22）**

1. ~~**相位条件化阶段**~~（`phase_stage_probe.py`）——**已否决**（周期
   修正为频谱中位 38 帧后重测）：absolute / phase-only / phase×relative 的
   LOO Brier 分别为 0.0518 / 0.0448 / 0.0722（常数基线 0.0400），AUC 0.446 /
   0.447 / 0.334。**这批数据上没有任何状态划分能跨 episode 泛化**；
   5 episode、y8 基础率 4.2%，功效很低，只能算“未观察到”。
2. **容量/数据比**：1500 帧长了 734 个原型，平均每原型 ~2 次访问，
   疼痛率与 MC 值是单样本估计。候选 A/B：`--max-nodes 1000 → 150`
   （3 种子），检验“统计质量”是否才是所有通道惰性的共同原因。
   **已关闭**：cap 150 / 250 / 1000 / 2000(birth .35) 均值
   80.3 / 117.9 / **152.9** / 124.3，two 方向都退化，默认在峰值。

**`pain_gamma` 验证（seed 100，5 个死亡 episode）**

| 死亡前 k 帧 | 20 | 16 | 12 | 8 | 0 |
|---|---:|---:|---:|---:|---:|
| mc（gamma 0.99） | 0.41 | 0.41 | 0.41 | 0.43 | 0.42 |
| mc（pain_gamma 0.9） | 0.067 | 0.123 | 0.144 | 0.164 | 0.217 |

→ 痛的折扣从 100 帧压到 10 帧后，端点价值成为**单调距离-死亡信号**，
覆盖 8–20 帧；可作为“视界外推”的廉价来源。

**质量门**：阶段同质性（within-stage residual 显著低于跨阶段）；转移校准
误差可控、applicability violation 低、support 不足 abstain；seed 101 的
decision-relevant miss/regret 下降；seed 100/102 selected survives 不降、
FP 不升。**首要案例 = seed 101 的 score misrank**；seed 100/102 默认
fine-only。

**反模式**：固定时间窗切阶段；按 context 距离合并；粗预测覆盖细预测；
用 salience 大小直接当事件层级；为 benchmark 写阶段词/几何。

**风险**：1500 帧流上阶段样本少 → support 门 + `UNKNOWN` 回退；粗层若
强行覆盖 fine 会让 seed 100/102 退化；吸引力距离必须在学到的状态空间
（阶段记忆叶）定义，不得用任务坐标。

### C. 质量门与 A/B 协议（适用于 A/B 两条线）

- 每个新机制默认关；关闭时决策路径**逐位不变**（回归测试钉住）；
- 准入：真实基准 3 种子配对；fast 屏只用于筛 bug，不作为效果证据；
- 探针先行：先证明机制在跑（覆盖/校准/召回），再谈生存收益；
- 结论与改动写入 README 归档并推送。

## 实验归档索引（README）

| 主题 | README 章节 |
|---|---|
| 训练内省 / 五能力 / 三条断开通路 | 训练内省 |
| 记忆强化 / 好奇心恒零机制 | 记忆回溯修正 + 决策侧好奇心（现状警示） |
| 帧级失败追踪 / 盲死与危险饱和 | 帧级失败追踪 |
| 0/1 死亡规则 → 硬否决 → 质量 → 覆盖优先 → 误差包络 | 0/1 死亡条件规则 … 决策相关不确定性 |
| 生成式展开 / 生成先验 / 多分支 / 预测通道 | 通用上下文与分层记忆 … 多分支节点系统 |
| 成对特征 / 覆盖率门槛 / 跨世界验证 / 动作边 Q | 实现重构与已关闭实验 |
| 历史标签口径（E1/E2、前兆、graded） | E1 / E2a / E2b |

## 历史附录（只读）

（以下为 2026-11 及更早的解剖 / 审计 / 瓶颈记录，不再维护；
新结论写入 README 归档或本文件「当前状态 / 待办」。）

## 四子代理解剖附录（2026-11-20）

### A · 信息流台账要点（43 读写点全图；此处记风险榜）
R1 原型危险关联是单调包络：pain 边只增不减（_learn_pain_association
   对 pain≤0 提前返回），graph.decay 全管线从未调用 → trace 饱和 →
   原型通道读"是否曾经相邻过"而非基率（node_runtime.py:757-778,
   824-842; node_graph.py:267-276, 323-328）。
R2 默认目标的 max 组合（原型/特征/记忆三通道 + 终端第四 max）
   从未校准——校准只在 contrast 路径的 pain_graded_batch 内生效。
R3 格子按 acting=history[lag] 键控，标签却在 pain_horizon 个状态上
   取 max（其余 lag 帧由其他 pending 动作掌管）→ 动作对比被共同
   原因稀释；排除谓词双向失火（缺口银行 horizon=4 被训不读；
   未标记根继承 runtime horizon=8 混入 threat）。
R4 **离线配对错位一行**：校准/残差扫描的 pending=history[:lag][::-1]
   是运行时给下一行状态的编码 → F6 校准在错位配对上拟合（真 bug）。
R5 事件记忆是冻结快照：memory.value 创建后永不更新；
   memory.immediate_pain 从未赋值（恒 0）；memories_for 按 max 合并
   去重并吞并共享事件；自环转移提前返回丢弃学习。
R6 value_baseline：一个标量池化所有 (根,动作,窗) 回报；只在根激活
   行上更新（偏样本）；rate 0.02 → 遗忘 50 事件。
R7 聚合不一致：风险=max、价值=mean（低水平有梯度承重，L1 否证）；
   subgraph 下"最具体=最小计数"把特异性与稀缺混同。
R8 动作边 Q 按 history[0]（刚发出）键控而转移结果由 history[lag]
   （真正掌管）决定——错位归属；且 policy 从不读（死写）。
R9 遗忘按"中位以下 + count≤1"删——恰好删除稀有原因证据。
R10 future_pain_labels 的 max 语义抹去窗口内的时序/多重性。
R11 条件缺陷：association 传播（若开）会把特征根值向 0 扩散；
    组件过滤器跨 episode 泄漏；obs_trust 不对称（pain 有、dopamine
    无）；MC/TD 两套 node.value 写法（目标含/不含 immediate）打架。

### 死机制清单（写了、管线从不消费）
_calibrate_threat（死代码）、pain_contrast_batch（仅测试）、
逐行预测误差回路（feature_calibration/surprise_buffer 仅诊断——
"反事实目标缓冲"从未被学习消费）、count_sum、
ActionEdge.prediction_error、eligibility_*、act()/observe()/
learn_transition 专属状态（_traces/last_*/action_nodes）、
rules_long/rules_value 对象（仅诊断）、非根特征统计（默认路径）、
excess 通道（control=0 时）、终端 _pain（contrast 下被弃）、
终端 dopamine 列（lexicographic 下几乎永不生效）、
row["terminal"] 键、value_snapshot/_k_step（value_horizon=0 时）、
graph.decay/graph.prune（从未调用）。

### B · 逐规则收敛判定（8 条主规则）
1 原型出生/指派：无标量不动点；pass 1 后冻结（非压缩映射）；
   顺序与流依赖 = 表征级种子彩票。pass1 出生 2201，此后 0。
2 节点值/多巴胺 EMA（MC 反向，α=0.2）：常数步长 → 只在分布意义
   收敛，残差 sd ≈ 0.33σ 永存；不动点 = E_μ[折扣未来 pain|节点]
   = 行为连续行动预测，非"最优可达"。实测 p50 三遍爬升
   0.196→0.353→0.478。
3 特征前兆格子：hits/count 收敛 a.s.，但锚点 base 是事件加权 EMA
   永不落定（残差 sd≈0.16σ_label）→ 复合读出无单一不动点；
   计数极偏（1/171/4331）而先验固定 p=2：宽根无感、薄根被支配；
   未观测动作的读出 = 池化行为边际（min_a 被中和）。
4 价值格子：sum/count 收敛，锚点 b 不收敛（规则5）→ 无联合
   不动点；value_horizon>0 时 bootstrap 通道错配（pain 值注入
   dopamine 目标）——默认关但路径有缺陷。
5 value_baseline EMA（rate 0.02，记忆 50 事件）：不动点 = 事件
   混合（权重由"哪些格子开火"决定）；实测 sd 4.28、终值 13.80 =
   最后 episode 的新近均值——无流级含义；既是收缩目标又是未覆盖
   读出（56.9% 状态坐在其上）；其噪声（±0.086 hazard 单位）与
   它要稳定的动作差（0.10–0.17）同阶。
6 动作边 Q：不动点 = 按经验下一状态分布的计数加权边际
   E_μ[R|s,a]（掌管动作错位）；边集 pass 2 后冻结；未知动作罚
   prior+0.5 → 策略回避无数据格子（而 VOI 预算默认关）。
7 MC 反向归属：递归本身精确；归属无收敛内容——重放只重平均；
   γ=0.99 在 33 步中位 episode 内几乎不衰减 → 终局事件几乎全额
   记给每个前缀存储 = "episode 命运"。**无任何优势分解**。
8 预测误差回路：池化均值混合代际（pass1 锚点主导 vs 后期格子
   主导——代际加权混合被流顺序冻结）；surprise 缓冲单调棘轮向
   陈旧（pass1 后冻结于 64）；无学习消费者（回路开放）。

### C · 手工常数表要点（16 行；此处记排序与四大横切缺陷）
横切缺陷：F1 校准粒度错配（组合拟合 vs 逐根消费；
_calibrate_threat 死代码）；F2 分级排除谓词过宽（缺口银行根被
静音——F5 对自己的产品失明）；F3 锚点只上不下（max 播种）；
F4 空格子 = 自由参数（漂移锚点定价未测动作）。
排序（收敛链恢复）：1 组合层校准（逐根/逐动作 PAV 或单调组合器）
→ 2 收缩锚点经验贝叶斯化（held-out 锚点、动作池化回退）→
3-4 价值组合 NNLS + 目标权重学习（w/K/control 一并，执行计划
数据）→ 5 危险账按查询步归一（消除 w 的 horizon/stride 依赖）→
6 分级排除谓词修复（恢复 F5 反馈）→ 8-9 归纳门控多重性校正
（BH/置换校准 z、Wilson 下界支撑门）与预算肘部 → 10-16 原型
sigma/记忆权重/标签窗/折扣/阈值格/VOI/平局列/回退语义。

### D · 文献方法表要点（五断点 × 成熟方法）
断点1 一次结构搜索: Friedman 梯度提升（残差→弱学习器→收缩系数ν
  追加；子采样+held-out 早停）；SLIPPER 自弃权规则 + Viola-Jones
  硬负例级联；steady-state GA + NAS 一次性代理教训。
  适配: 全栈 LS_Boost 代循环——校准残差直接算（rank 判据降为
  平局）、每规则系数 ν·c_m 加性投票替代 max、代际接受以 held-out
  残差损失改善为准、幸存负例作下一代负例。
断点2 原型冻结: ART 警戒+match tracking（预测误差抬高警戒→
  强制搜新类）；Growing Neural Gas + 效用修剪（局部误差最高处
  出生、最低效用节点删除并就近并格）；DP-means λ 出生 + RAN
  新颖性出生（距离×预测残差）。适配: 出生 = 距离超阈 OR 预测
  残差超阈；λ 单旋钮；max_nodes 满时按效用删并转移格子计数。
断点3 边际混杂（最高杠杆）: 逐决策截断 IS（Precup/Sutton；
  Retrace min(1,ρ)）；双重稳健（Jiang&Li；图作 V̂ 控制变量）；
  基线中心化优势（GAE；HCA 后验归因）。适配: 窗口内逐行权重
  Π min(1, π/b)；格子学习 E[pain|f,a] − min_a（可控优势）；
  DR 项 = ρ·(V̂_graph(s') − V̂_model(s'))。
断点4 手工组合: 经典 RBF 输出层闭式最小二乘（Broomhead&Lowe;
  Moody&Darken）；OLS 误差缩减比选中心（Chen 等）；Breiman 非负
  堆叠（NNLS 恰好归零通胀成员）+ 校准后置。适配: 设计矩阵 Φ
  （各通道×动作作列）对 precursor 标签 NNLS；OLS 误差缩减比
  控制根准入；现有 max/mean 保留为库成员之一。
断点5 手工目标: MaxEnt IRL（Ng&Russell；经验特征计数差梯度）；
  Bradley-Terry 偏好回归（Christiano 等；DAgger 回合即偏好对）；
  meta-gradient RL（Xu 等）+ PBT + OPE 选择；PBRS 势函数护栏
  （Ng-Harada-Russell：γΦ(s')−Φ(s) 不改最优策略）。适配:
  (计划账本分量 → 实际存活) 单调回归；执行计划对的权重回归；
  价值项写为势差。
**D 最高杠杆**: 断点3 修正（逐决策 IS + 截断 + 基线中心化优势
  + 图 TD 作 DR 控制变量）——三个更新循环的修改，非架构重写；
  次选 = BP4 NNLS 输出层。
- **评测方法裁定（2026-11-20，用户）**：训练应当只强不弱；出现
  非单调（如学习曲线 1000→1500 下落）时**先查训崩原因**，不得直接
  解读为机制结论。第一步：**评测用多条 episode**（`--eval-episodes ≥ 3`，
  每条新 env_seed = 新初始条件）排掉世界级偶发误差，再判读。
  训练/单条评测本身是确定性的（同 env_seed 同帧数），多次评测必须
  走更多 episode/世界，不是同种子重复。
- **运行卫生（2026-11-20 发现）**：kill 父进程会遗留 multiprocessing
  孤儿 worker（占内存空转）——这解释了 jobs 3 full 运行的 MemoryError
  （孤儿积累）。清理法：`Get-CimInstance Win32_Process` 按
  `parent_pid=<dead>` 过滤后 `Stop-Process`。杀臂时先清孤儿再起新臂。
- **纪律裁定（2026-11-20，用户）**：**少样本高提取是目标**。靠增加数据
  （更多评估世界、更多训练数据、跨种子池化、DAgger 数据回流）只会暴露
  学习效率不足——那是设计问题。修法一律走机制路线；数据扩张类方案
  （含 20 世界复验）不作门槛、不排优先级。
- 学习效率的可测口径：①held-out/self 差距（当前 ~67 帧）收窄；
  ②held-out 中位数越过 152（= 稳定过第 2 根）而**不动任何数据量**；
  ③样本效率曲线（held-out 性能 vs 训练步数）。
- 活跃协议：fast + held-out 评估（120–124）。full 协议（5000 步、H=10、
  cap=6000）的评估成本病理未解（batch 读出中的 Python 循环），暂不运行。
- 已实现未验证方向：`--obs-trust`、`--feature-value-prior`、
  `--feature-action-prior`（默认值保持历史行为）。

## 审计整合（2026-11，三份只读子 agent 报告）

代码路径 / 学习可辨识性 / 架构设计三份审计交叉确认。测量基准：fast 协议 8 种子
`sd=83.5, SE=29.5`，因此**帧数差异 <2 SE 一律不作结论**（131.6 vs 94.6 与
131.6 vs 93.9 都只有 ~1.25 SE）。

### 已确认的关键事实

- 部署决策 `policy_result` 只读 `pain_signal_batch`/`valence_state_batch`；不读
  action edge、action 节点、组件、`_score_actions`。`NodeRuntime.act` 只被测试调用
  （把 `act`/`decision`/`_score_actions` monkeypatch 成 raise 后 `policy_result` 仍跑通）。
- 行为策略是**状态无关** Bernoulli(≈0.14)：`corr(a, obs_i) ≤ 0.033`，`H(A)=0.411`、
  `H(A|bucket)=0.211`。此前"动作几乎变成观测的函数"的说法**撤回**。
- 分布错配：评估状态 **80–95%** 距最近原型超过 `birth_threshold`；训练死亡
  112 边界/9 管道，评估死亡 **6 管道/2 边界**（方向相反）。
- 风险读出稀疏：`pain_signal_batch` 在评估轨迹 0.3% 步骤非零，3/8 种子全程为 0，
  管道死亡前 6 帧全为 0。
- 特征是"死亡当帧分类器"：归纳用 `next_obs` + 当帧 pain；seed 100 死前
  `f2=0.278` vs 阈值 `0.274`（差 0.004 未触发）。
- 即使 50/50 采样，同状态也只有 ~80/2316 个双动作桶；`23/2316` 指标不可达，
  不得作为验收门槛。
- lag=3 下 action edge 按无效动作（`history[0]`）切分状态；有效动作是
  `history[3]`（已在 `pending` 坐标里）。`NodeRuntime.act` 在 lag>0 时语义错误。
- `exhaustive_search` 确实枚举两个动作且 lag 映射正确；问题是评分函数
  （静态状态风险、无终止模型、`step_stride=2` 跳过第一个受影响预测点）。

### 具体 bug（审计确认）

- `memory.value` 只在创建时赋值，**从不更新**（恒 1.0）；`_forget_memories` 的
  `value < median` 条件永不成立，遗忘从未生效；`_feature_memory_risk_batch`
  命中即返回精确 1.0。
- 特征值初始化 `node.value = max(value, rule.confidence)` 使用了 held-out test
  统计 → 选择标签泄漏进预测器初值。
- `_score_actions` 的未知动作是**惩罚**（`+unknown_penalty`）而非乐观；
  已知 pain=0.9 的动作（0.92）会赢过未知动作（1.0）。
- `_interact` 自环转移直接 return：pain 关联、memory、association 全部丢失，
  但 MC 价值照常更新。
- `graph.decay` 从不调用：pain association 单调饱和（10 epoch 后 0.812 仍在涨）。
- 组件被喂常量 `1.0`、同一转移 step 两次、`observe_valence(active=None)` 会
  credit 到上一个节点、`reset` 不清组件状态。"组件无效果"是接线同义反复。
- TD bootstrap 缺 `next.immediate_pain`（生产 MC 路径不受影响）。
- 搜索无终止/可行性模型：线性模型穿墙，唯一保护是越界后特征才触发。
- `--lag 4` 会 KeyError（latent 只有 0..3）；物理常量依赖 import 顺序
  （`latent_world` 的 env 覆盖可能不生效）。
- 文档与代码矛盾：README 声称 memory value 事件局部学习、组件参与决策、
  特征根作为决策候选源等，均与实际路径不符；0.977 offline AUC 不可复现。

### P0 先修测量

- [x] `--no-features` / `--instant-labels` 开关贯通 `fit`/`run_seed`/CLI（消除"改代码才能消融"）。
- [x] `trace_node_runtime` 改为插桩部署路径：OOD 比例、决策状态风险激活率、死亡类型、
  `P(death within k)` 前兆 AUC（k=1..lag+1）、kNN 双动作支撑（训练数据 k=8）、
  滞后槽位识别（Δ_j argmax + 分离度）、与 zero/one 计划同尺度的对照。
- [x] `plan_score`：按 `policy_result` 同一目标给固定计划打分（含与 exhaustive_search
  一致性的单元测试）。
- [x] 评估协议（2026-11）：`--eval-seeds`/`--eval-episodes`、per-seed 中位数/IQR、
  按训练模型聚类的 bootstrap 95% CI、`--report` JSON；`eval_protocol.py` 负责种子
  解析与统计。首次 fast held-out（训练 100–107，评估 120–124，短窗口）：
  n=40，均值 `116.4`、中位数 `152.0`、IQR `80.0`、区间 `42–232`，聚类 CI
  `[99.9, 133.9]`。remaining：hybrid 与 full 5000 步协议用同一 held-out 集复核。
- [ ] 删除或标注不可复现的 0.977 AUC 与 n=2 的"验证"；README 增加"实现/训练/被控制器读取"三栏。

### P1 正确性

- [x] 特征值初始化改用 train 统计（Laplace 先验），不再用 held-out test confidence。
  单独验证：fast 8 种子 131.6/115.0 与基线逐种子完全相同（数值上无副作用）。
- [x] E1 前置标签：特征归纳与特征 pain 更新改在 `obs_t` 上，标签
  `P_k(t) = max_{j≤lag} pain_{t+j}`（k=lag+1，截断于 done）；`--instant-labels` 回退。
  结果见下节：检测修复，决策未受益，5000 步规模变差。
- [ ] `memory.value` 改为 pain 事件局部 EMA + 真实遗忘判据（回归测试：pain memory
  复访安全步后 value 不再恒 1.0）。
- [ ] 风险融合 max(association, feature, feature-memory) 统一量纲/校准后再融合。
- [ ] 搜索加学到的终止/可行性（或拒绝支持外预测状态）；`step_stride` 覆盖首个受影响点。
- [ ] 小修：自环转移保留 pain 关联；未知动作改乐观；episode 末调用 `graph.decay`；
  组件单步/显式 credit/reset；TD bootstrap 补 immediate_pain；lag 合法值校验。

### P2 结构（待 P0/E1 结果决定方向）

- [x] **按有效动作条件化的特征统计**（`feature_action_stats`，2026-11）：格子
  `(feature, history[lag])` 记 count/hits，读出 `P(pain within lag+1 | feature, pending[0])`，
  无数据格子回退标量前驱值（`feature_action_prior=2.0` 伪计数收缩）。
  验证：5000 步格子 n=40–1756/动作，不对称性物理合理
  （近边界高 \|vy\|：不拍 0.588 vs 拍 0.848；贴管缺口在下：不拍 0.673 vs 拍 0.923）；
  `f1>1.243 & \|f2\|<0.222` 的 a1 格子 n=0——反事实缺口被显式暴露。
  **存活逐种子不变**（fast 122.0、full 87.1 与条件化前完全相同）：贴管探针显示
  两个计划的 feat_risk 在中段全程为 0——值条件化救不了"规则根本不触发"的地方。
- [x] **pain_horizon 窗口实验**（2026-11）：k=8 单库 fast 102.2 / full **190.5**（median 134.5，
  被 seed 103=636 单点撑起）；hybrid（k=4 短库 + k=8 长库，每节点带自己的
  `pain_horizon` 标签窗口）fast 110.9 / full **151.8**（median **141.0**，最稳定的臂）。
  对照 instant 119.6 / k=4 87.1。全部 <2 SE，不作强结论，但方向一致：
  长窗口给接近段正覆盖（`risk_active` 0.066→0.26，trace 验证），短窗口保临终检测；
  单一窗口会垄断 8 条规则预算（k=8 单库的前兆 AUC 反向到 0.37，死前 4 步风险 0.000）。
  新发现检测盲区：管道附近"中度错位 + 缓速漂移"（f3 ≈ 0.85–1.25，raw gap 0.17–0.25，
  恰好在碰撞死区边缘）两个库都不覆盖——训练里该类死亡极少（管道死亡只占 9–24/121），
  归入 R1 采集问题。
- [ ] **激活 DERIVATION 子图（下一批）**：OR/AND 分支已经作为共享子节点存在
  （`_feature_children`），但读出只按根节点取格子。给每个子节点自己的标量值和
  动作格子，读出对"根 + 当前激活分支"取 max。动机（数据实测，5000 步采集）：
  两管之间偏离中心时，上升态拍翅 P(8步内死) 0.44→0.57（有害），下降态拍翅
  0.116→0.085（有益）——"正常控制高度"的直觉在条件统计里存在，但被
  "上升/下降合并"的规则格率平均掉（上升态 n=1529 主导下降态 n=672）。
  同一状态只匹配一个分支，分支级格子自动分开方向；这同时把装饰性的
  derivation 结构变成真的计算 DAG（审计 P2 项）。
- [ ] 序列/计划级价值：单步 hazard 已接近"局部正确但短视"的天花板——
  上升时拍翅局部有害是真的，穿越需要序列，而终端价值几乎平坦
  （0.48–0.62，无法为穿越序列背书）。
- [x] **预测误差回路 v1**（2026-11）：决策时把读出预测（更新前的
  `(root, acting)` 格子读出，各自对自己窗口的标签）与实际前驱结果对照，
  记录校准统计（pred/actual 均值、Brier）并把 高置信×大误差 的事件存入
  上限 64 的 surprise 缓冲——即 VOI 预算的目标清单。trace 新增 [8] 节。
  实测：32 个校准格子 0 个系统性错标（计数格子训练后收敛到真实率）；
  缓冲满 64 条，top 全是 `pred=0.96–1.00, actual=0, conf=0.96–0.98`
  ——正是"系统认定必死但实际存活"的高置信错误预测时刻。
  纯观测：fast 8 种子 110.9 逐种子不变。
- [x] **VOI 反事实预算 v1**（2026-11，机制已验证，存活未证明）：两阶段采集
  （bootstrap ~85% → 引导 ~15% → 全量重训），引导 = 活跃高先验根里少数动作
  格子最空者优先，每格 ≤16 次测试，全局预算 `--voi-fraction`，propensity
  逐转移记录（1.0 = 引导覆盖）。机制验收达成：填充格子 15/16（多数规则的
  a1 格子从 0 填到 2–126），少数动作样本 194 → 数百。
  存活未证明（**修复 collect 数据流回归后的干净数字**）：fast 0.15 →
  117.1/90.5（paired +6.3 vs 混合 110.9，0.24 SE）；0.08 → 117.1/104.0
  （0.55 SE）；全量 0.15 → **153.2/152.0**（vs 混合 151.8/141.0，
  paired +1.5，0.04 SE；中位数 +11；种子间 sd 88→60）。均值持平、
  方差收紧、中位数方向略正。机制复核（干净流）：填充格子 16/16、
  空 a1 格子为零。默认 `voi_fraction=0.0`（关闭）。判读不变：
  覆盖必要但不充分。
- [x] **collect 数据流回归修复**（2026-11）：VOI 重构把 `rand` 与 `uniform`
  的抽取顺序反了 → 行为流分叉 → 所有 b1703ac 之后的 run 与之前不可比。
  修复抽取顺序；新增 `test_behavior_stream_is_pinned` 钉死行为流哈希；
  顺带修复审计 #20：`flappy` 默认物理改为 latent 基准值，import 顺序
  不再影响物理。
- [ ] fork 校准：环境确定性，同一状态两种动作都能重放；按纪律不进真实
  训练集，只用于验证延迟动作后果与校准 readout（审计 E4）。
- [ ] 标量 EMA 是近因估计器（有效窗口 ~20 次激活，随 replay 顺序在
  0.73–0.96 摆动，全数据率 0.81）：读出已以计数格子为主，标量只作收缩先验；
  后续把标量也改成计数式或显著加大窗口，消除 per-seed 顺序噪声。
- [x] **序列级价值 + 标量化目标**（2026-11；纪律裁定：任务无关的通用形式
  可以人为设计，守卫测试保证其中无任务知识，但每个形式改动必须 A/B 验收）：
  `(feature, acting)` 格子学 MC 折扣 dopamine 回报（learn_episode 反向 pass
  更新，收缩先验 = 运行中的全局 baseline）；搜索目标从词典序 max-risk 改为
  标量化 `Σ hazard − feature_value_weight × V_f(terminal)`。
  fast 8 种子：max→sum 本身 110.9→144.9（seed 104 到 cap、seed 107 到 232），
  加 w=0.02 → 121.1/104.5。
  **全量 8 种子（w=0.02）：213.9 / 194.0**（vs 混合 151.8 / 141.0，
  paired +62，1.15 SE，8 种子下低于结论线但中位数 +53；seed 106 152→392、
  seed 107 130→476——首次稳定越过 4–5 根管，此前全仓库最高 312）。
  方差增大（sd 80→144）：价值项放大"接受接近段风险"的赌注，兑现或失败。
  trace：前兆 AUC 0.92+、kNN 支撑 0.806/近管道 0.923、chosen 累计风险最低。
  - [x] 全量 w=0 / w=0.02 对照完成：max→sum 单独 paired −8.4（0.42 SE，
    均值持平、中位数 +13、无门槛突破，最高 232）；**价值项 +70.5（1.5 SE）、
    中位数 +40，两个门槛跨越（392/476）全部来自它**——max→sum 清路，
    序列级价值是杠杆。fast 规模两臂混合（seed 104/107 反向），全量才是
    价值复利的场。
  - [ ] w 用留出 episodes 校准（`prediction_report` 基础设施已就绪），
    把 w 从拍的变成校准的。
  - [x] cell-Q-only 消融：不搜索，直接 `argmax_a V_f(s,a)` + hazard 惩罚，
    测 MPC 壳的真实贡献。
- [ ] 三套控制器收敛为一个 `ActionPolicy`（collect/eval/node 三模式），`act` 与
  `policy_result` 不再是平行宇宙。
- [ ] 计划级后验 `p1 = Φ(Δ/se)`（se 来自世界模型残差 + 支撑度）替代固定温度；
  lag>0 时动作语义建在计划层，不建在 `Q(s_t, a_t)`。
- [ ] VOI 反事实预算：`deficit × consequence × unresolved`，**叠加**在现有探索之上，
  不替换采样器；记录 propensity；空格子（如上 a1 n=0）就是预算的目标清单。
- [ ] 规划器驱动 / DAgger 采集 + 惊讶门控节点出生，缩小 80–95% OOD。
- [ ] 价值从 exemplar 迁到 predicate 统计（或删除装饰性 DERIVATION/组件机制）。

### E1 验收（paired per-seed）

- 前兆 AUC（t-3）≥0.75（当前 online 0.44）；`risk_active_frac` ≥0.30（当前 0.003–0.37）。
- 方向准确率 ≥0.6（当前均值 0.408 且换种子变号）。
- paired frames ≥131.6 基线，单种子跌幅 <20%；Brier <0.10。
- 判读：AUC 升而 frames 不动 → 瓶颈是终端价值/horizon（R4），标签时机不是约束；
  frames 也升 → 继续投入 VOI 采集（E2）。

### E1 结果（已跑，2026-11）

| 臂 | fast 8 种子 | 5000 步 8 种子 |
|---|---|---|
| 基线（instant，无泄漏修复） | 131.6 / 115.0 | — |
| instant + 泄漏修复 | **131.6 / 115.0（逐种子相同）** | 119.6 / 112.5 |
| 前置标签（默认） | 122.0 / 144.5（Δ -9.6，0.4 SE） | **87.1 / 72.0**（Δ -32.5，1.45 SE，6/8 更差） |

trace（seed 100）：

- 前兆 AUC 0.498 → **0.980–0.990**（fast 与 5000 步都成立）；特征激活 0 → 0.046；
  终端 AUC 0.309 → 1.000（fast）。
- 5000 步规则集被 `|f1|`（vy 幅值）类规则主导（n=239–427，p 0.63–0.95）。
  全拍计划平均风险 **1.04** vs 全不拍 **0.67**：拍翅后的预测状态大量触发
  vy 幅值规则 → 搜索把"拍翅"本身当成危险。
- 未达验收：`risk_active_frac` 0.066/0.115 < 0.30；frames 未达 131.6；
  Brier 未计算。

结论与判读：

1. **标签时机假设成立**：旧特征层确实只在死亡当帧出现；前置标签把风险读出
   变成真正的前兆检测（AUC 0.50→0.98）。
2. **但静态状态风险势无法选择动作**：幅值原语混淆 vy 方向，窗口 4 的标签把
   "pending 动作已注定的死亡"也算进当前状态。逐步探针确认：中段（来得及决策）
   两个计划 feat_risk 均为 0，贴管处全拍计划从 step 5 起持续触发
   `f1>2.241 & |f2|<0.793`（数据里真实，lift 9.3x，death-in-4 0.72）。
   规则学到的是"将死之鸟 vy 大"——结果与动作在 P(pain|状态) 里不可分。
   这与三份审计的"需要 `P(pain | s, a)`"结论一致，是当前最优先的结构缺口。
3. 下一步（P2 最小版）：**按有效动作条件化的特征统计**。因果上干净：进入
   `s_{t+1}` 的 pain 由 `history[lag]` 时刻发出的动作造成；沙盒预测状态的
   `pending[0]` 恰好是该动作。特征节点按 `(feature, effective_action)` 维护
   count/mean，`_feature_risk_batch` 用 `pending[0]` 查表；验收为
   "全拍/全不拍计划风险不再系统性倒挂 + paired frames ≥ instant 臂"。

## 质量瓶颈（2026-11，held-out 尺子之后更新）

现状（全量最好臂 self-eval 213.9/194.0）的存活分布是**三峰**的：3 种子
0 根（72 地板）、3 种子 1–2 根、2 种子 **4–5 根**（392/476）。但 held-out
尺子显示 self-eval 里约一半是过拟合（full held-out ~105.6，见上一节），
真实水位约 1–2 根管。瓶颈排序更新：

```text
B0 泛化差距（修正后）：self 213.9 → held ~146.8（7/8，差距 ~67）；
    极端双峰仍在（6 个 ~100，1 个 393.8）；full 协议评估成本病理待解
B1 闭环（DAgger 已实现；fast +45–55 有效；full 验证提前终止，
    **保持默认关闭，降级**——它只抬均值不抬中位数，且按少样本裁定
    不再是方向）
B2 对齐正样本缺失：四臂 held-out 中位数全部钉死 152 = 只稳定过 1 根管
B3 统计功效：held-out 协议已就位；数据扩张类方案按裁定降级
B4 世界模型管道事件失明：dx 一步误差 max 0.79
B5 残余人为成分：w 未校准；stride A/B 已跑（中性）；H 偏紧
```

**按少样本原则重读这些结果——三个独立实验指向同一个设计缺陷**：

```text
bt=0.8（原型砍半 800→510）  → held-out +19.9     少记反而好
OOD trust（不支持处少信）    → +18.2            少信反而好
DAgger（+45–55 帧, 2.18 SE）→ 中位数钉死 152    多数据只抬均值
```

即：**当前表征在记忆训练世界（exemplar 原型铺满状态空间 + 每世界特定
的 cell 统计），而不是从样本中提取可迁移结构**。学习效率不足的根因
在表征层，补数据只会加剧过拟合（DAgger 数据也采自训练世界，已经
放大了这个问题）。

机制路线（按提取效率排序）：

1. **D1 价值通道对齐谓词**（B2 的少样本版）：用高回报覆盖归纳
   "对齐接近"特征（|gap| 适中 + dx 适中）——从已有的少量管旁样本
   里提取对齐概念，让 V_f 在对齐带上有自己的分辨率。文法复用，
   零新数据。
   - [x] **已实现**（2026-11，commit `3a2355d`）：第二 `FeatureRuleBank`
     以 MC 折扣回报分布的上分位（默认 top 20%，`--value-quantile`）为
     标签归纳（`future_return_labels` 与 MC 反向 pass 逐位一致）；
     `install_feature_rules(channel="value")`；价值读出 = 痛根 ∪ 价值根
     并集（保住已有机制），痛侧统计/校准/反事实引导不碰价值根；
     `--no-value-features` 关闭。
     **fast 8 种子 A/B（配对，pain-horizon 8、w=0.02）**：
     OFF 121.1/104.5 [72,72,154,81,238,72,152,128]（与历史基线逐位一致，
     复现性确认）→ ON **181.8/153.0** [72,**315**,154,**312**,232,72,145,152]；
     paired +60.6、median +48.5；4 种子大幅上升（两个 ~312 ≈ 3 根管，
     fast 协议首次）、2 微降、2 不动；节点仅 +28。归纳产出 8 条宽支撑
     谓词（如 `f3<-0.070 & (f2<0.516|f2>2.937)` n=292 p=0.449 vs 基率
     0.2）——不是痛银行那种细条。
     **验收状态：以 fast A/B 为准（资源裁定后 full 复跑不再排程）**；
     3 种子一轮的 fast 复验随时可加（~2–5 分钟）。
2. **D2 激活共享特征子图**：OR/AND 分支的格子独立训练、读出取
   "根+激活分支"——同一批样本的信息被更多组合复用；方向分离
   （上升/下降）正是"每个样本提取更多"的直接形态。DERIVATION
   从装饰变计算 DAG。
   - [x] **已实现**（2026-11）：`subgraph_cells` 开关（默认 ON，
     `--no-subgraph-cells` 关）；`_feature_descendants` 子树枚举；
     前兆统计按根窗口分组（修复：分支此前挂在默认窗口，长窗银行
     标签错位）；价值格子更新扩展到激活分支（共享子节点去重）；
     风险/价值读出按"根+激活分支"做分子/分母池化（每节点贡献
     观测计数 + 收缩伪计数）。单测：ON 下 OR 两分支读数分离、
     OFF 下相等（根池化病理）；AND 分支继承根窗口。
   - [x] **fast 第 1 轮（种子 100–102）**：ON [72,315,154] ==
     OFF [72,315,154]（OFF 与 D1 状态逐位一致）→ 配对 Δ = [0,0,0]。
   - [x] **fast 第 2 轮（种子 103–105，D1 活跃区）**：同样 [0,0,0]。
   - [x] **判读（2026-11-20）**：两轮 6 种子 × 2 读出形式
     （池化 v1 / 最具体成员 v2）评估轨迹**逐位不变**。差分诊断证明
     机制生效（59 子节点带价值格子、43 带风险格子；读出
     max|Δ| 风险 0.14 / 价值 1.06），但**谓词内部读出精化的扰动
     比 plan 决策边际小一个量级**（chosen 0.69 vs 次优 1.47；价值项
     还要乘 w=0.02）。D1 能移帧靠的是**新谓词**（区域性价值地形
     变化），不是既有谓词内的精化——反向确认了杠杆的位置。
     **处置：默认 OFF**（`--subgraph-cells` 可开），机制保留；
     分支继承根窗口的修复只在 ON 模式生效（OFF 保持历史分组行为）。
3. **D3 读出全面谓词化**：风险与价值读出只走特征格子（原型仅作
   事件记忆，不承载泛化）——bt=0.8 与 OOD trust 的收益说明
   记忆通道在伤害泛化；谓词化把泛化负载全部交给坐标函数。
4. **D4 少样本可信度校准**：`--feature-value-prior` /
   `--feature-action-prior` / `--obs-trust`（已实现未验证）+ OOD
   trust——把"什么时候该信小样本统计"本身变成学出来的量。

已不是瓶颈的：边界死亡、死亡检测（前兆 AUC 0.92+）、反事实覆盖
（16/16 格子）、目标函数病理（max-risk 词典序已改标量化）、价值梯度
存在性（392/476 证明可被搜索利用）、价值归属时程（k 步目标中性偏负，
MC 归属成立）、决策机制（cellq 消融证明 MPC 壳必要且有效）。

D1 → D2 → D3 → D4 全部零新数据；验收即上面三个学习效率口径。

按杠杆排序：

- **B1 无闭环**：术语对齐——"规划器" = 沙盒预测（`LinearObservationModel`
  对候选动作序列滚动 H 帧）+ 穷举计划搜索 + 标量化打分（`policy_result`，
  滚动时域）。仓库现有两种**真实**数据，但只有一种进训练：

  | 数据 | 内容 | 现状 |
  |---|---|---|
  | 行为数据（`collect`） | LayerPair 策略，episode 40–50 帧，25% 见到第一根管 | ✓ 进训练 |
  | 规划器数据 | 评估时实际执行的轨迹（156–476 帧，含连续对齐 4–5 根管的完整经验） | ✗ 评估完即丢 |

  闭环（DAgger）= 评估时顺手记录规划器实际执行的 obs/action/feedback
  （真实转移，与 collect 同格式），与行为数据合并重训，迭代 2–3 轮。
  纪律不破：禁止的是沙盒**反事实预测**（假想状态）进训练；规划器**实际
  执行**的轨迹每步都有真实反馈，允许回流。DAgger + surprise 门控节点
  出生防图爆炸——把强种子的发现变成所有种子的先验，1.75 根 →
  系统性 4–5 根的直接路径。
  - [x] 代码入口（2026-11）：`evaluate(record=True)` 记录实际执行的
    `obs/next_obs/history/signals/done/episode`；`fit(dagger_rounds=N)` 每轮
    在训练种子的独立世界（offset 50000）执行规划器、追加真实转移、从零重训；
    每轮 episode id 递增，避免轨迹被错误合并。CLI：`--dagger-rounds/
    --dagger-episodes/--dagger-cap/--dagger-offset/--dagger-horizon`。默认关闭。
  - [x] fast held-out A/B（2026-11）：8 个训练模型 × 5 个 held-out 世界。
    short-window：116.4 → **170.1**，paired +53.7（SE 24.6，2.18 SE），7/8 同向；
    hybrid：134.9 → **179.8**，paired +44.9（SE 23.6，1.90 SE），6/8 同向。
    planner 数据量约 110–800 步/模型/轮，图节点约 1808–2222。`cap=400` 会截断强种子。
  - [x] cell-Q-only 消融（2026-11，fast held-out，hybrid+DAgger）：
    同一批训练好的 cell，决策从 MPC 搜索换成单步 `argmin_a [hazard(s,a)
    − w·V_f(s,a)]` → **68.2 / 72.0**（CI [51.7, 87.0]）vs 搜索 **179.8**；
    paired **−111.7**（SE 17.1，**8/8 同向下降**）。结论：MPC 壳不是装饰——
    学到的 cell 单独远不足以驱动决策；价值/风险读出依赖世界模型把整条
    H 步轨迹铺开后再打分。Q 格子是搜索的输入，不是策略。
  - [x] full 尺度验证提前终止（2026-11，用户决定）：2/8 模型完成
    （101：88.0 持平；103：90.0→168.0，+78），不足以判定；**DAgger 保持
    默认关闭**，不再投入。理由：所有臂的 held-out 中位数都钉死在 152
    （= 只稳定过 1 根管），DAgger 只抬均值不抬中位数，绝对水位不变；
    当前最大瓶颈是泛化/记忆化（B0）和对齐覆盖（B2），不是闭环。
  - [x] ~~跨种子聚合~~（2026-11 提出，**同日否决**）：本质是追加数据量的
    杠杆，与"少样本高提取"目标相反，不做。泛化修法走机制路线：
    `birth_threshold` A/B 与 OOD 门控读出（见"泛化瓶颈"一节）。
- **B2 价值通道无专属特征**："对齐接近"（|gap|<0.17 且 dx 适中）是反死亡
  状态，现有特征全部从死亡覆盖归纳，最接近的只有 |f3|<0.007 的细条——
  V_f 在最需要梯度的对齐带上回退 baseline。用高回报覆盖做一次
  价值通道归纳（文法复用），锐化梯度。
- **B3 统计功效**：+70.5 仅 1.5 SE；8 种子 × 每种子 1 条确定性轨迹、
  训练种子=评估种子。升级：每 seed 3 条评估 episode、held-out 种子
  120–139、bootstrap CI。
- **B4 世界模型管道事件失明**：dx 一步误差 max 0.79（生成/弹出不连续），
  穿越计划恰好在 dx 上做文章。事件感知动力学，中改。
- **B5 残余人为成分**：w=0.02 未校准；`step_stride=2` 跳过第一个受影响
  预测点；H 对 24 帧接近段偏紧。`--step-stride` 已暴露；`--stride-includes-lag`
  的 fast A/B 已跑：paired +3.8（0.11 SE）、median −11.6、4/8 正向，**中性**，
  默认保持关闭。剩余：full 复核、w 校准、H 消融。

顺序（按少样本裁定修订）：D1 价值对齐谓词 → D2 子图激活 → D3 读出
谓词化 → D4 可信度校准，全部零新数据；B1/DAgger 降级为默认关闭；
数据扩张类方案不作门槛。

## Held-out 评估（2026-11）

> **口径警告**：下表全部是**前兆标签**（`P(pain within k)`）时代的读数。
> 2026-11-20 起默认改为点事件 pain，这些臂需在新口径下重新测量；
> 表内数字只作机制历史，不能与点事件基准直接比较。

协议：训练种子 100–107，评估种子 120–124；每个训练模型在 5 个 held-out 世界上
各跑 1 条确定性 episode；区间按训练模型聚类（8 个 cluster），不是按 episode。

| 臂 | 协议 | n | mean | median | IQR | cluster 95% CI |
|---|---|---:|---:|---:|---:|---|
| short-window, dagger=0 | fast | 40 | 116.4 | 152.0 | 80.0 | [99.9, 133.9] |
| short-window, dagger=2 | fast | 40 | **170.1** | 152.0 | 94.0 | [126.8, 212.9] |
| hybrid (`--pain-horizon 8`), dagger=0 | fast | 40 | 134.9 | 152.0 | 80.8 | [109.4, 160.9] |
| hybrid, dagger=2 | fast | 40 | **179.8** | 152.0 | 150.2 | [144.9, 214.0] |
| hybrid, dagger=2, `--stride-includes-lag` | fast | 40 | 183.7 | 152.0 | 97.5 | [135.3, 239.8] |
| hybrid, dagger=2, `--decision cellq` | fast | 40 | **68.2** | 72.0 | 39.2 | [51.7, 87.0] |
| hybrid, dagger=2, `--value-horizon 4` | fast | 40 | 162.6 | 152.0 | 156.0 | [118.1, 211.5] |
| hybrid, dagger=2, `--value-horizon 8` | fast | 40 | 167.1 | 152.0 | 153.8 | [123.4, 214.2] |
| hybrid, dagger=0 | full | 35（**7/8，提前停止**） | ~146.8 | 待定 | 待定 | 待定 |
| hybrid, dagger=2 | full | 10（2/8，提前停止） | — | — | — | — |
| hybrid, `--birth-threshold 0.6` | fast | 40 | 133.9 | 152.0 | 80.0 | [114.4, 151.3] |
| hybrid, `--birth-threshold 0.7` | fast | 40 | 124.8 | 152.0 | 80.0 | [102.6, 142.7] |
| hybrid, `--birth-threshold 0.8` | fast | 40 | **154.8** | 152.0 | 82.5 | [120.6, 187.5] |
| hybrid, `--birth-threshold 1.0` | fast | 40 | 151.2 | 113.5 | 80.5 | [103.3, 202.6] |
| hybrid, `--ood-trust-scale 0.45` | fast | 40 | **153.2** | 152.0 | 113.8 | [124.4, 182.2] |
| bt=0.8 / ood 0.45 / 组合，20 世界复验 | fast | 7/8、6/8、6/8 | 134.8 / 129.7 / 126.5 | — | — | 提前停止，不作结论 |
| 默认（bt=0.45），20 世界 | fast | 0/8 | — | — | — | 提前停止 |

DAgger 的 paired 每模型差（同训练种子、同 held-out 世界）：

```text
short-window: +62.4/+14.4/+114.4/+44.8/+98.0/+156.4/+0.2/-60.8
              mean +53.7  median +53.6  SE 24.6 (2.18 SE)  7/8 positive
hybrid:       +96.4/+35.4/+115.8/+78.2/+80.6/-32.0/+61.8/-77.0
              mean +44.9  median +70.0  SE 23.6 (1.90 SE)  6/8 positive
```

两个窗口臂方向一致：DAgger 把 fast held-out 均值提高约 35–55 帧，但 hybrid
的种子间方差明显变大（IQR 80.8→150.2），有两个种子反向。注意 fast 协议
`cap=400` 会截断强种子，真实收益可能被低估，也可能把“活得更久”混进
cap 效应；full 协议仍需复核。

B5 的 `step_stride` 覆盖第一个受影响点：hybrid + DAgger 下
`--stride-includes-lag` 的 held-out A/B 已跑（2026-11）：183.7 / 152.0 vs
179.8 / 152.0，paired `+82.0/-33.6/-99.8/-32.2/-112.8/+9.0/+184.6/+33.4`，
mean **+3.8**、median **−11.6**、SE 34.5（0.11 SE）、4/8 正向。**中性**，
维持默认关闭；开关保留，等 full 协议复核后再决定是否定案。

cell-Q-only 消融（同 agent，只换决策）：68.2 vs 179.8，paired **−111.7**
（SE 17.1，8/8 同向）——MPC 壳承重，cell 不是策略（见 B1 条目）。

价值 cell 的 k 步 bootstrap 目标（`--value-horizon 4/8`，vs MC 全 episode）：

```text
k=4: 162.6 / 152.0  paired -17.3 (SE 27.1)  3正/1零/4负
k=8: 167.1 / 152.0  paired -12.8 (SE 22.6)  4正/1零/4负
```

**中性偏负**：截短价值归属没有帮助，反而丢了长时程"对齐有回报"的信号
（IQR 从 150.2 涨到 ~155，弱种子 104/107 仍弱）。结论：MC episode 归属
不是价值 cell 的主要瓶颈；短时程 TD 假设在这个读出上不成立。

fast 协议：1500 步、3 epoch、1000 节点、H=8、cap=400。full 协议：5000 步、
5 epoch、5000 节点、H=10、cap=6000。short-window 指 `--pain-horizon 0`。

## 泛化瓶颈（2026-11，held-out 尺子暴露）

held-out 协议上线后暴露：**self-eval 与 held-out 有实质差距**。full 协议
（5000 步、5 epoch、5000 节点、H=10、cap=6000）的 held-out 在 **7/8 模型
完成后被提前终止**（用户决定——见下）：

```text
per model mean: 88.0 / 90.0 / 104.0 / 106.2 / 120.6 / 125.0 / 393.8（7/8）
partial mean ≈ 146.8
对照 full self-eval（hybrid + w=0.02）：213.9 / 194.0
```

**判读修正（2026-11）**：早先基于 6/8 的"训练越多 held-out 越差"（fast
134.9 > full ~105.6）是**不完整数据的假象**——第 7 个完成的模型
held-out 393.8，把 partial mean 拉到 146.8，高于 fast 的 134.9。仍然成立的是：

1. self-eval 213.9 vs held-out ~146.8：差距约 67 帧，**仍然显著但减半**；
2. **极端双峰**：6 个模型 ~88–125，1 个模型 393.8——种子抽签没有消失，
   只是形式从"三峰"变成"六低一高"；
3. `birth_threshold 0.8` 的 +19.9（完整 8/8 配对数据，两侧同协议）
   不受影响，是可靠的。

**full 协议为什么跑不完（已记录为病理）**：评估成本 ∝ 存活帧数 ×
H=10 的 1024 计划枚举，模型活得越久单集 eval 越贵；seed 104 的
393.8 均值让 5 个评估世界耗时数小时，整个 run 因此被终止。
若回到 full 协议，先要解决 batch 读出里的 Python 循环（memory 查找）
或降低单集成本；当前活跃协议是 fast + 20 评估世界。

机制解释（与既有审计吻合）：评估状态 **80–95%** 距最近原型超过
`birth_threshold`（OOD）。图按"距离 > 阈值就出生"无上限增长，5000 步
数据把训练世界的状态空间铺满；特征规则是阈值谓词，拟合训练分布里的
特定 gap 组合；价值 cell 学的是"这些训练轨迹最终活多久"。held-out 世界
的管道序列不同，这些原型的统计就不再成立。DAgger 数据也采自训练种子
世界，目前**放大**同一个问题（fast 下 +45–55 的收益有多少能在 full/
held-out 复现，正在验证）。

今天的消融排除了其他嫌疑：

- 不是归属时程：`--value-horizon 4/8` 中性偏负（−17.3 / −12.8）；
- 不是决策机制：`--decision cellq` 崩到 68.2（−111.7，8/8 ↓）——
  能力在"世界模型 + 轨迹展开"组合里，而这个组合本身是好的（误差 1e-9）；
- 不是动力学：一步误差 1e-9。

### 下一步实验（按杠杆排序，2026-11 用户裁定后修订）

**跨世界池化被否决**：本质是追加数据量的杠杆，与"少样本高提取"的目标
相反，不做。泛化问题的修法走机制路线：

1. **`birth_threshold` 泛化 A/B（已出结果）**：0.8 是甜点——held-out
   `154.8`（+19.9，SE 12.0），概念节点砍半（~800→~510）；1.0 太粗
   （中位数崩到 113.5）。**首个"少记反而好"的直接证据**。注意扫参
   选择偏差；20 世界复验已提前停止、部分均值更低（见"泛化瓶颈"一节），
   默认值保持不变，待默认臂的同尺度对照补齐后再定；
2. **OOD 门控读出（已实现，独立有效）**：`--ood-trust-scale` 按距离向
   基线收缩，+18.2（SE 14.9，6↑/1↓/1平）——80–95% OOD 查询不再拿到
   过度自信的统计。组合臂复验同样提前停止，保持默认关；
3. **B2 价值特征**：用高回报覆盖归纳"对齐接近"特征（抬中位数 152）；
4. **B4 事件感知世界模型**（管道生成/弹出）。

注意：任务基线 1026.9 证明观测信息足够；旧风险头路径（190.1）从未做过
held-out 测量，应尽快补上同尺度的对照，确认泛化差距是节点图特有的还是
所有学习器共有的。

## 当前状态

默认 `substrate_sandbox_search.py`（世界反馈价值模式）最近一次 8 种子测量
（jobs=8；terminal_fraction=`0.15`；risk_updates=`8000`；value_updates=`3000`）：

- 存活：平均 **190.1 帧**，中位数 **232**，范围 **57–312**
- 动力学识别正常：lag=`3`，动作增益约 `0.020`，一步预测误差约 `1e-9`
- 价值留出集：pain AUC `0.910`；终止风险 AUC `0.971`，Brier `0.019`；
  近管道风险 AUC `0.757`，缺口内/外风险均值 `0.012/0.124`
- 练习覆盖（8 个种子）：平均 episode 约 43.1 帧；管道死亡 128 次，
  边界死亡 792 次，管道窗口 921 步

当前结论：风险头已经能区分终止风险，搜索能避开眼前的危险并把存活从 14.5 帧提高到约 190 帧；
但它仍不能稳定对准管道缺口，和 1026.9 帧的任务基线差距很大。修正后的 beam 评分不再隐式改变
目标：2 种子 pilot 中 `terminal` 聚合只有 25 帧，`max` 聚合 H=78 约 155 帧，与 exhaustive
`max` 的 152 帧接近；默认 `terminal` exhaustive 约 272 帧。没有证据表明长时域 beam 带来增益。
搜索器本身仍没有任务专用 gap 规则。

节点中心第一版（`substrate_nodes.py` / `node_runtime.py`）已经开始：不再用 `terminal` 标签训练
风险头，风险由节点/边从 `dopamine/pain` 信号中用 MC/TD 信用分配学习；8 种子均值 107.9 帧，
见下文。它还没有超过旧路径，但把“风险”从人工标签改成了节点自身的期望 valence。

## 本轮修复（代码）

- `NodeGraph.prune` 现在会同时删除与被剪 edge node 相连的 association/membership 边，并从
  memory 的 `members` 中移除该 id；不再留下会触发 `KeyError` 的悬空边。
- 图记忆的 transition edge 现在按“当前转移”更新 value；此前每次遇到同一条边都会把 value
  重置为 base，导致边价值几乎不学习。新增回归测试验证危险边会收敛到高 pain。
- `TransitionReplay` 新增 `for_stream(count)`：按“每个 outcome 类保留全部转移”分配容量。
  sandbox、generic、图搜索和三个 trace 都改用它，避免 `capacity=len+1` 时静默丢掉一半存活样本。
- `--risk-action-dim` 现在必须落在 `[0, ACTION_DIM]`；`practice` 和所有 CLI 在开始长时间采集前校验。
- 风险特征统一为 **终止标签所属的 next state**；sandbox probe 的 `pipe_dx/y/vy` 元数据也改为 next state。
  generic、图搜索、图记忆和 mixed-node 的全局风险头使用同一约定。
- `TerminalRiskHead` 增加 `interaction_start`，交互项可只作用于观测块；按类别重采样后做先验截距校正，
  训练更新不受推理期校准偏移影响。
- beam 不再隐式改变评分目标：`engine` 只决定搜索方式，风险聚合完全由 `--risk-aggregate` 决定。
  beam 在 `terminal` 聚合下没有中间信号，必须显式使用 `max`/`sum` 才能作为长时域对照。
- `target_gap` 接收实际 horizon；任务基线不再固定按 `H=14` 选管道。
- 图搜索的 previous labels 改为按 action-prefix 缓存，支持 exhaustive/beam 和任意 batch 重排。
- 所有 CLI 增加范围校验；`kmeans`/图记忆/mixed memory 校验空数据；图记忆和 mixed memory 可重复 `fit`。
- `WorldFeedbackValueHead` 按 seed 初始化；`trace_generic_training` 改为使用与生产一致的
  normalization、priority 采样和风险先验校正。

## 节点中心改造（第一版）

目标：风险不再由 `terminal` 标签和 `TerminalRiskHead` 定义，而是由节点/边从 world 的
`dopamine/pain` 信号中自己学习。人类只提供 `obs/action/valence`，`done` 只用于 episode 边界。

已完成：

- `NodeGraph` 增加 `ActionEdge`：`(source, action) -> target`，含 count、`q_pain`、`q_dopamine`、
  预测误差；删除节点时同步删除动作边。
- `NodeRuntime`：在线概念节点出生（距离阈值）、Monte-Carlo 回报 + TD(λ) 信用分配、
  soft-neighbour 运行时评分、未知动作的 optimism/uncertainty、episode 边界 reset。
- `substrate_nodes.py`：learned observation model 做候选轨迹预测；节点 valence 只给终点打分；
  没有终止风险头、`GAP/2`、`target_gap`、碰撞窗口或 near-pipe 采样。
- 新增节点运行时/动作边回归测试；当前全量 62 个测试通过。

测量（8 种子，5000 步，H=10，10 epochs，max_nodes=5000，memory_weight=0.1）：
均值 108.6，中位数 85.5；状态尺度由观测标准差估计，不写任务常量。概念节点约 2220–2337，
没有触及 5000 上限；graph nodes 约 4900–5105。仍低于旧风险头路径 190.1 和任务基线
1026.9；这是第一版节点路径，不是最终性能结论。

下一步：

- [ ] 节点生命周期：按预测误差/valence surprise 出生，按效用/冗余剪枝；当前只有距离出生，
  没有在线剪枝。
- [ ] 事件记忆：memory 节点应按 valence 变化/事件分段，而不是整个 episode。
- [ ] 节点级规划：当前 MPC 仍依赖 `LinearObservationModel` 预测轨迹；下一步让动作边直接参与
  多步 rollout。
- [ ] 弱种子（25/34/72 帧）诊断：检查未知动作 fallback、节点覆盖和动作边不确定性。
- [ ] 评估：用“死亡前 k 帧 vs 匹配存活”的前兆指标替代 terminal 分类 AUC。
- [ ] 环境：free-threaded `python3.15t` 的 numpy DLL 被应用控制策略阻止；当前 GIL 构建
  自动用进程池，free-threaded 构建会自动切到线程池。

## 运行时组件与三态概念节点

- [x] `node_components.py`：`RuntimeComponent`（difference/capacitor/resistor）只做运行时数值计算，
  不进入 `NodeGraph.nodes`；`ComponentBank` 把组件绑定到基础节点。
- [x] 三态：先平滑（EMA），再饱和到 `[-1,1]`，按 `threshold` 舍入为 `+1/0/-1`；默认
  `[-1,-0.2]` 抑制、`(-0.2,0.2)` 保持、`[0.2,1]` 激活；`hysteresis` 可选，防止边界抖动。
- [x] 三态是普通 concept 节点（`role=component_state`、`component_id`、`sign`），参与关联、
  transition、memory、valence EMA 和持久化。
- [x] `NodeGraph.to_state/from_state` 持久化节点、边、动作边和组件状态；`ComponentBank.state_dict`
  持久化组件规格、当前三态、平滑值和节点 valence。
- [x] `NodeRuntime.bind_component` 可把组件绑到运行时节点；三态节点已参与运行时关联/转移。
- [x] 实际接入对照：8 种子、`difference+capacitor`、`min_count=8`、`max_components=128`，
  baseline 107.9 帧 vs components 107.9 帧，逐种子相同；图从 2000 增至约 2512 节点。
  **中性/负结果**：三态节点是基础节点激活的确定性函数，并行动作边没有带来新信息。
- [x] transition gating 尝试：5000 步/128 组件学到 156 个 gated key，评估 22 步命中 0 次，
  8 种子仍为 107.9 帧。**负结果**：signature 覆盖太稀疏，需要共享/回退或事件级记忆。
- [x] 旧节点系统检查：`graph_memory` 初始价值改为由 world pain signal 估计，不再用
  `terminal`；`NodeGraph` round-trip 覆盖全部边类型；三个 trace 脚本正常退出。
- [x] 实际追踪（`trace_node_runtime.py`，seed 100）：世界模型 y/vy 误差 ~1e-9；节点价值
  offline holdout AUC 0.977，但 online terminal AUC 0.442；近管道 chosen/other 动作边
  135/23.6，双动作支撑仅 0.52；方向准确率 0.696。瓶颈是终止状态/反事实动作的在线覆盖，
  不是世界模型。
- [x] 训练过程追踪（`trace_node_training.py`，max_nodes=5000）：约 2339 概念节点、graph nodes
  5068；整体双动作支撑 0.18、近管道 0.085；终止节点 75 个。
- [x] 通用 interaction：每条真实转移建立 association/transition；**没有原生事件定义**，
  memory 只由 world pain 信号创建；安全步只检索/加强已有 memory。5000 步后
  assoc/trans/mem = 2915/2640/89。
- [x] memory 通过 membership 关联；激活节点（含一跳 association）反查并加强 memory
  trace/membership；episode 结束 decay，低支持 memory 遗忘，高 value memory 保护。
- [x] 随机 memory 追踪（`trace_memory_change.py`，3 种子）：pain memory 约 8–89 个，
  value 全为 1.0、members≈3；learning 前后新增 0–1 个；首轮 memory 保留率高，但很多
  高 value memory 的 trace/membership 衰减到 0，保留但难以激活。
- [ ] value-aware memory retrieval：高价值 pain memory 的 trace/membership 可能衰减到 0，
  需要按 value 而不只按 trace 找回。
- [ ] memory 成员仍约 28 个，事件分段偏粗；需要更细的事件边界。
- [x] transition edge node 用 MC/TD target 更新价值；memory value 以默认 0.1 权重进入终点评分；
  对称 association 价值平滑实测有害（232→141 帧），默认关闭，association 只参与 memory 检索。
- [x] 决策改为沙盒轨迹搜索：线性模型预测 H=10 步，每 2 步沿 association 读
  `valence:pain` 信号取 max，终点用节点/memory pain + dopamine；不写管道/碰撞常量。
  8 种子仍 108.6 帧，耗时 126s；pain 信号在预测状态上仍很稀疏，逐状态避管没有明显生效。
- [ ] 训练连续 pain/terminal 读出（从 world pain/dopamine 学习，不用 terminal 标签），
  让它能泛化到沙盒预测的反事实状态；这是逐状态避管生效的前提。
- [x] 组合特征规则层（`node_features.py`）：输入坐标的 raw/`|x|` 阈值原语 + AND/OR 组合，
  按覆盖尚未解释的 pain 事件选规则，held-out episode 验证。
- [x] 规则在 epoch 0 后安装成 `role="feature"` 图节点：每步评估谓词并激活，节点保存
  `P(pain|feature)`；特征作为 action edge source 参与转移学习；memory 成员包含活跃特征，
  可按特征召回；沙盒预测状态直接算特征风险。fast 8 种子 131.6 帧，关闭特征层 94.6；
  seed 100 312 vs 113，seed 107 152 vs 77。
- [x] pain 事件会把活跃特征补写进已有 memory（不再只在 memory 创建时写）；fast 23 个
  memory 中 16 个有 feature 成员，特征召回生效。
- [x] 发现阶段补强：pain 样本少时自适应 minimum support；episode 交错 held-out；
  pain 阈值外扩 margin；覆盖选择后用未使用坐标/原语补充不同 cause。fast 规模已能选出
  撞管特征 `(|f2|<0.274 & |f3|>0.787)`（train 3/3，test 2/2）和边界特征。
- [x] 复合特征现在是共享特征子图：原语、AND/OR 组合都是节点，组合节点用 derivation 边
  指向子节点；相同原语只存一份。根特征节点承担 action source、memory 成员和风险读出。
- [ ] 特征发现还没有接入不变量分段（pain 状态集合的组内方差/单边性）；当前主要靠
  阈值原语 + AND/OR 搜索。
- [ ] 特征生命周期：等价特征合并、低效用/冗余特征剪枝、特征节点预算。fast 规则集里
  仍有很多边界变体，需要压缩。
- [x] 快速开发协议：1500 步、3 epoch、1000 节点、H=8、cap=400；单种子约 11s，8 种子约 41s。
- [x] 组件出生改为按节点 valence 方差（相对节点总体 median+std）自适应；低效用组件按
  sign-pain 方差和访问次数剪枝。5000 步 seed 100 自动出生 76 个组件，存活仍 232。
- [ ] 让 association/transition 结构进一步参与动作选择/信用分配；当前 memory 只以 0.1 权重
  进入终点价值，存活与不读取时基本持平。
- [ ] 节点生命周期：上限/剪枝/合并；避免新终止状态 alias 到安全节点。
- [ ] 用 on-policy/DAgger 采集 MPC 自己产生的终止状态与近管道反事实动作，再重训价值/Q。
- [ ] 给终止状态更连续的价值读出不只依赖最近原型；检查 boundary 附近节点的分辨率。
- [ ] 组件仍保留 `component_min_count`/`max_components` 作为样本/资源保护；下一步去掉，
  让组件算子选择和预算也由训练统计决定。
- [ ] 让三态节点承担基础状态没有的角色：eligibility/信用分配、事件 memory、转移条件；
  不要只增加同源动作边。
- [ ] 把三态节点接入 NodeRuntime 的 MC/TD 信用分配（当前只做 EMA valence）。
- [ ] 组件生命周期：哪些节点派生哪些组件、每节点组件数上限、按效用/冗余剪枝，避免图爆炸。
- [ ] 旧 `add_derived` 数值派生节点仅保留为实验兼容；新路径不再用它承载概念。

任务无关原型（`substrate_generic.py`）已跑通：规划器只读 `obs/action/feedback/terminal`，
`LinearObservationModel` 学习完整观测转移，价值/风险只吃观测和动作历史。3 个种子存活
73/232/319 帧（均值 208）；单步留出误差约 `4e-3` 到 `1e-7`，seed 100 的最大训练误差 `0.79`
来自管道生成的不连续事件。下一步需要非线性/事件感知的观测模型，而不是让适配器手写管道规则。

训练波次诊断（`trace_generic_training.py`，seed 100，5000 步，coverage 0.1）：

- 价值/风险随更新稳定上升：pain AUC 约 `0.83→0.94`，risk AUC 约 `0.96→0.975`，
  near-pipe AUC 约 `0.75–0.78`，Brier 约 `0.011`；没有训练崩溃。
- 每 500 步的近管道样本在 `3–81` 间波动，近管道实际拍翅样本在 `1–8` 间波动；
  数据覆盖是非平稳的。
- world model 的 `y/vy` 误差约 `1e-9`，但管道维度的误差在 seed 100 为
  `dx≈1.6e-3, gap≈6e-5`，其他种子约 `1e-8`；这对应管道生成/弹出现象。
- 当前主要失效点不是优化器，而是**数据覆盖 + 事件感知世界模型**。下一步应优先稳定这两项，
  而不是继续换种子或调最终搜索参数。
- 输入敏感度检查（`trace_sensitivity.py`，5000 步）：世界模型使用 `dx/gap/y/vy` 和滞后动作
  `a3`；风险头仍主要使用 `y`（扰动 `y` 使 AUC 下降约 `0.279`），`gap1/gap2/dx2` 约
  `0.004–0.006`；价值头主要使用 `vy/gap1`。结论：完整观测是存在的，但风险/价值没有真正
  利用管道和动作信息，不能把“输入中有”误认为“模型意识到了”。下一步需要硬负样本/近管道
  加权、动作条件和显式的管道风险辅助目标，而不是继续加普通特征。
- 多信号对照（`trace_signal_world.py`）：raw、`d1/d2`、多时间常数缓冲、全部信号的
  one-step/multi-step 误差没有稳定改善；seed 100 的 one-step 均值约 `6.20e-4`（raw）到
  `6.60e-4`（buffer），多步约 `3.8e-3`。当前观测已是完整状态，缓冲/差分冗余，低通还会
  抹平事件边缘；不能替代事件/动作条件模型。
- 输入敏感度（`trace_sensitivity.py`，5000 步）：世界模型对 `dx1/dx2/vy/gap/y` 和滞后动作
  `a3` 都敏感；风险头仍主要依赖 `y`（AUC 下降约 `0.279`），`gap1/gap2/dx2` 约
  `0.004–0.006`；价值头主要依赖 `vy/gap1`。风险头有完整输入，但“只看高度”仍能获得
  大部分全局 AUC。下一步需要硬负样本/近管道加权、动作条件和显式的管道风险辅助目标。
- 通用优先回放：replay 支持按世界模型预测误差加权，`substrate_generic.fit` 当前默认使用该
  优先级；3 个种子（coverage=0.1）存活 73/232/319。风险敏感度仍以 `y` 为主；
  世界误差优先级没有让风险头真正利用管道，下一步应使用风险/价值误差更新优先级。
- 混合节点原型（`mixed_nodes.py` / `trace_mixed_nodes.py`，seed 100，5000 步）：
  全局风险头 AUC `0.975/near 0.752`，Brier `0.011`；痛驱动混合节点 AUC `0.816`、
  near `0.773`、Brier `0.133`，near-inside/outside `0.061/0.753`；连续记忆版 6/64 节点，
  AUC `0.914/near 0.858`、Brier `0.021`，near-inside/outside `0.032/0.044`；
  分层版 26/64 节点（16 稀有槽位），AUC `0.947/near 0.773`、Brier `0.018`，
  near-inside/outside `0.015/0.035`。连续/分层版 near AUC 已接近或超过全局头，
  但绝对风险差仍小；下一步需要按验证增益/重建误差选稀有节点并做风险校准。
- 统一图基建：新增 `node_graph.py`，实现 concept/transition/memory/derived 节点，
  association/transition/membership/derivation 边，转移边上的 edge node，
  激活传播、痕迹、回忆、衰减和剪枝。当前只作为基建原型，尚未接入 Flappy 控制器。
- 图记忆接入：`graph_memory.py` 将观测原型作为概念节点、连续原型作为 transition
  edge node、episode 作为 memory node。seed 100 约有 48 概念/219 edge node/98 memory；
  全局风险头 AUC `0.975/near 0.752`、Brier `0.011`，图记忆风险 AUC `0.876/near 0.596`；
  图敏感度中 `gap2/gap1/y` 约 `0.114/0.084/0.069`，比全局风险头更依赖管道。
  edge value 现在按实际转移累积；此前每次遇到同一条边都会重置为 base。
- 图搜索接入：`substrate_graph_search.py` 用线性观测模型预测、图概念/边节点同时学习
  pain/dopamine，记忆节点价值参与事件级风险；搜索不读取管道/缺口常量。
  修正后 seed 100/101 为 312/232 帧（均值 272），图约 434–437 节点、2703–2803 边；
  与前文全局反馈同种子结果相同，当前图风险没有独立增益。旧的 4 种子
  312/232/480/312 不再作为当前结果。
- `substrate_graph_search.py` 已加入 `--seeds/--jobs` 进程级并行；图训练和搜索按 seed
  独立运行，避免串行等待。

动作轨迹诊断（`trace_feedback.py 100`）：管道附近 139 个决策中，“缺口在上应拍翅、缺口在下不应拍”
的方向准确率约 `0.29`；需要上拍时拍翅率约 `0.08`。风险头对全不拍/全拍的平均风险为
`0.26/1.00`，从未认为全拍更安全。它学到的主要是“管道附近有终止风险”，不是“当前拍翅能对准缺口”。
穿管时刻的选中计划/全不拍/全拍落入缺口比例：seed 100 为 `0.843/0.753/0.663`；
平均穿管前拍翅 `0.43` 次，平均 gap 误差约 `0.104`。这说明计划最多是宽缺口下的弱调整，
不是稳定的动作规则。
这解释了为什么存活变长但策略仍不会过管。2 种子 pilot：`--risk-aggregate max` 平均约 152 帧，
`sum` 约 91 帧，仅在预测穿管时刻查询的 `crossing` 约 232 帧，默认 `terminal` 约 272 帧；
说明只改变聚合/查询时点不够。当前风险头整体 AUC 约 0.97，但只看管道附近的近邻状态时
AUC 约 0.72–0.76；需要先提升近管道风险模型，再上动作条件化或完整碰撞/长时域预测。

P0 尝试（`--risk-interaction-dim 6`，有/无 `--near-pipe-fraction 0.4`）：交互项现在只作用于
观测块；近管道 AUC 约 `0.669/0.656`，两种配置的存活都只有 25 帧。默认已恢复
`risk_interaction_dim=0`、`near_pipe_fraction=0.0`，保留这些选项用于后续实验。
这说明增加交互特征/近管道采样并不是瓶颈；下一步应直接做动作条件化的碰撞/风险模型。

动作历史输入尝试（2 种子 pilot）：`--risk-action-dim 4` 平均存活 272 帧、近管道 AUC `0.710`，
与默认 272 帧、`0.719` 基本持平；与 `--risk-aggregate crossing` 组合降到 119.5 帧。
默认已恢复 `risk_action_dim=0`。结论：仅在风险头拼接最近动作历史不足以学会“拍翅避免撞管”，
需要动作条件的 n-step 风险、`Q(s,a)` 或完整碰撞模型。

事实动作预测检查：seed 100 近管道 50 个状态中只有 3 个实际拍翅，seed 103 的 80 个中只有
4 个；这些实际拍翅样本没有终止。状态风险对整体终止的 AUC 约 0.97，但加入动作历史后
near-pipe AUC 仍约 0.62–0.78，没有稳定提升。由于不能生成反事实结果，当前数据无法估计
“同一状态下拍翅/不拍翅”的差异；必须先通过探索在管道附近实际采集两种动作，才能检验动作效果。

## P0：先让价值头可验证

- [x] 建立价值评估集：每 5 个练习 episode 留出 1 个，分别统计终止状态与存活状态的预测疼痛。
- [x] 报告疼痛预测的区分度（AUC）和风险头校准误差（Brier），而不只看存活帧数。
- [x] 给价值头加真实转移回放；对终止样本做平衡采样，避免稀疏终止信号被大量存活样本淹没。
- [x] 比较在线 TD 与实际 episode 回报（Monte Carlo）两种目标，保留可复现的较优方案：
  MC 的 pain AUC 约 `0.912`，TD 约 `0.688`；当前默认使用 MC 回报。
- [x] 明确验收：价值头在留出集上明显高于随机；终止风险显著高于安全状态。当前风险头已过此门槛。

诊断依据：早期 TD 价值头的 pain AUC 约 `0.188`；加入 replay、MC 回报、next-state 风险特征、
重采样先验校正和带平方/交互特征的 `TerminalRiskHead` 后，pain AUC 约 `0.910`，风险 AUC 约
`0.971`，Brier 约 `0.019`。剩余问题是长时域管道对齐，不是疼痛预测本身。

## P1：补足练习数据覆盖

- [x] 统计当前 5000 步练习中的终止类型、管道窗口步数、管道死亡次数和 episode 长度分布，写入实验输出。
- [x] 改进探索：更早使用模型中性动作率，并用低拍翅率先验避免早期 episode 过早结束；新颖度仍只作为特征。
- [ ] 研究模型中性动作率探索与新颖度探索的组合，记录覆盖率和终止样本数量。
- [ ] 明确验收：留出集中管道死亡和边界死亡都有足够样本；当前 8 个种子共有 128 次管道死亡和
  792 次边界死亡，覆盖已明显改善，但仍不足以证明管道风险泛化。

当前覆盖检查（8 个种子，5000 步）：平均 episode 约 43.1 帧；管道死亡 128 次，
边界死亡 792 次，管道窗口 921 步。

## P2：处理规划时域

- [x] 量化 `H=14` 与第一根管道约 78 帧到达时间之间的差距；实现 beam H=78 作为扩展时域对照。
- [x] 让价值在搜索时域末端承担更长时间的风险：比较末端 value bootstrap 与 beam 扩展时域。
- [x] 在反馈价值模式中保留“无数值/特征无效”检查，不加入任务专用边界或管道规则。
- [ ] 明确验收：价值排序仍不能稳定区分“暂时安全但未来危险”和“可继续运行”。2 种子 pilot：
  beam H=78 默认 `terminal` 只有 25 帧（没有中间信号），显式 `max` 聚合约 155 帧；
  exhaustive `max` 约 152 帧，默认 `terminal` 约 272 帧。修正目标混淆后，没有证据表明
  长时域带来增益；下一步需要动作条件化的风险/价值，或能预测管道碰撞的长时域模型。

## P3：实验与文档

- [x] 重新跑并记录当前 gap 任务基线的完整 8 种子结果：平均 1026.9 帧，中位数 659，
  范围 123–3491。
- [x] 同步 README 的当前反馈模式测量：当前默认 8 种子平均 190.1 帧。
- [x] 每次价值改动都保留：测试、配置、随机种子、留出集指标和负结果。
- [x] 不把沙盒反事实预测写入真实转移或价值训练集。

## 完成标准

1. 价值头在真实留出转移上能稳定区分终止风险，而不只是拟合训练集。**已达到风险 AUC 0.971。**
2. 练习数据覆盖管道和边界相关状态。**部分达到：已有管道和边界死亡样本，但管道风险泛化仍不足。**
3. 搜索在 **不写 Flappy gap/碰撞专用评分** 的前提下，能利用学到的反馈价值避开危险候选。
   **部分达到：能避开眼前终止风险，但管道附近的方向动作准确率仅约 0.29，未学会对准缺口。**
4. 外部存活率提升，且不把任务基线（gap 评分）混入内在价值训练。
   **部分达到：从 14.5 帧提升到 190.1 帧，仍低于 1026.9 帧任务基线。**
