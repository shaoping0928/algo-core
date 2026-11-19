# TODO

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

## 质量瓶颈（2026-11，序列级价值之后）

现状（全量最好臂 213.9/194.0）的存活分布是**三峰**的：3 种子 0 根（72 地板）、
3 种子 1–2 根、2 种子 **4–5 根**（392/476，全仓库首次连续多管）。均值
1.75 根/seed vs 任务基线 ~12 根。瓶颈不再是均匀能力缺失，而是
"能否拿到对齐证据"的种子级抽签。

已不是瓶颈的：边界死亡、死亡检测（前兆 AUC 0.92+）、反事实覆盖
（16/16 格子）、目标函数病理（max-risk 词典序已改标量化）、价值梯度
存在性（392/476 证明可被搜索利用）。

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
  - [ ] full 协议 held-out 复核（hybrid、cap=6000）与 dagger=1 vs 2 的
    轮数消融；同时检查 DAgger 是否把评估轨迹变成训练集导致自评泄漏
    （当前实现用 offset 世界，与 `--eval-seeds` 世界不相交）。
  - [ ] **跨种子聚合**（2026-11 提出）：fast A/B 显示 DAgger 抬地板
    （floor≤90 从 18/40→11/40，<100 均值模型 3→1），但弱种子的 planner
    数据只有 70–80 帧/轮，闭环自举失败（seed 107 held-out 反而下降）。
    根源是每种子独立训练。候选：一个全局模型吃全部训练种子的
    行为+planner 数据（观测已按全局 std 标准化，无任务知识），或至少把
    强种子的 planner 数据并入弱种子训练池做迁移 A/B。这是攻击
    "种子抽签"的直接路径，排在 B2 之前。
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

顺序：B3+B5（尺子可信）→ B1（闭环）→ B2（价值特征）→ B4（事件模型）。

## Held-out 评估（2026-11）

协议：训练种子 100–107，评估种子 120–124；每个训练模型在 5 个 held-out 世界上
各跑 1 条确定性 episode；区间按训练模型聚类（8 个 cluster），不是按 episode。

| 臂 | 协议 | n | mean | median | IQR | cluster 95% CI |
|---|---|---:|---:|---:|---:|---|
| short-window, dagger=0 | fast | 40 | 116.4 | 152.0 | 80.0 | [99.9, 133.9] |
| short-window, dagger=2 | fast | 40 | **170.1** | 152.0 | 94.0 | [126.8, 212.9] |
| hybrid (`--pain-horizon 8`), dagger=0 | fast | 40 | 134.9 | 152.0 | 80.8 | [109.4, 160.9] |
| hybrid, dagger=2 | fast | 40 | **179.8** | 152.0 | 150.2 | [144.9, 214.0] |
| hybrid, dagger=2, `--stride-includes-lag` | fast | 40 | 183.7 | 152.0 | 97.5 | [135.3, 239.8] |
| hybrid, dagger=0 | full | 待填 | | | | |
| hybrid, dagger=2 | full | 待填 | | | | |

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

fast 协议：1500 步、3 epoch、1000 节点、H=8、cap=400。full 协议：5000 步、
5 epoch、5000 节点、H=10、cap=6000。short-window 指 `--pain-horizon 0`。

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
