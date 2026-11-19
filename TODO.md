# TODO

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
- [x] 训练过程追踪（`trace_node_training.py`，max_nodes=5000）：约 2337 概念节点、graph nodes
  5105；整体双动作支撑 0.18、近管道 0.085；终止节点 75 个。
- [x] 通用 interaction：每条真实转移建立 association/transition；event memory 的边界由
  `|Δpain|+|Δdopamine|` 的 running mean/std（surprise）自适应决定，没有 memory_mode/阈值配置；
  重复 epoch 不重复建 memory。5000 步后 assoc/trans/mem = 2640/2640/128。
- [x] memory 通过 membership 关联；激活节点（含一跳 association）反查并加强 memory
  trace/value/count 和 membership 边；episode 结束 decay，低支持 memory 遗忘。
- [x] 随机 memory 追踪（`trace_memory_change.py`，3 种子）：修正 memory value 被后续存活步
  洗掉/拉满、membership 权重饱和、高价值 memory 误删。修正后 learning 前后新增 1–2 个
  memory，旧 memory value 稳定；terminal value≈1、安全 value≈0；首轮保留 206/208。
- [ ] value-aware memory retrieval：高价值 terminal memory 的 trace/membership 可能衰减到 0，
  需要按 value 而不只按 trace 找回。
- [ ] memory 成员仍约 28 个，事件分段偏粗；需要更细的事件边界。
- [x] transition edge node 用 MC/TD target 更新价值；memory value 以默认 0.1 权重进入终点评分；
  对称 association 价值平滑实测有害（232→141 帧），默认关闭，association 只参与 memory 检索。
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
