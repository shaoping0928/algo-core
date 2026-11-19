# Algo Core：节点信号基建与 Flappy Bird 实验

这是从主工作区 `algo_lab/` 中整理出的独立实验目录，聚焦离散节点基座、节点强度信号算子、规则/关系发现，以及在 Flappy Bird 延迟世界中的价值学习与策略搜索验证。源文件从主工作区复制而来；本目录可独立配置 Git，不包含其他工作区项目或临时实验。

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
signal_bank.py            每节点信号源：差分、积分/泄漏、逆泄漏、均值
substrate_*.py            各算子、记忆、噪声、延迟和搜索的验证脚本
tests/                    unittest 回归测试
docs/substrate.md         A.1–A.28 机制与实验记录
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

## 价值与策略搜索

目标是让系统从世界提供的反馈中学习价值，而不是在搜索器里写任务专用的评分公式。当前实现把这一责任拆成三层：

1. **世界反馈**：`flappy.py` 每步输出 `dopamine` / `pain` 两个通道。它们只是环境输入，代码不根据缺口中心或管道位置推导价值。
2. **价值学习**：`substrate_value.py` 的 `WorldFeedbackValueHead` 从真实交互的 Monte-Carlo 回报学习 dopamine/pain 两路未来值；`TerminalRiskHead` 从同一 `pain` 标签学习终止风险。风险特征和标签都取 **next state**；真实转移进入 replay，终止样本按比例采样，采样后的先验偏移在推理时做截距校正。反事实 rollout 不参与价值或风险训练。
3. **方案选择**：`trajectory_search.py` 提供任务无关的 exhaustive 和 beam 搜索，支持多目标词典序比较。Flappy 适配器默认先最小化预测疼痛/终止风险，再用 pain 回报和 dopamine 排序；`--task-baseline` 才切换回 gap-center 评分与碰撞可行性过滤。

默认搜索是 `exhaustive H=14`；`--engine beam` 是搜索方式，不改变评分目标。beam 在默认的
`terminal` 聚合下没有中间风险信号，长时域对照应显式加 `--risk-aggregate max` 或 `sum`，
并保持两个引擎使用同一聚合方式。

这不是“价值函数天然等于存活”：反馈通道的语义仍由世界提供。当前实现也没有把新颖度直接当作奖励，它只是价值模型的一个输入特征。

## 当前测量概况

- 在合成延迟世界中，预测误差能学到三帧滞后和约 `0.020` 的动作增益。
- `u` 是 RC 型泄漏积分；`signal_bank.py` 提供节点级候选信号，均值器默认不启用。
- `recall_world.py` 的 recall 是“线索只出现一次、随后输入静默 d 步、最后报告线索符号”的延迟回忆任务，不是通用文本检索。它对照节点痕迹 `u`、每边转移残留 `res`、稀疏联想记忆 `M` 和递归权重学习。
- 节点痕迹与边残留分开：`u` 保存节点激活的衰减痕迹；`res` 让刚用过的边暂时更容易通过；`M` 用共激活关联补全回忆模式。它们是实验臂，尚未全部集成进 Flappy 沙盒控制循环。
- **当前世界反馈模式**（8 个种子，默认 exhaustive H=14）：存活平均 **190.1 帧**，中位数 232，范围 57–312；lag=3、增益约 `0.020`。这比最初的 14.5 帧明显提高，但仍远低于任务基线。
- 价值留出评估（每 5 个练习 episode 留出 1 个）：pain AUC `0.910`；终止风险 AUC `0.971`，Brier `0.019`；近管道风险 AUC `0.757`，近管道缺口内/外风险均值 `0.012/0.124`。风险特征已与终止标签统一到 next state，概率做了重采样先验校正。风险头已能排序终止风险，剩余瓶颈是长时域管道对齐，而不是疼痛预测本身。
- **当前任务基线**（8 个种子，5000 步练习）：存活平均 **1026.9 帧**，中位数 659，范围 123–3491。它仍使用 gap-center 评分和碰撞可行性过滤，只作对照。
- 长时域对照（修正后）：beam 不再隐式切换评分目标。2 种子 pilot：默认 exhaustive `terminal` 均值 272；exhaustive `max` 均值 152；beam H=78 `max` 均值 155；beam 默认 `terminal` 只有 25（没有中间信号，候选被任意截断）。因此“长时域 beam 更差”不再是目标混淆造成的假象，但也没有证据表明长时域带来增益；必须保持 `--risk-aggregate` 一致再比较。
- 风险聚合对照（2 种子 pilot）：默认 `terminal` 均值 272 帧；`max` 均值 152，`sum` 均值 91，只在预测穿管时刻查询的 `crossing` 均值 232。逐帧聚合并不优于终点价值；风险头整体 AUC 约 0.97，但近管道 AUC 约 0.72–0.76，仍不足。
- 风险头增强对照（2 种子 pilot）：`--risk-interaction-dim 6`（有/无 `--near-pipe-fraction 0.4`）的近管道 AUC 约 `0.669/0.656`，存活都只有 25 帧；默认已恢复无交互、无近管道过采样。问题不在特征容量或采样比例，而在缺少动作条件化的碰撞/风险预测。
- 动作历史输入对照（2 种子 pilot）：`--risk-action-dim 4` 把最近动作拼入风险头，平均存活 272 帧、近管道 AUC `0.710`，与默认的 272 帧、`0.719` 基本持平；与 `--risk-aggregate crossing` 组合降到 119.5 帧。没有证据表明拼接动作历史能改善近管道风险。默认 `risk_action_dim=0`；下一步需要动作条件的 n-step 风险/价值或 Q(s,a)。
- 事实动作覆盖诊断：seed 100 的近管道状态中只有 3/50 实际拍翅，seed 103 只有 4/80；这些拍翅样本没有终止。状态风险整体 AUC 约 0.97，但动作历史没有稳定提升近管道 AUC。由于不允许生成反事实结果，当前数据不足以估计同状态下拍翅/不拍翅的差异；需要先在管道附近实际采集两种动作。
- 练习覆盖（8 个种子）：平均 episode 约 43.1 帧；管道死亡 128 次、边界死亡 792 次、管道窗口 921 步。覆盖统计由 `substrate_sandbox_search.py` 输出。
- **动作学习轨迹仍然是负结果**：`trace_feedback.py 100` 追踪到 139 个管道附近决策，按“缺口在上则拍翅、缺口在下则不拍”计，方向准确率约 `0.29`；需要上拍时拍翅率约 `0.08`。风险头对全不拍/全拍的平均风险为 `0.26/1.00`，从未认为全拍更安全。它学到的是“管道附近有终止风险”，不是“当前拍翅能对准缺口”。
- 穿管时刻预测：seed 100 的选中计划/全不拍/全拍在缺口内的比例为 `0.843/0.753/0.663`；平均穿管前拍翅 `0.43` 次，平均 gap 误差约 `0.104`。也就是说，选中计划只有轻微、种子相关的调整，不是稳定的方向性动作规则。当前搜索仍主要靠宽缺口、初始高度和延迟拍翅延长存活。
- 沙盒反事实 rollout 单独记录为搜索经验，不作为真实状态转移训练世界模型或价值头；只有实践中观察到的转移更新模型和价值。

## 节点运行时：只给信号，不标注风险

`substrate_nodes.py` / `node_runtime.py` 是新的节点中心原型，和上面的风险头路径并列：

- 人类只规定 `obs`、`action`、`dopamine/pain`；`done` 只作为 episode 边界，用于 bootstrap 截断和 reset，不作为风险标签。
- `NodeGraph` 新增动作条件转移边 `(source, action) -> target`，记录 count、`Q_pain`、`Q_dopamine` 和预测误差。
- `NodeRuntime` 在线出生 concept 节点；用 Monte-Carlo 回报把 world pain 反向传播到此前被激活的节点和边；节点价值 = 未来回报 + 当前状态 immediate pain，避免“终止节点 V=0”把危险终点评成安全。
- 运行时用 soft neighbours 聚合节点价值；`substrate_nodes.py` 用学到的 `LinearObservationModel` 预测候选轨迹，并用节点学到的 valence 给终点打分。
- 这条路径不读取 `GAP/2`、`target_gap`、`BIRD_X`、碰撞窗口或 near-pipe 区域，也没有 `TerminalRiskHead`。

```bash
python substrate_nodes.py --seed 100 --seeds 8 --jobs 8 \
    --steps 5000 --coverage 0.1 --horizon 10 --epochs 10
```

当前测量（8 种子，5000 步练习，H=10，10 epochs，max_nodes=2000）：存活
**232/48/31/78/151/72/152/99** 帧，均值 **107.9**、中位数 **88.5**；每个种子都达到
2000 节点上限、约 3100–3160 条动作边。状态尺度不是硬编码常量，而是从真实观测的标准差估计。
它仍然低于世界反馈风险头路径（190.1）和任务基线（1026.9）；节点上限被顶满说明当前
“距离出生、无剪枝”的生命周期需要下一步处理，而不是靠任务常数调参。

脚本会检测 Python 是否启用 GIL：free-threaded 构建使用 `ThreadPoolExecutor`，普通构建使用
`ProcessPoolExecutor`。当前 `.venv` 是 GIL 构建；本机的 `python3.15t` 存在，但 free-threaded
numpy wheel 的 `_multiarray_umath` DLL 被应用控制策略阻止，暂时无法在该解释器加载 numpy。

## 运行时组件与三态概念节点

数值派生量不再作为概念节点。`node_components.py` 把电容、电阻、差分器等算子留在运行时组件里：

- 组件绑定到一个基础节点，只做数值计算，**不进入 `NodeGraph.nodes`**；
- 组件输出先平滑（EMA），再饱和到 `[-1, 1]`，最后按触发阈值舍入成三态：
  `[-1, -0.2] -> 抑制(-1)`，`(-0.2, 0.2) -> 保持(0)`，`[0.2, 1] -> 激活(+1)`；
- 这三个状态是普通 `CONCEPT` 节点（`component:<id>:+ / 0 / -`），带 `role/component_id/sign`
  元数据，参与关联、转移、memory、valence 学习和持久化；
- 组件可以设置 `hysteresis`，让已经跨过触发的状态在回程时不被边界抖动反复翻转；
- `NodeGraph.to_state/from_state` 持久化节点、边、动作边和组件三态节点；组件的瞬时数值不是概念，
  只有组件规格、三态状态和学到的节点/边被持久化。

`add_derived` 仍保留为旧实验的数值派生节点接口，但新的运行时路径不再用它承载概念。当前组件路径是
原型：`NodeRuntime.bind_component` 可以把组件绑到运行时节点，三态节点会参与关联/转移和 EMA
valence 更新；下一步是把它们接入 MC/TD 信用分配，并处理组件数量增长。

实际接入对照（8 种子，5000 步，H=10，10 epochs，`difference+capacitor`，
`component_min_count=8`，`max_components=128`）：

- baseline：107.9 帧均值、88.5 中位数
- components：107.9 帧均值、88.5 中位数，逐种子完全相同
- 图从 2000 个概念节点增加到约 2512 个节点，动作边约 3610–3660 条

这是一个**中性/负结果**：三态节点确实被创建、参与动作边并进入决策，但它们是基础节点激活的
确定性函数，派生出的并行动作边没有提供新信息，反而增加了图规模。要产生实际增益，三态节点需要
承担基础状态没有的角色，例如 eligibility/信用分配、事件记忆、或作为转移条件，而不是再加一套
同源的动作边。

进一步尝试把三态作为 **transition 条件**（gated action edge）：只有当当前 signature 有学到的
gated 边时才使用它，否则回退到基础边。5000 步、128 个组件时学到 156 个 gated key，但评估时
22 步中没有一步命中这些 key；8 种子存活仍为 107.9 帧，逐种子不变。原因是三态 signature 的
覆盖太稀疏，评估状态大量落在未见的 `(node, signature, action)` 上。下一步需要 signature 共享/
回退或事件级记忆，而不是继续增加条件分支。

旧节点系统检查：`graph_memory` 的初始节点价值现在由 world `signals` 的 pain 估计，而不是由
`terminal` 标签估计；`NodeGraph.to_state/from_state` 新增覆盖 association、transition、
memory、derivation、action edge 的 round-trip 测试。`trace_graph_memory.py`、
`trace_mixed_nodes.py`、`trace_sensitivity.py` 均正常退出。

### 实际追踪：哪个环节弱

`trace_node_runtime.py` 把链路拆成四段（seed 100，5000 步，H=10）：

```text
[1] world model one-step error
  y/vy  ~1e-9
  dx/gap mean ~1e-2/1e-3, max 0.79/0.15（管道生成事件）
[2] node value
  offline holdout AUC 0.977, near 0.914
  online terminal AUC 0.442, near 0.493
  terminal states nearest immediate pain 0.000
[3] action support
  both actions 0.530; near-pipe 0.521
  near-pipe chosen/other counts 135/23.6
[4] decision direction near pipe 0.696 over 69 events
```

结论：世界模型不是瓶颈；瓶颈是**离线到在线的终止状态/价值覆盖**。节点价值在行为策略的
留出集上 AUC `0.977`，但 MPC 评估时终止 AUC 只有 `0.442`，因为评估策略死亡的状态落在
训练里“安全”的节点上，nearest immediate pain 为 0。另一个弱点是近管道反事实动作支撑：
另一动作平均只有 `23.6` 条边，约一半近管道状态没有双动作支撑。方向准确率 `0.696` 说明
决策不是完全盲目，但价值排序和反事实 Q 仍不可靠。

训练过程本身（`trace_node_training.py`，seed 100，5000 步，5 epochs，max_nodes=5000）：

```text
wave 1:  363 concepts,  485 action edges, assoc=356,  trans=356,  mem=19
wave 10: 2057 concepts, 2819 action edges, assoc=2203, trans=2203, mem=122
wave 22: 2337 concepts, 3389 action edges, assoc=2632, trans=2632, mem=128
end:     2337 concepts, 3410 action edges, assoc=2640, trans=2640, mem=128
         graph_nodes=5105, terminal_nodes=75
```

节点上限扩大到 5000 后，5000 步数据实际只生成约 2339 个概念节点，没有触及上限。
现在概念间 interaction 已经真正建立：

- 每条真实转移都建立 association 和 directed transition；
- **没有原生事件定义**：memory 只由 world 的 pain 信号触发；安全步只检索/加强已有 memory，
  不按 episode 边界或 valence 变化阈值切事件；
- 训练时 pain 信号把当前/下一状态和 memory 与 `valence:pain` 节点关联；预测状态通过
  association 读出 pain 强度，而不是靠人工风险标签；
- memory 通过 membership 关联 concept/transition；被激活时加强 trace 和 membership，
  episode 结束 decay；高 value 的 pain memory 受保护，低支持/低 value 的遗忘；
- transition edge node 的价值用 MC/TD target 更新；memory value 以 `0.1` 权重进入终点价值；
  association 对称价值平滑实测有害（232→141 帧），默认关闭，association 只参与 memory/pain 检索。

实测：

- 8 种子（5000 步，H=10，epochs=10，max_nodes=5000，memory_weight=0.1）：
  均值 **108.6 帧**，中位数 **85.5**；memory_weight=0.5 时为 104.6。
- 5000 步训练过程：assoc/trans/mem = 2915/2640/89，graph nodes 5068。
- 随机 memory 追踪（3 种子）：pain memory 约 8–89 个，value 几乎都是 1.0，members≈3；
  很多 memory 的 trace/membership 会衰减到 0，虽然保留但很难再被激活。
- 自适应组件（`difference+capacitor`）在 seed 100 自动出生 76 个组件，存活仍是 232 帧。

但决策路径仍未读取 association/transition 结构；memory 只以较小权重参与终点价值。存活与
“只建立结构”基本持平，说明下一步需要更好的信用分配/校准，而不是继续增加结构数量。

### 快速开发协议

`python substrate_nodes.py --fast` 使用 1500 步、3 epoch、1000 节点、H=8、cap=400：

- 单种子端到端约 **9 秒**，8 种子并行约 **30 秒**；
- fast 8 种子：均值 **123.0 帧**，中位数 **79.5**；
- 同一批 fit 关闭特征层后均值 **94.6**（seed 100 为 312 vs 113）；
- 旧的 5000 步协议是 108.6 帧 / 131 秒，所以日常迭代用 fast 协议，里程碑再跑完整规模。

组合特征层（`node_features.py`）从输入坐标自动生成原语（raw 和 `|x|` 的阈值测试），
用 AND/OR 组合，按“覆盖尚未解释的 pain 事件”选择规则，并在 held-out episode 上验证。
发现阶段包含：

- pain 样本很少时自适应降低 minimum support，保留低支持但精确的假设；
- episode 交错划分 held-out，避免稀有 cause 全部落在训练集；
- pain 组内阈值加外扩 margin，保证精确的“边界值”组合仍然覆盖所有 pain 样本；
- 覆盖选择优先解释新的 pain 事件，再用“未使用过的坐标/原语”补充不同 cause。

规则在 epoch 0 后安装成图中的 `role="feature"` 节点：

- 每个状态都会评估谓词并激活对应特征节点，节点上保存学到的 `P(pain | feature)`；
- 特征节点作为 action edge 的 source 参与转移学习，决策时也会被加入候选 source；
- 创建 memory 时把当时活跃的特征节点写成成员，之后可以按特征召回 memory；
- 沙盒预测状态先算特征风险，再与节点/memory 的 pain 读出取 max。

fast 8 种子：特征节点参与训练后 **123.0 帧**，关闭特征层 **94.6**；seed 100 为 312 vs 113。
现在 fast 规模已经能自动选出撞管特征
`(|f2| < 0.274 & |f3| > 0.787)`（train 3/3，test 2/2），
以及边界特征 `|f0| > 2.397`。特征仍然不是手写的：谓词、阈值、AND/OR 结构和保留哪几条
规则都由数据决定。

### 决策时实际参考什么

`substrate_nodes.policy_result` 现在参考：

1. `LinearObservationModel` 预测每条候选动作序列的 H=10 步观测；
2. 每 2 步把预测状态映射到节点/memory，沿 association 读 `valence:pain` 的强度，按轨迹取 **max**；
3. 预测状态同时计算组合特征节点的激活，特征上学的 `P(pain|feature)` 直接进入风险；
4. 特征节点还能按 membership 召回相关 memory，memory 的 pain 值也进入风险；
5. 终点用节点/memory pain 价值和学到的 pain 信号，再用 dopamine 排序；
6. 词典序：先最小化最坏预测 pain，再最大化终点 dopamine；
7. 不使用 action edge 做搜索评分，也不读取管道/缺口/碰撞常量。

fast 规模下特征层把 seed 100 从 113 帧提高到 312 帧；8 种子均值 94.6 → 123.0。
撞管特征已经能自动进入特征集；当前剩下的问题是规则集里仍有很多边界变体，
需要特征生命周期（合并/剪枝/预算）来压缩冗余，并接入不变量分段来发现更抽象的特征。

### 随机 memory 追踪

`trace_memory_change.py` 随机抽 3 个种子、各抽 1 个 episode，对比学习前后的 memory：

```text
学习前 n=130 value=0.146/1.000 trace=0.428/0.860 members=43.0
学习后 n=131 value=0.145/1.000 trace=0.429/0.898 members=43.1
created=2 updated=129 forgotten=1
full training: n=111 value=0.000/0.000（修正前）
```

追踪暴露并修正了两个问题：

- **memory value 被后续存活步洗掉/拉满**：原先每个 episode 的 memory 都会在后续 pain=0 的
  转移上更新，最后要么趋近 0，要么被终止 pain 的 max 拉成 1。现在只有 **当前事件 memory**
  更新 value，旧 memory 被检索时只更新 trace/membership，不改事件 valence。
- **membership 权重饱和到 1**：原先任意成员被激活就强化该 memory 的所有 membership 边。
  现在只强化当前/下一节点对应的边，并在结束时 decay；权重从全是 `1.00` 变为约 `0.00–0.20`。
- **高价值 memory 被误删**：遗忘现在用 trace + 相对 value 中位数保护高 pain 的 terminal memory，
  不再只看低 trace/count。

修正后：

```text
学习前 n=206 value=0.461/1.000 trace=0.274/0.860 members=28.2
学习后 n=208 value=0.462/1.000 trace=0.273/0.898 members=28.2
created=2 updated=188 forgotten=0
full training: n=207 value=0.464/1.000，首轮 memory 保留 206/208
```

也就是：每个 episode 约新增 1–2 个 memory；旧 memory 多数只更新 trace/权重，value 稳定；
terminal event memory value≈1，安全 event memory value≈0；memory 成员数约 28，仍然偏
episode/窗口级，而不是严格的事件级。

剩余问题：

- 部分高价值 terminal memory 的 trace 会衰减到 0，membership 权重也接近 0，虽然被保留下来
  但很难再被激活；需要 value-aware retrieval，而不只靠 trace。
- memory 成员仍较多（约 28），事件分段还不够细。
- memory 以 0.1 权重进入终点价值时对存活没有可见影响（8 种子仍 108.6）。

其余训练过程问题仍然成立：

- 概念节点约 2337 个，没有触及 5000 上限；但新的终止状态仍可能 alias 到旧安全节点。
- 近管道双动作支撑只有约 `0.18` 整体、`0.085` 近管道；另一动作的 Q 长期缺少数据。
- 因此训练出的图仍然偏“僵化的离散记忆”：能拟合行为策略的留出集，却不能覆盖 MPC 自己产生的
  终止状态和新轨迹。

## 任务无关世界模型原型

`substrate_generic.py` 是独立原型：规划器只通过

```text
obs_t, action_t, feedback_t, terminal_t
```

与世界交互，不读取 `pipes`、`BIRD_X`、`GAP` 或 `SPEED`。`LinearObservationModel` 从真实转移学习

```text
(obs_t, action history) -> obs_{t+1}
```

价值/风险头只使用观测和动作历史；搜索用学到的完整观测模型做 rollout。当前 3 个种子（100–102）
的存活为 73/232/319 帧（均值 208），单步留出误差约 `4e-3` 到 `1e-7`；seed 100 的最大训练误差约 `0.79`，
来自管道生成造成的不连续事件，说明纯线性观测模型还不足。这个原型验证了接口分层，但还不是
性能更好的默认路径；当前默认仍是 `substrate_sandbox_search.py`。

`trace_generic_training.py` 抽样了 seed 100 的 5000 步训练波次：价值/风险指标随更新稳定上升
（pain AUC 约 `0.83→0.94`，risk AUC 约 `0.96→0.975`，near-pipe AUC 约 `0.75–0.78`，
Brier 约 `0.011`），没有出现训练崩溃。不稳定的部分主要是数据覆盖和世界模型事件：每 500 步的
近管道样本在 `3–81` 之间波动，近管道实际拍翅样本在 `1–8` 之间；seed 100 的
`dx/gap` 留出误差约 `1.6e-3/6e-5`，其他种子约 `1e-8`，对应管道生成的不连续事件。
也就是说，学习器不是主要失效点，**数据覆盖和事件感知世界模型才是**。

`trace_signal_world.py` 对 raw、`d1/d2` 差分、多时间常数缓冲、以及全部信号做了同一批真实 episode 的对照：seed 100 的 raw 单步平均误差约 `6.20e-4`，差分约 `6.35e-4`，缓冲约 `6.60e-4`，全部约 `6.58e-4`；多步约 `3.8e-3`，没有稳定改善。seed 101 的 raw/差分/缓冲/全部约 `1.0e-8/0.9e-8/1.1e-8/1.5e-8`。管道维度仍主要受生成/弹出事件限制。结论：当前完整观测下，缓冲和差分是冗余的，低通还会抹平事件边缘；不能替代事件/动作条件模型。

`trace_sensitivity.py` 的扰动重要度更直接（seed 100，5000 步）：世界模型对 `dx1/dx2/vy/gap/y` 和滞后动作 `a3` 都敏感（`dx1` 约 `0.132`）；风险头仍主要依赖 `y`（AUC 下降约 `0.279`），`gap1/gap2/dx2` 约 `0.004–0.006`，动作贡献更小；价值头主要依赖 `vy/gap1`（约 `0.160/0.091`）。也就是说：**风险头有完整观测输入，但“只看高度”仍能获得大部分全局 AUC；管道和动作信息只有很小贡献。** 这不是输入缺失，而是训练目标/样本分布问题。

`node_graph.py` 把统一节点图落到基建层：所有节点共享激活/电容痕迹/关联/绑定；关联边不带边节点，只有真正的有向转移边才分配 `edge node`；记忆节点用 membership 绑定多个节点；派生节点有深度上限；图提供激活传播、回忆、衰减和按权重/痕迹剪枝。它是节点/边语义的基建原型，尚未替换现有线性世界模型控制器。

`graph_memory.py` 把真实 Flappy 观测原型接到该图上：原型成为概念节点，连续原型成为 transition edge + edge node，episode 成为 memory node；pain 更新 next state 的概念价值和相应 transition 边价值。seed 100 的图约有 48 个概念、219 个 edge node、98 个 memory；全局风险头 AUC `0.975/near 0.752`，图记忆风险 AUC `0.876/near 0.596`；图记忆敏感度中 `gap2/gap1/y` 约 `0.114/0.084/0.069`，比全局风险头更依赖管道信息。图风险仍低于全局头，但 edge value 现在会跨重复转移累积，而不是每次重置为 base。

`substrate_graph_search.py` 已把图记忆风险接入搜索：世界模型预测观测，概念/边节点同时学习 pain/dopamine，记忆节点价值参与事件级风险，搜索用图风险 + 全局风险 + 图价值，边节点资格参与边价值，搜索不读取管道/缺口常量。修正后 seed 100/101 为 312/232 帧（均值 272），图约 434–437 节点、2703–2803 边、259–264 edge node、122–130 memory；与前文全局反馈模式的同种子结果相同，说明当前图风险还没有带来独立增益。旧的 4 种子 312/232/480/312 不再作为当前结果；不能只凭少量种子断言泛化。

混合节点原型（`mixed_nodes.py` / `trace_mixed_nodes.py`，seed 100，5000 步）：数据中存在
`dx1 < 0.28 AND gap1 < -0.402` 的联合状态（support 48，pain rate 19.3%，base 2.5%）。
全局风险头 AUC `0.975/near 0.752`，Brier `0.011`；痛驱动混合节点 AUC `0.816`、
near `0.773`、Brier `0.133`，near-inside/outside `0.061/0.753`，仍明显高估缺口外风险。
连续记忆版保留 6/64 节点，AUC `0.914`、near `0.858`、Brier `0.021`，
near-inside/outside `0.032/0.044`；分层记忆版保留 26/64 节点（16 个稀有槽位），
AUC `0.947`、near `0.773`、Brier `0.018`，near-inside/outside `0.015/0.035`。
连续/分层版在 near AUC 上已经接近或超过全局头，但 near-inside/outside 的绝对风险差仍小，
且样本有限；“混合节点存在”仍不等于“记忆生命周期和校准正确”。

通用优先回放：`TransitionReplay`/`ReturnReplay` 支持按世界模型预测误差加权采样，`substrate_generic.fit`
当前默认使用该优先级；3 个种子（coverage=0.1）存活 73/232/319。风险敏感度仍以 `y` 为主，
世界误差优先级没有让风险头真正利用管道；下一步应使用风险/价值误差作为优先级，而不是只看世界误差。

## 加法/乘法：原始输入实验

`raw_math_experiment.py` 生成两阶段字符流，只包含题目和最终答案：

```text
3+7=10
4*6=24
```

不包含偏积、进位过程或解法步骤。答案由外部算术 oracle 生成，这是监督信号，不是模型内部规则。
nanoGPT 使用现有 `model.py` 和训练器的普通堆叠路径（`n_loop=0`，其他扩展关闭）；没有写入加法或乘法代码。

复现命令示意（先在 `algo_core` 生成数据，再从 `nanogpt-eval` 调用原训练器）：

```powershell
$raw = "C:\temp\raw_arith"
$run = "C:\temp\raw_arith_run"
python ..\algo_core\raw_math_experiment.py prepare $raw

python train.py config/baseline.py "--dataset=$raw/memory" "--out_dir=$run" `
  --device=cuda --dtype=bfloat16 --compile=False --block_size=64 `
  --batch_size=64 --max_iters=600 --lr_decay_iters=600 `
  --always_save_checkpoint=True --init_from=scratch --n_loop=0

python train.py config/baseline.py "--dataset=$raw/form" "--out_dir=$run" `
  --device=cuda --dtype=bfloat16 --compile=False --block_size=64 `
  --batch_size=64 --max_iters=5000 --lr_decay_iters=5000 `
  --always_save_checkpoint=True --init_from=resume --n_loop=0
```

最近一次运行的训练顺序和独立随机探测（每个长度格 64 题）：

```text
阶段 1  记忆：全部 200 个一位数加/乘映射，打乱重复 64 次
        一位数加法 64/64；一位数乘法 64/64
        两位数及以上 0/64

阶段 2  形式：120,000 条随机二至三位数原始等式，继续训练，不含中间步骤
        位数       加法 exact       乘法 exact
        1          0/64             0/64      <- 阶段 1 的事实被遗忘
        2         59/64             2/64
        3         49/64             0/64
        4+         0/64             0/64
```

这是字符级 GPT 的基线，不应解释成显式节点记忆或策略搜索。它显示了事实记忆能形成，但后续训练会遗忘；
原始等式样本让加法在未见过的三位数组合上有一定泛化，乘法没有，位数外推也没有。
每格 64 个随机 probe；结果是当前一次种子运行，不是多种子均值。

`math_memory_search.py` 是另一项、数值输入上的机制隔离实验：关联存储记住一位数事实；`substrate_search`
再从通用候选基（原值、变换和两两乘积）选择关系。它恢复 `a+b` 和 `a*b`，并在无记忆命中的二至六位数上
逐题预测正确。这个结果只证明**给定数值输入和该候选基时**的关系搜索；解析字符、候选算子范围以及
把这种搜索接回 nanoGPT/节点网络，仍未解决。候选基包含乘积这一通用算子，因此也不能宣称系统从零发明了乘法。

## 边界与纪律

1. 动力学系数和延迟可以从真实观测学习；任务目标（例如缺口中心）、候选动作表示和搜索预算仍是外部提供的。
2. 沙盒预测不是现实证据。必须与后续实际转移分开存储、比较和学习。
3. 合成、无噪声实验的机器精度结果不代表真实感知输入下的性能。
4. 本目录的结论以 `docs/substrate.md` 和各验证脚本为准；负结果及其适用条件也保留。

主工作区和本目录是两份文件副本。后续修改应明确选择维护哪一份，再同步需要的变更。
