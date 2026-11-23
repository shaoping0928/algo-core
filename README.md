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
- 因此当前方向不是再加一层长期机制，而是先修“决策尺度上的危险读出/候选分辨力”；详见 [2026-11-23 归档](docs/archive/README-2026-11-23.md)。

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
