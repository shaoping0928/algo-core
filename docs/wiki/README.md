# algo_core · Code Wiki

本 Wiki 是对 `algo_core` 仓库的**结构化代码文档**：项目整体架构、主要模块职责、关键类与函数说明、依赖关系、运行方式，以及测试与回归保护。

> 文档分工：**README.md** 讲"结构与如何运行"，**TODO.md** 记录"发生了什么和为什么"（逐批实验、判负结论、设计裁定），**docs/substrate.md** 是基座机制规格 + 实验日志（0–15 节 + 附录 A.1–A.28）。本 Wiki 是这三者的**代码索引**，不重复实验结论。

## 一句话定位

`algo_core` 是从主工作区 `algo_lab/` 抽离出的独立实验目录，聚焦：**离散节点基座（substrate）→ 节点强度信号算子 → 规则/关系发现 → Flappy Bird 延迟世界中的价值学习与策略搜索**。全项目坚持**无梯度、纯计数、CPU 可解释**的路线，唯一第三方运行依赖是 NumPy。

## 文档地图

| 章节 | 内容 |
|---|---|
| [01 整体架构](./01-architecture.md) | 三层世界观、两条并行管线、训练/决策/评测数据流、核心不变量 |
| [02 主要模块职责](./02-modules.md) | 按目录分组，逐文件职责速查表 + 分组详解 |
| [03 关键类与函数](./03-core-api.md) | 重要类/函数签名、参数、方法职责（含 `NodeRuntime`、`NodeGraph`、搜索与价值头） |
| [04 依赖关系](./04-dependencies.md) | 外部依赖、内部 import 图、分层规则与耦合边界 |
| [05 运行方式](./05-runbook.md) | 环境搭建、测试、各入口命令、`--fast`/`--benchmark` 预设、确定性与线程 |
| [06 测试与回归](./06-tests.md) | 30 个测试文件覆盖范围、pinning 回归、架构中立性守卫 |
| [07 新增实验线](./07-experiment-lines.md) | 阶段/相位、0/1 死亡规则、上下文/动力学/记忆、动作/时程/选择诊断、共享工具 |

> **覆盖范围与时效**：01–06 章先落地，覆盖主线（环境/图与运行时/特征/搜索/价值/规则/测试）。仓库随后持续演进（`main` 领先 `origin/main` 38+ 提交，探针与实验脚本大量增加），因此新增的 `stage_* / phase_stage_probe / death_* / context_* / *_probe / 记忆树` 等模块集中在 **07 章**。本 wiki 的模块清单以 `git ls-files` 为准；若数字与文件列表不一致，以仓库实际为准。

## 快速事实

| 项 | 值 |
|---|---|
| 语言 / 依赖 | Python 3；`numpy>=1.24,<3`（见 `requirements.txt`） |
| 规模 | 约 120 个 `.py`（含 30 个测试文件），主线 `node_runtime.py` 约 3857 行、`substrate_nodes.py` 约 2427 行 |
| 主入口（活跃线） | `substrate_nodes.py`（节点中心规划） |
| 旧/对照入口 | `substrate_sandbox_search.py`（世界反馈价值）、`substrate_generic.py`（任务无关原型）、`substrate_graph_search.py` |
| 环境模型 | `flappy.py`（无延迟）+ `latent_world.py`（`lag=0..3` 延迟世界） |
| 默认物理 | `GRAVITY=0.0016`、`FLAP=0.020`、`SPEED=0.010`、`GAP=0.34`、`SPACING=0.80`（可用 `FL_*` 环境变量覆盖） |
| 观测维度 | 6 坐标（`y,vy,dx,dy,next_dx,next_dy`）+ 1 存活积分坐标（`Latent`）= 7 |
| 默认延迟 | `LAG=3` |
| 输出产物 | 无数据文件；`--report` 可写 JSON（逐 episode 行 + 汇总 + 聚类 bootstrap CI） |

## 名词速览

- **节点 / 边（node / edge）**：`node_graph.py` 的统一基底。节点有四类 `concept / transition(edge) / memory / derived`；边有四类 `association / transition / membership / derivation`，另有动作条件边 `ActionEdge`。
- **原型（prototype）**：观测空间中被"出生"离散出的概念节点，用距离阈值在线生成。
- **特征根 / 谓词（feature root / predicate）**：`node_features.py` 由输入坐标的阈值原语经 AND/OR 组合出的规则，落到图里成为 `role="feature"` 的节点。
- **格子（cell）**：`(feature, acting_action)` 上累积的 count/hits（痛率）或回报统计（价值）。
- **危险 / 价值（hazard / value）**：搜索目标的两块——逐步累积的危险份额与终点序列级价值。
- **点事件 pain（默认）**：世界只在撞边界/管道的那一步转移上给 `pain=1`；正常飞行 `pain=0`。
- **前兆标签（precursor，opt-in）**：`--precursor-labels` 打开的 `P(pain within k)` 口径，属历史实验开关。
- **TD(λ) 资格迹 / MC 信用分配**：`NodeRuntime` 从 `dopamine/pain` 通道自学节点与边的期望 valence。
- **VOI 反事实预算**：采集侧在"威胁高且少数动作格子空"处定向覆盖（默认关）。
- **DAgger 闭环**：把规划器**实际执行**的真实转移回流训练（默认关）。
- **决策侧好奇心**：目标里的证据势差项（PBRS，评测时归零）。

## 推荐阅读顺序

1. 先读本页 + [01 整体架构](./01-architecture.md) 建立全局图；
2. 要动手跑实验 → [05 运行方式](./05-runbook.md)；
3. 要改代码 → [02 主要模块职责](./02-modules.md) 定位文件，再查 [03 关键类与函数](./03-core-api.md) 的签名；
4. 要增删依赖或判断分层 → [04 依赖关系](./04-dependencies.md)；
5. 要了解探针/实验线 → [07 新增实验线](./07-experiment-lines.md)；
6. 提交前跑 [06 测试与回归](./06-tests.md) 的用例。
