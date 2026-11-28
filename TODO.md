# TODO（多 Agent 协同主线与任务看板）

> **维护约定（2026-11-28 整理）**：本文件面向**多 Agent 协同开发**，提供可直接冷启动接手的项目背景、当前现状、代码入口/复现口径、硬约束契约与可认领的下一步任务。2026-11-27 及更早的完整过程记录已无损归档至 `docs/archive/TODO-*.md`（见文末「历史归档索引」）。

---

## 一、项目背景与核心研究命题

`algo_core` 的目标是构建一套**严格任务无关（task-agnostic）的通用长程世界模型与决策基座**——不依赖手工任务规则、词法分词器、语法先验或硬编码惩罚，仅从原始离散符号流或连续观测-动作-反馈交互中自监督演化出**多时间尺度结构记忆与长程规划能力**。

项目目前沿两条已打通、正在汇合的主线演进：
1. **连续控制 / 长程阶段决策主线（Phase 1–17，已阶段性收口）**：解决帧级规划近视与变物理迁移问题。通过自监督阶段切分、生存方差树（`survival_tree`）、串联主动剪枝搜索、在线 LCB 转移/需求校准与节点变量抽象（`DynamicsClassNode`），在延迟控制基准 `cap=5000` 上将 600 局均分推进至 **`1996.0`**（零调参未见 240 局 **`2512.3`**），并在 `DelayedCartPole(lag=2)` 零改动跨任务通关（`369.3`，`18×` 随机）。详见 [TODO-2026-11-27.md](docs/archive/TODO-2026-11-27.md)。
2. **符号序列 / 语言上下文整合与原地状态空间主线（当前核心发力点）**：
   - **核心痛点**：早期高阶符号记忆本质上是局部 N-gram/块级统计，续写易退化为“只看前几个字的局部判断”；
   - **核心理念（用户明确要求）**：每接收或输出一个符号，必须在**已有的持久节点状态上做连续原地修改**（而非每步重建无状态表示），且节点的时间常数（快/慢衰减）与上下文整合策略必须**由数据与反馈端到端自学习**（核心主题自动长成大电容慢衰减、瞬态语法细节自动长成小电容快充快放）。

---

## 二、当前现状与可复用资产（接手即用）

### 2.1 运行环境与硬件底座
- **Python 解释器**：`C:\workspace\code\.venv-ft\Scripts\python.exe`（Python `3.14.5t` free-threaded，`gil_enabled=False`，24 核 CPU，NumPy `2.5.4`）。
- **GPU 计算底座**：**AMD Radeon RX 9070 XT（`gfx1201`，32 CUs，13.53 GB VRAM）**，通过零外部依赖原生 Windows `OpenCL.dll` 驱动（[core/compute/opencl_gpu.py](core/compute/opencl_gpu.py)）。当前 `32×32` 寄存器分块 SGEMM 实测 **`5.60 TFLOPS`**，支持全量 64M token 流常驻独立显存零拷贝 BPTT（注：Windows 任务管理器需在 GPU 视图切换为 `Compute_0 / Compute_1` 及「专用 GPU 内存」观察）。
- **语料路径**：`C:\workspace\code\data\wiki_zhihu\corpus.txt`（中文维基 + 知乎混合语料）。

### 2.1b 主要启动命令与输入 / 输出目录

> 以下命令均在仓库根 `c:\workspace\code\algo_core` 下用 PowerShell 执行；`$PY = C:\workspace\code\.venv-ft\Scripts\python.exe`。脚本内路径为硬编码常量（脚本顶部），换机器请先改 `ROOT / corpus_path / ckpt_*`。

| 用途 | 命令 | 耗时 / 资源 |
|---|---|---|
| **核心单测（42 条）** | `& $PY -m unittest tests/test_context_integrator.py tests/test_scalability_and_arbitration.py tests/test_circuit_gpu.py tests/test_policy_pool.py tests/test_structural_memory.py tests/test_recency_kernel.py tests/test_context_decision.py tests/test_temp_context.py tests/test_sequence_stage.py tests/test_class_policy.py tests/test_virtual_node_dag.py -v` | ~15s，需 GPU 的用例会自动使用 OpenCL |
| **64M 符号层从零构建**（16M seed + 3×16M 流式摄取 + 保存检查点） | `& $PY scripts/lm/train_and_eval_64m_full_v3.py` | ~10 min，RSS ≤ 8.6 GB；覆盖写 `checkpoints/wiki_zhihu_64m_v2000.pkl.gz` |
| **64M 2 层 SSM 训练 + `@260M` 评测 + P1/P2/P3 续写**（当前冠军流程，需先有上面的符号层检查点） | `& $PY scripts/lm/train_and_eval_64m_2layer_ssm.py` | ~14 min（训练 718s，吞吐 ~178 ktok/s）；RSS ≤ 9.6 GB，显存 ~2.3 GB；训练完成后**覆盖写**同一检查点 |
| **仅评测续写**（加载已有检查点，不训练） | `& $PY scripts/lm/eval_64m_ssm_rollouts.py` | ~1.5 min |
| **256M 全量构建 / 评测**（可选，对照上限） | `& $PY scripts/lm/train_and_eval_256m.py` | ~9 min，RSS ≤ 11.5 GB；写 `checkpoints/wiki_zhihu_256m_v2000.pkl.gz` |
| **长任务提示** | 以上脚本均会持续 print 日志；建议后台运行并将输出重定向，如 `& $PY scripts/lm/train_and_eval_64m_2layer_ssm.py *> results/lm_run_$(Get-Date -f yyyyMMdd_HHmm).log` | — |

**输入**
- 语料：`C:\workspace\code\data\wiki_zhihu\corpus.txt`（字符级，词表 `V=2000` 由前 16M 字符按频次构建，`<UNK>=0`）；训练区间 `[0, 64M)`，评测区间偏移 `260,000,000`（取 20,000 字，使用 `ev[5000:11000]`，`n=5876`），**禁止用于训练**。
- 检查点：`checkpoints/wiki_zhihu_64m_v2000.pkl.gz`（默认；`StructuralSubstrateBundle.load(path)`）。

**输出**
- 检查点目录：`checkpoints/`（`wiki_zhihu_{4m,16m,64m,256m}_v2000.pkl.gz`，64M 当前 `256.95 MB`，含 2 层 SSM）。
- 评测与续写结果：当前脚本仅打印到 stdout（`Ablation`、`UnionBlend` 扫 `alpha`、τ 谱、P1/P2/P3 续写），**无结构化结果文件**；建议将日志重定向到 `results/`（见上表），并在认领任务时把关键指标回填本文件。
- 单次实验的临时/一次性脚本请放各自 agent 的 scratch 目录，不要入库；可复用脚本放 `scripts/lm/`。

**最小代码调用（复用检查点做续写/评测）**
```python
import sys, os
ROOT = r"c:\workspace\code\algo_core"; sys.path.insert(0, ROOT)
for dp, _, _ in os.walk(os.path.join(ROOT, "core")):
    if "__pycache__" not in dp: sys.path.insert(0, dp)
from substrate_bundle import StructuralSubstrateBundle
b = StructuralSubstrateBundle.load(os.path.join(ROOT, "checkpoints", "wiki_zhihu_64m_v2000.pkl.gz"))
ctrl = b.make_controller(use_arbitrator=True, lookahead_depth=2, lookahead_beam=5)  # 2 层 SSM 默认 stateful_weight=0.60
ctrl.stateful_weight = 0.65; ctrl.self_stateful_beta = 0.0   # 当前冠军口径
```
详细续写循环（`decide_with_lookahead` + 温度 0.65 采样 + `observe(source="self")`）见 `scripts/lm/eval_64m_ssm_rollouts.py`。

### 2.2 现役代码架构速查
| 模块路径 | 职责 | 关键类 / 入口 |
|---|---|---|
| [core/compute/opencl_gpu.py](core/compute/opencl_gpu.py) | 原生 OpenCL 计算内核 | `OpenCLAccelerator`（`5.60 TFLOPS` 32×32 SGEMM、GRU/RC 步进前反向、Softmax-CE-Grad、AdamW、显存常驻 token 切片、GPU 束电路场） |
| [core/memory/learnable_ssm_policy.py](core/memory/learnable_ssm_policy.py) | **当前核心状态空间策略** | `LearnableStateSpacePolicy`（1/2 层残差门控 RC 状态空间、96 步 GPU-BPTT、原地状态更新、高 $\tau$ 慢通道选择性锚定） |
| [core/memory/stateful_context_policy.py](core/memory/stateful_context_policy.py) | 前代固定 EMA 门控策略 | `StatefulContextPolicy`（保留兼容与 GPU 批量推理对照） |
| [core/memory/context_integrator.py](core/memory/context_integrator.py) | 块级多线索上下文整合 | `ContextualIntegrator`（均值中心化嵌入、内容门控贝叶斯融合、14K 领域块与整块承诺） |
| [core/memory/temp_context.py](core/memory/temp_context.py) | 工作记忆与电路场宿主 | `TemporaryContextMemory`、`ElectronicCircuitNodeField`（自学习每节点 R/C 时间常数） |
| [core/memory/substrate_bundle.py](core/memory/substrate_bundle.py) | 统一构建与检查点序列化 | `StructuralSubstrateBundle`（24 核并行拟合、流式分块摄取、16 GB 治理、`save/load`、`make_controller`） |
| [core/planning/context_decision.py](core/planning/context_decision.py) | 统一决策与前瞻控制器 | `ContextDrivenDecisionController`（4 专家 LOO 仲裁 + `UnionBlend` + 深度 2 束宽 5 批量前瞻） |
| [core/stage/](core/stage/) / [core/env/](core/env/) / [core/runner/](core/runner/) | 连续控制/变物理主线 | `StageRegimeSummarizer`、`OnlineRegimeContinuationUpdater`、`OnlineFeatureDemandUpdater`、`DynamicsClassNode` |

### 2.3 当前最新检查点与验证基线（HEAD `c5c91d9`）
- **最新默认 64M 可复用检查点**：`checkpoints/wiki_zhihu_64m_v2000.pkl.gz`（`256.95 MB`，已内含 64M 符号层 + 2 层 96 步 BPTT `LearnableStateSpacePolicy`，`StructuralSubstrateBundle.load(...)` 约 13s 加载即可直接评测/续写/增量实验）。另有 `4m / 16m / 256m` 历史检查点位于 `checkpoints/`。
- **核心单元测试（42/42 绿灯，~15s）**：
  ```powershell
  C:\workspace\code\.venv-ft\Scripts\python.exe -m unittest `
    tests/test_context_integrator.py tests/test_scalability_and_arbitration.py `
    tests/test_circuit_gpu.py tests/test_policy_pool.py tests/test_structural_memory.py `
    tests/test_recency_kernel.py tests/test_context_decision.py tests/test_temp_context.py `
    tests/test_sequence_stage.py tests/test_class_policy.py tests/test_virtual_node_dag.py -v
  ```
- **严格零重叠 64M 演进基线（`@260M ev[5000:11000]`, `n=5876`, `V=2000`）**：

| 阶段 / 模型配置 | 规模 | Top-1 | Top-8 | SameCls | SemAlign | CenteredSem | 留出 CE / PPL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `STATE_ONLY`（1 阶原地 Ridge） | 64M | 8.80% | 27.60% | 9.00% | — | — | — |
| `GATED_ONLY`（1 层 256 门控） | 64M | 20.68% | 45.06% | 23.35% | — | — | — |
| `SELECTIVE_ONLY`（1 层 512 门控 GPU） | 64M | 21.82% | 47.31% | 24.29% | — | — | — |
| `DEEP_2L_ONLY`（固定 EMA 2 层 GPU） | 64M | 23.64% | 50.00% | 27.74% | — | — | 4.321 / 75.3 |
| 1 层 48 步 `SSM_ONLY`（7.79 MB） | 64M | 28.85% | 54.03% | 32.74% | 74.85% | 53.39% | 3.997 / 54.4 |
| 纯符号记忆层 `L4` 基线 | 64M | 29.17% | 55.14% | 32.61% | 77.37% | 55.11% | — |
| 历史 256M 全量符号基线（4× 数据） | 256M | 29.70% | 56.70% | 32.50% | — | — | — |
| 1 层 64M `UnionBlend (a=0.45)` | 64M | 31.52% | 59.17% | 34.72% | 77.84% | 55.89% | — |
| **2 层 96 步 `SSM_2L_ONLY`（13.80 MB）** | **64M** | **31.76%** | **57.11%** | **35.33%** | 75.05% | 54.77% | **3.805 / 44.9** |
| **2 层 64M `UnionBlend (a=0.65)`（当前冠军）** | **64M** | **32.93%** | **61.10%** | **36.20%** | 77.28% | **56.57%** | — |

- **已验证涌现特性与盲测基线**：
  - **两级自学习 RC 时间常数**（$\tau = 1/\mathrm{mean}(z_t)$）：Layer-1 快句法 `[p50=1.36, p95=5.43]` 步；Layer-2 慢篇章 `[p50=5.30, p75=41.71, p95=73.90]` 步。
  - **3 段 100 字盲测续写**（`UnionBlend a=0.65, slow_anchor=0.42`, depth=2 beam=5）：
    - `P1(立冬/日历 @260M+16367)`：`80.7 ms/step` | `3gram_rep=6.1%` | `centered_cos(p)=0.878` | `centered_cos(gt)=0.701`
    - `P2(个税/BI数据 @260M+19122)`：`71.2 ms/step` | `3gram_rep=12.2%` | `centered_cos(p)=0.890` | `centered_cos(gt)=0.842`
    - `P3(Wiki-上海广场 @260M+82)`：`64.2 ms/step` | `3gram_rep=17.3%` | `spaces=0` | `centered_cos(p)=0.736` | `centered_cos(gt)=0.285`

### 2.4 当前明确暴露的 4 个短板
1. **单篇低频实体长程保真不足**：`P1`（立冬/农历/节假日）在续写第 20–80 字后仍会被高频知乎回答模板（“所以说这个是我们的主题”“我们可以看出”）稀释，纯固定维度的 512 维慢向量难以精确召回单篇低频实词。
2. **字符级词表 `V=2000` 瓶颈**：`<UNK>` 约 `2.5%`，且纯单字符输入迫使底层状态消耗大量步数拼字成词。
3. **推理前瞻延迟偏高**：深度 2 束宽 5 前瞻目前 `64–81 ms/step`，2 层 SSM 尚未与符号/电路场统一 GPU 合批前瞻。
4. **跨模态统一尚未闭环**：端到端 GPU-BPTT `LearnableStateSpacePolicy` 尚未反哺连续控制 / 变物理多世界基准。

---

## 三、多 Agent 协同硬约束（所有 Agent 必须遵守）

1. **任务无关红线**：`core/` 内严禁引入中文分词器、正则/标点特判、词性表、硬编码复读惩罚或任务专用几何特征。
2. **默认规模与零泄漏红线**：序列实验默认使用 **64M**（`0..64M`），评测严格锁定 **`@260M ev[5000:11000]`** 及 `P1/P2/P3`，严禁将 `@260M` 任何片段用于训练或超参过拟合。
3. **内存与显存红线**：主机 RSS 严格 `≤ 16 GB`（当前 ~`8.2–9.6 GB`）；RX 9070 XT 显存可用空间 `13.53 GB`（当前训练约用 `2.33 GB`，可安全扩至 `4–8 GB`）。
4. **原地状态与数值一致红线**：所有新状态机制必须支持严格原地 `observe()` 与无副作用 `fork_step()`；新增 OpenCL GPU 内核必须补充 **GPU vs CPU 数值一致性单测**（`atol <= 5e-4`）。
5. **仓库整洁与提交规范**：临时实验脚本写到 scratch 目录、不提交大体积二进制检查点（`checkpoints/*.pkl.gz` 不入 git）；每次合入前须跑通 42 条核心单测，并如实更新本文件实测数据。

---

## 四、下一步可认领任务看板（Next Tasks）

> 各任务相互解耦，可由不同 Agent 并行认领推进；认领时请在任务标题后标注 `[IN PROGRESS: <agent>]`，完成后勾选并回填实测指标与 commit hash。

### [P0-1] 任务无关快权重关联记忆（Fast-Weight Hebbian Associative State）
- **目标**：解决固定维度慢向量稀释单篇低频实体（如 `P1` 的“立冬/农历/节假日”）的问题，实现**篇内罕见实体精确长程召回**。
- **方案要点**：
  - 在 `LearnableSSMState` 中新增一块可原地演化的轻量低秩快权重矩阵 $M_t \in \mathbb{R}^{d_k \times d_v}$（例如 $64 \times 64$，仅 `16 KB`/状态）：
    $$M_t = \lambda_t M_{t-1} + \eta_t (k_t \otimes v_t), \qquad m_t = \mathrm{LayerNorm}(q_t M_t)$$
  - 写入门 $\eta_t$ 与衰减门 $\lambda_t$ 由网络端到端从 BPTT 自学习（自然给低频/高惊喜度实体高写入、给高频虚词低写入），读出 $m_t$ 残差叠加至 `s1 + s2 + m_t`；
  - 同步实现 OpenCL 前/反向内核并补齐 GPU/CPU 数值一致性单测 + 合成“首次出现即记住（one-shot associative recall）”单测。
- **涉及文件**：`core/compute/opencl_gpu.py`、`core/memory/learnable_ssm_policy.py`、`tests/test_context_integrator.py`
- **验收门槛**：
  - 42+ 核心单测全绿（含 GPU/CPU 数值一致与 one-shot 关联召回单测）；
  - 64M `@260M` 上 `SSM_ONLY` 留出 PPL 低于 `44.9`，`UnionBlend` Top-1 突破 **`33.5%+`**；
  - `P1(立冬/日历)` 100 字续写中段能自然维持“立冬/农历/时间/节假日”相关实词，显著减少知乎通用口癖。

### [P0-2] 双流多粒度输入（Atom + Self-Discovered Chunk）与 RX 9070 XT 显存扩容
- **目标**：打破 `V=2000` 纯单字瓶颈，充分利用 RX 9070 XT 剩余 ~11 GB 显存将等效上下文视野推至 **200+ 字**。
- **方案要点**：
  - **路径 A（双流输入）**：在 `LearnableStateSpacePolicy` 输入层融合字符嵌入 `E_atom[sym]` 与 `HierarchicalChunkMemory` 自发现的块嵌入 `E_chunk[chunk_id]`（零外部分词器），减少底层拼字消耗；
  - **路径 B（规模与深度扩容）**：将词表扩至 `V=4000~5000`（`<UNK> < 0.5%`），评估 `H=768` / 3 层残差门控 RC（或对角选择性线性递推 + SwiGLU 门控）与 `T=128~192` 长程 BPTT（目标活跃显存 `4–8 GB / 13.53 GB`），并在默认 64M 跑通后再做一轮 256M 全量扩展对比。
- **涉及文件**：`core/memory/learnable_ssm_policy.py`、`core/memory/substrate_bundle.py`、`core/compute/opencl_gpu.py`
- **验收门槛**：
  - 主机 RSS `≤ 16 GB`，训练吞吐不低于 `100 ktok/s`；
  - 同口径 `@260M` 上 64M `UnionBlend` Top-1 突破 **`34.5%+`** / Top-8 突破 **`63.0%+`**（扩词表时需同时报告同口径 `V=2000` 子集指标以确保可比）。

### [P1-1] 前瞻推理全链 GPU 合批加速（目标 `< 15 ms/step`）
- **目标**：消除 `decide_with_lookahead` 当前 `64–81 ms/step` 延迟，为更深前瞻（depth 3–4）与实时交互铺路。
- **方案要点**：
  - 分析 `decide_with_lookahead` 逐模块耗时热点（Python 字典查表 vs `LearnableStateSpacePolicy` `fork_step`/readout vs 电路场）；
  - 将 `LearnableStateSpacePolicy` 的权重常驻 RX 9070 XT 推理缓存，实现批量 `fork_step_batch` + `predict_batch_with_pmi` 一步 GPU 完成，并减少每分支 Python 对象复制开销。
- **涉及文件**：`core/memory/learnable_ssm_policy.py`、`core/planning/context_decision.py`、`core/compute/opencl_gpu.py`
- **验收门槛**：
  - 深度 2 束宽 5 前瞻在 `P1/P2/P3` 上实测 **`≤ 20 ms/step`**（力争 `< 15 ms/step`），且与 CPU 路径预测分布数值等价（单测锁定）。

### [P1-2] 测试期惊喜度门控原地自适应（Episode-Time Fast Adaptation）
- **目标**：让系统在阅读未见长篇/新专有名词时，仅凭单次前向流就能原地记住篇内新概念，且不污染全局权重、不引发崩溃。
- **方案要点**：
  - 阅读外部 prompt/文档时，按 token 预测惊喜度 $-\log p(y_t \mid s_{t-1})$ 门控触发轻量级原地更新（仅更新局部 readout 偏置或 P0-1 快权重，受 Wilson-LCB / KL 散度护栏约束，episode `reset()` 时清空或回滚）；
  - 设计严格的长文档（如 `@260M` 连续 2000–5000 字）前段 vs 后段 PPL 对比评测。
- **涉及文件**：`core/memory/learnable_ssm_policy.py`、`core/planning/context_decision.py`
- **验收门槛**：
  - 连续未见长文后半段（500–2000 字）留出 PPL 显著低于冻结推理基线，且 100 字自由续写无退化或复读反弹。

### [P2-1] 跨模态统一状态底座验证（语言序列 + 连续控制合体）
- **目标**：验证 `LearnableStateSpacePolicy` 的原地门控 RC 动力学同样胜任连续控制与变物理环境，实现两条主线的架构统一。
- **方案要点**：
  - 将 `LearnableStateSpacePolicy` 输入端泛化为支持连续观测向量投影 `x_t = obs_t @ W_obs`（与离散 `E_[tok]` 并列），接入 `DelayedCartPole(lag=2)` 与变物理多世界基准做多步观测/终局风险预测或隐状态表征；
  - 验证在变物理切换场景中，门控 RC 隐状态是否自动分化出快动力学通道与慢物理参数通道。
- **涉及文件**：`core/memory/learnable_ssm_policy.py`、`core/runner/`、`tests/test_cartpole_cross_task.py`
- **验收门槛**：
  - 不破坏既有 `DelayedCartPole`（`369.3`）与多世界变物理双门指标，并在隐状态追踪/变物理切换预测上呈现可量化正向增益。

---

## 历史归档索引

2026-11-27 及更早的非近期记录已原样迁出至以下只读归档文档：

| 日期 / 范围 | 归档文件 | 主要内容 |
|---|---|---|
| 2026-11-24 ~ 2026-11-27 | [TODO-2026-11-27.md](docs/archive/TODO-2026-11-27.md) | Phase 1–17 阶段总结/方差树/串联主动剪枝/在线 LCB/点事件失败记忆/变物理节点变量/关系差特征/残差门控世界重拟合，及早期多视窗转移与阶段记忆基座记录 |
| 2026-11-23 | [TODO-2026-11-23.md](docs/archive/TODO-2026-11-23.md) | 方向校正、晚间更新、里程碑/阶段总结、Batch 6（图重写/规则库）、P0 time-in-state、Batch 7（B0–B1-plan）、Batch 8（6 步 executor / span / 事件线） |
| 2026-11-21 | [TODO-2026-11-21.md](docs/archive/TODO-2026-11-21.md) | 历史状态快照、Batch 4 尾项、记忆树/阶段图/多尺度历史设计稿、第二层诊断、质量门与 A/B 协议、旧 README 归档索引 |
| 2026-11-20 及更早 | [TODO-appendix-2026-11-20.md](docs/archive/TODO-appendix-2026-11-20.md) | 四子代理解剖附录（信息流/收敛/常数/文献）、死机制清单、三子代理审计整合、早期质量/泛化瓶颈与节点中心改造记录 |
