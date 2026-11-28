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
| **核心单测（48 条）** | `& $PY -m unittest tests/test_context_integrator.py tests/test_scalability_and_arbitration.py tests/test_circuit_gpu.py tests/test_policy_pool.py tests/test_structural_memory.py tests/test_recency_kernel.py tests/test_context_decision.py tests/test_temp_context.py tests/test_sequence_stage.py tests/test_class_policy.py tests/test_virtual_node_dag.py -v` | ~18s，需 GPU 的用例会自动使用 OpenCL |
| **64M 符号层从零构建**（16M seed + 3×16M 流式摄取 + 保存检查点） | `& $PY scripts/lm/train_and_eval_64m_full_v3.py` | ~10 min，RSS ≤ 8.6 GB；覆盖写 `checkpoints/wiki_zhihu_64m_v2000.pkl.gz` |
| **64M 2 层 SSM 训练 + `@260M` 评测 + P1/P2/P3 续写**（原 `H=512` 基线流程，需先有上面的符号层检查点） | `& $PY scripts/lm/train_and_eval_64m_2layer_ssm.py` | ~14 min（训练 718s，吞吐 ~178 ktok/s）；RSS ≤ 9.6 GB，显存 ~2.3 GB；训练完成后**覆盖写**同一检查点 |
| **64M `V=2000` P0-1+P0-2 升级训练 + 严格盲测**（`H=768×2` + 门控快权重 `FW=64` + 双流 Chunk） | `& $PY scripts/lm/train_and_eval_64m_p01_p02.py` | ~22 min（训练 1216s，吞吐 `105.3 ktok/s`）；RSS `8.85 GB`，显存 `4.23 GB`；写独立 `checkpoints/wiki_zhihu_64m_v2000_p01_p02.pkl.gz` |
| **64M `V=4096` 扩词表 P0-1+P0-2 全量构建与盲测**（`<UNK>=0.29%` + 符号层 + 双流 Chunk + `FW=64` + `H=768×2`） | `& $PY scripts/lm/train_and_eval_64m_v4096_p01_p02.py` | ~30 min（符号层 ~5 min + GPU-BPTT 1437s，`89.1 ktok/s`）；RSS `9.50 GB`，显存 `4.46 GB`；写独立 `checkpoints/wiki_zhihu_64m_v4096_p01_p02.pkl.gz` |
| **仅评测续写**（加载已有检查点，不训练） | `& $PY scripts/lm/eval_64m_ssm_rollouts.py` | ~1.5 min |
| **256M 全量构建 / 评测**（可选，对照上限） | `& $PY scripts/lm/train_and_eval_256m.py` | ~9 min，RSS ≤ 11.5 GB；写 `checkpoints/wiki_zhihu_256m_v2000.pkl.gz` |
| **长任务提示** | 以上脚本均会持续 print 日志；建议后台运行并将输出重定向，如 `& $PY scripts/lm/train_and_eval_64m_2layer_ssm.py *> results/lm_run_$(Get-Date -f yyyyMMdd_HHmm).log` | — |

**输入**
- 语料：`C:\workspace\code\data\wiki_zhihu\corpus.txt`（字符级；前 16M 字符构建词表，`<UNK>=0`：`V=2000` 时 `@260M` UNK=`2.62%`，`V=4096` 时 UNK=`0.29%`）；训练区间 `[0, 64M)`（或 256M 扩展 `[0, 256M)`），开发选参区间 `@257M`，封存评测区间 `@260M`（取 20,000 字，使用 `ev[5000:11000]`，`n=5876` / `5990`），**禁止用于训练或选参**。
- 检查点：`checkpoints/wiki_zhihu_64m_v4096_p01_p02.pkl.gz`（最新 `V=4096` 旗舰）或 `checkpoints/wiki_zhihu_64m_v2000_p01_p02.pkl.gz`（`V=2000` 对照）；原基准 `checkpoints/wiki_zhihu_64m_v2000.pkl.gz` 保留未覆盖。

**输出**
- 检查点目录：`checkpoints/`（`wiki_zhihu_64m_v4096_p01_p02.pkl.gz` `304.03 MB`、`wiki_zhihu_64m_v2000_p01_p02.pkl.gz` `275.02 MB`、原 `wiki_zhihu_{4m,16m,64m,256m}_v2000.pkl.gz`）。
- 评测与续写日志：`results/`（最新留档 `results/train_64m_p01_p02_v2000.log`、`results/train_64m_p01_p02_v4096.log`、`results/v4096_p01_p02_12prompts.log`、`results/ssm_training_check/256m_quality/`）。
- 单次实验的临时/一次性脚本请放各自 agent 的 scratch 目录，不要入库；可复用脚本放 `scripts/lm/`。

**最小代码调用（复用最新 `V=4096` P0-1+P0-2 检查点做续写/评测）**
```python
import sys, os
ROOT = r"c:\workspace\code\algo_core"; sys.path.insert(0, ROOT)
for dp, _, _ in os.walk(os.path.join(ROOT, "core")):
    if "__pycache__" not in dp: sys.path.insert(0, dp)
from substrate_bundle import StructuralSubstrateBundle
b = StructuralSubstrateBundle.load(os.path.join(ROOT, "checkpoints", "wiki_zhihu_64m_v4096_p01_p02.pkl.gz"))
ctrl = b.make_controller(use_arbitrator=True, lookahead_depth=2, lookahead_beam=5)
ctrl.stateful_weight = float(b.metadata.get("union_blend_alpha_dev", 0.45))  # V=4096 为 0.45；V=2000_p01_p02 为 0.65
ctrl.self_stateful_beta = 0.0
```
详细续写循环（`decide_with_lookahead` + 温度 0.65 采样 + `observe(source="self")`）见 `scripts/lm/eval_64m_ssm_rollouts.py`。

### 2.2 现役代码架构速查
| 模块路径 | 职责 | 关键类 / 入口 |
|---|---|---|
| [core/compute/opencl_gpu.py](core/compute/opencl_gpu.py) | 原生 OpenCL 计算内核 | `OpenCLAccelerator`（`5.60 TFLOPS` 32×32 SGEMM、前向融合 GRU/RC 步进、快权重 Delta-Rule LDS 单核扫描 `fast_weight_scan_{fwd,bwd}`、双流块嵌入累加 `embed_add_1`、Softmax-CE-Grad、AdamW、显存常驻切片） |
| [core/memory/learnable_ssm_policy.py](core/memory/learnable_ssm_policy.py) | **当前核心状态空间策略** | `LearnableStateSpacePolicy`（1/2/3 层残差门控 RC、P0-1 门控快权重关联矩阵 $M_t$、P0-2 双流 Atom+Chunk 输入、96 步 GPU-BPTT、原地状态更新、高 $\tau$ 慢通道选择性锚定） |
| [core/memory/stateful_context_policy.py](core/memory/stateful_context_policy.py) | 前代固定 EMA 门控策略 | `StatefulContextPolicy`（保留兼容与 GPU 批量推理对照） |
| [core/memory/context_integrator.py](core/memory/context_integrator.py) | 块级多线索上下文整合 | `ContextualIntegrator`（均值中心化嵌入、内容门控贝叶斯融合、14K 领域块与整块承诺） |
| [core/memory/temp_context.py](core/memory/temp_context.py) | 工作记忆与电路场宿主 | `TemporaryContextMemory`、`ElectronicCircuitNodeField`（自学习每节点 R/C 时间常数） |
| [core/memory/substrate_bundle.py](core/memory/substrate_bundle.py) | 统一构建与检查点序列化 | `StructuralSubstrateBundle`（24 核并行拟合、流式分块摄取、16 GB 治理、`save/load`、`make_controller`） |
| [core/planning/context_decision.py](core/planning/context_decision.py) | 统一决策与前瞻控制器 | `ContextDrivenDecisionController`（4 专家 LOO 仲裁 + `UnionBlend` + 深度 2 束宽 5 批量前瞻） |
| [core/stage/](core/stage/) / [core/env/](core/env/) / [core/runner/](core/runner/) | 连续控制/变物理主线 | `StageRegimeSummarizer`、`OnlineRegimeContinuationUpdater`、`OnlineFeatureDemandUpdater`、`DynamicsClassNode` |

### 2.3 当前最新检查点与验证基线（2026-11-28 P0-1 + P0-2 更新）
- **最新 64M 可复用检查点**：
  - `checkpoints/wiki_zhihu_64m_v2000.pkl.gz`（`256.95 MB`，原 64M 2L `H=512` 基准）；
  - `checkpoints/wiki_zhihu_64m_v2000_p01_p02.pkl.gz`（`275.02 MB`，64M `V=2000` `H=768×2 + FW64 + DualChunk`）；
  - `checkpoints/wiki_zhihu_64m_v4096_p01_p02.pkl.gz`（`304.03 MB`，64M `V=4096` `<UNK>=0.29%` + `H=768×2 + FW64 + DualChunk`）。
- **核心单元测试（48/48 绿灯，~17.9s）**：
  ```powershell
  C:\workspace\code\.venv-ft\Scripts\python.exe -m unittest `
    tests/test_context_integrator.py tests/test_scalability_and_arbitration.py `
    tests/test_circuit_gpu.py tests/test_policy_pool.py tests/test_structural_memory.py `
    tests/test_recency_kernel.py tests/test_context_decision.py tests/test_temp_context.py `
    tests/test_sequence_stage.py tests/test_class_policy.py tests/test_virtual_node_dag.py -v
  ```
- **严格零重叠 64M 演进基线（`@260M ev[5000:11000]`, `n=5876`, 同口径 `V=2000`；末两行附全量 `V=4096`）**：

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
| 2 层 96 步 `SSM_2L_ONLY`（`H=512`, 13.80 MB） | 64M | 31.76% | 57.11% | 35.33% | 75.05% | 54.77% | 3.805 / 44.91 |
| 2 层 64M `UnionBlend`（冻结 `@257M a=0.55`） | 64M | 32.90% | 61.06% | 36.15% | 77.57% | 56.62% | — |
| **P0-1+P0-2 `V=2000` SSM-only**（`H=768×2, FW64, DualChk`） | **64M** | **33.17%** | **59.33%** | **36.57%** | 75.48% | 55.63% | **3.637 / 37.98** |
| **P0-1+P0-2 `V=2000` UnionBlend**（冻结 `@257M a=0.65`） | **64M** | **34.33%** | **62.29%** | **37.49%** | 77.69% | 57.32% | — |
| **P0-1+P0-2 `V=4096` SSM-only（`V=2000` 子集 `n=5876`）** | **64M** | **34.04%** | **60.89%** | **37.71%** | 74.98% | 66.00% | **3.532 / 34.21** *(raw 34.78)* |
| **P0-1+P0-2 `V=4096` UnionBlend（`V=2000` 子集, 冻结 `a=0.45`）** | **64M** | **34.58%** | **62.39%** | **37.68%** | **78.15%** | **69.37%** | — |
| **P0-1+P0-2 `V=4096` 全量（`n=5990`, `<UNK>=0.29%`）** | **64M** | **34.24%** *(SSM 33.71%)* | **61.72%** *(SSM 60.22%)* | **37.35%** | 77.23% | 68.55% | **3.618 / 37.27** *(SSM)* |

> **质量评测协议复核（2026-11-28）**：旧 64M 训练脚本在 `@260M ev[5000:11000]` 上扫 alpha 并选最优值，再用同片段报告冠军指标；因此 `a=0.65` 和相应 UnionBlend 指标属于开发结果，不能作为盲测结论。现改为 `@257M` 选择、`@260M` 冻结参数后单次报告。现有 64M 检查点的拆分复测在 `@257M` 选得 `a=0.55`；固定后 `@260M n=5876` 为 Top-1 `32.90%`、Top-8 `61.06%`、SameCls `36.15%`、SemAlign `77.57%`、CenteredSem `56.62%`，SSM CE/PPL `3.805 / 44.90`。Alpha 的盲测 Top-1/Top-8 与历史数值接近，但基线/超参对照应以后续严格拆分实验为准。P1/P2/P3 续写只在冻结 alpha 后运行一次并保留作盲测记录。

- **已验证涌现特性与最新盲测续写（2026-11-28 P0-1+P0-2）**：
  - **两级自学习 RC 时间常数**（$\tau = 1/\mathrm{mean}(z_t)$）：`V=4096` 版 Layer-1 快句法 `[p50=1.50, p95=5.58]` 步，Layer-2 慢篇章 `[p50=4.93, p75=43.04, p95=93.92]` 步；`V=2000` 版 Layer-2 `p95=96.47` 步。
  - **3 段 100 字盲测续写**（depth=2, beam=5，冻结 `@257M` `alpha`）：
    - `P1(立冬/日历 @260M+16367)`：`V=4096` `24.5 ms/step` | **`3gram_rep=1.0%`** | `cos(p)=0.933` | **`cos(gt)=0.800`**（`V=2000` 为 `17.9 ms/step`, `rep=4.1%`, `cos(gt)=0.670`）
    - `P2(个税/BI数据 @260M+19122)`：`V=4096` `22.6 ms/step` | **`3gram_rep=12.2%`** | `cos(p)=0.896` | **`cos(gt)=0.882`**（`V=2000` 为 `18.8 ms/step`, `rep=12.2%`, `cos(gt)=0.853`）
    - `P3(Wiki-上海广场 @260M+82)`：`V=2000` `19.0 ms/step` | **`3gram_rep=6.1%`** | **`cos(gt)=0.795`**；`V=4096` 消除了「淮海中路/淮海公园」`<UNK>`（`23.2 ms/step`, `rep=16.3%`, `cos(p)=0.716`）

### 2.4 当前明确暴露的 4 个短板（下一阶段强化靶点）
1. **单标量快权重衰减门退化，尚未锁死跨句实体**：自然语料下单个标量 $\lambda_t$ 收敛到 `~0.008`（退化为 1–2 步局部双线性），导致长程具体实词仍会滑向通用口癖（如 `P1` 丢掉「立冬/农历」）。
2. **单字触发导致的“联想跳轨”与实体幻觉**：实测 8 条情景续写中，`i=179` 从「指甲」因“甲”字跳到「甲骨文/甲状腺/甲午战争」，`i=111` 从「武洛社」跳到「武田信重/武则天」，`P3 V=4096` 因「香港」复读「香港大学/南京大学」；另在混入篇内检索时 `i=187` 复读升至 `31.6%`。
3. **`V=4096` 尚未扩展至 256M 语料且吞吐略低于 100 ktok/s**：当前 `V=4096` 仅在 64M 训练（`89.1 ktok/s`），Top-8 `62.39%`（距 `63.0%+` 仅差 `0.61 pp`），输出层投影仍有提速空间。
4. **跨模态统一尚未闭环**：端到端 GPU-BPTT `LearnableStateSpacePolicy` 尚未反哺连续控制 / 变物理多世界基准。

---

## 三、多 Agent 协同硬约束（所有 Agent 必须遵守）

1. **任务无关红线**：`core/` 内严禁引入中文分词器、正则/标点特判、词性表、硬编码复读惩罚或任务专用几何特征。
2. **默认规模与零泄漏红线**：序列实验默认使用 **64M**（`0..64M`），评测严格锁定 **`@260M ev[5000:11000]`** 及 `P1/P2/P3`，严禁将 `@260M` 任何片段用于训练或超参过拟合。
3. **内存与显存红线**：主机 RSS 严格 `≤ 16 GB`（当前 ~`8.8–9.5 GB`）；RX 9070 XT 显存可用空间 `13.53 GB`（当前 64M 训练约用 `4.2–4.5 GB`，可安全扩至 `8 GB`）。
4. **原地状态与数值一致红线**：所有新状态机制必须支持严格原地 `observe()` 与无副作用 `fork_step()`；新增 OpenCL GPU 内核必须补充 **GPU vs CPU 数值一致性单测**（`atol <= 5e-4`）。
5. **仓库整洁与提交规范**：临时实验脚本写到 scratch 目录、不提交大体积二进制检查点（`checkpoints/*.pkl.gz` 不入 git）；每次合入前须跑通 48 条核心单测，并如实更新本文件实测数据。

---

## 四、下一步可认领任务看板（Next Tasks）

> 各任务相互解耦，可由不同 Agent 并行认领推进；认领时请在任务标题后标注 `[IN PROGRESS: <agent>]`，完成后勾选并回填实测指标与 commit hash。

### 已结项基建与吞吐筛查结论（保留归档口径，不再重复投入）
- **推理与训练基建已达标**：CPU 侧后缀缓存 + NumPy 连续数组合并将前瞻延迟从 `~88–114 ms` 压至 `~18–24 ms/字`；GPU BPTT 去除切片复制 + 前向 GRU 融合（`gru_projection_fused_kernel`）达 `125 ms/批`。
- **已验证不采纳路线（日志留存于 `results/ssm_training_check/`）**：梯度行采样、混合 BPTT 长窗启发式调度、冻结预测增益选窗、截断 96 步 carry 边界、SGEMM K32 tile、GRU 反向融合均未提供稳定净收益；默认保持完整 `T=96` 连续反传。

### [已完成阶段] P0-1 & P0-2 基础架构合体（2026-11-28 阶段收口）
- **P0-1（快权重关联记忆 `fw_dim=64`）**：`opencl_gpu.py` 单核 LDS 扫描（`fast_weight_scan_{fwd,bwd}`）+ 无命中静音 Soft-Norm + 读出门控 $o_t$；40 局随机键值 one-shot 关联召回 `≥ 34/40`（GPU/CPU 误差 `≤ 5e-7`）。
- **P0-2（双流 Atom+Chunk + `H=768×2` + `V=4096`）**：`HierarchicalChunkMemory.match_stream_chunks()`（64M 流对齐 `7.8–8.3s`，命中率 `34.9%`）；`<UNK>` 降 9× 至 `0.29%`；64M `@260M` 盲测将纯 SSM PPL 从 `44.91` 压至 `34.21`（`V=2000` 子集），`UnionBlend` Top-1 达 `34.58%` / Top-8 `62.39%`。

---

## 五、下一阶段强化方向（可认领任务）

### [S1 / P0-1 强化] 多头各向同性快权重 + 慢槽保真（解决单字跳轨与实词丢失）
- **痛点**：当前单标量 $\lambda_t$ 被局部句法拉低至 `~0.008`，仅起到 1–2 步局部双线性作用，未能保护跨句实体（如 `P1` 的「立冬/农历」、`i=179`「指甲」防跳轨）。
- **方案要点**：
  - 将单个 `64×64` 快权重拆为 **多头各向同性快权重**（如 4 头 × `32×32` 或 8 头 × `16×16`）：每头保持严格各向同性标量衰减 $\lambda^{(h)}_t$（已验证不可用按行各向异性衰减，会破坏键内积对齐），其中设置不同慢头下限/偏置（如 2 个慢头 $\lambda^{(h)} \in [0.90, 0.995]$ 专司篇内实体绑定，2 个快头自由学习 $\in (0, 1)$）；
  - 可在 `fast_weight_scan_{fwd,bwd}` 内直接循环或按 `(b, head)` 映射到 workgroup（4 头 × `32×32` 每 workgroup 仅 ~`4.2 KB` LDS，大幅提升 GPU 占用率）。
- **验收门槛**：48+ 核心单测全绿；`P1` 续写中后段保留「立冬/农历/节假日」等实词，情景集防跳轨明显改善，同口径 `@260M` Top-1 突破 `35.0%+` / PPL 低于 `33.5`。

### [S2 / P0-2 强化] `V=4096` 扩展至 256M 全量语料 + 输出层提速（冲刺 Top-8 ≥ 63.0%）
- **方案要点**：
  - 基于 `checkpoints/wiki_zhihu_64m_v4096_p01_p02.pkl.gz` 扩展构建 256M `V=4096` 符号层，并在 `[0, 256M)` 语料上对 `H=768×2 + 多头快权重 + DualChunk` 做 1–2 轮 GPU-BPTT 续训（严格封存 `@260M`）；
  - 优化 `V=4096` 输出层 SGEMM / `softmax_ce_grad`（当前 `89.1 ktok/s`），力争恢复 `≥ 100 ktok/s`。
- **验收门槛**：RSS `≤ 16 GB`，`@260M` 同口径 `V=2000` 子集 Top-8 突破 **`63.0%+`**、PPL 低于 **`33.0`**，8/12 条情景集实体召回显著提升。

### [S3 / P1-2 强化] 测试期惊喜度门控自适应 & 情景检索防复读融合
- **痛点**：当前情景记忆混入（`with_memory`）虽提高召回，但会在局部触发高复读（如 `i=187` 升至 `31.6%`）或英文串扰。
- **方案要点**：
  - 在阅读外部文档/prompt 时按惊喜度 $-\log p(y_t \mid s_{t-1})$ 门控触发测试期快权重/局部偏置原地自适应，并受 KL 散度 / Wilson-LCB 护栏约束（`reset()` 清空）；
  - 设计 `@260M` 连续 2000–5000 字长文档前段 vs 后段 PPL 对比与 12 条情景集复读率验收。
- **验收门槛**：长文档后半段 PPL 显著低于冻结推理基线，情景集召回提升的同时 3-gram 重复率不反弹。

### [S4 / P1-1 收口] `V=4096` 推理前瞻 GPU 合批（目标 `< 15 ms/step`）
- **方案要点**：针对 `V=4096` 下 `decide_with_lookahead`（当前 `22.6–24.5 ms/step`）做推理侧 `fork_step_batch` + readout GPU 合批，压至 `< 15 ms/step` 并锁定与 CPU 分布等价单测。

### [S5 / P2-1] 跨模态统一状态底座验证（语言序列 + 连续控制合体）
- **方案要点**：为 `LearnableStateSpacePolicy` 补充连续观测投影 `x_t = obs_t @ W_obs`，接入 `DelayedCartPole(lag=2)` 与变物理多世界基准验证快慢通道分化，不破坏既有 `369.3` 基准。

---

## 历史归档索引

2026-11-27 及更早的非近期记录已原样迁出至以下只读归档文档：

| 日期 / 范围 | 归档文件 | 主要内容 |
|---|---|---|
| 2026-11-24 ~ 2026-11-27 | [TODO-2026-11-27.md](docs/archive/TODO-2026-11-27.md) | Phase 1–17 阶段总结/方差树/串联主动剪枝/在线 LCB/点事件失败记忆/变物理节点变量/关系差特征/残差门控世界重拟合，及早期多视窗转移与阶段记忆基座记录 |
| 2026-11-23 | [TODO-2026-11-23.md](docs/archive/TODO-2026-11-23.md) | 方向校正、晚间更新、里程碑/阶段总结、Batch 6（图重写/规则库）、P0 time-in-state、Batch 7（B0–B1-plan）、Batch 8（6 步 executor / span / 事件线） |
| 2026-11-21 | [TODO-2026-11-21.md](docs/archive/TODO-2026-11-21.md) | 历史状态快照、Batch 4 尾项、记忆树/阶段图/多尺度历史设计稿、第二层诊断、质量门与 A/B 协议、旧 README 归档索引 |
| 2026-11-20 及更早 | [TODO-appendix-2026-11-20.md](docs/archive/TODO-appendix-2026-11-20.md) | 四子代理解剖附录（信息流/收敛/常数/文献）、死机制清单、三子代理审计整合、早期质量/泛化瓶颈与节点中心改造记录 |
