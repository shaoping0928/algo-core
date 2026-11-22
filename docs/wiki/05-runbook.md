# 05 · 运行方式

## 5.1 环境搭建

需要 Python 3 与 NumPy。唯一第三方依赖在 `requirements.txt`：

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate
python -m pip install -r requirements.txt
```

所有命令的**工作目录**是 `algo_core/`。

## 5.2 快速自检（无需训练）

```bash
python -m unittest discover -s tests
```

## 5.3 主线：节点中心规划（`substrate_nodes.py`）

```bash
# 默认（点事件 pain 标签），3 种子一波并行
python substrate_nodes.py --seed 100 --seeds 3 --jobs 3 \
    --eval-seeds 120-124 --eval-episodes 3

# 快速开发预设：1500 步 / 3 epoch / 1000 节点 / H=8 / cap=400
python substrate_nodes.py --fast --seeds 3 --jobs 3 --eval-seeds 120-124

# 标准尺（benchmark）：1500 帧离线腿只回放 1 遍 + 一轮 5000 帧自由控制 + 评测 cap 5000
python substrate_nodes.py --benchmark --seeds 3 --jobs 3 --eval-seeds 120-124

# 闭环 DAgger：规划器自采真实轨迹回流，迭代 3 轮
python substrate_nodes.py --fast --seeds 3 --jobs 3 \
    --dagger-rounds 3 --dagger-episodes 8 --dagger-cap 400 \
    --eval-seeds 120-124

# 决策消融：把 MPC 搜索换成单步读格子
python substrate_nodes.py --fast --seeds 3 --jobs 3 --decision cellq --eval-seeds 120-124

# 输出逐 episode 行 + 汇总 + 聚类 bootstrap CI 到 JSON
python substrate_nodes.py --seed 100 --seeds 3 --jobs 3 \
    --eval-seeds 120-124 --report report.json
```

要点：

- `--eval-seeds` 为空时会打印警告并回退旧的 self-eval（训练种子 = 评估种子）；**正式比较务必显式给 held-out**（推荐 `120-139`）。
- 世界 id 走 `eval_seed * 10000 + episode`，与 `collect` 命名一致。
- `--fast` 与 `--benchmark` 是互斥的预设改写器（后者在 `--dagger-*` 未设时自动补成 1 轮 / 5000 帧）。
- 可用 `--no-features`、`--precursor-labels`、`--curiosity-weight 0` 等旗标做消融（完整清单见 [03 关键类与函数 §3.6](./03-core-api.md#36-substrate_nodespy)）。

## 5.4 对照路径

```bash
# 世界反馈价值模式（默认）；按种子并行
python substrate_sandbox_search.py --jobs 8

# 任务 gap 评分基线，仅用于对照
python substrate_sandbox_search.py --task-baseline --jobs 8

# 长时域 beam 对照（注意：必须显式指定 --risk-aggregate max|sum）
python substrate_sandbox_search.py --engine beam --jobs 8

# 任务无关原型：只通过 obs/step/feedback 交互
python substrate_generic.py --seed 100

# 图记忆风险驱动的搜索原型
python substrate_graph_search.py --seed 100 --seeds 4 --jobs 4 --coverage 0.1 --cap 1200
```

## 5.5 诊断脚本（训练/机制追踪）

```bash
python trace_node_runtime.py --fast          # 部署路径 8 节瓶颈诊断（seed 100）
python trace_node_training.py                # 训练过程结构增长
python trace_memory_change.py                # 记忆节点学习前后变化
python trace_generic_training.py --seed 100 --coverage 0.1
python trace_signal_world.py --seed 100 --coverage 0.1
python trace_sensitivity.py --seed 100 --coverage 0.1
python trace_mixed_nodes.py --seed 100 --coverage 0.1
python trace_graph_memory.py --seed 100 --coverage 0.1
python trace_feedback.py 100                 # 管道附近方向正确率
python trace_rule_induction.py --seed 100 --steps 5000 --coverage 0.1

python training_introspection_probe.py --seed 100 --rounds 2 --report intro.json
python feature_generalization_probe.py --seed 100 --eval-seeds 120-124
python memory_credit_probe.py --seed 100 --eval-seeds 120-124 --report mc.json

# 时间尺度（柱间距）不变性：同一模型在 0.8/1.6/2.4 下重评；--policy reference 用强控制器
python spacing_invariance_probe.py --seeds 24 --jobs 6 --eval-seeds 200-203 --report si.json
python spacing_invariance_probe.py --policy reference --eval-seeds 120-139 --cap 6000
```

## 5.6 基座/规则层/其他验证脚本

这些脚本**没有 CLI 参数**（或只有位置参数），直接运行：

```bash
# 规则层
python rule_mainline.py
python rule_loop.py 2000
python rule_policy.py
python rule_trigger.py
python live_core.py 6

# 延迟回忆
python recall_world.py 200 600 0        # episodes_per_d, units, split(0/1)

# 数学实验
python math_memory_search.py

# 信号/算子/延迟基础验证（无参数）
python substrate_signals.py
python substrate_signals_noise.py
python substrate_selfdelay.py
python substrate_delay.py
python substrate_layers.py
python substrate_state.py
python substrate_basis.py
...
```

> 这些脚本是**独立实验/验证器**，不是集成好的生产运行时，也不会自动训练出通用策略。它们的结论记录在 `docs/substrate.md`。

## 5.7 原始算术语料（对接外部 nanoGPT）

```powershell
$raw = "C:\temp\raw_arith"
$run = "C:\temp\raw_arith_run"
python raw_math_experiment.py prepare $raw

# 阶段 1 记忆（一位数事实）
python train.py config/baseline.py "--dataset=$raw/memory" "--out_dir=$run" `
  --device=cuda --dtype=bfloat16 --compile=False --block_size=64 `
  --batch_size=64 --max_iters=600 --lr_decay_iters=600 `
  --always_save_checkpoint=True --init_from=scratch --n_loop=0

# 阶段 2 形式（多位数原始等式，continue）
python train.py config/baseline.py "--dataset=$raw/form" "--out_dir=$run" `
  --init_from=resume --max_iters=5000 --lr_decay_iters=5000 --n_loop=0

# 逐位精确匹配评估
python raw_math_experiment.py eval --nanogpt-root <path> --checkpoint <ckpt> --meta-dir $raw/memory
```

## 5.8 协议与预设速查

| 预设 | steps | epochs | max_nodes | horizon | cap | 备注 |
|---|---:|---:|---:|---:|---:|---|
| 默认 | 5000 | 5 | 5000 | 10 | 6000 | 完整规模 |
| `--fast` | 1500 | 3 | 1000 | 8 | 400 | 单种子约 11s，8 种子约 41s |
| `--benchmark` | 1500 | 1 | 1000 | 8 | 5000 | 离线 1 遍 + 5000 帧在线 |

评估：每种子 `--eval-episodes` 条独立 episode；统计按**训练模型聚类**（同一模型的多个 episode 不当独立证据）。

## 5.9 确定性与线程

- **确定性**：同一命令重复运行逐位相同（含 DAgger 多轮与多进程 `--jobs`）。
- **行为流钉死**：`substrate_generic.collect` 的采样器 RNG 顺序被 `test_behavior_stream_is_pinned` 用 MD5 钉死——任何改动行为流的 PR 必须**有意**更新该哈希（当前七坐标流哈希 `6bb2a643…`）。
- **并行模型**：GIL 构建用 `ProcessPoolExecutor`，free-threaded 用 `ThreadPoolExecutor`；`FREE_THREADED` 常量在 `substrate_nodes.py` 顶部判定。

## 5.10 常见坑

1. **`lag` 合法范围是 0..3**：`--lag 4` 会 KeyError（`latent` 只定义 0..3）。
2. **kill 父进程会遗留 multiprocessing 孤儿 worker**（占内存空转，导致 MemoryError）。清理法：`Get-CimInstance Win32_Process` 按 `parent_pid=<dead>` 过滤后 `Stop-Process`；起新臂前先清孤儿。
3. **改行为流 = 破坏可比性**：`collect` 中 rng 抽取顺序一旦改变，之后所有臂的"paired 比较"都被混杂（历史上有过一次真实回归）。
4. **不要把小量种子结论当强结论**：3 种子 × 少量 episode 分辨率有限；`<2 SE` 的差异通常不作结论。
5. **反事实预测永不入训练**：DAgger 只回流"实际执行"的转移。

## 5.11 新增实验线的运行方式

这些探针大多以 `--seed`（训练种子）为主，先 `fit` 主线模型再评估；可加 `--report x.json` 落盘（详见 [07 新增实验线](./07-experiment-lines.md)）。

```bash
# 阶段/相位线：先导出评估流，再做各探针
python stage_support_probe.py --seed 100 --dump-streams streams.npz --report support.json
python stage_graph.py --streams streams.npz --sweep-k 2,3,4 --report graph.json
python stage_sweep_probe.py --seed 100 --penalties 0.02,0.05,0.1,0.25,0.5,1.0 --report sweep.json
python stage_content_probe.py --seed 100 --penalty 1.0
python phase_stage_probe.py --seed 100 --eval-seeds 120-124
python state_partition_probe.py --seed 100 --counts 4,8,16,32 --detrend
python event_lead_probe.py --seed 100 --events stage      # 或 --events feature

# 0/1 死亡规则线
python death_rule_probe.py --seed 100 --test-seeds 122-124 --refine-rounds 2 --report rules.json
python death_failure_probe.py --seed 100 --refine-seeds 120-121 --test-seeds 122-124
python death_feature_probe.py --seed 100 --eval-seeds 120-124
python death_pattern_probe.py --source-worlds 8 --test-worlds 6
python death_transition_quality.py --seed 100 --branches 10

# 上下文 / 动力学 / 记忆 / 生成
python dynamics_context_probe.py --seed 0 --train-steps 1200 --episodes 4
python generated_prior_probe.py --train-steps 1200 --warmup 10
python context_benchmark.py --seeds 100-102 --jobs 3 --report ctx.json

# 动作 / 时程 / 选择诊断
python action_conditioning_probe.py --streams streams.npz --k 2 --permutations 200
python causal_horizon_probe.py --seed 100 --max-lag 20
python doom_horizon_probe.py --seed 100 --max-lag 12
python oracle_selection_probe.py --seed 100 --eval-seeds 120-122
python coarse_search_probe.py --seed 100 --macro-frames 4 --macro-steps 3
python score_column_probe.py --seed 101 --eval-seeds 120-124
python risk_contrast_probe.py --seed 100 --eval-seeds 120-124
python metric_probe.py --seed 100 --steps 1500
python n_tie_probe.py --seed 100 --epochs 2
python planner_rollout_quality.py --seed 100 --eval-seeds 120-124 --cache cache.pkl

# 帧级学习轨迹 / 接线审计
python learning_trace.py --seed 100 --eval-seeds 120-124
python wiring_audit.py --seed 100 --steps 1500 --report wiring.json
```

> 提示：多个探针共用同一份 `fit` 结果，可用 `probe_cache.load_or_fit(...)`（对应 CLI 的 `--cache` / `--refit`）在各臂之间复用，避免重复拟合。所有探针都只读主线，不改默认训练/决策路径。

## 5.12 拓扑读出（readout propagation）

`--readout-propagation GAIN` 把原型读出从"几何 kNN"改成"沿学到的图边松弛后再读"：

```
几何 kNN 种子 q0 ──沿边松弛 g 步──► q ──► 读节点 value / 危险率
q ← (1-g)·q + g·(q @ A)      A = 行归一化的原型邻接（由图的 transition/association 边构成）
```

- 相关参数：`--readout-propagation-steps`（默认 1）、`--readout-propagation-edges {transition,association,both}`（默认 transition）。
- 实现落点：`NodeRuntime._prototype_transition`（CSR 转移算子）/ `_relax_step` / `_propagated_readout`，被 `valence_state_batch` 与 `_hazard_channels` 的原型通道消费。
- **默认 `GAIN=0` 关闭**，走历史几何读出（数值不变）；只有 `>0` 时新代码才生效。

```bash
# A/B：几何读出 vs 拓扑读出（其余条件完全一致）
python substrate_nodes.py --fast --seeds 3 --jobs 3 --eval-seeds 120-124
python substrate_nodes.py --fast --seeds 3 --jobs 3 --eval-seeds 120-124 \
    --readout-propagation 0.5 --readout-propagation-steps 2
```

> 效率（拓扑读出）：松弛**从不物化稠密 `batch × P` 块**。种子每行只有 `soft_neighbors`（默认 5）个非零，每步只把这些非零沿其出边做一次 ragged 展开，因此工作量跟随**种子支撑集**（O(batch·K·出度)）而不是 P（P 可达数千）。混合与去重（重复的 `(row, prototype)` 需**相加**而非覆盖）用单次 `np.unique` + `np.bincount` 完成，最终收缩是每个输出列一次 `np.bincount`；全程 C 级 NumPy，无 Python 逐行循环。实测单次读出 P=755、E=816、batch=1024 时 **18.6ms → 1.0ms（18.7x）**；eval 整体 **~24s → ~0.8s**，修正后拓扑读出相对几何读出的额外开销可忽略。

> 效率（几何/default 读出）：`_prototype_hazard_from_weights` 把历史「逐行 Python 遍历 + 每行重建记忆字典」的读出向量化——一次 ragged 展开加 `np.maximum.reduceat` 求 `(row, memory)` 的最大权重（与 `memories_for` 的语义一致），实测 **6.45ms → 0.125ms（51.7x）**；profile 中 `memories_for` 调用数从 **477k → 0**，`pain_signal_batch` 累计 **10.1s → 4.9s**，默认配置 eval **8.4s → 6.3s**。该改动对 `gain=0` 的读出数值保持一致（种子 100 逐位相同，其余 ≤1e-16 的重排差异），帧数不变。

> 派生结构缓存：转移算子（O(E) 边扫描）、逐原型关联率、逐原型读出列、原型→记忆 incidence 都只在**每个读出相位构建一次**（`NodeRuntime._readout_cache`）：训练由 `_interact` 失效，结构变化由 `_readout_signature` 守卫失效。incidence（`_prototype_memory_index`）是**单遍**构建：每个节点的 membership 权重只解析一次并复用给所有与之相连的原型，避免逐原型重复扫描；`_prototype_columns` 与 `_prototype_hazard_from_weights` 共用它，所以每行读出的记忆并集与最大强度都只付一次 cost。

> 当前剩余热点：`_soft_weights_batch` 的几何 kNN（`argpartition` 占其约 1/3），**几何与拓扑两条路径都必须付**，属于读出本身的固有开销。

## 5.13 阶段级决策与长视界搜索

决策路径由 `--decision` 选择：`search`（默认，逐帧 MPC）、`stage`
（块常数日程）、`cellq`（单步读格子）；另有 `--beam-width` 切换 search 的搜索器。

### 阶段级（`--decision stage`）

把决策从"逐帧计划"换成"**阶段结果**"：无关紧要的漂浮阶段不必关注每一步的具体
操作，只需知道这个阶段结束后能达到什么效果——一种剪枝/去重。

```
候选 = 块常数日程（|A|**blocks 个）：每个动作保持 block-frames 帧
       动作字母表取自 agent.action_space（任务无关，零任务常数）
评分 = 现有 hazard 累加 + 端点 hazard + 价值项（六列账本恒等式不变）
执行 = 最优日程的第一个动作
```

- 参数：`--decision stage --stage-blocks B --stage-block-frames K`（默认 4/4 = 16 帧）。
- 落点：`substrate_nodes.stage_schedules` / `stage_result`，经 `decision_action` 分派。
- **默认 `--decision search`**，关闭时逐位不变；291 测试覆盖。

### 长视界（`--beam-width`）

`--beam-width W > 0` 把 search 从穷举 `2**horizon` 换成 beam：**逐帧动作分辨率
不变**，候选数被 W 截断，唯一超参是宽度（任务无关）。

```bash
# 等价性校验：W ≥ 2**horizon 时必须与穷举逐位一致
python substrate_nodes.py --horizon 6  --beam-width 64 --seeds 24 --jobs 6 --eval-seeds 200-203
# 长视界（逐帧分辨率不变，候选有界）
python substrate_nodes.py --horizon 12 --beam-width 64 --seeds 24 --jobs 6 --eval-seeds 200-203
```

> 验收（详见 `TODO.md`「阶段级决策」）：`beam h6 w64` 与 `search h6` **逐位一致**；
> `h12`（`w1024`，≈穷举）仍 −9.33（CI 跨 0）；小宽度长视界 −22（帧级近视目标
> 贪心剪枝会剪掉"短期分低但能长期存活"的路径）。块常数日程 −2/−23/−60 单调变差。
> **结论：帧视界不是正确的轴**，真正起作用的是候选集的**时间分辨率**。

### 评估指标：进度里程碑

报告与打印除 `frames` 外增加 **progress milestones**（`env.passed`：翻越一根
柱子 +1）。世界只把它当**评测事实**，**不进观测、不进学习路径**。

```bash
python substrate_nodes.py --cap 600 --seeds 24 --jobs 6 --eval-seeds 200-203
# → "frames overall: ..." / "progress milestones overall: ..."
```

> 量程：`zero` 0.00、`scripted` 0.30、`recovered(hor=10)` 34.40。基线 `search h6`
> 为 mean 0.29 / median 0（24-of-96 episode ≥ 1，max 4）——当前评测**主要落在
> 第一根柱子之前的区间**，帧数差异主要反映近段能力。

### 时间尺度（间距）不变性

`--eval-spacing S` 把**评测世界**的柱间距改成 S（0 = 世界默认 0.80；训练世界不受影响）。
间距只改变两次通行之间的**时长**，通行几何不变——所以它是检验"决策是否依赖时间尺度"
的旋钮。

```bash
# 学到的策略：每种子训练一次，再在 3 个间距上重评同一个模型
python spacing_invariance_probe.py --seeds 24 --jobs 6 --cap 1200 \
    --horizon 6 --spacings 0.8,1.6,2.4 --eval-seeds 200-203 --report si.json

# 强参考控制器（无需训练）：量程大，是检验指标本身的不变性的灵敏仪器
python spacing_invariance_probe.py --policy reference --eval-seeds 120-139 \
    --cap 6000 --spacings 0.8,1.6,2.4
```

> 原始 `frames` 与原始 `passes` 都是**时间量**，必然随间距移动（同一帧预算在更大
> 间距下含更少通行）。所以探针额外报**无时间量纲**的
> `passes · spacing / (frames · speed)` = "已达成通行 / 实际提供的通行"。
> 验收（详见 `TODO.md`）：该统计量 reference 为 0.975/1.000/1.039（不变），
> learned h6 为 0.132/0.122/0.137（不变，但只有 reference 的 1/7.5）——即
> **指标时间不变性成立，而当前规划器的缺陷是普遍决策质量，不是时间尺度**。

### 里程碑分段与转移记忆（`milestone_graph.py`）

任务无关的**相位复位**检测：只看观测行，输出"一个周期结束、下一个开始"的
帧索引——一个与时长无关的转移单位。在此之上是**里程碑级转移记忆**。

```python
from milestone_graph import (segment_boundaries, milestone_statistics,
                             milestone_memory, next_milestone)
boundaries = segment_boundaries(observations)      # (frames, dim) -> indices
stats = milestone_statistics(observations, feedback=world_feedback)
memory = milestone_memory(observations, feedback=world_feedback)
# memory["keys"]        # 每个里程碑的身份（边界状态的分位桶）
# memory["durations"]   # 每段帧数（分布，不是 horizon）
# memory["transitions"] # key -> key 的计数 / 平均时长 / 平均反馈
next_milestone(memory, memory["keys"][0])          # 高层向下预测
```

分段规则：`3 × q90(分数)` 做阈值 + 每段保留极大值（NMS 8 帧）。分数是
**逐坐标归一化**的：每个坐标的 `|leaky step|` 除以**它自己**的 q90，再取跨坐标 max
（`boundary_scale` / `boundary_scores`）。**逐坐标归一化是必需的**：观测里有近似
线性斜坡的坐标（每帧 ~1），它的 leaky 步长是常数，跨坐标 max 因此被钉成常数
（实测每帧恰好 1.0000），阈值永远不可达、**复位检出 0 个**。**尺度取高分量、不取
中位数**（复位在尾部）；**NMS 必需**（capacitor 把一次复位抹成 1/β 帧的尾巴）。
多 episode 校准时统计量**只从流内帧池化**（`pooled_boundary_model`）——episode
接缝不是相位复位，拼接会把 q90 抬高（实测 3.0 → 6.68）。
身份规则：边界状态投影到它们的**主方向**再对**标量**分桶（`boundary_axis` /
`frame_project` / `frame_keys`），坐标先**逐坐标标准化**（`sigma`）。**不要**回到
"逐坐标各分一级再取组合"：那是各维级别的**乘积**，在有相位的流上给出 23–25 个身份
/ 33 个里程碑（几乎一个里程碑一个标签——指纹而非概念），跨尺度 key 集 Jaccard
仅 0.43–0.56。投影方案给出 2 个身份 / 33 个里程碑。
**但两点未收敛**：(a) 未标准化时主方向 **100% 由那个单调坐标承担**（与时间相关
+0.46）——"主方向即时钟"；标准化后仍 +0.25，未清干净；(b) 验收指标不能用"key 集
Jaccard"——2 级方案的标签集恒为 `{0,1}`，该指标**平凡**；要比较的是**同一个
里程碑**在两个时标下是否同名。
验收（检测器）：对已知柱子回收帧，reference 流上 precision = **1.00**、
recall = 0.86–1.00；时长分布中位在 2× 时标下**精确 ×2**。
内在约束 `β·A > k·d` 见模块 docstring。**已接入 planner**（下两节的 `goal` /
`rate` 两种形态），默认关。

> **根因提醒（2026-11-22）**：`fit` 的训练流**几乎没有完整相位**。死亡点解剖：
> **第一个柱子在第 78 帧才到达鸟**，一次"通过"要到 **~120 帧**；而行为策略
> （`collect`）平均 **46.5 帧**、planner **~72 帧**，**都死于下坠、从未接触到柱子**。
> 证据：把 `spacing` 从 0.80 扫到 0.10、把 `speed` 减半，episode 长度**完全不变**
> （46.5 / 52.6 帧，0 回收）——死因与柱子无关。行为策略的拍翅率 ≈8–11% 而悬停阈值
> = `GRAVITY/FLAP` = **8%**，刚好卡在线下。各策略完成一次通道的比率：reference
> **22/4 世界**、planner **2/64 episode**、行为策略 **0**。
>
> **最小可行路径**：`collect` 现在把**世界参数**当**任务输入**透传
> （`spacing` / `gravity` / `flap` / `speed`，默认全部不变、不含逻辑）。实测
> `gravity=0, steps=6000` 时**学习者自己的流**给出 **80 个里程碑 / 67 episode**、
> 身份 2 个、转移表非空——这一层首次在自己的流上有内容可学。但无重力世界是**退化**
> 的，且会改变整个 agent 的训练分布，属**任务/curriculum 决策**。
>
> **统一的敌人是"单调/无界坐标"**：`_alive`（累计飞行距离，obs 的第 7 维）每帧 ~+1，
> 它把检测器的跨坐标 max 钉成常数、把池化阈值抬高、把身份的主方向占成时钟。
> 凡是用"数值水平"的统计量，都要先对单调成分免疫。

### 里程碑塑形（`--milestone-mode goal`，选项 1，默认关）

`fit` 从 agent 自己的训练流取里程碑，把每个里程碑的**起始观测**作为候选目标，
按频次选模态边界状态；`--milestone-weight w > 0` 时 planner 的终端目标加
`− w · min_g ‖(endpoint − goal)/scales‖`（记进现有第 4 列 shaping 槽，
六列账本恒等式保持）。**默认 0 时逐位不变。**

```bash
python substrate_nodes.py --horizon 6 --milestone-mode goal --milestone-weight 0.02 \
    --seeds 24 --jobs 6 --eval-seeds 200-203 --report ms.json
```

> 验收（详见 `TODO.md`）：24 种子配对呈**单调退化，无正区间**——w=0.005 为 null
> （−1.24，CI 跨 0）、w=0.02 为 −10.05、w=0.05 为 −43.35。诊断：目标是**绝对
> 状态向量**，该项因此是**静态位置吸引子**，与任务竞争（把 y 拖向 ≈0.12 与
> 平稳飞行冲突）。里程碑**身份**刻意用分位桶避开绝对值，**目标空间没有**——
> 修复方向是让高层传**方向/约束**而非目标点。

### 里程碑动作承诺（`--milestone-mode rate`，选项 2，默认关）

`fit` 另外按**里程碑身份**学一个**时长加权动作率** `agent.milestone_rate[key]`；
决策时 `evaluate` 用 `MilestoneTracker` 逐帧给出当前身份，取该身份的率，
planner 的终端项加 `w · mean_t |a_t − rate|`（第 4 列 shaping 槽，账本恒等式保持）。
默认 `off` 时逐位不变。

```bash
python substrate_nodes.py --horizon 6 --milestone-mode rate --milestone-weight 0.16 \
    --seeds 24 --jobs 6 --eval-seeds 200-203 --report ms2.json
```

> 验收（详见 `TODO.md`）：**不予采纳**，且**根因是上游的**——`fit` 目前从一条
> **没有完整相位**的流里学里程碑（复位检出 0，转移表为空），所以这些数字测的是
> **全局拍翅率平移**而非里程碑承诺。数据本身：200–203 上 +7.9…+10.0（14/24 种子为正，
> 4/4 世界为正）、300–303 上 −5.4…−20.8（4/4 世界为负）——效应很大但**符号由世界
> 决定**。另两条与根因独立的断点：`key_rate` 只编码**频率**（planner 与 reference
> 的动作率都 ≈0.08，表现却差数十倍——优势在**时机**）；逐帧标量代价**没有地方放**
> 通道尺度的转移图。

### 里程碑记忆覆盖（`milestone membership coverage`）

报告与打印给出**里程碑记忆对 planner 流的覆盖**。身份必须是**一套共享字母表**：
切点与边界尺度要在**整条流池化**的一帧上校准（每 episode 各自校准会让 key 不可比）。
但注意：**覆盖率高不等于这一层有效**——若训练流里没有相位（每个 episode 只有 1 个
"里程碑"），覆盖率会**平凡地**接近 1。判断这一层是否真的学到了东西，要看
**检出边界数与 `key→key` 转移表是否非空**。
