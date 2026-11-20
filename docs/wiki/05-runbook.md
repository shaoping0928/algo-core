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
