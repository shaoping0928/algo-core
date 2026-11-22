# 04 · 依赖关系

## 4.1 外部依赖

| 依赖 | 版本 | 用途 |
|---|---|---|
| Python | 3.x | 唯一语言 |
| NumPy | `>=1.24,<3`（`requirements.txt`） | 全部数值计算 |

其余全部来自标准库：`argparse`、`json`、`os`、`sys`、`math`、`random`、`pickle`、`time`、`collections(deque)`、`itertools`、`dataclasses`、`typing`、`contextlib`、`concurrent.futures`、`unittest`。

`raw_math_experiment.py` 在 `eval` 子命令里**动态导入**外部 nanoGPT 的 `train.py`/`eval_local`，因此它有一个可选的外部运行依赖，但 `prepare` 子命令不需要。

仓库**无数据目录、无二进制资产、无配置文件**（除 `requirements.txt`）；根目录只有 `README.md`、`TODO.md`、`requirements.txt`、`.gitignore`，`docs/` 下是 `substrate.md`（+ 本 wiki）。

## 4.2 内部依赖图

```
                        ┌──────────────┐
                        │   flappy.py  │  (无延迟世界 + 物理常量)
                        └──────┬───────┘
                               │
                        ┌──────▼────────────┐
                        │ latent_world.py   │ 延迟世界 + 测量工具
                        └──┬────────────┬───┘
             ┌─────────────┘            └──────────────┐
             │                                          │
  ┌──────────▼──────────┐               ┌───────────────▼──────────────┐
  │ substrate_search.py │               │ predict_layer.py (LayerPair) │
  └──────────┬──────────┘               └───────────────┬──────────────┘
             │                                          │
   ┌─────────▼─────────┐                                │
   │ 规则层 / 基础验证  │                                │
   │ substrate_rules.py │                                │
   │ rule_*.py          │                                │
   └───────────────────┘                                │
                                                        │
  ┌────────────────────────── 运行时基底 ──────────────────────────────┐
  │                                                                    │
  │  node_graph.py ──┬── node_components.py (ComponentBank)            │
  │                  ├── node_features.py (FeatureRuleBank)            │
  │                  ├── graph_memory.py (PrototypeGraphMemory)        │
  │                  └── node_runtime.py (NodeRuntime)                 │
  │                                                                    │
  │  node_runtime 依赖：node_graph + node_features + node_components    │
  │  signal_bank.py 独立（仅 numpy）                                    │
  │  mixed_nodes.py 独立（仅 numpy）                                    │
  └────────────────────────────┬───────────────────────────────────────┘
                               │
  ┌────────────────────────────▼───────────────────────────────┐
  │                        搜索与价值层                          │
  │  trajectory_search.py (无本地依赖)                           │
  │  substrate_value.py   (无本地依赖)                           │
  │  substrate_generic.py → latent_world, predict_layer,         │
  │                         substrate_value, trajectory_search   │
  │  substrate_sandbox_search.py → latent_world, predict_layer,  │
  │                         substrate_value, trajectory_search   │
  │  substrate_graph_search.py → graph_memory, latent_world,     │
  │                         substrate_generic, substrate_value,  │
  │                         trajectory_search                    │
  │  substrate_nodes.py → eval_protocol, latent_world,           │
  │                         node_features, node_runtime,         │
  │                         substrate_generic, trajectory_search │
  └──────────────────────────────────────────────────────────────┘

  eval_protocol.py 独立（仅 numpy），被 substrate_nodes 与三个 probe 使用
```

## 4.3 分层约定

自下而上四层，**上层可依赖下层，下层不得反向依赖上层**：

| 层 | 模块 | 规则 |
|---|---|---|
| L0 数值地基 | `trajectory_search.py`、`substrate_value.py`、`signal_bank.py`、`mixed_nodes.py`、`eval_protocol.py`、`node_features.py`、`node_graph.py` | 只依赖 numpy / 标准库；不得导入任何 CLI 或世界模块 |
| L1 世界与运行时 | `flappy.py` → `latent_world.py` → `substrate_search.py`；`node_components.py`、`node_runtime.py`、`graph_memory.py`、`predict_layer.py` | `node_runtime` 只依赖 L0 的 `node_graph/node_features/node_components`；不导入 `flappy/latent_world` |
| L2 组合与适配 | `substrate_generic.py`、`substrate_sandbox_search.py`、`substrate_graph_search.py`、`substrate_nodes.py` | 可依赖 L0/L1；`substrate_sandbox_search` 独占 Flappy 任务几何 |
| L3 实验与诊断 | `substrate_*.py` 验证脚本、`trace_*.py`、`*_probe.py`、`rule_*.py`、`recall_world.py`、`math_memory_search.py`、`raw_math_experiment.py` | 最上层，可依赖一切 |

## 4.4 耦合边界与纪律

1. **任务无关边界（硬约束）**：`node_runtime.py`、`node_graph.py`、`node_features.py`、`substrate_nodes.py` **不得**出现 Flappy 术语（`BIRD_X`、`GAP`、`pipes`、`flap` 等）。这条边界由 `test_node_runtime.py::test_core_does_not_embed_task_geometry` 守卫——它做源码级检查。
2. **Flappy 几何只允许出现在**：`flappy.py`（世界本体）、`latent_world.py`（世界扩展）、`substrate_sandbox_search.py`（适配器/基线）、以及各 `trace_*`/验证脚本的诊断代码。
3. **沙盒预测不回流**：`evaluate(record=True)` 只记录规划器**实际执行**的真实转移；任何"反事实预测状态"不得进入 `collect` 数据或世界模型/价值头训练。
4. **凭据/外部导入**：只有 `raw_math_experiment.py eval` 需要外部 nanoGPT 路径（参数显式传入），其余模块自包含。
5. **导入顺序敏感点（已修复）**：`flappy.py` 的默认物理已是 latent 基准值，`FL_*` 环境变量仍可覆盖；`import flappy` 不会再解析 CLI 参数或产生输出（有子进程测试守卫）。历史上 `latent_world` 通过 `os.environ.setdefault` 在导入前设物理值，import 顺序会改变物理——该地雷已消除。

## 4.5 关键 fan-in / fan-out

| 模块 | 被谁依赖（fan-in 高 = 影响面大） |
|---|---|
| `trajectory_search.py` | `substrate_generic`、`substrate_sandbox_search`、`substrate_graph_search`、`substrate_nodes` |
| `substrate_value.py` | `substrate_generic`、`substrate_sandbox_search`、`substrate_graph_search`、`trace_*` |
| `latent_world.py` | 几乎所有 L2/L3（含 `SEEDS`、`latent`、`CAP`） |
| `node_graph.py` | `node_runtime`、`node_components`、`graph_memory`、`trace_node_*`、probe |
| `node_runtime.py` | `substrate_nodes`、`trace_node_*`、`*_probe` |
| `node_features.py` | `node_runtime`、`substrate_nodes`、`feature_generalization_probe` |

| 模块 | 依赖谁（fan-out 高 = 集成点） |
|---|---|
| `substrate_nodes.py` | 6 个内部模块（`eval_protocol`/`latent_world`/`node_features`/`node_runtime`/`substrate_generic`/`trajectory_search`） |
| `substrate_graph_search.py` | 5 个内部模块 |
| `substrate_generic.py` | 4 个内部模块 |

## 4.6 并行与运行时依赖

- `substrate_nodes.main()` 按 `sys._is_gil_enabled()` 检测 free-threaded 构建：GIL 构建用 `ProcessPoolExecutor`，free-threaded 用 `ThreadPoolExecutor`（常量 `FREE_THREADED`）。因此多进程分支会重新导入整个依赖树——**模块级不应有副作用**（这正是"import 不解析 CLI"守卫存在的原因）。
- 当前 `.venv` 是 GIL 构建；free-threaded `python3.15t` 存在但 numpy 的 `_multiarray_umath` DLL 被应用控制策略阻止加载（见 `README.md`）。

## 4.7 新增实验线的依赖（见 [07 新增实验线](./07-experiment-lines.md)）

| 层 | 模块 | 依赖 |
|---|---|---|
| L0 纯数值/数据层（仅 numpy / 标准库） | `rule_expressions`、`stage_graph`、`action_effect`、`context_model`、`branches`、`variable_registry`、`memory_tree`、`generative_expansion`、`probe_cache` | 不 import 项目其他模块（`probe_cache` 默认在函数内延迟 import `substrate_nodes.fit`） |
| L1 组合库 | `death_rules`（→ `node_features.Primitive`、`rule_expressions`）、`death_rule_loop`（→ `death_rules`）、`contextual_rules`（→ `death_rules`）、`pattern_library`（→ `death_rules`）、`hierarchical_memory`（→ `memory_tree`、`variable_registry`）、`planner_rollout_quality`（→ `trajectory_search`、`substrate_nodes`、`death_transition_quality`、`probe_cache`） | 见括号内 |
| L3 只读探针（最上层） | `stage_*_probe`、`phase_stage_probe`、`state_partition_probe`、`death_*_probe`、`death_transition_quality`、`action_conditioning_*`、`causal_horizon_probe`、`doom_horizon_probe`、`oracle_selection_probe`、`coarse_search_probe`、`outcome_coarse_probe`、`score_column_probe`、`risk_contrast_probe`、`metric_probe`、`n_tie_probe`、`event_lead_probe`、`warning_lead_probe`、`dynamics_context_probe`、`generated_prior_probe`、`context_benchmark`、`learning_trace`、`wiring_audit` | 普遍 import `substrate_nodes`（`fit/evaluate/policy_result/_curiosity_off/_encoded_pending/collect/_train`）、`latent_world`，常配合 `probe_cache`、`eval_protocol` |

纪律不变：探针只读主线，**不改变默认训练与决策路径**；探针之间的复用只经由 `probe_cache` / `eval_protocol` / `planner_rollout_quality` 这类无副作用工具。
