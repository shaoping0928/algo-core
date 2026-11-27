# TODO

> **维护约定（2026-11-26 整理）**：本文件只保留**近期任务状态、未完成事项与当前主线实验记录**。2026-11-23 及更早的 Batch 记录、历史设计稿与只读审计附录已迁移至 `docs/archive/TODO-*.md`（见文末「历史归档索引」）；已关闭的 README 实验归档仍在 `docs/archive/README-*.md`。

## P0：长程阶段总结与阶段转移学习重构（2026-11-26 诊断与路线）

**背景与目标**：帧级学习不再是当前瓶颈，且帧级不是最终目标。面向长程任务预测，核心缺口在于**阶段性任务总结（Stage Summarization）**与**阶段级状态转移学习（Stage Transition Learning）**。

**五维只读诊断结论（2026-11-26，seed 100；Passive 6500 帧 / Passive+DAgger 6500 帧 / 20 个长程 Planner 评测局均长 198.8 帧、含 31 次过柱）**：

1. **阶段切分 56%–70% 是 float32 舍入噪声与常数计数器伪边沿**：`GenericKeySignalDetector` 对 `|Δobs|` 取 98% 分位并按 `>=` 触发，导致等速漂移维 `dx/next_dx`（每帧 `-0.01`）阈值落在 `0.01000005 / 0.01000011`，99% 触发都是 1-ULP 舍入抖动（Passive `627/628` 与 `548/549`；DAgger `394/401` 与 `449/456`；长程 Planner `80/111` 与 `131/162`）；常数存活计数维 `obs[:, 6]`（每帧 `+1.0`）每帧全触发，上升沿去重变成固定第 1 帧伪阶段。合计 Passive/DAgger 中 **85%–91%** 的阶段是确定性浮点时钟碎片。
2. **阶段总结丢弃长程状态信息**：`stage.key` 仅记录进入瞬时的触发 token，不含阶段内持续状态/趋势。实测 DAgger 流上预测剩余存活帧数：`stage.key` $R^2$ **0.124**（劣于纯时钟 **0.213**）；预测存活 ≥80 帧（过柱尺度）时 `stage.key` AUC **0.516**（随机），而阶段入口连续状态本身已达 **0.746 AUC / 0.376 $R^2$**。长程流中 `StageOutcomeReadout` $R^2 = \mathbf{-0.482}$。
3. **单步 91% 准确率存在未来动作泄漏与开局时钟虚高，多步长程坍塌**：`accuracy_coverage_probe` 查询传入了测试阶段完整的未来 `action_trace`；剥离未来动作并排除开局确定性 4 阶段（`idx 0–3`）后，长程 Planner 流 `idx 4+` 因果 `pooled` NLL 增益为 **−0.373**（top-1 `56.1%`），跨流（DAgger→Planner）为 **−0.646**；多步递推 `k=1..8` 阶段（13–98 帧）终止 `brier_skill` 在 `k=1,2,3,5` 全为负，中后段终止 AUC 掉至 **0.595**。
4. **跨分布状态→阶段解析大面积 OOD（82%–87% 拒答）**：Passive/DAgger 拟合的 `StageStateResolver` 在长程 Planner 评测上仅支持 **17.7% / 13.2%**，`outside_radius` 高达 **54.9% / 74.3%**——主因之一是无界递增计数器维 `obs[:, 6]` 在局超过训练期长度（~73 帧）后把所有长程状态拉出原型半径。
5. **精确 RLE `action_trace` 切碎阶段动作可控性**：Passive/DAgger 中 **55%**、长程 Planner 中 **76.7%** 的训练 option 是孤例（singleton），长程验证流 **30.6%** 遇未见 option，19 个 stage key 仅 3 个有 ≥2 个受支持 option。

**三阶段推进路线（严格自监督、任务无关、默认栈不变）**：

- [x] **Phase 1 — 修复阶段切分与信号检测器（`GenericKeySignalDetector`，2026-11-26 完成）**：
  - 在 `stage_outcome_memory.py` 中改为相对稳态步长基准（仅当 ≥50% 转移聚集在稳态步长附近时扣减）+ 尺度自适应 `noise_floor` + 常规离散动作回声平台守卫，保留稀有跳变检测。
  - 新增 3 条回归单测（`tests/test_stage_outcome_memory.py`：float32 匀速漂移 ULP 抖动、单调常数计数器、常规高频平台），179/179 阶段测试通过。
  - 三流验证：float32 ULP 误报清零、常数计数维伪阶段清零，过柱跳变检测达 **1/1（Passive）、7/7（DAgger）、31/31（长程 Planner）零误报**；阶段时长恢复至 30–66 帧真实尺度，也直接暴露出纯事件切分过粗与长阶段精确 RLE 96% singleton 的结构问题（进入 Phase 2/3）。
- [x] **Phase 2 — 升级阶段总结（`stage_regime.py`，2026-11-26 完成）**：
  - 新增严格任务无关的 `detect_pure_drift_dims`（纯自监督识别并剥离零跳变单调计数器维，保留周期复位维）与 `StageRegimeSummarizer`（结合硬事件边界 + 非漂移观测上的确定性体制聚类 + `min_dwell/hysteresis` 平滑，支持在线 `resolve` 与显式 OOD 半径门）。
  - 跨流验证（拟合于 Passive/DAgger → 评测 20 个均值 198.8 帧/最长 472 帧的长程 Planner 世界）：`in-radius` 覆盖率由 **17.7% / 13.2%** 跃升至 **89.4% / 78.1%**。
  - D4 长程信息量（DAgger 6500 帧）：剩余存活 $R^2$ 由 `0.124` 提升至 **0.403**（超过纯时钟 `0.329` 与线性观测 `0.331`，逼近二次观测 `0.424`），`surv>=20/40` AUC 达 **0.905 / 0.837**。
- [x] **Phase 3 — 可控阶段级行为抽象与严格因果多步长程转移门（`stage_regime.py` + `stage_regime_probe.py`，2026-11-26 完成）**：
  - 用低维可控的 `(duty_cycle_tier, early_impulse)` 阶段行为签名（≤5 类、**0 singletons**，替代 55%–96% singleton 的精确 RLE）构建 `CalibratedRegimeTransitionModel`：含训练折内 4 折 episode-OOF 收缩、离散停时 PMF 向量化卷积，以及严格概率守恒的因果 `k`-stage `rollout()`（支持 pooled 与多步宏动作计划两种模式）。
  - 反证核查（完全剔除终局截断，仅看纯存活的下一体制预测）：宏动作 profile 在三条流上全部产生正向增益（Passive `+0.213`、DAgger `+0.121`、长程 Planner `+0.252` nats，长程 top-1 `40.6% → 60.4%`）；DAgger → Planner 跨流 `idx 4+` 因果 NLL 增益由 **−0.646 → +0.912**。
  - **3 种子独立验证（`stage_regime_probe.py --seeds 100 101 102 --steps 6500`）—— 3/3 过门**：
    - `dur_p50` 6.0–8.0（`p90` 15.0–17.0），5 个行为签名（`singletons = 0`），可控体制 **15/16、14/16、16/16**（mean max TV `0.710 / 0.803 / 0.817`）；
    - 1 步因果 NLL 增益 `pool ALL` `+1.089 / +1.054 / +0.929`，`pool idx4+` `+0.824 / +0.715 / +0.670`；纯非终局动作增益 `ΔNLL = +0.213 / +0.296 / +0.160`（top-1 `0.483→0.560`、`0.368→0.549`、`0.457→0.551`）；
    - 多步因果递推 `k=1/2/3`（约 9–21 帧）：终止 AUC `0.904–0.956 / 0.873–0.915 / 0.747–0.819`，`brier_skill` 全部显著为正（`k=2`: `+0.540 / +0.504 / +0.429`）；
    - D4 剩余存活 $R^2$：`regime` `0.313 / 0.318 / 0.254` 全面胜过 `clock`（`-0.000 / 0.156 / -0.080`）与线性观测（`0.171 / 0.235 / 0.138`）；`surv>=20` AUC `0.882 / 0.863 / 0.870`（vs 时钟 `0.737 / 0.761 / 0.694`）。
- [x] **Phase 4 — 长程 DAgger 验证与默认关闭的续策值配对 A/B（`stage_regime.py` + `stage_regime_ab_probe.py`，2026-11-26 完成首轮）**：
  - **Phase 4A（6 种子 DAgger 长程门，`b0-100..105.pkl`）**：
    - **6/6 种子** `k=1/2/3`（约 11 / 20 / 28 帧）因果多步终止 AUC 全正：`k=1` `0.809–0.949`、`k=2` **`0.846–0.940`**（Brier skill **`+0.325–+0.553`**）、`k=3` `0.715–0.876`（Brier skill `+0.101–+0.413`）；
    - **6/6 种子** `idx 4+` 因果 1 步 NLL 增益全正（`pool4+`: `+0.592–+1.772`；`act4+`: `+0.533–+1.425`），可控体制 `13–15/16`（mean max TV `0.706–0.912`），`surv>=20/40` AUC 达 **`0.780–0.905` / `0.734–0.858`**；
    - 负对照记录：将 `(ctx, act)` 单向收缩至同上下文 `(ctx, None)` 会丢失全局后继支撑并在未见邻接体制上产生零概率 NLL 尾部爆炸（使 6 种子 `nonterm dNLL` 由 4/6 降到 1/6），已及时回退至包含全局支撑先验的收缩口径。
  - **Phase 4A 决策端点审计（3 种子 `100..102` × `eval 120–124`）**：
    - `CalibratedRegimeTransitionModel.continuation_values(k=4)` + `RegimeContinuationEvaluator`（显式 OOD 门返回基线中性风险，经 `as_stage_value_callback()` 接入既有默认关闭的 `stage_value` 接口，权重 0 时逐位不变）。
    - `H=6` 候选端点支持率 **85.4%–96.2%**、单决策不同体制数 **2.64–2.97**（max 5–6，彻底打破旧 `stage_type_value` 的 1.29 塌缩）；`H=8` 支持率 **57.4%–84.5%**、不同体制 **2.85–4.40**。
    - `pooled` 续策风险无乐观偏差，在真实 held-out 轨迹上对 **15 / 30 / 60 帧内死亡**的预警 AUC 达 **`0.765–0.889` / `0.705–0.906` / `0.577–0.838`**。
  - **Phase 4B 配对 A/B（`stage_regime_ab_probe.py`，`mode=pooled, k=4, cap=1200`，`eval 120–129` 每种子 10 个独立世界）**：
    - **`H=6` 6 种子确认（`100..105` × 10 世界 = 60 个配对世界）——通过 6 种子确认门（5/6 种子强正向）**：
      - control (`w=0.00`)：均值 **115.7**、中位 **75.0**、过柱 **33**；
      - `w=0.25`：均值 **127.1**（`+11.5`）、中位 **115.5**、过柱 **43**（`+10`）、4/6 正；
      - **`w=0.50`**：均值 **144.8**（**`+29.2` / `+25.2%`**）、中位 **133.5**（**`+58.5`**）、过柱 **55**（**`+22` / `+66.7%`**）、**5/6 种子正向**（`[+14.9, +14.2, +37.4, −1.7, +74.8, +35.4]`）、60 局胜/平/负 **`34 / 13 / 13`**——仅用 6 步帧搜索（每决策候选数减少 4×）即追平 8 步基线（143.4）；
      - `w=1.00`：均值 **116.3**（`+0.6`）、过柱 **35**（倒 U 型响应，过大权重压过帧级避障）。
    - **`H=8` 3 种子筛查（`100..102` × 10 世界 = 30 个配对世界）**：
      - control (`w=0.00`)：均值 **143.4**、中位 **152.0**、过柱 **26**；
      - `w=0.05`：均值 **157.8**（`+14.4`）、过柱 **32**（`+6`）、2/3 正（`[+1.5, −3.0, +44.7]`，W/T/L `9/14/7`）；
      - `w=0.10 / 0.25`：seed 101（`+29.0 / +19.2`）与 seed 102（`+38.5 / +38.7`）继续大涨，但 seed 100 回撤（`−65.6 / −84.9`），根因与其 `H=8` 端点支持率仅 `57.4%`（高存活 OOD 端点与受支持端点间产生对比偏置）一致。
- [x] **Phase 5 — 决策级 OOD 覆盖率门控 + 6 种子双视界确认 + CLI 接线（2026-11-26 完成）**：
  - **机制改造**：`RegimeContinuationEvaluator` 新增 `min_decision_coverage` 与 `ood_policy="neutral_mean"`（当单次决策候选端点支持率低于 `0.80` 时整步弃权返回 0，高于门槛时少数 OOD 端点填本决策受支持均值，彻底消除高存活区 OOD vs supported 对比偏置）。
  - **`H=6` 6 种子确认（60 个配对 held-out 世界 `120–129`，`min_cov=0.80, ood=neutral_mean`）——再创新高（5/6 强正向）**：
    - `w=0.25`：`115.7 → 132.6`（`+16.9`），过柱 `33 → 45`（`+12`），W/T/L `32/17/11`，4/6 正，`active=0.87`；
    - **`w=0.50`**：均值 **`115.7 → 149.6`（`+33.9` / `+29.3%`）**、中位 **`75.0 → 138.0`（`+63.0`）**、过柱 **`33 → 56`（`+23` / `+69.7%`）**、胜/平/负 **`38 / 15 / 7`**（胜负比 **5.4 : 1**）、**5/6 种子正向**（`[+32.0, +11.5, +49.1, −3.9, +109.5, +5.3]`），`active=0.88`。
  - **`H=8` 6 种子确认（60 个配对 held-out 世界 `120–129`，`min_cov=0.80, ood=neutral_mean`）——**6/6 全正向通过确认门****：
    - control (`w=0.00`)：均值 **175.2**、中位 **152.0**、过柱 **76**（逐种子 `[210.3, 107.0, 112.8, 176.0, 243.2, 202.1]`）；
    - **`w=0.10`**：均值 **`175.2 → 187.2`（`+12.0` / `+6.8%`）**、过柱 **`76 → 85`（`+9` / `+11.8%`）**、胜/平/负 **`19 / 27 / 14`**、**`6 / 6` 种子全部正向（零回撤种子）**：`seed_deltas = [+4.2, +5.0, +38.5, +6.1, +17.8, +0.2]`，`active=0.67`（33% 高 OOD 决策干净弃权；seed 100 由未加门时的 `−65.6` 修复至 `+4.2`）。
  - **CLI 接线（默认关闭）**：`substrate_nodes.py` 新增 `--regime-value-weight`（默认 `0.0`）、`--regime-n-regimes`、`--regime-k-stages`、`--regime-min-coverage`（默认 `0.80`）、`--regime-ood-policy`（默认 `neutral_mean`）；单测锁定 `weight=0.0` 时逐位不变。
- [x] **Phase 6 — 行为无损效率优化 + 短视界（`H=6 / H=4`）平滑核累积 LCB 长程续策（2026-11-26 完成）**：
  - **代码效率优化（100% 逐位一致，764+ 回归测试全过）**：
    - `node_runtime.py`：向量化 `_compiled_feature_plan()` / `_feature_mask_table()`、预缓存 `_prototype_geometry()`（`-2*P^T` BLAS 矩阵 + in-place 范数加）、预聚合 `_prototype_memory_attribute_vectors()` 消除 `_memory_values_batch()` 三重 Python 循环，并为 `valence_state_batch` / `pain_signal_batch` / `_hazard_channels` 增加可选 `soft_weights` 复用。
    - `substrate_nodes.py`：`evaluate_terminal` 共享单次 `_soft_weights_batch`；跳过 `q_table is None` 时的无用 `previous_encoded` 计算；在 `exhaustive_search` 的 `predict_batch` 与 `evaluate_step` 中按前缀步长去重（`H=6` 总 `_soft_weights_batch` 行数 `320 → 106`，`H=8` `1536 → 426`），**单线程 5 世界 `H=6` 从 `2.908s → 1.008s`（2.88× 提速，逐位一致）**。
    - `stage_regime.py` + `stage_regime_ab_probe.py` + `stage_regime_probe.py`：距离计算改用预缓存 BLAS 矩阵，并接入 Python 3.14t Free-Threaded (`gil_enabled=False`) `ThreadPoolExecutor` 多线程并行（`--jobs`）。**6 种子 × 10 世界 `H=6` 完整配对 A/B（120 局）总耗时从 ~100s+ 降至 `3.65s`（>20× 墙钟提速），`--jobs 1` 与 `--jobs 16` 逐位一致**。
  - **短搜索深度长程续策增强（`stage_regime.py` + `substrate_nodes.py` opt-in CLI，默认仍关闭）**：
    - `StageRegimeSummarizer` 训练期自校准局部体制间距 `kernel_scale_`（到第 2 近中心距离平方中位数）；
    - `CalibratedRegimeTransitionModel.continuation_values` 新增多阶段折现累积存活 `cum_survival` 与保守下置信界模式 `mode="lcb"`（对受支持宏动作 profile 施加 `lcb_z / sqrt(n_ep)` 惩罚并锚定 pooled，既消除小样本乐观偏差又释放真实可控宏动作收益）；
    - `RegimeContinuationEvaluator` 新增 `soft_top_k`（高斯核平滑插值 top-4 受支持体制，消除短视界 Voronoi 胞内零梯度塌缩，保留 OOD 覆盖率门与 `neutral_mean`）。
  - **180 个独立 held-out 世界配对确认（6 种子 `100..105` × 10 开发世界 `120–129` + 20 全新未见世界 `200–219`，`cap=1200`，`mode=lcb(z=1.0), cum_survival, k=8, soft_top_k=4, tau=0.7, min_cov=0.80, ood=neutral_mean`）**：
    - **开发 60 世界（`eval 120–129`）**：
      - **`H=4`（仅 16 候选/步，`1.60s`）**：`73.7 → 126.0`（**`+52.3` / `+71.0%`**，中位 `72.0 → 106.0`，过柱 `12 → 39` `+27`，`38W/19T/3L`，**`6/6` 种子全正**）——仅用 `H=4` 即超越 `H=6` 基线（`115.7`）；
      - **`H=6`（64 候选/步，`3.65s`）**：`115.7 → 156.7`（**`+41.0` / `+35.4%`**，中位 `75.0 → 146.0`，过柱 `33 → 61` `+28`，`33W/15T/12L`，**`6/6` 种子全正**）。
    - **全新未见 120 世界（`eval 200–219`，零调参样本外验证）**：
      - **`H=4`（16 候选/步，`2.61s`）**：`75.6 → 120.8`（**`+45.2` / `+59.8%`**，过柱 `26 → 71` `+45`，`71W/34T/15L`，`5/6` 正）；
      - **`H=6`（64 候选/步，`7.56s`）**：`124.7 → 165.8`（**`+41.1` / `+33.0%`**，中位 `79.5 → 147.5`，过柱 `77 → 136` `+59`，`61W/23T/36L`，`4/6` 正）——**在 120 个未见世界上以 1/4 搜索深度、~3.5× 更快速度直接超越 `H=8` 基线（`154.8` 帧、`120` 过柱、`17.35s`）**；
      - **`H=8`（256 候选/步，`w=0.50`）**：`154.8 → 208.9`（**`+54.2` / `+35.0%`**，中位 `130.0 → 152.0`，过柱 `120 → 202` `+82`，`51W/42T/27L`，`4/6` 正）。
    - **合并 180 世界总览**：`H=4 + regime`（`122.5` 帧 / `110` 过柱）追平 `H=6` 基线（`121.7` / `110`）；`H=6 + regime`（`162.8` 帧 / `197` 过柱）超越 `H=8` 基线（`161.6` / `196`）。
- [x] **Phase 6B — 特征提取诊断 + 闭式自监督生存方差树体制分区（`partition="survival_tree"`，2026-11-27 完成）**：
  - **决策与特征提取诊断结论**：
    - 底层规则提取器对 `t-1` 碰撞面极准（`die<=1` AUC `0.987`），但 4 帧外迅速失效（`die<=10/20`: `0.567 / 0.527`），且 `epochs=1` 下 `feature_value` 与 `event_memory` 空转（AUC `0.500`）、`f6 (alive)` 泄漏进 5 条风险规则与 3 条价值规则；
    - `kmeans K=16` 阶段表征在等权标准化下将 33.3% 欧氏权重浪费在 100% 共线维 `(dx, next_dx)`（$R^2=0.797$），挤扁速度维 `vy`（$R^2=0.363$）与对准维 `dy`（$R^2=0.578$），导致对 `pipe_lower` 预警平直（`0.56` vs `pipe_upper/ceiling 0.74–0.80`），贡献了 75% 的回退局；
    - 负对照记录：14 组纯线性变换（去共线、反比加权、PCA 白化、单维 $\eta^2$）在 DEV 60 世界上均未超过 `kmeans K=16`，因白化放大噪声特征向量且单维加权无法切出多维交互危险区。
  - **核心改进（`stage_regime.py` + `stage_regime_ab_probe.py` + `substrate_nodes.py` opt-in `--regime-partition survival_tree`）**：
    - `_fit_survival_tree_centroids()`：无超参、纯闭式贪心二叉分区，仅在训练终局轨迹上按 `log1p(rem)` 剩余存活方差降幅切分（受独立局数与样本数守卫），天然对共线重复维免疫并自动捕获多维交互险区（6 种子 90 次切分中冗余维 `next_dx` 切分 **0 次**，`next_dy` 仅 **6 次**，93.3% 容量集中于核心动力学与对准维）。
  - **180 个独立 held-out 世界配对确认（6 种子 `100..105`，`cap=1200`）——全视界 `6/6` 种子全正**：
    - **DEV 60 世界（`eval 120–129`）**：
      - `H=4`（16 候选）：`73.7 → 129.8`（**`+56.1`**, 过柱 `12 → 42`, **`6/6` 正**）；
      - `H=6`（64 候选）：`115.7 → 186.2`（**`+70.5` / `+60.9%`**, 过柱 `33 → 84` `+51`, **`6/6` 正**，超越 `H=8` 基线 `175.2`）。
    - **全新未见 120 世界（`eval 200–219`，零调参样本外确认）**：
      - **`H=4`（16 候选，`2.54s`）**：`75.6 → 137.7`（**`+62.1` / `+82.1%`**, 中位 `72.0 → 139.0`, 过柱 `26 → 99` `+73`, `82W/27T/11L`, **`6/6` 正**, `act=0.96`）——超越 `H=6` 基线（`124.7` / `77` 过柱）；
      - **`H=6`（64 候选，`8.86s`）**：`124.7 → 219.7`（**`+95.1` / `+76.3%`**, 中位 `79.5 → 152.0`, 过柱 `77 → 218` `+141`, `85W/15T/20L`, **`6/6` 正**, `act=0.95`，`seed_deltas=[+34.3, +60.7, +68.5, +151.8, +187.2, +67.8]`）——**超越 `H=8` 基线（`154.8` / `120` 过柱）`+64.9` 帧，并与 `H=8 + survival_tree`（`219.2` / `217` 过柱）完全持平，算力仅为 1/4**；
      - **`H=8`（256 候选，`w=0.50`）**：`154.8 → 219.2`（**`+64.4` / `+41.6%`**, 过柱 `120 → 217` `+97`, `63W/28T/29L`, **`6/6` 正**, `act` 从 `0.63 → 0.84`）。
- [x] **Phase 7 — 串联 2–3 帧主动剪枝快搜 + 自监督可控特征需求启发式（启发式与最终价值函数彻底分离，2026-11-27 完成）**：
  - **核心机制（严格任务无关、默认栈不变）**：
    - `ControllableFeatureDemand`（[stage_regime.py](file:///c:/workspace/code/algo_core/stage_regime.py)）：通过世界模型多步动作敏感度 + 信噪比门槛（`rel_effect >= 0.02, snr >= 0.20`）自动剔除不可控相位/漂移维；在训练期高存活态上计算**占有率不变的安全包络中点 `0.5 * (q05 + q95)`**，按存活对齐相关性与临终对比度自监督赋权；
    - `chained_shallow_result`（[substrate_nodes.py](file:///c:/workspace/code/algo_core/substrate_nodes.py)，opt-in `--decision chained`）：将 `H=6 / H=9` 拆为多跳 `2–3` 帧主动剪枝快搜，**特征需求仅作中间跳剪枝启发式排序函数，绝不修改最终校准价值函数**（已由单测锁定 `beam_width >= 2**H` 时得分逐位不变）。
  - **180 个独立 held-out 世界配对确认（6 种子 `100..105`，`cap=1200`）**：
    - **`Chain 2×3 (H=6), B=2 + env-dem (h=0.10)`**（每步仅 **`18` 行风险评估**——比 `H=6` 穷举少 **6×**、比 `H=8` 穷举少 **24×**）：
      - **DEV 60 世界（`eval 120–129`，`2.13s`）**：从无需求 `153.6 (62过柱)` 跃升至 **`213.1` 帧（中位 `152.0`，`105` 过柱，`+59.5` 帧）**；
      - **全新未见 120 世界（`eval 200–219`，`3.80s`）**：从无需求 `177.0 (157过柱)` 跃升至 **`261.0` 帧（中位 `152.0`，`281` 过柱，`+84.0` 帧 / `+124` 过柱，5/6 种子正向 `[+212.9, +100.0, -5.0, +146.0, +334.6, +29.2]`）**——**以 `1/24` 风险评估量、`5.7×` 更快耗时超越 `H=8` 基线（`154.8` / `120` 过柱，`21.84s`）`+106.2` 帧，甚至超越完整穷举 `H=6 + survival_tree`（`219.7` / `218` 过柱，`12.14s`）`+41.3` 帧（主动剪枝起到结构正则化作用）**！
      - **合并 180 世界总览**：均值 **`245.0` 帧 / `386` 过柱**（vs `H=6` 基线 `121.7 / 110p` 超过 **2× 存活、3.5× 过柱**；vs `H=8` 基线 `161.6 / 196p` 领先 **`+83.4` 帧**）。
    - **`Chain 3×3 (H=9), B=4`（每步 `66` 行风险评估）**：DEV 60 世界 `187.8 → 212.4`（`105` 过柱）；未见 120 世界 `202.4–207.5`（`196–202` 过柱，`5.59s`）；诊断确认 `chunk_size=3` 与 `lag=3` 冲突导致第 0 跳在动作尚未物理生效前盲剪枝，而 `c=2, B=4` 完整保留所有首跳分支至 `step=3` 生效（120 未见世界达 **`274.6` 帧 / `302` 过柱**）。
- [x] **Phase 8 — 评测期固定阶段划分在线转移/LCB 更新与主程序接入（2026-11-27 完成）**：
  - **在线评测对照诊断（6 种子 `100..105` × 40 未见世界 `eval 200–239` = 240 局/组，`cap=1200`）**：
    - 负对照（频繁推倒重拆 `survival_tree`）：`Online Stage`（`168.9`）与 `Online Stage+Node`（`209.2`）在 `Ep 01–10` 同步暴跌 `-87.9 / -88.4` 帧（少量长局拉偏树分裂边界引发冷启动震荡），随后分别以 **`+37.3` / `+48.4` 帧 / 10 局** 的配对斜率爬升，`Online Stage+Node` 在 `Ep 31–40` 反超冻结版 **`+59.3` 帧**（`237.4` vs `178.1`，过柱 `123` vs `78`）。
  - **固定阶段划分在线更新确认（6 种子 `100..105` × 60 未见世界 `eval 200–259` = 360 局/组，`cap=1200`）**：
    - 固定已拟合的 `StageRegimeSummarizer` 划分树，仅保留 `6500` 启动帧 + 最近 `12000` 在线帧滑动窗口，每 2 局在线重估 `CalibratedRegimeTransitionModel` 与 LCB 续策值：
      - **`Frozen Chained`（`Chain 2×3, B=4`）**：均值 `219.5`、总过柱 `657`；
      - **`Fixed-Partition Online Stage`（固定阶段树 + 在线更新转移/LCB）**：均值 **`291.0`（`+71.5` / `+32.6%`）**、总过柱 **`979`（`+322` / `+49.0%`）**、配对胜/平/负 **`121 / 189 / 50`**；6 个 10 局分块配对差全部为正（**`+17.0, +95.7, +62.3, +83.0, +88.5, +82.1`**，彻底消除前 10 局冷启动坍塌，前 20 局快速爬升并在后续稳定维持 `+62~+89` 帧优势）；
      - **`Fixed-Partition Online Stage+Node`（再叠底层 `learn_episode`，`max_nodes=1500`）**：均值 **`282.7`（`+63.2`）**、总过柱 **`942`（`+285`）**、胜/平/负 **`127 / 184 / 49`**，6 个 10 局分块全正（`+9.2, +54.0, +59.5, +101.7, +96.6, +58.4`，斜率 `+14.0/10ep`），当前设定下增益主要由阶段层转移/LCB 校准贡献。
  - **主程序接线（严格任务无关，默认关闭、默认栈逐位不变）**：
    - [stage_regime.py](file:///c:/workspace/code/algo_core/stage_regime.py) 新增 `OnlineRegimeContinuationUpdater`：冻结阶段划分、单局增量分段缓存、支持滑动帧窗上限与实时委托回调 `as_stage_value_callback()`（与全量重分段数学等价）；
    - [substrate_nodes.py](file:///c:/workspace/code/algo_core/substrate_nodes.py) 新增 opt-in CLI 参数 `--regime-online-every`（默认 `0` = 冻结评测，正整数按局频次更新）与 `--regime-online-frames`（默认 `12000` 帧滑动窗口），并在 `run_seed()` / `--report` 中输出在线更新统计；
    - [tests/test_stage_regime.py](file:///c:/workspace/code/algo_core/tests/test_stage_regime.py) 新增冻结划分/全量等价性/滑动窗口与 `run_seed` 接线回归测试。
- [x] **Phase 8B — 死因定量解剖 + 主干索引纯在线点事件失败记忆树与 600 局确认（2026-11-27 完成）**：
  - **6 种子 × 20 未见世界真环境反事实解剖（120 局，均值 `330.9` / `386` 过柱）**：
    - 确认世界模型运动学精度已非瓶颈（5/6 种子 6 步位置误差 `0.000`）；`Chain 2×3` 剪枝丢活路仅占总死局 `11.4%`；
    - 两大主瓶颈合计解释 **`82.5%`** 死亡：**① `t-6` 提前锁死（`43.0%`，而 `t-9` 仍有 `90.4%` 可救）**；**② 临界多维组合边界风险失明/倒挂（`39.5%` 总死局 / `69.2%` 可救死局，即使穷举 `H=6` 仍选死路，`89.2%` 致命路径预测帧风险 `< 0.15`，`75.4%` 死于同类组合边界）**。
  - **三轮对照迭代规律（`eval 200–239`，240 局/组）**：
    - 负对照 R1（`rem<=5` 先导标签 + 硬轴分裂树，`w=1.0`）：`172.8`（`-116.5`），先导标签污染必经安全走廊、硬阶跃打崩强种子（`seed 104: 736 → 139`）；
    - 负对照 R2（纯 `done=True` 点事件平滑核，但 `prior=1.0, w=1.0`）：`257.4–263.2`（`4/6` 弱中种子上涨，但单次偶发死亡权重过重压低强种子）；
    - 确认版 R3（`TrunkPointFailureMemory`：固定主干叶索引 + 启动期作安全背景密度 + **纯在线 `done=True` 点事件** + 平滑局部高斯核密度比 `k_f / (k_f + k_s + prior)`，`sigma=0.25, prior=4.0, w=0.25`）：`289.3 → 314.6`（`+25.3`，`5/6` 种子正向，`40W/183T/17L`，末 10 局 `+50.8`）。
  - **100 未见世界超长程确认（6 种子 `100..105` × 100 世界 `eval 200–299` = 600 局/组，`cap=1200`，`161.5s`）——`6/6` 种子全胜 `fixed_16`**：
    - `frozen`（冻结 `Chain 2×3`）：均值 `226.6`、中位 `152.0`、总过柱 `1148`；
    - `fixed_16`（固定 16 格在线转移/LCB）：均值 `307.9`（`+81.4`）、中位 `156.0`、总过柱 `1758`（`4/6` 胜 `frozen`）；
    - **`cal_online_w25`（+ `TrunkPointFailureMemory`）**：均值 **`338.0`（`+111.4` / `+49.2%` vs `frozen`，`+30.1` vs `fixed_16`）**、**中位跃升至 `232.0`（`+80.0` vs `frozen`、`+76.0` vs `fixed_16`）**、总过柱 **`1985`（`+837` / `+72.9%`）**；配对胜/平/负 vs `frozen` `245/269/86`（`5/6` 正）、vs `fixed_16` **`154/361/85`（`6/6` 全正，零回退种子：`[+43.5, +25.4, +31.1, +12.4, +62.4, +5.5]`）**；10 个 10 局分块全部胜 `frozen`（`Ep 01–50: +104.0 → Ep 51–100: +118.8`），9/10 分块胜 `fixed_16`（`seed 104` 达 `790.4` 帧、26 局打满 `1200` 上限）。
  - **主程序接线（严格任务无关，默认关闭、默认栈逐位不变）**：
    - [stage_regime.py](file:///c:/workspace/code/algo_core/stage_regime.py) 新增 `TrunkPointFailureMemory` 并集成到 `OnlineRegimeContinuationUpdater`（可选 `failure_memory` + `failure_memory_weight`）；
    - [substrate_nodes.py](file:///c:/workspace/code/algo_core/substrate_nodes.py) 新增 opt-in 参数 `--regime-failure-weight`（默认 `0.0`，推荐 `0.25`）、`--regime-failure-sigma`（默认 `0.25`）、`--regime-failure-prior`（默认 `4.0`），并输出在线失败记忆统计；
    - [tests/test_stage_regime.py](file:///c:/workspace/code/algo_core/tests/test_stage_regime.py) 新增局部点事件危险定位/OOD 归零/有界水库与 `run_seed` 接线测试。
- [x] **Phase 9 — 任务无关节点/状态变量抽象与类节点按世界变量实例化（2026-11-27 完成）**：
  - **核心结构（严格任务无关，复用 [variable_registry.py](file:///c:/workspace/code/algo_core/variable_registry.py)，默认关闭、默认栈逐位不变）**：
    - 新增 [node_variables.py](file:///c:/workspace/code/algo_core/node_variables.py)：
      - `ParameterizedState`：将观测与当前推断的隐变量签名、激活通道和可信度绑定为一等状态对象；
      - `DynamicsClassNode`：抽象动力学类节点，将**慢共享类模板（`class:dynamics`）**与**快实例变量槽位（`var:dyn:ch{j}`，经 `VariableRegistry` 的 `tracks` 关系推导可信度）**分离，配备**耦合跳变整步掩码**（`q90` 步长阈防离散复位污染）与**残差显著性死区门**（`deadzone_z=3.5, min_gain_ratio=0.30`，未变世界保持零扰动），并支持残差突变检测与局中快切换；
      - `variable_similarity()`：无量纲变量空间高斯相似度，单世界时恒为 `1.0`。
    - [stage_regime.py](file:///c:/workspace/code/algo_core/stage_regime.py)：`TrunkPointFailureMemory` 与 `OnlineRegimeContinuationUpdater` 扩展可选 `var_signature` 作用域绑定——多世界切换时按推断变量相似度加权/筛选同世界经验并即时重实例化，同时以启动数据作跨世界共享类先验，不传 `var_signature` 时逐位不变；
    - [substrate_nodes.py](file:///c:/workspace/code/algo_core/substrate_nodes.py)：新增 opt-in 参数 `--world-variables`（默认关闭）、`--world-var-reset-episode`、`--world-var-sigma`（默认 `0.75`），并在 `evaluate()` 增加默认关闭的 `adapt_world=False` 钩子。
  - **双门实测验证（全部基于仅在默认重力 `g=0.0016` 训练的模型，零物理先验）**：
    - **Gate 1 — 8 个未见物理场景 × 90 局 = 2880 局（`cap=1200`）**：
      - 自监督提取快变量与真实隐藏重力偏移的皮尔逊相关系数达 **`|r| = 0.99978`**；局中 `t=60` 突然切重力中位检测延迟仅 **`2.0` 帧**；
      - `full_var_online` 在默认世界（`280.9 → 333.9`, `+53.0`）、`0.50×` 轻重力（`204.7 → 251.6`, `+46.9`）、`1.38×` 重重力（`185.2 → 315.4`, `+130.3`）、`1.75×` 超重（`141.1 → 290.9`, **`+149.8`**, 过柱 `76 → 245` `3.2×`）、4 参数重变（`196.4 → 274.0`, `+77.6`）与局中切重力（`179.0 → 301.6`, `+122.6`）全部大幅胜出（7/8 场景大胜；仅 `multi_light` 因多参数相互抵消 `323.1 → 301.2`）。
    - **Gate 2 — 交替多世界防遗忘门（`10轻 0.55× → 10重 1.65× → 10轻 → 10重`，6 种子 × 40 局 = 240 局/组）**：
      - `static_world`：`157.3`（`255p`，重返 `B3+B4 = 147.3`）；
      - `dyn_unscoped`（不区分变量的在线记忆）：`255.1`（`547p`，重返 `B3+B4 = 216.5`，因跨世界混入体验导致轻/重回归段掉分）；
      - **`dyn_var_scoped`（按变量签名的类节点实例化）**：**`284.1`（`633p`，相对静态 `+126.8`，相对不区分变量 `+28.9` / `+86p`）**，重返旧世界段 `B3+B4` 达 **`268.5`（比不区分变量高 `+52.0` 帧：`B3 Light 259.9 vs 195.3`, `B4 Heavy 277.1 vs 237.6`）**。
  - **回归测试**：新增 [tests/test_node_variables.py](file:///c:/workspace/code/algo_core/tests/test_node_variables.py)（含任务无关性锁、死区零扰动锁、`|r|>0.999` 隐变量提取与局中切换恢复、跨世界记忆隔离与 `run_seed` 接线测试）。
- [x] **Phase 10 — OMNI 机制交叉吸收验证：工作记忆槽位 + 来源级因果 `AbstainGate`（2026-11-27 完成）**：
  - **第一项（2400 局 = 6 种子 × 4 组 × 100 未见世界 `eval 200–299`）**：
    - **保留原语**：在 [node_variables.py](file:///c:/workspace/code/algo_core/node_variables.py) 合入 `WorkingMemorySlots`（`UNBOUND` 哨兵、`(scope, entity, role)` 槽位竞争、侧抑制与老化）并接入 `DynamicsClassNode`；
    - **负对照否决**：在 `TrunkPointFailureMemory` 上叠硬 `LEARN/QUARANTINE` 门控（`328.4 vs 338.0, -9.5`）与 `node_rel_age` 衰减（`316.7 vs 338.0, -21.3, 0/6` 正）均为负——因平滑贝叶斯先验 `prior=4.0` 已柔性保护高安全区，硬隔离会饿死弱种子，而相对时钟衰减会让系统在成功避障后遗忘真实物理死墙；不合入。
  - **第二项（4320 局并行验证）**：
    - **负对照否决**：静态 `branch_entropy` 降权（360 局：`324.2 vs 330.4, -6.2`）误伤已校准种子；不合入。
    - **过门晋升① — 阶段续策局后因果 Wilson-LCB `SourceAbstainGate`（360 局，`eval 200–259`）**：
      - `base_cal_w25` `330.4 (1157p)` → **`stage_abstain` `340.7 (+10.3, 1203p +46)`**；5/6 已校准种子精确零触碰（`0.0` 变化），**外科手术式自动识别并拉升唯一倒挂种子 `seed 103`：`291.0 → 353.0（+62.0 帧）`**！
    - **过门晋升② — 动力学类节点因果 1 步预测 Wilson-LCB `SourceAbstainGate`（2880 局，8 变物理场景）**：
      - 默认世界自动弃权 `94.8%`（`279.9 vs 280.9` 零退化），变化世界自动激活 `94%–96%`；**`full_var_abstain` 达成 `8/8` 全部物理场景全正**（把唯一负场景 `multi_light` 翻正至 `326.2 vs 323.1`，`mid_switch_g` 升至 **`307.9 (+128.9)`**，`1.75×` 超重 **`282.9 (+141.8)`**，`1.00×` 默认 **`331.0 (+50.1)`**）。
  - **主程序 opt-in 接线（默认关闭，默认栈逐位不变）**：
    - [node_variables.py](file:///c:/workspace/code/algo_core/node_variables.py) 新增 `wilson_lcb()`、`SourceAbstainGate` 及 `DynamicsClassNode(abstain_lcb=0.0)`；
    - [stage_regime.py](file:///c:/workspace/code/algo_core/stage_regime.py) 为 `OnlineRegimeContinuationUpdater` 增加 `stage_abstain_lcb=0.0`；
    - [substrate_nodes.py](file:///c:/workspace/code/algo_core/substrate_nodes.py) 新增 `--regime-abstain-lcb`（默认 `0.0`，推荐 `0.45`）与 `--world-var-abstain-lcb`（默认 `0.0`，推荐 `0.55`）；
    - [tests/test_node_variables.py](file:///c:/workspace/code/algo_core/tests/test_node_variables.py) 补齐 `WorkingMemorySlots`、`SourceAbstainGate` 与因果自监视单测。
- [ ] **Phase 11 — 后续可选方向（默认栈仍不变，按需推进）**：
  - 在 `cap=5000` 完整 benchmark 口径上验证 `Chain 2×3 + OnlineRegimeContinuationUpdater + TrunkPointFailureMemory + DynamicsClassNode + SourceAbstainGate` 的长尾上限；
  - 将抽象类节点进一步扩展至“关系型/对称特征类模板”（跨观测通道共享边界/接近关系模板）及长程多世界变量树先验召回。

## P0：多视窗状态转移机制重设计（当前主线）

**核心判断**：这里不是再找一个更好的 span hazard / 触发阈值；问题在于
训练对象与预测对象不是同一个转移。统一成**动作对齐的经验转移测度**：

```text
(source node, 实际作用动作) -> (successor node, feedback, terminal)
```

- 训练只消费真实转移；实际作用动作取 `history[lag]`，不是刚发出的
  `history[0]`。
- 同一个 kernel 在任意窗口递推，返回 successor 分布、反馈/terminal
  累积量、support、UNKNOWN；可选目标节点集的 first-passage CDF，其
  差分就是停时分布。窗口是**查询分辨率**，不是多套模型。
- 候选预测输入实际的候选动作序列；只在已观测的 source/action 支持
  上传播。缺边 / OOD → UNKNOWN，不当安全、不补反事实标签。
- fine MPC 保持执行权；新 kernel 先作为 shadow prediction 评估，
  过 held-out 多视窗转移门后才做配对决策 A/B。

**已落地的底座**：`multiview_transitions.py` 在 `fit()` 后从实际行训练
kernel；`predict_views()` 对任意窗口递推并显式传播未知质量。旧
`_transition_table()` 已修正 action delay 对齐，并保留每条边的动作计数，
不再把动作合并为多数票。默认 planner 尚未消费该对象。

**第一版结论（seed 100/101，leave-episode-out，初步）**：h1 successor
top1 0.668/0.686，full support 0.812/0.780；h4 top1 0.389/0.399，
full support 0.385/0.419；h8 full support **0.064/0.055**；h16
**0.034/0.048**。UNKNOWN 随 rollout 深度累积，精确 frame-node kernel
不能直接支撑长视窗。修正过的 probe 按每个 horizon 独立纳入起点，并以
事件时刻 / 右删失 survival likelihood 计终止；不会再把死在短窗内的样本
筛掉。Terminal NLL 只在 full-support 子集上计算，且该子集会偏向某些
结果，不能单独当 calibration 通过证据。

**留出限制**：当前 probe 留出的是 transition edges / episode rows，
但 fine node vocabulary 来自已拟合 agent（看过这些 episode 的状态）；
所以它只诊断“固定表示下边表能否迁移”，不是完整的 representation
generalization gate。预测等价 quotient 必须在每折训练 episode 上拟合，
测试 episode 只能映射到该折的冻结表示；映射失败计 UNKNOWN。

**机制收敛**：不能只把稀疏一步图反复相乘；也不能用精确 prototype
连续 run 直接作宏段（真实流的 prototype run length p50=1、p90≈5）。
下一版必须学习**预测等价的粗状态商**：合并那些在真实 action-conditioned
successor / feedback 分布上不可区分的细状态，再从真实轨迹直接记录粗状态
之间的 `(出口状态, 停时, feedback, terminal/censoring)` 分布。合并深度由
held-out 预测似然与支撑选择，不设固定节点数/层数。候选动作序列按
`history[lag]` 顺序原样输入，不压成无序动作直方图；不支持就 UNKNOWN。
细层 executor 保持兜底。

**下一步 P0（先验证实现，再谈接线）**：

1. `multiview_transition_probe.py` 已完成初版 rolling-origin /
   leave-episode-out：h=1/4/8/16 successor、终止生存似然与 UNKNOWN。
   结论是**一步核多步支撑坍塌**；不接 planner。
2. **折外表示已完成**：`multiview_transition_probe.py` 默认 `fold-refit`，
   每折只用训练 episode 重建 state vocabulary；评测节点无映射 → UNKNOWN。
   seed 100/101 五折结果已收齐（每折最多500起点）。
3. **探索性 quotient 原型已落地**：
   `predictive_state_quotient.py` 按真实 action-conditioned successor /
   feedback profile 聚合细节点。seed 100 双折 pilot：flat K=1 虽有支撑，
   但 top1≈1 是常数退化；K=4/16 在 h4 后支撑仍坍塌。→ 不能选一个全局 K。
4. **五折跨种子结果：有支持率/混合分辨率正信号，但尚未通过**：
   `hierarchical_transitions.py` 的 16→4→1 backoff 对比 fine-only：
   - seed100/101 的 h8 full support **0.831/0.794**（fine **0.077/0.065**），
     mixed-resolution top1 **0.728/0.788**；
   - h16 full support **0.832/0.793**（fine **0.048/0.049**），
     mixed-resolution top1 **0.909/0.972**；
   - h16 的回退量约 44% 在父层、35%/38% 在根层。
   这些 top1 将不同抽象分辨率混在一起，不能单独作为预测能力门槛。
   seed100 两折分层诊断中，h16 fine 层 top1≈0.30、q4≈0.68、root=1
   （单状态恒真），平均状态分辨率约645/1000；q16 层无可评估支持。
   所以支持率改善不是全由 root 解释，但跨种子的逐层质量尚未确认；
   默认 planner 继续不接线。
5. **[x] 跨种子独立逐层审计**：`multiview_transition_probe.py` 已新增
   fine/q16/q4/root 各自独立 kernel 的评估，不再把混合回退分辨率当逐层
   质量；报告训练折 endpoint base-rate、UNKNOWN、coverage、log-loss、
   top-1 和 fine-node resolution。seed100/101，fold-refit 五折，窗口
   h=1/4/8/16，每局最多500个起点。
   - h1 有支持子集上的非根层信号跨 seed 一致（fine 条件 log-loss gain
     +3.26/+3.42，q16 +1.20/+1.15，q4 +0.25/+0.16），但全样本 log-loss
     因 UNKNOWN 反而劣于 base-rate。
   - 长窗覆盖不能成立：h4 q16 为 0/0，q4 为 0.26/0；h8 fine 为
     0.035/0.032、q4 为 0.169/0；h16 fine 为 0.001/0.005、q4 为
     0.062/0。q4 的 h8/h16 有支持子集也未稳定胜过匹配 base-rate。
   - root 的 h4–h16 支持率虽约 0.81–0.90，但 root 是单类恒真：
     base top-1=1、base log-loss=0；混合粒度 top-1 不能作为预测能力证据。
   → **长视窗门失败**：只确认 h1/部分 h4 的局部信号，不足以支持粗层
   长程转移；停止 quotient 调 K，不接 planner。
6. **[x] 不启动动作序列 shadow**：第5项长窗非根层 coverage / 跨 seed
   质量门未过；不在 `policy_result()` 插 shadow，也不进入配对决策 A/B。
7. 当前 hierarchy 仍是粗状态上的 frame kernel 递推，不是 semi-Markov。
   直接 first-passage/stopping-time 学习暂缓；本轮没有通过 shadow 前置门，
   不从未验证的 quotient 继续扩建。
8. **[x] delayed-action / terminal 边界审计**：`NodeRuntime` 的 action
   条件学习与 `_transition_table` 均按 `history[lag]` 归因；当前
   `history[0]` 用于读取新发动作/推进 pending queue，不是转移归因。
   旧 planner graph 的错位在现行表中已修正。另给 `_transition_table`
   加 terminal 行截断，防止同一 episode id 下的脏尾行生成死后边；
   `test_terminal_row_does_not_create_an_outgoing_graph_edge` 覆盖该边界。
   全量测试 **524 passed**。

详细契约与边界见 [`docs/multiview-transition-design.md`](docs/multiview-transition-design.md)。

## 通用阶段记忆基座（2026-11-24，用户确认）

**目标校正**：这里要解决的是通用的决策近视，不是给单个任务加一个
更长的帧搜索窗口。基座从任务提供的观测、实际动作、反馈和终止信息中
发现关键信号与阶段，并学习阶段间的非帧转移；不认识任务几何、事件名称
或手写控制律。阶段持续时间是转移结果，不是固定 horizon。

**自监督输入边界（用户确认）**：唯一外部来源是世界交互：观测、实际作用
动作、世界反馈与 episode 终止。prediction error / novelty / key signal、
阶段身份、StageDemand 和 feature→successor 关联都由内部模型和这些真实记录
推导；生产运行不接受 task-provided 阶段标签、目标阶段、启发权重或控制规则。
`evaluate` 的 provider 仅是内部学习器的组合接口，不从 task 字典或 CLI 注入；
`run_seed` 不读取 provider 字段。合成输入只用于单元测试。

**结构约定**：

- 阶段实例记录一次真实 episode 中的入口/出口状态、关键信号、按序动作
  轨迹、持续时间、反馈以及 terminal / censoring。
- 阶段上下文形成可变深度记忆树；在线从当前阶段的最具体历史上下文向父级
  回退。每个上下文下的兄弟阶段只由真实有向转移边定义，树上的相似/兄弟
  关系本身不代表可达转移。
- 转移记录为 `(stage context, ordered action option) -> (sibling outcome,
  duration, feedback, terminal/censoring)`；只消费实际发生的转移。
  episode 支撑不足或动作选项未见时返回 `UNKNOWN`，root 仅表示粗分辨率基准，
  不把常数 root 的覆盖冒充预测能力。
- 阶段分段/身份、阶段树合并与有向转移图分离：前者产生可寻址的阶段记忆，
  后者才学习阶段的时间顺序和结果。

**第一片已实现（纯数据层，默认未接 planner）**：新增
`stage_outcome_memory.py`：

- `KeySignal` 使用不透明 channel / polarity，不含任务语义；
  `GenericKeySignalDetector` 从训练 episode 为 observation-change、
  feedback-change 和可选 prediction-error 通道拟合分位阈值，held-out 只做
  frozen transform；`segment_episode()` 按信号上升沿切出变长阶段，不设固定帧长。
- `StageOccurrence` 保存入口/出口向量、信号强度、保序 RLE action trace、
  持续时间、累计反馈、终止/删失标记，以及逐转移 active-feature 快照。
- `StageOutcomeMemory` 从真实相邻阶段实例统计有向结果；查询按最近阶段历史
  的变阶后缀上下文从具体层回退到 root，动作轨迹精确匹配；报告 sibling
  结果、episode support、持续时间分位数、反馈统计、terminal/censoring。
- `DemandAssociationSample.from_stage_episodes()` 把真实阶段内的 feature
  snapshots 与实际下一阶段配对，避免把整个阶段末尾才出现的 feature 当成
  阶段入口特征；默认池化动作选项，精确 option 条件另行显式提供。
- `tests/test_stage_outcome_memory.py` 的 11 条测试覆盖折内信号阈值、变长分段、信号持续值
  去重、兄弟结果、父级回退、动作顺序、episode 支撑、UNKNOWN 与删失。
  全量测试 **530 passed**。

**需求导向排序（纯启发式，默认关闭）**：新增
`stage_demand_heuristic.py` 与 `trajectory_search.beam_search(rank_step=...)`：

- `StageDemand` 可由高层已支持的兄弟阶段预测生成，表达目标阶段及其 context /
  action option；它不携带或修改最终价值。
- `StageDemandHeuristic` 只从真实样本统计
  `(stage context, action option, active feature) -> observed next stage`，
  以目标阶段在该 context 下的 base rate 为基线，返回平滑后的方向性 log-lift
  作为 priority；跨 episode 支撑不足则 UNKNOWN。该相关性用于排序，不宣称因果。
- `beam_search` 的可选 `rank_step` 只用于中间 frontier 保留；不改变 feasibility、
  objective `scores`、terminal 价值或最终 score。无 rank hint 时保留旧路径。
- Runtime adapter 使用候选状态的只读 feature masks；`policy_result`、
  `decision_action`、`evaluate` 已接入 `StageDemand + StageDemandHeuristic`，
  `stage_demand_provider` 每次决策提供当前需求，并可通过
  `observe_transition` 仅接收真实执行后的转移/反馈。provider 必须由内部阶段
  学习器产生，不从 task 输入；默认无 provider、rank hint 关闭。
- 验证：需求关联 / base-rate lift / UNKNOWN / 回退 6 条测试，rank-only 与终局
  价值分离 1 条测试，provider 到真实反馈更新链 5 条测试；全量测试
  **530 passed**。
- **当前 held-out 决策 A/B（2026-11-25，seed 100 训练，eval worlds 120–127，
  cap 400 / horizon 8 / beam 4）**：baseline 存活 `25` 帧 × 8；stage rank
  为 `25/37/25/31/25/37/53/38`，paired Δ 均值 **+8.9 帧**
  （95% t 区间约 **+0.8 到 +16.9**），5 胜 / 3 平 / 0 负；两臂均 **0 pass**。
  target frontier 有 8 次进入最终保留集，但 **199 个 demand 中 0 次选中目标
  endpoint**；provider 支持 199/271 次，但仅 124/10746 个候选状态得到非零
  rank。结论：能带来小幅存活提升，目标阶段尚未控制最终选择。
- **需求特征增权改造（2026-11-25）**：训练可选择 `hybrid` 特征源，把
  namespaced branch roots 与 nearest prototypes 同时作为真实阶段转移的
  association features；selector 按 transition episode support 与 outcome-value
  margin 生成 `demand_strength`，只缩放 feature log-lift 的临时 beam priority；
  `OnlineStageDemandProvider` 在同一 stage depth 内锁定 demand，到实际阶段边界
  才释放/重选。最终 objective/value/feasibility 不变，rank 仍 default-off。
  49 项阶段/需求测试通过，最终全量 **727 passed**。
  - 8-world A/B（hybrid，eval 120–127、H=8 / beam=4）：baseline/ranked 均
    **25 帧 × 8、0 pass**；192/200 次 demand supported，但只有
    **52/10368** 个候选获得非零 priority；15/192 个 demand 有目标候选，最终
    仅 6/192 留在 final beam、**0/192 被选中**。→ 混合原语扩大来源但没有提高
    决策覆盖，当前不应当作收益改造。
  - 同配置 root rank + stage intent lock 复现 `25→33.9` 帧小幅提升，但 final
    target 仍 **0/199**；sticky demand 未解决“目标入 beam、不入最终计划”。
  - **StageExecutor 8-world A/B**：exact option + depth/support gate 时 survival
    与 baseline 一致（80.9 均值），option share **1.5%**，每局平均拒绝约
    89 个候选；去掉门后 share **15.6%**，但配对变化
    `−43/0/0/−71/0/+80/0/−43` 帧，均值 **80.9→67.1**、passes `1→0`。
    validated option-class + gates 未选出 option。→ 不放宽执行门；精确 trace
    克隆不是可靠的高层意图执行器。
- **高度—缝隙对齐特征审计（seed 100 train 1500 steps、epochs=1；eval worlds
  120–127，hybrid demand，H=8 / beam=4）**：
  - 环境观测 `dy`（f3）定义为 gap center − bird height。value-rule induction
    确实提出了 `(|f0| < 0.115 & |f3| < 0.238)`，训练 `n=87, p=0.448`；
    原规则选择折的 holdout 是 **n=0**，不能算成已验证价值特征。
  - 200 个 held-out 实际状态上，`feature_value_batch` 在三个 `|dy|` 分桶的
    mean value 都是 **0**、mean support 都是 **0**。对齐 predicate 在
    `|dy|<0.05` 状态中有 **37.5%** 激活，但没有可用 value cell。
  - beam 回调中分离累计 priority 后，10,368 个 candidate-state 中只有
    **52** 个有直接非零 demand uplift；`|dy|<0.05` 区间 uplift **0/1184**，
    `0.05–0.15` 为 **28/2148**，`>0.15` 为 **24/7036**。以 `|dy|<0.1`
    为诊断标签的 uplift AUC **0.502**（近随机）；非零分数与 `-|dy|` 的相关
    约 **0.31**，样本太少且不构成区分证据。
  - baseline 与 demand-ranked 共 200 个决策的首动作、预测 endpoint 完全一致；
    demand-active 的 192 个 endpoint 中仅 **16.7%** 比当前状态更靠近 gap，
    mean `|dy|` 从 **0.202** 到预测 endpoint 的 **0.373**。realized next-4
    net feedback 在 close/mid/far 组分别 `4.00/3.96/3.04`，仅是同策略流描述值，
    不代表因果收益。
  - 判定：**几何线索以一个 value-rule 候选形式存在，但没有可泛化的 learned
    value；高层 demand 也没有把该线索增权到足以改变底层动作。** 下一步需在
    held-out episode 上学习并校准 `(stage context, generic feature, next-stage)`
    association，要求 close-gap feature 在高层目标相关时有独立 episode 支持、
    正向 uplift，并在同状态动作候选中提高 gap 收敛率；达标前不写 Flappy 专用
    `dy` 目标或手工价值。
- **多 epoch 复核（同 seed/data、epochs=5）修正上一条的 value 结论**：
  `FeatureRuleBank` 在 epoch 0 之后才安装 `rules_value`，所以 epochs=1 时其
  value cells 确为空；epochs=5 有后续 replay。`feature_35 =
  (|f0|<0.115 & |f3|<0.238)` 的 cell 现有 action0 `n=332 / 9 episodes`、
  action1 `n=16 / 4 episodes`。 held-out 200 个实际状态按 `|dy|` 分桶：
  global value `<.05/.05–.15/>.15` 为 **28.69/28.28/25.89**，support
  **28.76/29.15/18.90**；隔离 `feature_35` 时 `<.05` 读数 **27.24**
  （support 3.38），其他区间回退至 global prior **25.60**（support 0）。
  → 小 gap 与较高预测 return 有真实但有限的证据；rule 内部选择折的 `test n=0`
  仍说明该 predicate 的 induction 稳定性不足。
  - 将 value-only roots 纳入 hybrid demand token 后，66 个 `feature_35` 激活候选
    中仅 24 个有当前目标 successor 的关联支持，其平均 per-feature lift
    contribution 为 **−0.243**。即全局 return 读数认为它有价值，不代表当前被
    selector 选出的 successor 也需要它；target-specific association 的方向
    目前与“对齐”要求相反。
  - epochs=5 的 8-world A/B：baseline/ranked **25→34.9** 均值、median
    `25→30`，但 **0 pass**，最终 endpoint 仍 `0/196` 选中目标。rank nonzero
    为 `4990/10584`，feature 覆盖已提高，但尚未验证为对齐收益。
- **复盘能力边界**：`_record_prediction` / `prediction_report()` 能保存和报告
  高置信预测误差；`surprise_buffer` 当前只被追加、截断和展示，没有训练器或
  假设生成器消费它。因此系统会“报错”，不会自动归因“哪个需求/特征/搜索剪枝
  导致错误”或提出经验证的修复。`_advance_memory_traces` 是 valence
  retrospective credit assignment，不是错误诊断器。
- **3-frame chained search 与真实失败回溯**：
  - 新增 opt-in `chunked_beam_search`，每 3 帧扩展并剪枝一次；携带完整 action
    prefix/state（包括由 adapter 传入的 pending queue）。`search_chunk_size=0`
    仍走旧 per-frame beam；`SearchResult.rank_priority` 与 objective score 分开。
    probe 可输出真实失败前的 source/next observation、action、signals、各 chunk
    的目标保留情况，并给出固定诊断假设；未执行分支预测不写回训练数据。
  - 同一训练与 8 个 held-out worlds，H=8：per-frame 与 chunk=3 的 baseline
    均值均 **25**，ranked 均值均 **34.9**、median **30**、passes 均 **0**；
    每决策候选从 **54** 增至 **94**，chunking 没带来决策收益，且当前实现仍用
    同一 learned world predictor，并非独立低精度模型。
  - chunk=3 的三个 checkpoint：目标分别出现在 `14/196`、`16/196`、`10/196`
    个 demand 决策中，保留 `13`、`11`、`8` 次；最终选中目标 endpoint 仍
    **0/196**。终止前的 32 个回溯决策中，目标在所有检查点都未进入 frontier，
    所选 plan rank-credit 为 0。H=12（四段）后仍 0 pass，终止前 55 个 demand
    决策目标均未进入 frontier，final-retained 仅 `2/196`。
  - 诊断：短段串联能解释路径何处缺失，但扩大 frame horizon 本身未解决问题；
    有些非临终决策里目标路径留到了最终 beam，却仍未被 endpoint objective 选择。
    后续需要先学出与目标 successor 正相关、episode 支持充分的前置特征/option，
    再考虑显式 intent-conditioned candidate cohort；低精度 screening 还需单独做
    false-prune 校准后才能减少候选成本。
  - **逐转移 residual 复核**：新增的诊断插桩用训练所得 `LinearObservationModel`
    对失败前真实 `obs + history -> next_obs` 计算归一化一步 RMS。H=8 的 32 条
    唯一真实转移（frame 与 chunk 报告重复）mean/median/p95/max 为
    `6.72e-8 / 6.10e-8 / 1.23e-7 / 1.34e-7`；H=12 的 55 条为
    `8.06e-8 / 6.50e-8 / 1.86e-7 / 1.89e-7`。观测动作 0/1 和距终止 0–11 帧
    均无大的真实一步残差簇，故这轮失败不支持“实际一步延迟转移预测失配”。
    该测量不覆盖反事实动作或多步误差累积；此前另一探针在 lag 生效窗口报告的
    约 `0.33` 残差尚未与本轮口径对齐，不能据此排除多步 rollout 误差。
  - **诊断结论**：H=8 中 `8/196` 个 demand 的目标进入最终保留集，但 `0/196`
    被 objective 选中；H=12 为 `2/196` 与 `0/196`。另一方面，终止前 H=8 的
    `32/32`、H=12 的 `55/55` 个 postmortem 决策目标从未进入任何 frontier；
    这是候选缺失事实，不单独证明动力学上不可达。`search_chunk_size=3` 在 H=8
    候选工作量由 54 增至 94/decision，却与逐帧搜索产生完全相同的各 seed 存活
    和通关结果。session `debug-quick-proposal-diagnosis.md` 保持 OPEN：
    尚无证据足以选择业务逻辑修复；当前机制证明了诊断可解释性，未证明提议质量
    或决策收益。
- **策略提议—快速筛选—真实信用回写 v1（2026-11-26）**：
  - `OnlineStageDemandProvider` 从 outcome readout 的受支持 successor 中取最多
    3 个 proposal，共享同一套 beam/chunk 搜索；各目标分别算 rank priority，
    每个候选状态只用最高的受支持 proposal priority，不为每个目标重复 rollout。
    `search_chunk_size=3` 继续负责短段剪枝，rank 仍不进入最终 objective。
  - 新增 `StageStrategyCredit`，按 `(源阶段历史, 目标 stage root)` 记录结果；
    仅用环境实际到达的下一个阶段匹配目标树节点来计分。目标没命中时只给主
    proposal 记负结果；实际命中的备选得到正结果；未执行的其他备选不伪造负例；
    terminal 是实际未命中，episode reset/不完整结果按删失不计分。每个策略每个
    独立 environment episode 最多计一次，Beta 后验置信度达门槛后只调整未来 rank
    priority。修正 provider reset 时忽略 env seed、可能把多个 episode 合并计数的问题。
  - **8-world prequential smoke**（seed 100 训练 1500 steps/5 epochs，eval 120–127，
    H=8、beam=4、chunk=3）：56 个 intent 得到真实终点观察，其中 primary 命中
    **32**、进入其他阶段 **20**、终止前未命中 **4**；16/56 次提出了多个目标，
    共生成 72 个策略 proposal。42 次后续 proposal 使用了已支持信用。每 decision
    扩展数仍为 **94**（未因多个目标重复搜索）；提议胜出的 candidate rank evidence
    集中在 `tree 12`（5288 次），其他目标未观察到受支持的 rank。baseline/ranked
    存活仍为 **25/34.875** 帧、passes **0**；proposal-frontier final-retained
    **8/284**、selected endpoint **0/284**。→ 端到端真实反馈回写和共享短段筛选
    已接通，但本次小型在线 smoke **没有带来决策收益**，不能声称目标策略已收敛。
  - **chunk group quota（2026-11-26）**：`rank_group_quota=1` 在 chunk prune
    时为每个有正向、受支持的 proposal rank 至少留一条路径，再按原顺序补满；
    `rank_mode=tie` 只在 objective near-tie 集内预留，明显更优路径仍不被启发式挤掉。
    H=8 同一 8-world smoke 中，每 decision candidates 仍 **94**、ranked **34.875**
    帧、passes **0**，与 quota-off slate 基线一致；但实际只有 `tree 12` 形成正向
    proposal group，在它出现的 **216/216** 个 chunk frontier 都至少保留了该组，
    没有第二个可用的正向 group 可供配额竞争。→ quota 机制通过了行为测试，
    真实流暴露的限制仍是只有一个 successor 获得特征支持，而不是 quota 不够。
    rank 计算额外 CPU 开销尚未测量。
  - **值不确定度驱动的 slate 宽度**：selector 将 best-vs-runner-up 的归一化
    `value_confidence` 暴露给 provider；与 primary 的 episode-match posterior 取较
    保守值后，用 `ceil(K * (1-confidence))` 控制 slate 宽度，主策略 UNKNOWN 或真实
    mismatch 时仍取最大 K。信用仍只由真实后继转移更新。
  - **自适应 slate + quota held-out smoke**（同 seed 100 / eval 120–127，H=8、
    beam=4、chunk=3）：56 个 intent 中 **24** 次 slate=2、**32** 次 slate=3，
    没有 slate=1；原因是这批状态里没有同时满足高 value margin 与主策略稳定信用
    的决策。rank-group quota 保留 `tree 12` 于 **216/216** 个有支持的 frontier，
    仍没有第二个 proposal 形成可用的正向 rank group。baseline/ranked 仍为
    **25/34.875** 帧、passes **0**、94 candidates/decision、selected endpoint
    **0/284**。→ 自适应宽度与分组保留接线有效，但当前数据没有形成多个可竞争的
    细策略；不能用 quota 或增大 proposal 数代替缺失的特征证据。实际宽度统计为
    slate=2 **24**、slate=3 **32**、slate=1 **0**，说明当前模型没有足够证据可安全
    收缩为单一方案。
  - **剩余 gap**：本版候选来自 stage-outcome readout 的 top-3 successor，不会从
    value residual / 节点变化自动挖掘新的深层 feature policy；目标相似度目前是
    learned stage tree root 命中，而不是 duration/feedback 的联合似然。虽然已有
    group quota，但当 proposal 没有正向特征证据时没有组可保留。下一步需从真实
    预测残差生成通用特征候选，并用独立 episode 验证其 successor/feedback 增益，
    而不是放宽 quota 或把 UNKNOWN 当成正证据。
  - **验证**：新增 credit posterior、unique-episode、terminal/censor、备选命中、
    top-K proposal、adaptive slate、group quota、tie-mode objective guard 和 rank-only
    接线测试；全量 **751 tests passed**。
- **逐决策 trace（seed 100 train；eval 120 与 126；horizon 8 / beam 4）**：
  在相同状态上并行比较 objective-only 与 stage-rank，再只执行 ranked action。
  - eval 120：25 次决策全部选 action 0；stage demand 24/25 次有输出，但目标
    root 从未出现在候选 frontier，rank 未改变任一保留路径；第 25 帧以
    `pain=1` 终止。
  - eval 126：53 次决策中 provider 仅支持 25 次；t=26–52 为 `no_demand`。
    t=2（目标 tree 2，p=1.0，support=38）与 t=10（tree 12，p=0.833，
    support=36）两次把首动作从 0 改为 1。t=2 rank 将 objective cost
    从 **0.1921** 提高到 **0.2036**；t=10 从 **0.1488** 提高到 **0.1747**。
    全轨迹的候选 frontier 中目标状态出现 25 次、保留 21 次，但选中的 53 个
    endpoint **0 次**解析为需求目标。53 次账本的 value share 均为 0，最终仍在
    第 53 帧以 `pain=1` 终止（状态 `y≈−0.53`，未过柱）。
  - 按实际延迟动作生效时点，t=5、t=13 的一步预测归一化误差分别约
    **0.334 / 0.336**，恰好对应此前两次 action 1 的 lag=3 生效附近。
    结论：当前排序器能偶尔改动作并改善存活，但 target association 没转成
    endpoint/value 目标；有效决策还受需求断流、目标 endpoint OOD、序列价值
    不发声及动作生效附近模型误差共同限制。

**重排后的 gap 与后续方向（严格自监督）**：

1. **[x] P0：自动经验管线（`stage_experience.py`，2026-11-24）**。
   唯一外部来源仍是世界交互：按 episode 分组，acting action 取
   `history[lag]`（缺失则整局丢弃，绝不退回 `history[0]`），feedback 取
   真实 `signals[:2]`，active-feature 用 runtime 冻结的
   `_branch_active_features` 只读快照；`GenericKeySignalDetector` 只在传入
   rows 上拟合，`transform(rows, agent)` 复用冻结阈值，不更新阈值。
   产出 `StageOccurrence` 序列与 pointwise `DemandAssociationSample`
   （逐转移快照配真实下一阶段，不是只配阶段入口）。census 报告
   episodes/stages/keyed share/duration/feature coverage/demand 数。
   `substrate_nodes.py --stage-memory`（默认关）把 batch 挂到
   `agent.stage_experience`，planner 不读取，默认决策逐位不变。
   - 合成测试 7 条；全量 537 passed。
   - fast 真实流 smoke（seed 100、1500 帧）：38/38 局、367 stages
     （9.66/局）、duration 中位 4（q90 6）、keyed 0.90、9 个 signal
     channel、1257 demand samples；**feature coverage 仅 0.03
     （8 个 feature id，44/1500 行）**——内部 feature 根很稀疏，
     下一层不能把 demand 关联当主信号，resolver 应主要依赖状态/阶段，
     feature 只作可选证据。
   - **特征族群审计初筛**（`feature_family_audit_probe.py`，3 seeds ×
     6500 frames，outer episode-held-out；规则只在 induction episodes 生成、
     calibration episodes 选取，test episodes 只评分；目标为真实未来 4 步
     pain feedback）：
     - observation 基线 Brier skill **0.085/0.221/0.372**、NLL gain
       **+0.031/+0.053/+0.093**，三 seed 同向；拼入 pending queue 的读数与
       observation 完全相同，暂未观察到边际增益。
     - pairwise relation Brier skill **0.028/−0.296/+0.169**、NLL gain
       **+0.020/−0.035/+0.053**，方向不稳定。
     - temporal delta/acceleration 覆盖 **0.619/0.734/0.650**，但 Brier
       skill **−0.016/+0.025/+0.074**、NLL gain **−0.002/+0.001/+0.021**；
       高激活率未转成稳定预测增益。
     - 前一实际转移的折外 world-model residual 覆盖 **0.659/0.731/0.521**，
       Brier skill **+0.027/−0.038/+0.035**、NLL gain
       **+0.009/−0.012/+0.013**，同样未稳定。
     - state+temporal+residual 的 Brier skill **−0.030/+0.084/+0.255**，
       NLL gain **+0.012/+0.028/+0.067**；各 seed 均低于 observation 单独
       的增益，且覆盖更低。→ 暂不增广生产特征，也不把该结果外推为阶段转移
       特征充分性；这是短窗 pain 预测筛查。
   - **EpisodeBalancedFeatureBank 已实现为离线/默认不接 planner 的算法原型**
     （`predictive_feature_bank.py`）：typed threshold/AND/OR feature specs；
     train-only 分位候选；active/inactive 概率按 episode 等权并以 prior episode
     mass 收缩；要求两侧有独立 episode support；calibration 折按 Brier 增益与
     覆盖建立 Pareto frontier，只有 episode-bootstrap Brier 下界及 NLL gain
     同时为正才标记 supported；特征只在训练范围内适用，OOD 返回 UNKNOWN。
     `feature_family_audit_probe.py` 已切换使用此 bank，不再用旧 frame-count
     `FeatureRuleBank` 做该族群审计。
   - **三 seed 外层验证**（各 6500 frames，induction/calibration/test 按 episode
     隔离，pain horizon=4；每族每 seed 校准折选中 1–2 个 Pareto 特征，
     被选特征在 outer episode-macro Brier/NLL 上均为正，适用 episode coverage
     均为 1.0）：

     | 特征族 | Brier gain 100/101/102 | NLL gain 100/101/102 |
     |---|---|---|
     | observation | +0.0388/+0.0496/+0.0573 | +0.1324/+0.1762/+0.1630 |
     | state + pending | 与 observation 相同 | 与 observation 相同 |
     | pairwise relation | +0.0277/+0.0324/+0.0469 | +0.0753/+0.1118/+0.1442 |
     | temporal | +0.0219/+0.0299/+0.0345 | +0.0650/+0.0953/+0.1104 |
     | prediction error | +0.0075/+0.0106/+0.0189 | +0.0288/+0.0530/+0.0621 |
     | state + temporal + residual | +0.0513/+0.0501/+0.0573 | +0.1525/+0.1772/+0.1630 |

     同一 outer test 上相对 observation 的增量：pairwise / temporal /
     prediction-error 在 3/3 seed 的 Brier、NLL 均更差；pending 增量为 0；
     combined 为 Brier **+0.0125/+0.0005/0.0000**、NLL
     **+0.0201/+0.0010/0.0000**。cross-family activation + predictive-rate
     等价审计已加入 probe；1500-frame smoke 中识别到 1 个别名组，仍需在大样本
     上确认并合并。→ 新 bank 证明这些视图各自能抽出同分布 pain 预测器，
     但目前没有稳定证据表明它们给 observation 增添信息；不接运行时。
     结果仍是同一交互环境分布上的 4-step pain，不代表阶段转移或决策价值。
   - 单测覆盖 episode 支持、episode-weighted 阈值、训练范围/OOD UNKNOWN、
     训练阈值冻结、Pareto 校准、时间差分重置和残差因果对齐；全量
     **714 tests** 通过，`py_compile` 与 `git diff --check` 通过。
   - 单测覆盖 episode 支持、训练阈值冻结、Pareto 校准、OOD 弃权、时间差分重置
     与残差因果对齐；全量测试待本轮最终复跑。
   - **在线自监督 learner（`online_feature_learner.py`，2026-11-25）**：
     从真实转移流按 bounded episode reservoir 发现 typed threshold/AND/OR
     候选，候选定义冻结后只用后续 episode 的延迟 pain 标签验证；世界模型
     residual 按 predict-before-update 生成。episode 尾部未观测满 horizon
     时记为 censored，不当作安全；pending/history 不一致时弃权。验证按
     episode 汇总 Brier/NLL 增益、双侧支持和保守下界，达到规定的有标签
     episode 数后只判定一次，支持不足则到 `verification_timeout` 拒绝。
     validated 特征遇到 OOD 或滚动增益退化会降级；`UNKNOWN` 不补伪读数。
   - `NodeRuntime` 的真实 `observe`、`learn_transition`、`learn_episode` 与
     `evaluate` 路径可选接入；`--online-feature-learning` 默认关闭，批量
     `fit` 重放只在第一遍把 episode 计入在线统计。readout 目前只输出独立
     pain/support 状态，不进入 planner 排序或最终价值。
   - **当前 prequential held-out 实测**（训练 seed 100 / 3000 转移、74 episodes；
     eval worlds 120–127、8 episodes / 200 rows；预测未来 4 步 pain）：199/200
     有支持读数；pooled AUC **0.565**、Brier **0.127 vs base 0.134**
     （skill **+0.051**）、NLL gain **+0.020**。episode-macro Brier skill
     **+0.054**（95% t 区间 **−0.046 到 +0.154**），NLL gain **+0.020**
     （区间 **−0.020 到 +0.061**）；后 5 个 episode 的 skill 均为零附近或负。
     训练期 270 个候选中 7 个通过初始验证，但 7 个都退化；260 个被拒绝，
     3 个仍在验证。→ 目前没有稳定的预测增益证据。该评估是逐 episode 在线
     更新的 prequential 流，不是冻结模型的独立静态测试；样本也仍小。
   - 定向测试 **9 passed**；全量 **723 passed**；`py_compile` 和
     `git diff --check` 通过。120 条交互转移、2 epoch smoke 得到
     `transitions=120`（未把重放重复计数）、4 个 episode；短 smoke 提出到
     256 候选预算且仍在 verifying，因此仅验证接线/时序，不代表候选已可靠
     或有决策增益。后续需用足够多独立 episode 评估候选验证率、OOD/退化率、
     候选冗余和跨环境质量，再讨论 planner 接入。
2. **[x] P0：当前状态到历史阶段 resolver（`stage_state_resolver.py`，2026-11-24）**。
   从 runtime 自己的 prototype（不是 task 几何/欧氏身份）学习
   `state prototype -> 实际包含该状态的 stage key` 的分布；`fit_rows` 只吃
   train episodes，`resolve` 用冻结表与半径，绝不更新。UNKNOWN 带原因：
   `no_prototype / outside_radius / unseen_prototype / insufficient_support /
   ambiguous`。半径取训练最近距离分位与 runtime `birth_threshold` 的较大者
   （超出即 runtime 会新建 concept，不能冒领旧阶段）。
   - episode-held-out 审计（`stage_resolver_probe.py`，fold-refit 表示，
     seed 100，5 折，被动 5000 帧/128 局）：**coverage 0.468、supported
     accuracy 0.740、同 supported 子集常量 key 基线 0.435 → +0.305**，
     5/5 折同向（0.720–0.780 vs 0.476–0.541）。`min_episodes=1` 对照：
     coverage 0.571、accuracy 0.735、基线 0.449 → +0.286。
   - UNKNOWN 归因（min_episodes=2）：outside_radius 1400、
     insufficient_support 829、ambiguous 426、unseen_prototype 7。
     **约一半状态被 OOD 门拒绝**是当前主要限制；下一步预测等价阶段树
     应把碎片化的 stage 类型/原型证据池化。
3. **[x] P0：预测等价的多级阶段树（`predictive_stage_tree.py` +
   `stage_tree_probe.py`，2026-11-24）**。叶节点是真实 stage key；每个节点的
   档案由**真实** successor/terminal 分类分布、log-duration 高斯、
   feedback 对角高斯组成，全部用经验贝叶斯收缩到全阶段 base。
   merge 只在 validation episodes 上通过预测似然门才接受：pooled 的
   joint log-lik 必须 ≥ 分开模型 − `tolerance`，且该对至少有
   `min_validation_support` 个 validation stage；因此父节点是预测等价
   决策，不是固定 K、task geometry 或原始状态欧氏相似。未知 stage key
   没有叶节点，只能回退到全阶段 base（或 UNKNOWN），不会编造兄弟。
   修过两个真实边界：`-inf - -inf → nan` 绕过 merge 门（现在非有限增量
   不参与 merge），以及未见 outcome 的零概率（现在分类似然有 1e-6 下限）。
   - episode-held-out 审计（60/20/20 episode split，被动 5000 帧；test
     253/243/245 stages；3 seeds）：
     - **阶段档案本身强于 base**：joint NLL 树/叶 −4.41/−3.87/−3.48 vs
       base −5.89/−6.05/−5.43；top1 0.57–0.60 vs 0.37–0.38。
     - **默认支撑门（2/2）下 merge 回退几乎不被触发**（252/253、242/243、
       244/245 在叶层），树 vs 叶 joint Δ = **+0.015 / 0.000 / −0.005**
       → 不稳定。
     - **hi10 门（10/3）下树被真正使用**（level 1–3 有 9/9/7 个 test
       stage），joint Δ = **+0.023 / +0.136 / +0.083（3/3 正）**，
       top1 +0.000/+0.021/+0.012。
   - 判定 `stage_tree_pools_when_support_is_thin`：merge 对象真实、阶段
     档案有强 held-out 信号，但**树的增量只在低支撑门显形**，不是默认
     收益；hi30 极端门（seed 100）反而略差。树只作为条件回退层，不作
     决策门证据。
4. **[x] P0：内部高层需求选择器（`stage_demand_selector.py`，2026-11-24）**。
   `StageOutcomeReadout` 只用真实阶段结果训练：按 episode 折外拟合
   `StageOutcomeMemory`，对每个真实 source 阶段取它实际走到的 sibling 的
   预测特征，目标是**之后的真实净反馈和**（dopamine − pain，逐阶段累加；
   不是反事实）。特征包括概率、目标阶段 log-duration、feedback 均值/方差、
   terminal/censored、目标阶段 terminal/censored rate、support；岭回归，
   截距不惩罚。`StageDemandSelector` 枚举 memory 支持的 sibling，过滤
   terminal/censored（含目标阶段曾 terminal/censored 的 outcome），按学到的
   value 取最大者生成 `StageDemand`；无支持 / readout 样本不足 / 只有终局
   兄弟 → None。报告 value、runner-up、margin、candidates 与
   `provenance=world_feedback_outcome_readout`，`affects_value=False`。
   - 同时修正了 `StageOutcomeMemory` 的一个真实语义缺陷：outcome 的
     duration/feedback 原来取**源阶段**，现在取**目标阶段**（terminal/
     censored 无目标时取源阶段收尾），并新增
     `destination_terminal_rate / destination_censored_rate`；选择器与
     `StageDemand.from_transition_prediction` 都会拒绝曾终局的目标。
   - 合成测试 7 条；**在线 provider / planner 接线未做**：选择器是纯函数，
     需要等状态→阶段 resolver + 在线阶段上下文 provider 接上后再走
     rank-only 质量门（item 6）。
5. **[x] P1：需求关联及 option 口径（`stage_demand_association_probe.py`，
   2026-11-24，机制审计完成；可作条件信号，覆盖仍低）**。
   - 新增 `feature_source`：`branch`（诱导 feature roots，稀疏）或
     `prototype`（runtime 最近原型，稠密且对任意候选状态可用）；
     `--stage-feature-source`，census 记录来源。
   - 新增 `DemandAssociationSample.from_stage_prefixes`：每个 transition 的
     option 只取**该 transition 之前**已执行的动作（不含即将作用/后续动作），
     offset 0 无历史按 pooled；`StageDemandHeuristic` 的 option 查询只做
     同一前缀的 suffix backoff，**绝不回退到 pooled 表**。
   - episode-held-out 审计（train-only representation；3 seeds；exact vs
     tree-root context，pooled vs causal prefix option）：
     - `branch` 覆盖率 **0.003** → 不可用；`prototype` 覆盖 1.000。
     - exact+pooled：realized 覆盖 0.33–0.36，top1 0.78–0.84 vs chance
       0.85–0.89 → **低于 chance**。
     - exact+prefix：覆盖 0.18–0.24，top1 0.72–0.76，仍低于 chance。
     - tree-root+pooled：top1 ≈ chance（−0.006/+0.002/+0.009）。
     - **tree-root+prefix：3/3 高于 chance**——coverage
       0.178/0.187/0.236，top1 **0.861/0.838/0.837** vs chance
       0.836/0.825/0.817，majority 0.703/0.641/0.704；realized priority
       +0.016/+0.021/+0.013 vs others −0.119/+0.008/−0.055。
       放宽到 1/1 门（seed 100）：coverage 0.409，top1 0.902 vs chance
       0.883。
   → 判定 `tree_root_prefix_association_above_chance`：阶段树根池化 + 因果
   前缀 + suffix backoff 是可行机制，但绝对增益只有 +1.3–2.5pp、覆盖
   18–24%（放宽后 41%）。**不能单独作为 rank hint 接线**；item 6 需要
   覆盖门和在线 prefix/context provider，且优先扩大覆盖而不是调 prior。
6. **[x] P1：rank-only 搜索质量门（已完成首轮，未通过）**。
   新增 `stage_online.py`（在线因果阶段 context：每步重放冻结 detector/segmenter
   到当前 prefix，不读未来动作）与 `OnlineStageDemandProvider`（option-pooled
   memory 预测 → selector 选非终局目标 → 映射 stage-tree root → 当前 partial
   stage 内有效，边界变化即重算；UNKNOWN 不注入）。同时补
   `StageOutcomeMemory.predict_pooled`、`from_stage_roots` 根级样本、
   `make_node_root_rank_step`、`rank_stats`/`plan_out.candidate_count` 诊断。
   固定预算 gate（`stage_rank_gate_probe.py`：同 beam/horizon/cap，seed 100，
   3 worlds，beam 4，horizon 8，cap 400）：
   - 默认 2/2 门：provider 57/75 决策产出 demand，但 heuristic 只支持
     **0–2 / 3078** 候选状态 → rank 空转，frames 25.0 = 25.0，
     扩展 54/决策不变。
   - 根级 1/1 门：feature present **1990/2322**、heuristic supported
     **1608**、nonzero **1442**、target present **440** → rank 真正激活；
     但 **frames 25.0 → 18.3**（仍 3 worlds、beam 4），候选/决策 54 不变。
   - **5 worlds / beam 16 复核（同 1/1 根级）**：baseline **73.8**、
     ranked **32.2**（−41.6，−56%）；provider 78/161 决策有 demand，
     heuristic 8853/12324 支持、target present 2214，扩展仍 158/决策。
     → 负向结论不是小样本/窄 beam 的偶然。
   - **置信/覆盖门（target probability ≥0.6、prediction support ≥4）**：
     provider 仍有 72/161 决策产出 demand（low_support 26、low_probability 6），
     heuristic 8456/11376 支持，但 ranked frames **仍为 32.2**（baseline 73.8）。
     → 不是低置信 demand 的问题，而是 rank 目标/排序语义本身会把 beam
     带离高价值路径。
   - **近并列 rank 语义（`rank_mode=tie`）**：只让 priority 在
     `score[0] ≤ best + rank_margin` 的带宽内重排。beam16/5 worlds、1/1
     根级下，margin **0.0 → 32.4**、**0.05 → 32.2**（baseline 73.8）。
     仍严重伤害 → 不是 priority 覆盖了明显更优 score，而是中间分数并列/
     目标语义本身；tie 模式不能救。
   - target-path recall（该次历史评测）：当时仅有候选级
     `target_present` 计数，完整 frontier trace 尚未实现；后续已加入只读
     stage-tagged frontier 统计，DAgger 对照结果见本节后文。
   - 修掉两个实现 bug：根级 adapter 对状态**双重缩放**导致 0 特征；
     feature 无证据时未继续向父 context 回退（现在会回退）。
   → 判定 `rank_hint_activates_only_under_weak_support_and_hurts`：
   **质量门未通过，不接线**；5 worlds/beam 16 的复核进一步把负向幅度放大到
   −56%，置信/覆盖门也没有改善。下一步不是调 heuristic 权重，而是：
   ① 用 stage-tagged frontier 日志做真正的目标路径 recall；
   ② rank 语义重设计：近并列（tie）已试并失败（32.2–32.4 vs 73.8），
      下一步应转向可验证的 value target，而不是继续改 priority 排序；
   ③ demand target 质量与支持/置信度校准完成后再重跑同预算 A/B，
      3/6-seed 通过前不进入 shadow。
7. **[x] P1：非帧级 option 搜索（完整枚举基础层，2026-11-24，已落地、未接线）**。
   新增 `stage_option_search.py`：把 `StageOutcomeMemory` 的阶段→阶段转移
   组合成**完整路径**。这是非帧转移的第一个决策侧消费者，目标先正确性、
   不做剪枝：
    - 节点 = 阶段上下文，边 = 真实执行过的 option，边结果 =
      `(destination, terminal/censored, duration, feedback)` 的完整分布；
    - **无 beam / top-k / 分数剪枝**：每个节点展开 memory 支持的全部 option，
      每条边的全部 outcome 都被跟随；
    - 每条边展开前校验 outcome 概率和为 1（`max_edge_deviation` 记录），
      不满足即抛错；
    - terminal/censored 立即终止；到 `max_transitions` 记 `open`（深度截断，
      不是未知转移）；无支持 option 的节点记 `unresolved`，**不重归一化**；
    - `min_context_depth=1` 拒绝只靠 root 池化的预测：root 是跨源阶段的
      边际，不是本阶段的转移，不能拿来编造边；
    - 超过 `max_leaves` 默认抛 `StageOptionSearchCapacityError`，只有显式
      `allow_truncation=True` 才允许截断并标 `complete=False`，不会静默丢分支。
   探针 `stage_option_search_probe.py`（seed 100、1500 步、depth 2）：
    - memory 9 类 / 71 个 option / 1059 条支持边；
    - 离线 118 次搜索：13361 leaves（terminal 2775、open 10586、censored 0）、
      单次最多 166 leaves / 128 plans、**terminal 可达 106/118**、
      unresolved 30 节点（平均未解析质量 0.11）；
    - 在线 40 个真实决策：5220 leaves（terminal 1036、open 4184）、
      最多 163 / 127、**terminal 可达 38/40**、unresolved 4（0.04）；
      33086 次 option 查询中 4106 supported、28980 unknown、8476 次 root
      回退被拒；**每个阶段上下文平均 8.8–9.9 个 option 受支持**
      （词表 71、约 62 个 unknown/节点），说明阶段级搜索空间是稀疏但可枚举的；
    - 2 次阶段转移的期望 elapsed ≈ **8.0 帧**（单阶段中位 4），说明当前
      8 帧视界只能容纳约 2 个阶段——粗层的自然单位是“阶段”，不是“帧”。
   仍缺（下一层，不在本项）：option 归纳（当前只支持 exact trace）、
   duration 的生存/竞争风险似然、阶段级价值、option 执行器与边界重规划。
   **未接 planner，默认路径逐位不变**。
8. **[x] P1：停时分布（stop-time distribution，2026-11-24，已完成、未接线）**。
   目标：把 duration 从经验分位数升级为**离散停时分布**，并让搜索按分布
   合成，而不是把均值相加。这是“概念级转移”的第一步：时间是输出，不是
   帧计数。
   - 数据层：`_OutcomeCounts.summary` 增加
     `duration.distribution = ((t, p), ...)`（真实 duration 直方图归一化）
     与 `duration.count`；mean/median/q90/min/max 保留为描述统计。
   - 搜索层：`OptionStep.duration_distribution`、
     `OptionLeaf.elapsed_distribution`；路径 elapsed = root 偏移 + 各
     **后继边** duration 分布的离散卷积。terminal/censored 边仍不加时长
     （它描述当前阶段，避免重复计数）。
   - 契约：每个分布概率和为 1；缺失分布直接抛错（不用均值伪造点质量）；
     卷积支持集超过 `max_elapsed_support` 抛
     `StageOptionSearchCapacityError`，不静默截断。
   - 提供 `distribution_mean` / `distribution_quantile` /
     `survival_curve`（P(T > t)），供后续 value/executor 使用；
     `PlanSummary` 增加按 outcome 质量加权的 elapsed 分布。
   - 门：单测验证直方图、卷积、偏移、逐叶归一、plan 聚合与容量拒绝；
     probe 增加 elapsed q90 与 P(elapsed ≤ horizon) 读数。
   - 结果（seed 100、1500 步、depth 2）：offline 118 次搜索 elapsed
     mean **8.0 帧**、q90 **14.6 帧**、**P(elapsed ≤ 8) = 0.703**；
     online 40 决策 mean **7.8**、q90 **14.35**、within-8 **0.712**；
     卷积支持集最大仅 18 个时间点。→ depth-2 阶段计划里约 **30%** 的
     时长质量超出当前 8 帧视界，q90 约为视界的 1.8 倍：这从停时上确认了
     “8 帧视界 = 约 2 个阶段、且经常装不下”的判断。
   - 仍不做：competing risk / 删失校准、跨任务迁移；这些需要 held-out
     生存审计，后续单独开项。
9. **[x] P1：option 等价类（predictive option quotient，2026-11-24，已落地；直方图反馈后门通过）**。
   新增 `predictive_option_quotient.py`：把同一源阶段下、后继分布/时长/反馈
   可预测等价的 action trace 合并成类；类按源阶段作用域；合并是训练档案上
   的模型选择（每样本最多损失 `tolerance` nats），整个商在 held-out episodes
   上接受/拒绝。`StageOptionSearch(option_quotient=...)` 切换 class 模式：
   每个类一条分支，使用类池化分布；被拒的商禁止使用。
   - 第一版用 Gaussian 反馈模型：seed 100、1500 步、depth 2、86 traces 下
     所有 tolerance 都被拒（Δ −0.126…−0.235）；通道分解 categorical +0.015、
     duration −0.013、**feedback −0.128** → 拒绝几乎全部来自反馈通道。
   - **改为 per-dimension quantile histogram（8 bins）后**：tolerance 0.0 →
     86→67 classes（−22%）、**accepted=True、Δ +0.015**；tolerance 0.05 →
     86→**59 classes（−31%）**、**accepted=True、Δ +0.024**；
     0.005–0.02 → 63 classes、Δ −0.009（仍拒，边界内）。判定
     `option_classes_accepted_with_histogram_feedback`：可行性问题来自
     反馈的 Gaussian 方差惩罚，不是后继分布不可合并。
   - class 模式首次跑通（tolerance 0.05，71 次验证搜索）：leaves
     **102→205/搜索**、plans **83→181**、terminal 可达 **62→64**、
     unresolved **25→0**、elapsed q90 **14.1→16.7**、within-8
     **0.720→0.571**。→ 覆盖与可达性变好，但分支数翻倍、计划变长。
   - 仍只做离线搜索，**未接 planner**；下一步是同预算 trade/决策 A/B，
     或给 class 模式加 value 以处理翻倍的分支。
   - 仍不做：跨任务 option 归纳、unseen trace 到已有类的因果映射（当前
     unseen trace 仍 UNKNOWN）。
10. **[~] P1：阶段定义扩充（内生静态转移 / 共变 bundle / 相似度分层，2026-11-24）**。
    用户方向（修正理解）：**静态扩展的本质是一次特征提取，也就是内生的、
    非时间性的状态转移**——用系统已学到的状态特征、节点关联等，在**同一
    时刻**把当前状态整合成更丰富的表示，功能是“整合当前状态信息”；它应当
    广泛存在于每个状态。这与**外延的、向后推进的时间转移**相反：后者是
    `(s,a) -> s'`，推进时间；前者是 `s -> E(s)`，不推进时间。
    - A（已完成）：
      - `static_transition.py`：`endogenous_transition` 只沿**非时间边**
        做闭包：association（对称共现）、membership（member→memory，反向
        不展开）、derivation（source→derived，反向不回填）；**绝不跟随
        transition 边**。强度按 hop 衰减、`min_strength` 截断、`max_steps`
        限制；`StaticState` 暴露节点强度/kind/depth、`vector` 与 `change`。
      - `stage_change_bundles.py`：消费 `E(s)` 序列，按 `ΔE` 的符号 token
        挖掘反复共变的 bundle（episode 支撑、bundle 大小、lift 门），
        关键节点权重=逆变化频率；bundle 匹配生成边界，`segments` 输出区间
        与加权剖面（供 B）。
      - `static_transition_probe.py`：对比 depth 0/1/2/3 的整合规模、kind
        分布、边类型计数，以及整合后共变 bundle 的丰富度。
    - 结果（seed 100、1500 步、38 局、367 个状态）：
      - depth 0（仅 seed）：7.51 节点/状态、26 个独立节点、76 个 bundle
        （47 个 size-2、29 个 size-3）、边界 4.95/局；
      - depth 1：11.31 节点/状态（+50.6%）、30 个节点、122 个 bundle
        （78 size-2、44 size-3）、边界 7.13/局；边计数只有
        **derivation 2352**（association/membership 为 0）；
      - depth 2/3 与 depth 1 完全相同、converged=1.0 → 当前图上的内生闭包
        **一跳饱和**。
      → 机制成立，且整合后共变 bundle 更丰富（size-3 从 29 增到 44）；
      但当前 runtime 图的非时间结构主要是 derivation 边，association/
      membership 几乎未参与，闭包深度只有 1。要得到真正多跳的“状态整合”，
      需要更密的关联/记忆边，或把 membership/association 的强度学习接上。
    - 旁证（`stage_change_bundle_probe.py`，prototype 快照源）：57 个 bundle，
      全部 size-2 的 prototype 交接 `(concept_i,-1)/(concept_{i+1},+1)`，
      3.6 边界/局——说明二值 prototype 源本身不足以表达多节点协同。
    - A2（已完成，`static_graph_strengths.py`，2026-11-24）：
      - 从真实共激活学习 association 的 Jaccard 强度（支撑门）、membership
        的结构绑定强度；**只读 overlay，不写 runtime 图**。
      - `endogenous_transition(strengths=...)` 用 overlay 替换原始边权，
        支持多跳；新增可选**度归一化**（每节点出边质量归一），防止 hub
        淹没状态。
      - 结果（seed 100、1500 步、367 状态、30 个独立节点）：

        | 变体 | 节点/状态 | 边 | bundles | size-3 | 边界/局 |
        |---|---:|---|---:|---:|---:|
        | graph d1 | 11.31 | derivation 2352 | 122 | 44 | 7.13 |
        | learned d1 | 21.83 | assoc 54565 | 112 | 43 | 6.16 |
        | learned d2 | 24.59 | assoc 253349 | 87 | 33 | 5.05 |
        | normalized d1 | 12.62 | assoc 54565 | 119 | 43 | 7.13 |
        | normalized d2 | **12.98** | assoc 83597 | **128** | **50** | 7.03 |

      - 判定：原始强度会**过度整合**（d2 覆盖 24.6/30 节点，bundle 反降到
        87）；**度归一化**把整合控制在 12.6→13.0 节点、闭包到 d2–d3，且
        bundle / size-3 最高（128 / 50）。→ 可用变体是 **normalized**。
        membership 有 63 条结构绑定，但当前 feature seed 不是 memory
        member，实际未触发（association 是主通道）。
      - 下一步 B：在 normalized `E(s)` 剖面上做相似度聚类与 held-out 门。
    - B（已完成首轮，`stage_type_clusters.py`）：把 `E(s)` 剖面按关键节点
      加权余弦相似度做 complete-linkage 聚类；类型数不预设，合并到相似度门
      以下停止；held-out episodes 上评估 coverage / margin / 预测增益
      （type 条件下的下一 stage 分布 vs 全局 base）。结果（seed 100、
      1500 步、38 局、366 samples、295 train / 71 validation）：

      | min_sim | types | coverage | margin | gain | 门 |
      |---:|---:|---:|---:|---:|---|
      | 0.50 | 19 | 1.000 | 0.119 | **+0.211** | 过 |
      | 0.70 | 43 | 1.000 | 0.088 | **+0.232** | 过 |
      | 0.80 | 66 | 0.915 | 0.074 | +0.204 | 过 |
      | 0.90 | 109 | 0.732 | 0.069 | +0.314 | 过 |
      | 0.95 | 140 | 0.521 | 0.071 | +0.401 | 过 |

      → 所有阈值都过门；**0.70 是全覆盖 + 最高增益的平衡点**（43 types、
      +0.232）；更细的类型提高增益但 coverage 下降（0.95 只剩 0.52）。
      这是 1 seed / 71 验证样本的方向性结果；类型数偏多（43/295），下一步
      需要用更大样本或层级合并控制碎片化。
    - C（已完成首轮，`stage_type_layers.py`）：对 `E(s)` 建 complete-linkage
      合并树，按 held-out 门选择前沿；选择规则是“在 coverage/gain 过门的
      前沿里，取层数最少且 gain 在最优 `tolerance` 内的那个”。结果
      （seed 100、1500 步、38 局、295/71）：

      | sim | tol | layers | coverage | gain |
      |---:|---:|---:|---:|---:|
      | 0.5 | 0.01 | 168 | 1.000 | +0.281 |
      | 0.5 | 0.05 | 36 | 1.000 | +0.235 |
      | 0.5 | 0.10 | 17 | 1.000 | +0.202 |
      | 0.5 | 0.20 | 3 | 0.986 | +0.111 |
      | 0.9 | 0.05 | 19 | 0.690 | +0.344 |

      → 所有设置都过门；**tolerance 是真正的层数控制**：0.05 → 36 层、
      全覆盖、+0.235（接近 B 的 43 types/+0.232）；0.10 → 17 层、全覆盖、
      +0.202；0.20 → 3 层、+0.111。held-out 门只能验证“可达且有用”，
      不能单独决定层数；下一步需要跨 seed/折稳定性或下游决策目标来选
      tolerance。1 seed / 71 验证样本，方向性结论。
      - **跨折稳定性门（`stage_layer_stability.py`、
        `layer_stability_probe.py`）**：5 折 × 3 fold seeds。任一层数配置
        在折间的 layer count 从 **2 到 137** 剧烈变化，agreement 仅
        **0.20–0.40**；**没有一个候选达到 0.60**，3/3 fold seeds 均无
        稳定候选，尽管 mean gain 仍为 0.13–0.39。
        → 判定 `layer_count_not_stable`：当前数据量下相似度层级不可识别，
        **C 不接线**。需要更多 episode，或最小层支撑/高相似度约束后重跑。
    - B 决策侧（已完成首轮，`stage_type_value.py` +
      `make_stage_value_penalty` + `stage_type_value_gate_probe.py`）：把类型
      值作为**软终点项**加到 terminal cost（`stage_value_weight`，默认 0），
      同预算 gate（seed 100、beam 4、3 worlds、cap 400）：
      - 类型值模型：43 types、coverage 1.0、gain +0.232（B 的已过门模型）；
      - weight 0 / 0.1 / 0.5 → frames **25.0 / 25.0 / 25.0**，完全相同；
      - 终点候选的 penalty 在单次决策内 distinct 平均 **1.29/4**（max 2），
        即预测终态几乎总是落到同一个 tree root/type。
      → 判定 `stage_type_endpoint_value_inert`：终点项没有杠杆，**不接线**。
      下一步若继续，只能在中间 beam 逐步打分（更接近已失败的 rank hint），
      或改用 stage-level value 而非 frame endpoint。
    - Stage-level value + executor（已完成首轮，`stage_executor.py` +
      `stage_executor_probe.py`）：在阶段边界用 `StageOptionSearch` 选第一个
      option，按期望净反馈 − terminal/duration 罚分排序并执行其动作 trace；
      非边界/无 option 时回退帧级 planner。同预算（seed 100、3 worlds、
      cap 400、horizon 8、exhaustive baseline）：
      | 变体 | baseline | executor | passes |
      |---|---:|---:|---:|
      | full trace | 101.3 | **23.7** | 0 |
      | one action | 101.3 | **33.3** | 0 |
      | class mode | 101.3 | **34.7** | 0 |
      → 判定 `stage_executor_unsafe_without_frame_safety`：被选 option 在
      stage memory 里 terminal mass=0，但执行期间仍会死；阶段级安全不等于
      帧级安全。**不接线**。下一步必须先有帧级安全包络（在 fine model 下
      检查 option 动作的 hazard/feasibility，不安全即回退），或只允许 option
      作为软约束。
    - 视野诊断（`lookahead_diagnosis.py`，2026-11-24）：Flappy 的下一根柱子
      在 **~80 帧**外（spacing 0.8 / speed 0.01）。world model rollout 的
      **next_dy MAE**：1 帧 0.001、8 帧 0.009、32 帧 **0.046**、64 帧
      **0.101**（gap 0.34 的 13% / 30%）；next_dx：0.009 / 0.075 /
      0.114 / 0.234。阶段类型对 next_dx / next_dy 的 **eta² = 0.672 /
      0.885**，raw R² ≈ 1.0 → **信息没有丢**。但 8 帧 planner 和 depth-2
      阶段搜索（≈8 帧）只覆盖柱间距的 **~10%**；exhaustive 深度 20 需要
      9^20，不可行。弱策略没有 pipe collision，柱前 hazard 未测到。
      → 判定 `lookahead_insufficient`：**分层没有延长视野，只降低了决策
      频率**；长程外推误差在下一根柱子处约为 gap 的 30%。
      - 关键信息口径（`clearance_lookahead.py`）：32 帧 **next_dy 误差
        0.046 = gap 的 13%**，类型 eta² **0.885** → 按“关键信息保持一定
        准确率”的标准，**信息层是够的**；30% 是全状态 MAE（含 dx/next_dx），
        不是决策相关量。clear/hit 分类在当前弱策略下退化（到达柱子的样本
        全是 clear；碰撞/落地提前死亡被截断），需要探索性/oracle 策略才能
        测 hit。→ **瓶颈不是信息精度，而是规划深度与决策接口**。
    - 准确率–覆盖率曲线（`accuracy_coverage_probe.py`，2026-11-24，
      31 train / 7 validation 局、71 个验证转移）：
      - 整体：supported coverage **0.845**、top1 **0.817**、base 0.352、
        lift **+0.465**；
      - support 门：≥1/2 → 0.845 / 0.817；≥5 → 0.592 / **0.881**；
        ≥10 → **0.549 / 0.923**；
      - context depth 门：≥1 → 0.746 / 0.868（NLL 1.25）；≥2 →
        0.549 / 0.897（NLL 0.87）；≥3 → **0.394 / 0.893（NLL 0.198）**；
      - hazard 按 prototype 距离分桶：本 seed 的 AUC 0.53–0.74
        （每桶正例 0–8），**不复现 fast smoke 的 0.94**；安全包络的距离门
        还需要更多死亡事件。
      → 判定 `accurate_subset_is_support_and_depth`：**可靠子集约
      40–55%**（depth≥3 或 support≥10），top1 ~90%；depth 主要改善
      **校准**（NLL 2.96→0.20），support 主要改善 top1。消费者应按
      `depth≥2/3 + support≥5/10` 门行动，门外 abstain。
    - 选择性消费者（`stage_executor.py` 的 `min_context_depth`/`min_support`/
      `max_hazard`，2026-11-24）：只在可靠门内选 option，门外回退；
      同预算（seed 100、3 worlds、cap 400、horizon 8）：
      - 无门 / depth≥1+support≥2：**23.7**（fast，仍崩）；
      - **depth≥2 + support≥5**：fast **101.3 = baseline**、full
        **72.0 = 72.0**；gate rejects 25–53/局，option share 7%，
        hazard aborts 0。
      → 判定 `reliability_gate_safe_but_inert`：**门解决了安全，但没有收益**。
      原因：option 词表是**行为克隆**的 trace，在行为 trace 里选不可能超过
      行为策略本身。要收益必须做 **option improvement**（在 survival/
      clearance 目标下做阶段级动作搜索或合成新 trace），而不是继续加门。
    - 约束：节点强度/权重只来自 runtime 自身，不引入任务常量；未知节点
      保持 0，不伪造边；默认只读，不替换 detector/segmentation/planner。
    - **下一步（二选一，2026-11-24 收口）**：
      1. **Option improvement（阶段级动作改进）**
         - 目标：在“局部准确的转移记忆”之上，产生**不属于行为克隆**的
           阶段级动作，使 gated consumer 在可靠门内真正改善 frames/passes；
         - 候选实现：
           a. 在 survival/clearance 目标下对 option 动作做**阶段级搜索**
              （同一 start、同一 stop-time 约束下比较动作串）；
           b. 从真实反馈中**合成新 trace**，而不是从行为流里选 trace；
         - 门：可靠门（depth≥2 + support≥5）保持开启；门外回退 fine
           planner；同预算 baseline vs improved-option，3-seed 筛选、
           6-seed 确认（≥5/6 同向）；不通过则关闭 executor 线。
      2. **把阶段层定格为只读基座**
         - A/A2/B 只保留 readout：stage types、outcome memory、resolver、
           stop-time、coverage/accuracy 曲线；
         - 关闭并归档 rank、endpoint value、executor 三条决策侧线，保留
           负结果与诊断（rank −56%、endpoint inert、executor safe-but-inert）；
         - 默认栈、8 步基线、planner 路径保持不变；后续只在新机制
           （不是本转移记忆）上重新开决策侧。
11. **P1：真实数据与决策验收**。resolver / stage tree 已有单任务被动流的
   episode-held-out 审计（5 折 / 3 seeds，fold-refit 表示），但仍无
   planner A/B、跨 seed 决策稳定性、在线 provider 校准或 shadow 运行；
   所有表示/阈值/identity/demand/heuristic 只在训练 episode 拟合；先 shadow，
   再 3-seed 筛选、6-seed 确认。UNKNOWN 不作安全结论。

**阶段记忆本身的未完成项（此模块目前不宣称已经学到任务阶段）**：

1. 当前 detector 只覆盖观测变化、反馈变化和调用方可提供的 prediction
   error；尚未加入新颖度、action-conditioned 转移分布变化或多信号边界
   校准。quantile detector 已在 fast 真实流 smoke 上跑通（38 局/1500 帧，
   9 个 channel），但尚无 held-out 阈值稳定性或边界质量校准。
2. stage identity 仍是精确 key-signal token 组合，但已补两层：连续状态→历史
   stage type 的 resolver（fold-refit 表示、held-out 冻结、UNKNOWN 带原因）
   和 held-out 预测等价的 stage-type merge 树。**仍未实现**：从连续状态直接
   合成新的 stage 身份（当前只会检索已见 key 或回退 base）、跨签名的 split、
   以及多任务流上的稳定 identity 校准。
3. 选项 trace 目前严格精确匹配；已实现 predictive option quotient（按后继/
   时长/反馈等价合并 trace），Gaussian 反馈模型曾被 held-out 门拒绝
   （Δ −0.128），改用 per-dimension histogram 后门通过（Δ +0.024，
   86→59 classes）；class 模式目前只做离线搜索，未接 planner；仍无
   unseen trace → 已有类的因果映射，新 action trace 一律 UNKNOWN。
4. duration 目前报告经验分位数与离散分布（`duration.distribution`），
   搜索按分布卷积；删失样本单独标记，但**尚未**实现 competing
   risk / survival likelihood 或 held-out 校准，不能把这些分布当作
   已验证的生存预测。
5. 已在单任务真实被动流上做 episode-held-out resolver（5 折）与阶段树
   （3 seeds、60/20/20）审计；**仍未做**：duration 生存校准、feedback /
   terminal 的联合校准、父级收益的多种子显著性、root 常数退化检查，以及
   跨任务流验证。fast smoke 只验证管线可运行与计数。
6. 自动管线已把真实 rows 变成 stage / demand 样本，resolver / tree /
   demand readout 也都在 `--stage-memory` 下生成。`OnlineStageDemandProvider`
   与 `evaluate(stage_demand_provider=...)` 已实现，但必须由调用方显式构造
   和传入；标准默认路径仍不消费 stage memory。provider 接线可运行不代表
   目标路径召回或行为收益已过门。

**审查修复（2026-11-24）**：阶段树 validation detector 现在只在训练
episodes 上拟合，held-out episodes 用冻结阈值 transform；readout 的 OOF
训练和线上查询统一使用 pooled outcome memory。`StageExecutor` 仅接受单阶段
评分（多阶段续策尚无定义），把 terminal/censored mass 纳入风险门；任何
unresolved 或不完整/超容量搜索都弃权并回退。Resolver UNKNOWN/OOD 不再通过
最近 prototype 映射成阶段提示或价值。新增组合边界回归测试；全量
`unittest` **690 passed**。以上均保留 opt-in，不改变默认 planner。

**决策近视目标诊断（2026-11-24，6 帧 screening）**：

- 同一评估世界配对，训练 seed 100/101/102，各测 eval seed 120/121/122；
  1500 train steps、400-frame cap、horizon 6，executor 使用
  `context_depth >= 2 + support >= 5` 门。三个训练 seed 的 baseline →
  executor 平均 survival 分别为 **93.7→93.7、70.3→42.0、113.7→117.0**；
  9 个配对结果为 **5 平、3 退化、1 改善**（最大退化 45 帧），总体均值
  **92.6→84.2**。Option share 仅 **5%–18%**。结论：当前 executor 没有
  可靠地解决近视；可靠性门仍不能把行为克隆 option 变成改进行动。
- 深度 2 的 stage-search 只读审计（seed 100）显示 option-sequence
  条件分布的等权 macro：elapsed mean **8.30 帧**、q90 **10.32 帧**、
  已知路径的 `P(elapsed <= 6)` macro **0.477**；40 个在线 context 中，
  深度 1 时 terminal outcome 至少对一个已见 option 可达的有 **36** 个。
  这证明阶段时间尺度能越过 6 帧，但不是“无可行解”的校准预测，也不证明
  存在可执行的更优 option。
- 诊断中发现旧 `stage_option_search_probe.summarize` 把不同 action option
  的条件概率相加（terminal mass 可超过 1）；现已改成逐 option sequence
  独立统计并显式标为等权 macro，增加回归测试。当前全量
  `unittest` **691 passed**。
- 额外做了 eval-only exact-world 分支抽查：3 个训练 seed × 3 个 held-out
  世界，共 72 个靠近首个通行点的状态（距事件约 8–78 帧）。64/72 个状态
  中，两种首动作的所有 6 帧续串均存活；62/72 个状态的 6 帧存活比例几乎
  相同（差 <0.05），而 15 帧续串存活比例已相差 ≥0.05。柱子事件本身仍约
  78 帧外，因此 15 帧只是动作敏感性的早期代理，不是事件通过率。被冻结的
  pooled stage outcome 在距事件约 64–69 帧时通常支持但 `terminal_rate=0`；
  风险多在约 19–29 帧才上升，部分支撑仅 2–11 episodes。分支计数是均匀
  枚举动作串的评估指标，不是策略概率或学习标签。→ 进一步确认：短帧域的
  死亡可行性饱和，而现阶段 outcome memory 没有提供足够早且动作条件化的
  区分；粗层需要对更远事件建模并有可改进 option，不能只堆叠当前短阶段。
- **[x] P0 executor option trace 与实际动作坐标对齐（2026-11-24）**：
  新增 `executable_option_suffix`，按“已作用的 stage 前缀 + newest-first
  issued queue 的 oldest-first 顺序”校验候选 option；只把尚未提交的 tail
  作为新 issued actions。Option quotient 还要求全部 class member 都匹配已
  提交前缀，否则 UNKNOWN/回退。lag=0/3、边界动作错配、pending 错配和
  class 成员不兼容均有回归测试。定向 **58**、全量 **697** tests 通过。
  修复前后同 seed 100 / 3 eval worlds：survival 均值都为 **93.7**；
  option 选择由 **4.7 次/局、share 5.4%** 降至 **1.3 次/局、share 1.4%**，
  新 alignment rejects 为 **3–5/局**。这验证了旧路径确有不兼容候选、修复后
  安全回退；尚无收益证据，需与正确 action schedule 下的新实验区分。
- 前缀修复后的门限敏感性筛查（seed 100、eval 120/121/122、horizon 6）：
  `depth>=2 + support>=5` 时 option share **1.4%**，survival 与 baseline 持平；
  放宽为 depth/support=0 后 share **8.9%–17.4%**、alignment rejects
  **31–47/局**，三世界 baseline→executor 分别 **72→135、137→46、72→72**
  帧，均值 **93.7→84.3**。执行 option 状态 OOD rate **0.944**，fallback
  OOD rate **0.973**。→ 严格门把系统压成近似 inert；宽松门放出高方差、无一致
  收益的动作。仍是 1 个训练 seed/3 个 eval world；需先解决候选 option 的
  支撑与长提前量，不能靠继续放宽门解决。
- 复跑当前 online rank gate（训练 seed 100、eval 120/121/122、horizon 6、
  beam 4）：root 模式 provider 需求 **73/87** 次可用，但候选阶段特征只有
  **21/2774**、目标特征 **5/2774**；prototype 模式启发式仅支持
  **26/2736** 个候选状态。三世界平均 survival：baseline 25，root rank 29，
  prototype rank 25；passes 都为 0。样本不足以作显著性判断，但明确显示
  provider 产出率远高于候选状态可识别率；rank hints 没有足够 frontier
  覆盖，目标路径 recall 仍未测。
- **DAgger 数据上的 rank 正向 screening**（`stage_rank_gate_probe.py` 已支持
  同样的 DAgger 参数，prototype feature，3 train seeds × eval 120/121/122，
  horizon 6/beam 4）：每个训练 seed 的 baseline 都是 `[25,25,25]` 帧；rank arm：
  seed100 `[37,135,25]`、seed101 `[56,71,34]`、seed102 `[69,41,72]`，**8/9
  增长、1/9 持平**，均值 **25→60 帧**，仅出现 **1 次 pass**。候选 rank 支持仍
  只有 **4.8%–8.4%**；provider 有 demand 的决策为 108/197、86/161、150/182。
  seed100 加 `target probability>=0.6 + support>=4` 后，provider 输出降至
  60/197、绝对 rank hints 333→329，survival 与 ungated 同为65.7 帧/1 pass；
  tie-only margin 0.05 也与 order arm 持平。→ 是“更多事件经验让稀疏 rank hint
  开始影响行为”的正向信号，但3个 train seed、重复 eval worlds 仍是 screening；
  新增 default-off frontier trace 后，**134/344** 个有效 demand 的 selectable
  候选在至少一层命中 target root，且这134个目标均至少一次被保留；
  **75/344** 个 demand 的 target root 留在最终 beam frontier。selected plan
  endpoint 只有 **19/344** 命中目标 root（53/344 endpoint 可被 resolver 支持）；
  candidate root 映射共 **2831/13072**。→ rank 能把目标带进搜索，但当前 frame
  objective 很少最终选择该目标，符合“需求排序与最终价值分离”，也暴露了短程
  objective 缺少可验证的长程 option value。frontier trace 是默认关闭的只读
  callback，回归测试确认开启前后搜索结果一致；本轮全量 **698** tests 通过。
  仍不能判为宏观决策质量门通过。
- **等预算自监督采集筛查（1500 passive + 5000 planner frames，seed 100/101/102）**：
  passive 6500-frame arm 共 499 局，其中 **11 局 ≥78 帧**（2.2%/局）；
  DAgger arm 共 293 局，其中 **35 局 ≥78 帧**（11.9%/局，约 5.4 倍/局、
  3.2 倍/总帧）。三个 seed 的 reach-event 计数均同向：passive
  3/166、6/161、2/172；DAgger 8/101、11/96、16/96。DAgger 还把 stage q90
  从 passive 的约 6 帧提高到 seed100 的 13 帧，但仍远短于约 78 帧事件。
  → planner 引导的真实交互值得作为事件覆盖采集臂继续筛查；这只是数据支持
  改善，不证明 outcome calibration、动作因果效应或最终决策收益。下一步须做
  episode-held-out 转移质量评估，并保持不同采集臂严格隔离。
- **中间训练/决策 trace（seed 100 单次诊断）**：用固定最后 8 局作 held-out，
  对前 8/16/24/30 局逐步重训 detector + pooled outcome memory。test pooled
  top1 为 **0.692/0.766/0.808/0.773**，全局 base 为 **0.346/0.351/0.370/0.360**；
  pooled 查询的 context depth>0 占 supported 样本 **0.897/0.948/0.986/0.960**，
  并非只有 root 覆盖造成的 top1 提升；
  但 pooled NLL **3.264/2.864/2.189/2.564** 均差于 base
  **3.116/1.950/2.103/1.904**。这显示阶段上下文提高了最常见 successor 的
  命中率，却没有改善概率校准，且末档不单调。精确完整 option trace 的 top1
  更高，但使用了测试 episode 的未来动作，只能作 oracle-conditioned 上界，
  不能当作线上成绩。
  - 同一训练模型在 eval seed 120、horizon 6/beam 4 的 25 决策 trace 中，
    provider **24/25** 次产出 demand；但 rank adapter 对 **912/912** 个候选
    都返回 UNKNOWN/零优先级。episode 死亡前最后一决策的 terminal risk 仅
    **0.0476**（support 21）。故中间链路是“memory 有预测 → selector 有需求
    → candidate resolver 全 UNKNOWN → demand 没进入 beam 排序”，不是 provider
    缺请求。该 trace 为 1 seed/1 world，作为接线诊断，不作为收益评测。

**实施顺序（更新）**：delayed-action/option-prefix 对齐、DAgger 事件覆盖筛查、
stage-tagged frontier trace 均已落地。当前 DAgger rank 筛查显示目标 root 能进入
beam，但只有约 5.5% 的 demand 最终被 frame objective 选为目标终点。

- **episode-held-out 转移校准 probe 初版已实现**（`accuracy_coverage_probe.py`）：
  从原始真实交互行按 episode 切分；训练折单独拟合 key-signal detector、
  `StageOutcomeMemory` 和相同有序 option 的 root prior，验证折只经冻结
  detector transform。此前该 probe 先拟合全量 memory 再切 episode 的实现已移除，
  不再把验证转移计入训练计数。
- 指标在相同受支持验证样本上比较上下文模型与 option-only prior 的 episode-macro
  multiclass Brier/NLL，并报告 transition-micro 指标、支持覆盖、UNKNOWN、终局和
  删失诊断；删失样本不计入终局/后继校准分数，绝不折算为安全。
- 当前校准 gate 要求至少 20 个模型与 option-only prior 均可评分的验证 episode、
  未删失可比较覆盖率至少 10%、按 episode bootstrap 的 Brier skill 95% CI 下界
  大于 0，且 episode-macro NLL 优于 option-only prior。小样本 smoke run
  （seed 100、300 frames）仅有 1 个
  支持验证 episode，CI 不可估，gate 未通过；这是样本不足的接口检查，不是质量结论。
- **初轮三 seed 校准筛查（每 seed 6500 frames，5 折中 1 折 held-out）**：
  uncensored supported coverage 为 **0.908/0.892/0.911**，episode-macro Brier
  skill 为 **0.545/0.591/0.461**，其 episode-bootstrap 95% CI 下界为
  **0.450/0.467/0.378**；但相对 exact-option root prior 的 episode-macro NLL
  gain 为 **−0.031/+0.070/−0.821**，只有 **1/3 seed** 通过完整 gate。
  → 阶段上下文在这批 held-out episode 上有 Brier 改善，但概率尾部仍不稳定；
  不可仅凭高 top-1、Brier skill 或单 seed 通过而开放阶段级长程价值。
- **训练折内嵌套 OOF 收缩诊断**：预先固定 alpha 候选
  `{0, 0.5, 1, 2, 4, 8}`，在外层训练 episodes 内用 4 折 episode-OOF NLL 选值，
  再只应用于外层验证折；收缩权重以独立 episode support 计，不更改在线 memory。
  三 seed 分别选择 **1/1/0.5** episode pseudo-support；外层 episode-macro NLL gain
  为 **+0.336/+0.347/+0.261**，Brier skill 为 **0.548/0.588/0.478**，95% CI 下界
  为 **0.468/0.472/0.402**，收缩读数 **3/3** 通过 gate。说明 train-only shrinkage
  在本筛查中稳定了 NLL，但仍是同一交互环境分布上的一次外层折，不足以接入运行时。
- **验证**：新增 nested OOF / shrinkage / censor / UNKNOWN 回归覆盖；全量
  **703 tests** 通过，`py_compile` 和 `git diff --check` 通过。
- **下一步**：在不同交互 seed 与独立环境条件上复现 train-only shrinkage 的 NLL
  改善；稳定后再设计默认关闭的阶段转移概率读数接线，并单独验证决策收益。
  在此之前 provider 和 trace 均保持 opt-in/default-off；不把 task
  标签用于阶段身份、目标或选项，也不把 heuristic rank 当成最终价值。

## 历史归档索引

2026-11-23 及更早的非近期记录已原样迁出至以下只读归档文档：

| 日期 / 范围 | 归档文件 | 主要内容 |
|---|---|---|
| 2026-11-23 | [TODO-2026-11-23.md](docs/archive/TODO-2026-11-23.md) | 方向校正、晚间更新、里程碑/阶段总结、Batch 6（图重写/规则库）、P0 time-in-state、Batch 7（B0–B1-plan）、Batch 8（6 步 executor / span / 事件线） |
| 2026-11-21 | [TODO-2026-11-21.md](docs/archive/TODO-2026-11-21.md) | 历史状态快照、Batch 4 尾项、记忆树/阶段图/多尺度历史设计稿、第二层诊断、质量门与 A/B 协议、旧 README 归档索引 |
| 2026-11-20 及更早 | [TODO-appendix-2026-11-20.md](docs/archive/TODO-appendix-2026-11-20.md) | 四子代理解剖附录（信息流/收敛/常数/文献）、死机制清单、三子代理审计整合、早期质量/泛化瓶颈与节点中心改造记录 |
