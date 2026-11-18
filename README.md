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
math_memory_search.py     算术事实记忆 + 关系搜索实验
raw_math_experiment.py    nanoGPT 原始等式数据生成与实验
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
2. **价值学习**：`substrate_value.py` 的 `WorldFeedbackValueHead` 从真实交互的 Monte-Carlo 回报学习 dopamine/pain 两路未来值；`TerminalRiskHead` 从同一 `pain` 标签学习终止风险。真实转移进入 replay，终止样本按比例采样；反事实 rollout 不参与价值或风险训练。
3. **方案选择**：`trajectory_search.py` 提供任务无关的 exhaustive 和 beam 搜索，支持多目标词典序比较。Flappy 适配器默认先最小化预测疼痛/终止风险，再用 pain 回报和 dopamine 排序；`--task-baseline` 才切换回 gap-center 评分与碰撞可行性过滤。

默认搜索是 `exhaustive H=14`；`--engine beam` 是长时域对照，不是当前最好配置。

这不是“价值函数天然等于存活”：反馈通道的语义仍由世界提供。当前实现也没有把新颖度直接当作奖励，它只是价值模型的一个输入特征。

## 当前测量概况

- 在合成延迟世界中，预测误差能学到三帧滞后和约 `0.020` 的动作增益。
- `u` 是 RC 型泄漏积分；`signal_bank.py` 提供节点级候选信号，均值器默认不启用。
- `recall_world.py` 的 recall 是“线索只出现一次、随后输入静默 d 步、最后报告线索符号”的延迟回忆任务，不是通用文本检索。它对照节点痕迹 `u`、每边转移残留 `res`、稀疏联想记忆 `M` 和递归权重学习。
- 节点痕迹与边残留分开：`u` 保存节点激活的衰减痕迹；`res` 让刚用过的边暂时更容易通过；`M` 用共激活关联补全回忆模式。它们是实验臂，尚未全部集成进 Flappy 沙盒控制循环。
- **当前世界反馈模式**（8 个种子，默认 exhaustive H=14）：存活平均 **209.9 帧**，中位数 232，范围 55–312；lag=3、增益约 `0.020`。这比最初的 14.5 帧明显提高，但仍远低于任务基线。
- 价值留出评估（每 5 个练习 episode 留出 1 个）：pain AUC `0.899`；终止风险 AUC `0.971`，Brier `0.029`；风险均值终止状态 `0.715`、安全状态 `0.051`。风险头已能排序终止风险，剩余瓶颈是长时域管道对齐，而不是疼痛预测本身。
- **当前任务基线**（8 个种子，5000 步练习）：存活平均 **1026.9 帧**，中位数 659，范围 123–3491。它仍使用 gap-center 评分和碰撞可行性过滤，只作对照。
- 长时域对照：beam H=78、宽度 256 在较早配置下约 105 帧，低于 exhaustive H=14；当前默认不启用 beam。
- 风险聚合对照：`terminal` 约 209.9 帧，`max` 约 120.9，`sum` 约 92.5，仅在预测穿管时刻查询的 `crossing` 约 105.8；说明只改聚合/查询时点不够。风险头整体 AUC 约 0.97，但管道附近近邻状态 AUC 仅约 0.68–0.76。
- 风险头增强对照：加入交互特征和近管道采样后，近管道 AUC 仍约 `0.68–0.71`，存活反而降到约 `59–83` 帧；默认已恢复无交互、无近管道过采样。问题不在特征容量或采样比例，而在缺少动作条件化的碰撞/风险预测。
- 动作历史输入对照：`--risk-action-dim 4` 把最近动作拼入风险头，近管道 AUC 约 `0.733`、存活约 `170.1` 帧；与 `--risk-aggregate crossing` 组合约 `103.5` 帧。仅拼接动作历史仍不够，默认 `risk_action_dim=0`；下一步需要动作条件的 n-step 风险/价值或 Q(s,a)。
- 事实动作覆盖诊断：seed 100 的近管道状态中只有 3/50 实际拍翅，seed 103 只有 4/80；这些拍翅样本没有终止。状态风险整体 AUC 约 0.97，但动作历史没有稳定提升近管道 AUC。由于不允许生成反事实结果，当前数据不足以估计同状态下拍翅/不拍翅的差异；需要先在管道附近实际采集两种动作。
- 练习覆盖（8 个种子）：平均 episode 约 43.1 帧；管道死亡 128 次、边界死亡 792 次、管道窗口 921 步。覆盖统计由 `substrate_sandbox_search.py` 输出。
- **动作学习轨迹仍然是负结果**：`trace_feedback.py 100` 追踪到 139 个管道附近决策，按“缺口在上则拍翅、缺口在下则不拍”计，方向准确率约 `0.24`；需要上拍时拍翅率约 `0.09`。风险头对全不拍/全拍的平均风险为 `0.48/1.00`，从未认为全拍更安全。它学到的是“管道附近有终止风险”，不是“当前拍翅能对准缺口”。
- 穿管时刻预测：seed 100 的选中计划/全不拍在缺口内的比例为 `0.82/0.75`，seed 103 为 `0.843/0.843`；平均穿管前拍翅 `0.37/0.34` 次。也就是说，选中计划只有轻微、种子相关的调整，不是稳定的方向性动作规则。当前搜索仍主要靠宽缺口、初始高度和延迟拍翅延长存活。
- 沙盒反事实 rollout 单独记录为搜索经验，不作为真实状态转移训练世界模型或价值头；只有实践中观察到的转移更新模型和价值。

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
的存活为 152/232/76 帧，单步留出误差约 `5e-3` 到 `1e-7`；seed 100 的最大训练误差约 `0.79`，
来自管道生成造成的不连续事件，说明纯线性观测模型还不足。这个原型验证了接口分层，但还不是
性能更好的默认路径；当前默认仍是 `substrate_sandbox_search.py`。

`trace_generic_training.py` 抽样了 seed 100–102 的训练波次：价值/风险指标随更新稳定上升
（pain AUC 约 `0.87→0.95`，risk AUC 约 `0.90→0.96`，near-pipe AUC 约 `0.88–0.92`），
没有出现训练崩溃。不稳定的部分主要是数据覆盖和世界模型事件：每 500 步的近管道样本在
`0–81` 之间波动，近管道实际拍翅样本在 `0–9` 之间；seed 100 的 `dx/gap` 留出误差约
`1.6e-3/6e-5`，其他种子约 `1e-8`，对应管道生成的不连续事件。也就是说，学习器不是主要
失效点，**数据覆盖和事件感知世界模型才是**。

`trace_signal_world.py` 对 raw、`d1/d2` 差分、多时间常数缓冲、以及全部信号做了同一批真实 episode 的对照：seed 100 的 raw 单步平均误差约 `6.20e-4`，差分约 `6.35e-4`，缓冲/全部约 `6.6e-4`；seed 101 的 raw/差分/缓冲/全部约 `1.0e-8/0.9e-8/1.1e-8/1.5e-8`。多步误差没有稳定改善，管道维度仍主要受生成/弹出事件限制。结论：当前完整观测下，缓冲和差分是冗余的，低通还会抹平事件边缘；不能替代事件/动作条件模型。

`trace_sensitivity.py` 的扰动重要度更直接：世界模型对 `dx/gap/y/vy` 和滞后动作 `a3` 都敏感；风险头却几乎只依赖 `y`（打乱 `y` 使 AUC 下降约 `0.36–0.42`），`dx/gap` 和动作历史的贡献接近零；价值头主要依赖 `vy/y/gap`，动作贡献也很小。也就是说：**风险头有完整观测输入，但没有真正使用管道和动作信息。** 这不是输入缺失，而是训练目标/样本分布让“只看高度”就能获得较高全局 AUC。

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
