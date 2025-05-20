# 多臂老虎机实验框架 (Multi-Armed Bandit Experiment Framework)

这个项目实现了几种经典的多臂老虎机（Multi-Armed Bandit, MAB）算法，并提供了用于比较它们性能的工具。

查看[**实验分析报告**](reports.md)了解各算法性能比较和详细分析。

该项目代码基于[《动手学深度强化学习》](https://hrl.boyuai.com/chapter/1/%E5%A4%9A%E8%87%82%E8%80%81%E8%99%8E%E6%9C%BA)中的实现，并进行了扫展和重构。

## 项目结构

项目包含以下文件：

- `bandit_env.py`: 实现了 `BernoulliBandit` 类，模拟伯努利多臂老虎机环境
- `solver.py`: 实现了基础的 `Solver` 类，作为所有求解算法的基类
- `epsilon_greedy.py`: 实现了基于ε-贪婪策略的求解算法
- `decaying_epsilon_greedy.py`: 实现了ε值随时间衰减的贪婪算法
- `ucb.py`: 实现了上置信界(Upper Confidence Bound, UCB)算法
- `thompson_sampling.py`: 实现了汤普森采样(Thompson Sampling)算法
- `utils.py`: 包含绘图和结果可视化的工具函数
- `main.py`: 主程序入口，提供命令行参数解析和实验运行功能
- `reports.md`: 实验分析报告，包含各算法的性能比较和现象分析

## 环境设置

### 前提条件

- Python 3.6+
- pip (Python包管理器)

### 安装依赖

创建虚拟环境（推荐但非必需）：

```bash
# 创建虚拟环境
python -m venv mab-env

# 激活虚拟环境（Linux/Mac）
source mab-env/bin/activate

# 激活虚拟环境（Windows）
# mab-env\Scripts\activate
```

安装必要的依赖包：

```bash
pip install numpy matplotlib

# 如果使用wandb可视化，也需要安装wandb
pip install wandb
```

如果你计划使用Weights & Biases (wandb)进行实验跟踪和可视化，你需要先注册一个账户并登录：

```bash
wandb login
```

## 使用方法

### 基本用法

运行默认的衰减ε-贪婪算法：

```bash
python main.py
```

### 命令行参数

该程序支持以下命令行参数：

- `--algorithm`: 选择要运行的算法
  - `epsilon-greedy`: 标准ε-贪婪算法
  - `decaying-epsilon-greedy`: ε值随时间衰减的贪婪算法（默认）
  - `epsilon-comparison`: 比较不同ε值的贪婪算法
  - `ucb`: 上置信界(Upper Confidence Bound)算法
  - `thompson-sampling`: 汤普森采样(Thompson Sampling)算法
  - `all`: 运行所有算法并进行比较
- `--ucb-coef`: UCB算法的系数，控制不确定性比重（仅在algorithm=ucb或all时有效）
- `--steps`: 算法运行的步数（默认为5000）
- `--k`: 老虎机的拉杆数量（默认为10）
- `--seed`: 随机种子（默认为1）
- `--backend`: 选择结果可视化的后端
  - `local`: 本地显示图表（默认）
  - `wandb`: 使用Weights & Biases进行可视化和实验跟踪
- `--project-name`: Weights & Biases项目名称（仅在`--backend wandb`时有效）
- `--run-name`: Weights & Biases运行名称（仅在`--backend wandb`时有效）

### 示例命令

运行ε-贪婪算法：

```bash
python main.py --algorithm epsilon-greedy
```

比较不同ε值的贪婪算法：

```bash
python main.py --algorithm epsilon-comparison
```

运行所有算法并在本地比较：

```bash
python main.py --algorithm all
```

使用Weights & Biases进行结果可视化：

```bash
python main.py --algorithm all --backend wandb --project-name "mab-experiments"
```

### 特殊用例说明

#### Epsilon比较模式

当运行 `--algorithm epsilon-comparison --backend wandb` 时，系统会为每个不同的epsilon值（默认为[1e-4, 0.01, 0.1, 0.25, 0.5]）创建一个独立的W&B run。这样可以在W&B界面上直观地比较不同epsilon值对性能的影响。每个run名称包含对应的epsilon值，例如：`EpsilonGreedy-epsilon=0.01`。

## 算法说明

### ε-贪婪算法 (Epsilon-Greedy)

这是最基本的探索-利用平衡策略。算法以1-ε的概率选择当前估计奖励最高的拉杆（利用），以ε的概率随机选择一个拉杆（探索）。

### 衰减ε-贪婪算法 (Decaying Epsilon-Greedy)

这个算法让ε值随着时间推移而衰减，使得算法在早期更多地进行探索，而在后期更倾向于利用已知的好拉杆。

### 上置信界算法 (Upper Confidence Bound, UCB)

上置信界算法是一种基于不确定性的策略算法。它在选择拉杆时考虑了每个拉杆的当前期望奖励估计值和不确定性。对于尝试次数少的拉杆，其不确定性更高，因此更倾向于探索这些拉杆。UCB 算法的关键是计算每个拉杆的期望奖励上界，并选择上界最大的拉杆。

### 汤普森采样算法 (Thompson Sampling)

汤普森采样算法是一种基于概率分布的策略算法。它假设每个拉杆的奖励服从一个概率分布（通常是 Beta 分布），并在每一步中从每个拉杆的当前分布中进行采样，选择样本中奖励最大的拉杆。在获取新的奖励后，更新相应拉杆的奖励分布。这种方法在实践中效果很好，并且可以实现对数级别的渐近最优累积懊悔。

## 可视化选项

### 本地可视化

默认情况下，实验结果会通过matplotlib在本地显示为图表。这些图表显示了累积懊悔随时间的变化情况。

### Weights & Biases (wandb)

如果选择wandb后端，实验数据会被上传到Weights & Biases平台，支持：

- 实时实验跟踪
- 交互式图表和比较
- 实验参数记录
- 团队共享和协作

每次运行算法时，系统都会自动创建一个新的W&B实验 run，记录以下内容：

- 实验配置参数：
  - 算法名称和总数
  - 拉杆数量 (K)
  - 实验步数 (steps)
  - 随机种子 (seed)
  - bandit最佳拉杆编号和概率
- 每一步的懊悔值随时间的变化曲线
- 完整的数据表格，方便导出和后续分析

所有算法都使用统一的 `regret` 指标名称，这样在W&B界面上可以轻松地在同一个图表中比较不同的算法或不同的运行结果。

要使用此功能，请确保已安装wandb并已登录：

```bash
pip install wandb
wandb login
```

当运行 `--algorithm all` 时，无论选择哪种后端，系统都会为每个算法创建一个独立的记录（当使用 wandb 时，每个算法会生成一个独立的 run）。

所有算法都使用相同的指标名称 `regret`，这样在使用 W&B 时，您可以在界面上轻松地选择不同的 run 来直接比较任意算法组合的性能。

### 项目W&B页面

你可以通过以下链接查看此项目的Weights & Biases记录，比较不同算法的regret指标：

[https://wandb.ai/zuozuo/mab-experiments](https://wandb.ai/zuozuo/mab-experiments/workspace?nw=nwuserzuozuo&panelDisplayName=regret&panelSectionName=Charts)

在该页面中，你可以：
- 查看每种算法的累积懊悔（regret）随时间的变化趋势
- 比较不同算法的性能差异
- 分析算法参数（如UCB的系数、epsilon值等）对性能的影响

## 扩展项目

如果你想添加新的求解器算法，只需：

1. 创建一个新的Python文件（如`your_solver.py`）
2. 从`solver.py`中导入并继承`Solver`类
3. 实现`run_one_step`方法
4. 在`main.py`中导入并添加你的新算法

## 使用 Weights & Biases Sweeps 进行超参数优化

Weights & Biases Sweeps 是一种超参数优化工具，可以帮助你系统地探索不同的参数组合，找到最佳的算法配置。

### 执行 Sweep 实验

1. 确保已安装 Weights & Biases：

```bash
pip install wandb
```

2. 初始化 sweep：

```bash
wandb sweep sweep_config.yaml
```

3. 运行 sweep agent（将输出的 SWEEP_ID 替换为初始化时返回的 ID）：

```bash
wandb agent zuozuo/$PROJECT_NAME/$SWEEP_ID
```

其中 `zuozuo` 是您的 W&B 用户名，`$PROJECT_NAME` 是项目名称（默认为 `mab-sweep-experiments`），`$SWEEP_ID` 是初始化时返回的 ID。

4. 要加快执行速度，可以同时运行多个 agent：

```bash
# 在不同的终端窗口中运行多个相同的命令
wandb agent zuozuo/$PROJECT_NAME/$SWEEP_ID
wandb agent zuozuo/$PROJECT_NAME/$SWEEP_ID
# 你可以根据计算机性能运行多个并行 agent
```

5. sweep_config.yaml 文件已经经过优化，采用网格搜索方法探索关键参数组合，大大减少了执行时间。

### 探索的主要参数

Sweep 配置会探索以下参数：

1. **算法类型**：测试各种算法（epsilon-greedy、decaying-epsilon-greedy、UCB、Thompson Sampling）
2. **问题大小**：不同的拉杆数量 K（5、10、20、50）
3. **运行时长**：不同的步数（1000、5000、10000、20000）
4. **算法特定参数**：
   - epsilon-greedy 的 epsilon 值（0.00001 到 0.5）
   - UCB 的系数（0.1 到 2.0）
5. **随机种子**：多个随机种子确保结果的可靠性

### 在 W&B 上查看结果

Sweep 完成后，您可以在 W&B 界面上查看结果。平台提供了丰富的可视化工具来分析：

- 平行坐标图（查看参数之间的相关性）
- 重要性分析（确定哪些参数对算法性能影响最大）
- 最佳参数组合（获取表现最佳的配置）

这些分析可以帮助验证报告中的发现并探索新的见解。

## Beta 分布可视化工具

项目包含一个专门用于可视化 Beta 分布的工具脚本 `visualize_beta_distribution.py`。这对理解 Thompson Sampling 算法中使用的 Beta 分布非常有帮助。

### 为什么需要理解 Beta 分布？

Beta 分布在 Thompson Sampling 算法中扮演着重要角色，用于表示每个拉杆的奖励概率分布。了解不同 α 和 β 参数如何影响分布形状，有助于更深入地理解算法的工作原理。

### 使用方法

该工具支持两种可视化模式：本地显示或在 Weights & Biases 中创建交互式图表。

#### 本地显示模式

在本地显示 Beta 分布图（默认行为）：

```bash
python visualize_beta_distribution.py
```

或者明确指定使用本地模式：

```bash
python visualize_beta_distribution.py --backend local
```

#### Weights & Biandes 交互式图表模式

将 Beta 分布作为交互式图表记录到 W&B（使用默认项目名）：

```bash
python visualize_beta_distribution.py --backend wandb
```

自定义 W&B 项目名称：

```bash
python visualize_beta_distribution.py --backend wandb --project_name "your-custom-project"
```

### W&B 交互式图表的优势

当选择 `--backend wandb` 时，脚本会在 W&B 平台上创建可交互的线图，而不是上传静态图像。这为您提供以下优势：

- **交互式探索**：在 W&B 界面中直接放大、缩小和平移图表
- **选择性显示**：可以选择显示或隐藏特定的 Beta 分布
- **数据导出**：轻松导出原始数据进行进一步分析
- **精确值查看**：悬停在曲线上可查看确切的数据点值
- **团队共享**：轻松与团队成员分享结果

此外，脚本还会记录每个 Beta 分布的摘要统计信息（均值、众数、方差），方便进行定量分析。

## 参考资料

- [动手学深度强化学习 - 多臂老虎机](https://hrl.boyuai.com/chapter/1/%E5%A4%9A%E8%87%82%E8%80%81%E8%99%8E%E6%9C%BA)
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
- Multi-Armed Bandit Problem: https://en.wikipedia.org/wiki/Multi-armed_bandit
- Beta Distribution: https://en.wikipedia.org/wiki/Beta_distribution
