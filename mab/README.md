# 多臂老虎机实验框架 (Multi-Armed Bandit Experiment Framework)

这个项目实现了几种经典的多臂老虎机（Multi-Armed Bandit, MAB）算法，并提供了用于比较它们性能的工具。

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

- 实验配置参数（算法名称、bandit参数等）
- 每一步的懊悔值随时间的变化曲线
- 完整的数据表格，方便导出和后续分析

所有算法都使用统一的 `regret` 指标名称，这样在W&B界面上可以轻松地在同一个图表中比较不同的算法或不同的运行结果。

要使用此功能，请确保已安装wandb并已登录：

```bash
pip install wandb
wandb login
```

当运行 `--algorithm all` 时，无论选择哪种后端，系统都会为每个算法创建一个独立的记录（当使用 wandb 时，每个算法会生成一个独立的 run，名称带有 "-individual-run" 后缀）。

所有算法都使用相同的指标名称 `regret`，这样在使用 W&B 时，您可以在界面上轻松地选择不同的 run 来直接比较任意算法组合的性能。

## 扩展项目

如果你想添加新的求解器算法，只需：

1. 创建一个新的Python文件（如`your_solver.py`）
2. 从`solver.py`中导入并继承`Solver`类
3. 实现`run_one_step`方法
4. 在`main.py`中导入并添加你的新算法

## 参考资料

- [动手学深度强化学习 - 多臂老虎机](https://hrl.boyuai.com/chapter/1/%E5%A4%9A%E8%87%82%E8%80%81%E8%99%8E%E6%9C%BA)
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
- Multi-Armed Bandit Problem: https://en.wikipedia.org/wiki/Multi-armed_bandit
