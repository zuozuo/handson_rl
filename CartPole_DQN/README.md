# CartPole DQN Implementation

这是一个使用深度Q网络（Deep Q-Network, DQN）算法解决CartPole问题的实现。CartPole是一个经典的强化学习问题，目标是通过左右移动小车来平衡一个竖直的杆子。

## 项目结构

- `dqn_agent.py`: 包含DQN智能体的实现，包括Q网络、经验回放缓冲区等
- `train.py`: 主训练脚本，用于训练和评估DQN智能体
- `models/`: 保存训练好的模型
- `results/`: 保存训练结果，如奖励曲线图

## 依赖库

要运行此代码，您需要安装以下Python库：

```bash
pip install gym
pip install torch
pip install numpy
pip install matplotlib
pip install wandb
```

或者直接使用requirements.txt安装所有依赖：

```bash
pip install -r requirements.txt
```

## 如何运行

1. 确保已安装所有依赖库
2. 登录Weights & Biases（如果还没有账号，请先注册）：

```bash
wandb login
```

3. 运行训练脚本：

```bash
python train.py
```

训练过程将打印每个episode的奖励、平均奖励和当前的epsilon值。同时，所有训练指标都会被记录到Weights & Biases平台上，您可以在线查看实时训练曲线和指标。训练完成后，将生成奖励曲线图并保存在`results`目录中。

### 命令行参数

训练脚本支持多种命令行参数来自定义训练过程和Weights & Biases配置：

```bash
# 使用自定义超参数训练
python train.py --num_episodes 1000 --learning_rate 0.0005 --gamma 0.98

# 自定义Weights & Biases配置
python train.py --wandb_project "my-dqn-project" --wandb_entity "your-username" --wandb_tags "experiment,cartpole"

# 禁用Weights & Biases
python train.py --no_wandb

# 查看所有可用参数
python train.py --help
```

主要参数包括：

- `--env_name`: 环境名称（默认：CartPole-v1）
- `--num_episodes`: 训练的总episode数（默认：500）
- `--learning_rate`: 学习率（默认：0.001）
- `--gamma`: 折扣因子（默认：0.99）
- `--wandb_project`: Weights & Biases项目名称（默认：cartpole-dqn）
- `--wandb_entity`: Weights & Biases实体名称
- `--wandb_tags`: Weights & Biases标签，用逗号分隔
- `--no_wandb`: 禁用Weights & Biases
- `--log_model`: 将模型检查点保存到Weights & Biases
- `--log_video`: 将环境视频记录到Weights & Biases
- `--video_interval`: 记录视频的间隔（episode数，默认：100）

## 实现细节

### DQN算法

DQN算法结合了Q-learning和深度神经网络，主要包含以下关键组件：

1. **Q网络**：一个深度神经网络，用于近似动作价值函数Q(s,a)
2. **目标网络**：Q网络的一个副本，用于计算目标Q值，以提高训练稳定性
3. **经验回放**：存储和重用过去的经验，打破样本之间的相关性
4. **epsilon-贪婪策略**：平衡探索与利用

### Weights & Biases集成

本项目使用Weights & Biases (wandb) 来跟踪和可视化训练过程。记录的指标包括：

1. **训练指标**：
   - 每个episode的奖励
   - 平均奖励（最近100个episodes）
   - epsilon值
   - 损失值

2. **Q网络指标**：
   - Q值统计（最小值、最大值、平均值）
   - 目标Q值统计
   - 梯度范数

3. **评估指标**：
   - 定期评估的平均奖励

4. **可视化**：
   - 奖励曲线图
   - 环境视频（可选）

5. **模型保存**：
   - 可以将模型检查点保存到wandb（可选）

#### 视频记录

可以通过以下命令启用环境视频记录：

```bash
python train.py --log_video --video_interval 100
```

这将每隔100个episode记录一次评估过程的视频，并上传到wandb平台。

#### 模型保存

可以通过以下命令将模型检查点保存到wandb：

```bash
python train.py --log_model
```

这将在每次评估后将模型检查点上传到wandb平台，方便后续下载和使用。

### 超参数

主要超参数包括：

- `num_episodes`: 训练的总episode数
- `max_steps`: 每个episode的最大步数
- `batch_size`: 每次更新的批量大小
- `learning_rate`: 学习率
- `gamma`: 折扣因子
- `epsilon_start`: 初始epsilon值
- `epsilon_end`: 最小epsilon值
- `epsilon_decay`: epsilon的衰减率
- `buffer_capacity`: 经验回放缓冲区的容量
- `target_update`: 目标网络更新频率

## 预期结果

成功训练后，智能体应该能够在CartPole环境中获得较高的奖励（接近500分）。训练过程中，您可以观察到：

1. 随着训练的进行，平均奖励逐渐增加
2. epsilon值逐渐减小，表示智能体从探索转向利用
3. 最终的评估奖励应该接近环境的最大步数

## 参考资料

本实现基于以下资源：
- [DQN算法介绍](https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95)
- [DQN论文：Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
