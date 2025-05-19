# 多臂老虎机算法性能分析报告

## 实验环境

本实验在 10 臂伯努利老虎机环境中测试了不同的多臂老虎机算法。每种算法都运行了 5,000 次试验，并使用固定的随机种子 (np.random.seed(1)) 以确保结果的可重复性。

## 实验参数

各算法使用的具体参数如下：

1. **EpsilonGreedy**:
   - 固定 epsilon 值，测试了多个值：0.0001, 0.01, 0.1, 0.25, 0.5
   - 其中 epsilon=0.01 是默认值

2. **DecayingEpsilonGreedy**:
   - 初始 epsilon 值：1.0
   - 衰减率：使用 1/t 的衰减方式，其中 t 是当前步数

3. **UCB (Upper Confidence Bound)**:
   - 系数 (coef)：1.0
   - 使用公式：估计值 + coef * sqrt(log(总步数)/(2*(尝试次数+1)))

4. **Thompson Sampling**:
   - 使用 Beta 分布模型
   - 每根拉杆的先验分布：Beta(1, 1)
   - 根据观测到的奖励更新参数：α += 奖励值，β += (1-奖励值)

## 实验结果分析

根据 Weights & Biases 平台记录的运行结果，不同算法的累积懊悔（regret）按性能从好到差排序为：

1. **Thompson Sampling** - 累积懊悔约 57
2. **UCB** - 累积懊悔约 70
3. **EpsilonGreedy(epsilon=0.0001)** - 累积懊悔相对较低
4. **DecayingEpsilonGreedy** - 累积懊悔中等
5. **其他更大 epsilon 值的 EpsilonGreedy 算法** - 累积懊悔较高

![算法性能比较图](https://wandb.ai/zuozuo/mab-experiments/workspace?nw=nwuserzuozuo&panelDisplayName=regret&panelSectionName=Charts)

## 特殊现象分析：为什么 EpsilonGreedy(epsilon=0.0001) 比 DecayingEpsilonGreedy 效果更好？

在实验结果中，我们观察到一个有趣的现象：使用极小探索率 (epsilon=0.0001) 的 EpsilonGreedy 算法性能优于 DecayingEpsilonGreedy 算法。以下是可能的原因分析：

1. **初期探索量的差异**:
   - DecayingEpsilonGreedy 在初期有较高的探索率（从1.0开始），导致初期累积更多的懊悔
   - EpsilonGreedy(epsilon=0.0001) 从一开始就有极低的探索率，更倾向于利用而非探索

2. **探索与利用的平衡**:
   - 在这个特定的 10 臂老虎机问题中，可能只需要非常有限的探索就能找到较好的拉杆
   - epsilon=0.0001 意味着平均每 10,000 次拉动中只有 1 次是探索，这在 5,000 次试验中仍提供了足够的探索机会

3. **衰减速率的影响**:
   - DecayingEpsilonGreedy 的衰减可能不够快，导致在中后期仍在进行不必要的探索
   - 当问题相对简单（如 10 臂伯努利老虎机）且有限的试验次数时，快速锁定最佳臂可能比长期探索更重要

4. **随机种子的影响**:
   - 算法都使用了固定的随机种子，这个特定的种子可能恰好使得极低探索率的 EpsilonGreedy 在早期就找到了最优拉杆

## 总体结论

1. **汤普森采样和 UCB 算法表现最佳**：这两种算法能够更智能地平衡探索与利用，这与理论预期一致。

2. **极低探索率的 epsilon-贪婪在特定问题上表现良好**：这说明在某些相对简单的问题中，过度探索可能会导致不必要的懊悔。

3. **算法性能取决于具体问题**：这些结果是针对特定问题实例和参数设置的。在不同的环境、更复杂的问题或更长的时间尺度上，结果可能会有所不同。例如，在有大量拉杆或拉杆奖励分布更接近的情况下，具有适当探索策略的算法可能会表现得更好。

## 未来工作

1. 测试不同问题难度下的算法性能
2. 探索 DecayingEpsilonGreedy 的不同衰减策略
3. 测试 UCB 算法的不同系数值对性能的影响
4. 在非静态环境中比较这些算法的表现
