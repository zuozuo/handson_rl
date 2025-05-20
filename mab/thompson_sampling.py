# thompson_sampling.py
"""
汤普森采样 (Thompson Sampling) / 后验采样 (Posterior Sampling)

核心思想:
汤普森采样通过为每个臂的奖励概率维护一个后验分布（通常是 Beta 分布），
并在每一步从这些分布中采样来选择臂，从而平衡探索和利用。

Beta 分布详解:
Beta 分布是一种定义在 [0, 1] 区间的连续概率分布，由两个正形状参数 α (alpha) 和 β (beta) 控制。
它通常用来表示一个概率值本身的不确定性（例如，某个老虎机臂产生奖励的概率）。

主要特性和参数影响:
1.  α = 1, β = 1: 均匀分布 (表示对概率完全不确定，0到1之间任何值等可能)。
    这是本算法中每个臂初始的先验分布。
2.  α > 1, β > 1: 分布呈单峰形状。
    - α 和 β 越大，峰越尖锐，表示对概率的估计越确定。
    - 若 α = β, 峰在 0.5 (对称)。
    - 若 α > β, 峰偏向 1。
    - 若 α < β, 峰偏向 0。
3.  α < 1, β < 1: U 形分布，概率集中在 0 和 1 附近。
4.  直观理解: 可以将 α 视为 "观测到的成功次数 + 先验成功次数"，β 视为 "观测到的失败次数 + 先验失败次数"。
    在本实现中，初始 _a (对应 α) 和 _b (对应 β) 都设为 1，代表每个臂初始有1次虚拟成功和1次虚拟失败。
    每次选择臂 k 并获得奖励 r (1为成功, 0为失败):
    - _a[k] += r  (如果成功, _a[k] 增加 1)
    - _b[k] += (1 - r) (如果失败, _b[k] 增加 1)
5.  共轭先验: Beta 分布是伯努利分布 (单次成功/失败试验) 似然函数的共轭先验。
    这意味着如果先验是 Beta 分布，观察到伯努利试验结果后，后验分布仍然是 Beta 分布，
    只需简单更新 α 和 β 参数即可。这使得贝叶斯更新非常高效。

可视化 Beta 分布 (使用 visualize_beta_distribution.py):
项目中的 `visualize_beta_distribution.py` 脚本可以帮助理解不同 α 和 β 参数如何影响 Beta 分布的形状。

如何运行脚本:
1.  仅在本地显示图像 (默认行为):
    ```bash
    python visualize_beta_distribution.py
    ```
    或明确指定:
    ```bash
    python visualize_beta_distribution.py --backend local
    ```

2.  将图像记录到 Weights & Biases (使用默认项目名 "handson_rl-mab-visualizations"):
    ```bash
    python visualize_beta_distribution.py --backend wandb
    ```

3.  将图像记录到 Weights & Biases (指定自定义项目名):
    ```bash
    python visualize_beta_distribution.py --backend wandb --project_name "your-custom-project"
    ```
确保已安装必要的库 (numpy, matplotlib, scipy, wandb)，可以通过以下命令安装:
    ```bash
    pip install -r requirements.txt
    ```
"""

# 导入需要使用的库
import numpy as np
from solver import Solver

class ThompsonSampling(Solver):
    """ 汤普森采样算法,继承Solver类 """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)
        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += (1 - r)  # 更新Beta分布的第二个参数
        return k
