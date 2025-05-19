# 导入需要使用的库
import numpy as np
from solver import Solver

class UCB(Solver):
    """ UCB算法(Upper Confidence Bound),继承Solver类 """
    def __init__(self, bandit, coef=1.0, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef  # 控制不确定性比重的系数

    def run_one_step(self):
        self.total_count += 1
        # 计算上置信界
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k
