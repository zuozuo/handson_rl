# 导入需要使用的库
import numpy as np
import matplotlib.pyplot as plt
from bandit_env import BernoulliBandit
from epsilon_greedy import EpsilonGreedy
from decaying_epsilon_greedy import DecayingEpsilonGreedy
from utils import plot_results

def main():
    np.random.seed(1)  # 设定随机种子,使实验具有可重复性
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print("随机生成了一个%d臂伯努利老虎机" % K)
    print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
          (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

    # 运行epsilon-贪婪算法
    # np.random.seed(1)
    # epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    # epsilon_greedy_solver.run(5000)
    # print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
    # plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

    # 运行不同epsilon值的贪婪算法进行对比
    # np.random.seed(0)
    # epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    # epsilon_greedy_solver_list = [
    #     EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
    # ]
    # epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
    # for solver in epsilon_greedy_solver_list:
    #     solver.run(5000)
    #
    # plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

    # 运行epsilon值衰减的贪婪算法
    np.random.seed(1)
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
    decaying_epsilon_greedy_solver.run(5000)
    print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
    plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

if __name__ == "__main__":
    main()
