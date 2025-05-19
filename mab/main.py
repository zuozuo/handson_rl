# 导入需要使用的库
import numpy as np
import matplotlib.pyplot as plt
import argparse
from bandit_env import BernoulliBandit
from epsilon_greedy import EpsilonGreedy
from decaying_epsilon_greedy import DecayingEpsilonGreedy
from utils import plot_results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多臂老虎机算法实验")
    parser.add_argument(
        "--backend",
        type=str,
        default="local",
        choices=["local", "wandb"],
        help="绘图后端：local表示本地显示，wandb表示使用Weights & Biases"
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="mab-experiments",
        help="Weights & Biases项目名称 (仅在backend=wandb时有效)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Weights & Biases运行名称 (仅在backend=wandb时有效)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="decaying-epsilon-greedy",
        choices=["epsilon-greedy", "decaying-epsilon-greedy", "epsilon-comparison", "all"],
        help="要运行的算法，'all'表示运行所有算法"
    )
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()
    np.random.seed(1)  # 设定随机种子,使实验具有可重复性
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print("随机生成了一个%d臂伯努利老虎机" % K)
    print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
          (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

    # 用于存储所有运行的求解器和名称，以便在运行所有算法时使用
    all_solvers = []
    all_solver_names = []

    # 运行epsilon-贪婪算法
    if args.algorithm == "epsilon-greedy" or args.algorithm == "all":
        np.random.seed(1)
        epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
        epsilon_greedy_solver.run(5000)
        print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
        
        # 如果不是运行所有算法，则立即绘图
        if args.algorithm != "all":
            plot_results(
                [epsilon_greedy_solver], 
                ["EpsilonGreedy"],
                backend=args.backend,
                run_name=args.run_name,
                project_name=args.project_name
            )
        else:
            # 否则将求解器添加到列表中
            all_solvers.append(epsilon_greedy_solver)
            all_solver_names.append("EpsilonGreedy")
    
    # 运行不同epsilon值的贪婪算法进行对比
    if args.algorithm == "epsilon-comparison" or args.algorithm == "all":
        np.random.seed(0)
        epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
        epsilon_greedy_solver_list = [
            EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
        ]
        epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
        for solver in epsilon_greedy_solver_list:
            solver.run(5000)
        
        # 如果不是运行所有算法，则立即绘图
        if args.algorithm != "all":
            plot_results(
                epsilon_greedy_solver_list, 
                epsilon_greedy_solver_names,
                backend=args.backend,
                run_name=args.run_name,
                project_name=args.project_name
            )
        else:
            # 如果运行所有算法，我们只选择最佳的epsilon值。这里选择中间值0.1
            best_idx = 2  # 0.1的索引
            all_solvers.append(epsilon_greedy_solver_list[best_idx])
            all_solver_names.append(epsilon_greedy_solver_names[best_idx])
    
    # 运行epsilon值衰减的贪婪算法
    if args.algorithm == "decaying-epsilon-greedy" or args.algorithm == "all":
        np.random.seed(1)
        decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
        decaying_epsilon_greedy_solver.run(5000)
        print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
        
        # 如果不是运行所有算法，则立即绘图
        if args.algorithm != "all":
            plot_results(
                [decaying_epsilon_greedy_solver], 
                ["DecayingEpsilonGreedy"],
                backend=args.backend,
                run_name=args.run_name,
                project_name=args.project_name
            )
        else:
            # 否则将求解器添加到列表中
            all_solvers.append(decaying_epsilon_greedy_solver)
            all_solver_names.append("DecayingEpsilonGreedy")
    
    # 如果运行所有算法，最后绘制比较图
    if args.algorithm == "all":
        print('\n所有算法已运行完成，正在绘制比较图...')
        plot_results(
            all_solvers, 
            all_solver_names,
            backend=args.backend,
            run_name=args.run_name or "all-algorithms-comparison",
            project_name=args.project_name
        )

if __name__ == "__main__":
    main()
