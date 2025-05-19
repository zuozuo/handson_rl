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
        
        # 对每个算法都生成独立的绘图/记录，无论是单独运行还是在all模式下
        run_name_suffix = ""
        if args.algorithm == "all":
            run_name_suffix = "-individual-run"  # 添加后缀以区分单独运行和最终比较运行
        
        plot_results(
            [epsilon_greedy_solver], 
            ["EpsilonGreedy"],
            backend=args.backend,
            run_name=(args.run_name + run_name_suffix if args.run_name else "EpsilonGreedy" + run_name_suffix),
            project_name=args.project_name
        )
        
        # 如果是运行所有算法，则将求解器添加到对比列表中
        if args.algorithm == "all":
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
        
        # 运行所有epsilon值的求解器
        for solver in epsilon_greedy_solver_list:
            solver.run(5000)
        
        # 关键修改：为每个不同的epsilon值生成单独的run记录
        # 这样在wandb界面上可以直接对比不同epsilon值的性能
        for idx, (solver, solver_name) in enumerate(zip(epsilon_greedy_solver_list, epsilon_greedy_solver_names)):
            run_name_suffix = ""
            if args.algorithm == "all":
                run_name_suffix = "-individual-run"  # 添加后缀以区分单独运行和最终比较运行
            
            # 为每个epsilon值生成一个单独的run
            epsilon_run_name = f"EpsilonGreedy-{solver_name}{run_name_suffix}"
            if args.run_name:
                epsilon_run_name = f"{args.run_name}-{solver_name}{run_name_suffix}"
            
            plot_results(
                [solver], 
                [solver_name],
                backend=args.backend,
                run_name=epsilon_run_name,
                project_name=args.project_name
            )
            
            print(f'完成运行 {solver_name} 的累积懊悔为：{solver.regret}')
        
        # 如果是单独运行所有对比，而不是在all模式下，还可以生成一个汇总进行本地显示
        if args.algorithm == "epsilon-comparison" and args.backend == "local":
            # 绘制本地比较图，仅当使用本地后端时
            plot_results(
                epsilon_greedy_solver_list, 
                epsilon_greedy_solver_names,
                backend="local",  # 强制使用本地后端显示汇总图
                project_name=args.project_name
            )
        
        # 如果运行所有算法，我们选择最佳的epsilon值添加到对比列表中
        if args.algorithm == "all":
            best_idx = 2  # 0.1的索引
            all_solvers.append(epsilon_greedy_solver_list[best_idx])
            all_solver_names.append(epsilon_greedy_solver_names[best_idx])
    
    # 运行epsilon值衰减的贪婪算法
    if args.algorithm == "decaying-epsilon-greedy" or args.algorithm == "all":
        np.random.seed(1)
        decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
        decaying_epsilon_greedy_solver.run(5000)
        print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
        
        # 对每个算法都生成独立的绘图/记录，无论是单独运行还是在all模式下
        run_name_suffix = ""
        if args.algorithm == "all":
            run_name_suffix = "-individual-run"  # 添加后缀以区分单独运行和最终比较运行
        
        plot_results(
            [decaying_epsilon_greedy_solver], 
            ["DecayingEpsilonGreedy"],
            backend=args.backend,
            run_name=(args.run_name + run_name_suffix if args.run_name else "DecayingEpsilonGreedy" + run_name_suffix),
            project_name=args.project_name
        )
        
        # 如果是运行所有算法，则将求解器添加到对比列表中
        if args.algorithm == "all":
            all_solvers.append(decaying_epsilon_greedy_solver)
            all_solver_names.append("DecayingEpsilonGreedy")
    
    # 如果运行所有算法，最后再绘制一个总的比较图
    if args.algorithm == "all":
        print('\n所有算法已运行完成，正在绘制最终比较图...')
        plot_results(
            all_solvers, 
            all_solver_names,
            backend=args.backend,
            run_name=(args.run_name or "all-algorithms-comparison"),
            project_name=args.project_name
        )

if __name__ == "__main__":
    main()
