# 导入需要使用的库
import numpy as np
import matplotlib.pyplot as plt
import argparse
from bandit_env import BernoulliBandit
from epsilon_greedy import EpsilonGreedy
from decaying_epsilon_greedy import DecayingEpsilonGreedy
from ucb import UCB
from thompson_sampling import ThompsonSampling
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
        choices=["epsilon-greedy", "decaying-epsilon-greedy", "epsilon-comparison", "ucb", "thompson-sampling", "all"],
        help="要运行的算法，'all'表示运行所有算法"
    )
    parser.add_argument(
        "--ucb-coef",
        type=float,
        default=1.0,
        help="UCB算法的系数，控制不确定性比重 (仅在algorithm=ucb或all时有效)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="算法运行的步数"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="老虎机的拉杆数量"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="随机种子"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Epsilon value for epsilon-greedy algorithms (used when algorithm is epsilon-greedy)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)  # 使用用户指定的随机种子
    bandit_10_arm = BernoulliBandit(K=args.k)
    print(f'随机生成了一个{args.k}臂伯努利老虎机')
    print('获奖概率最大的拉杆为{}号,其获奖概率为{}'.format(
        np.argmax(bandit_10_arm.probs) + 1, bandit_10_arm.probs[np.argmax(bandit_10_arm.probs)]))
    
    # 用于存储所有求解器和名称，以便在--algorithm all模式下最终比较
    all_solvers = []
    all_solver_names = []

    # 运行标准ε-贪婪算法
    if args.algorithm == "epsilon-greedy" or args.algorithm == "all":
        np.random.seed(args.seed)
        # 使用命令行传入的epsilon值，如果 sweep 中定义了的话
        epsilon_to_use = args.epsilon 
        epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=epsilon_to_use)
        epsilon_greedy_solver.run(args.steps)
        print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
        
        # 对每个算法都生成独立的绘图/记录
        algorithm_name = f"EpsilonGreedy(epsilon={epsilon_to_use})"
        plot_results(
            [epsilon_greedy_solver], 
            [algorithm_name],
            backend=args.backend,
            run_name=(args.run_name if args.run_name else algorithm_name),
            project_name=args.project_name,
            steps=args.steps,
            K=args.k,
            seed=args.seed
        )
        
        # 如果是运行所有算法，则将求解器添加到对比列表中
        if args.algorithm == "all":
            all_solvers.append(epsilon_greedy_solver)
            all_solver_names.append(algorithm_name)
    
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
            solver.run(args.steps)
        
        # 关键修改：为每个不同的epsilon值生成单独的run记录
        # 这样在wandb界面上可以直接对比不同epsilon值的性能
        for idx, (solver, solver_name) in enumerate(zip(epsilon_greedy_solver_list, epsilon_greedy_solver_names)):
            # 对每个算法都生成独立的绘图/记录
            epsilon_run_name = f"EpsilonGreedy-{solver_name}"
            if args.run_name:
                epsilon_run_name = f"{args.run_name}-{solver_name}"
            
            plot_results(
                [solver], 
                [solver_name],
                backend=args.backend,
                run_name=epsilon_run_name,
                project_name=args.project_name,
                steps=args.steps,
                K=args.k,
                seed=args.seed
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
        np.random.seed(args.seed)
        decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
        decaying_epsilon_greedy_solver.run(args.steps)
        print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
        
        # 对每个算法都生成独立的绘图/记录
        plot_results(
            [decaying_epsilon_greedy_solver], 
            ["DecayingEpsilonGreedy"],
            backend=args.backend,
            run_name=(args.run_name if args.run_name else "DecayingEpsilonGreedy"),
            project_name=args.project_name,
            steps=args.steps,
            K=args.k,
            seed=args.seed
        )
        
        # 如果是运行所有算法，则将求解器添加到对比列表中
        if args.algorithm == "all":
            all_solvers.append(decaying_epsilon_greedy_solver)
            all_solver_names.append("DecayingEpsilonGreedy")
    
    # 运行UCB算法
    if args.algorithm == "ucb" or args.algorithm == "all":
        np.random.seed(args.seed)
        ucb_solver = UCB(bandit_10_arm, coef=args.ucb_coef)
        ucb_solver.run(args.steps)
        print('UCB算法的累积懊悔为：', ucb_solver.regret)
        
        # 对每个算法都生成独立的绘图/记录
        plot_results(
            [ucb_solver], 
            [f"UCB(coef={args.ucb_coef})"],
            backend=args.backend,
            run_name=(args.run_name if args.run_name else "UCB"),
            project_name=args.project_name,
            steps=args.steps,
            K=args.k,
            seed=args.seed
        )
        
        # 如果是运行所有算法，则将求解器添加到对比列表中
        if args.algorithm == "all":
            all_solvers.append(ucb_solver)
            all_solver_names.append(f"UCB(coef={args.ucb_coef})")
    
    # 运行汤普森采样算法
    if args.algorithm == "thompson-sampling" or args.algorithm == "all":
        np.random.seed(args.seed)
        thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
        thompson_sampling_solver.run(args.steps)
        print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
        
        # 对每个算法都生成独立的绘图/记录
        plot_results(
            [thompson_sampling_solver], 
            ["ThompsonSampling"],
            backend=args.backend,
            run_name=(args.run_name if args.run_name else "ThompsonSampling"),
            project_name=args.project_name,
            steps=args.steps,
            K=args.k,
            seed=args.seed
        )
        
        # 如果是运行所有算法，则将求解器添加到对比列表中
        if args.algorithm == "all":
            all_solvers.append(thompson_sampling_solver)
            all_solver_names.append("ThompsonSampling")
            
    # 所有算法已分别生成各自的独立记录，不再绘制汇总图
    if args.algorithm == "all":
        print('\n所有算法已运行完成。')

if __name__ == "__main__":
    main()
