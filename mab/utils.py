# 导入需要使用的库
import numpy as np
import matplotlib.pyplot as plt
import wandb

def plot_results_to_local(solvers, solver_names):
    """生成累积懊悔随时间变化的图像并展示在本地。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


def plot_results(solvers, solver_names, backend="local", run_name=None, project_name="mab-experiments"):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称。
    
    参数：
    - solvers: 求解器列表
    - solver_names: 求解器名称列表
    - backend: 绘图后端，"local"表示本地显示，"wandb"表示使用Weights & Biases
    - run_name: wandb运行的名称，只在backend="wandb"时使用
    - project_name: wandb项目名称，只在backend="wandb"时使用
    """
    if backend.lower() == "local":
        return plot_results_to_local(solvers, solver_names)
    elif backend.lower() == "wandb":
        return plot_results_to_wandb(solvers, solver_names, run_name=run_name, project_name=project_name)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose either 'local' or 'wandb'.")


def plot_results_to_wandb(solvers, solver_names, run_name=None, project_name="mab-experiments"):
    """将累积懊悔随时间变化的结果记录到wandb。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称。"""
    # 初始化wandb run
    if run_name is None:
        run_name = f"{'-'.join(solver_names)}"
    
    run = wandb.init(project=project_name, name=run_name, reinit=True)
    
    # 创建一个表格记录每一步的结果
    columns = ["step"] + solver_names
    data_table = wandb.Table(columns=columns)
    
    # 获取最长的regrets列表长度
    max_steps = max([len(solver.regrets) for solver in solvers])
    
    # 为每一步记录每个solver的regret值
    for step in range(max_steps):
        row = [step]
        for solver in solvers:
            # 如果solver的regrets长度小于当前步数，则使用最后一个值
            if step < len(solver.regrets):
                row.append(solver.regrets[step])
            else:
                row.append(None)
        data_table.add_data(*row)
    
    # 记录表格
    wandb.log({"regrets_table": data_table})
    
    # 创建折线图
    data = {}
    for idx, solver in enumerate(solvers):
        # 对每个步骤记录regret值
        for step, regret in enumerate(solver.regrets):
            wandb.log({
                f"regret/{solver_names[idx]}": regret,
                "step": step
            })
        
        # 保存所有regrets为一个列表
        data[solver_names[idx]] = solver.regrets
    
    # 记录bandit信息
    wandb.log({
        "bandit/arms": solvers[0].bandit.K,
        "bandit/best_idx": solvers[0].bandit.best_idx,
        "bandit/best_prob": solvers[0].bandit.best_prob
    })
    
    # 结束wandb运行
    run.finish()
