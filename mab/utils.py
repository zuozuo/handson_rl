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
    而solver_names也是一个列表,存储每个策略的名称。
    
    每次调用都会创建一个新的wandb run。
    """
    # 初始化wandb run
    if run_name is None:
        run_name = f"{'-'.join(solver_names)}"
    
    # 确保wandb初始化一个新的run
    # reinit=True确保即使同一个进程中多次调用也会创建新的run
    # 需要启用allow_val_change允许在同一进程中修改config参数
    run = wandb.init(
        project=project_name, 
        name=run_name, 
        reinit=True,
        settings=wandb.Settings(start_method="thread")
    )
    
    # 收集每个solver的信息，用于配置
    config = {
        "algorithm_names": solver_names,
        "num_solvers": len(solvers),
        "bandit_arms": solvers[0].bandit.K,
        "bandit_best_idx": solvers[0].bandit.best_idx,
        "bandit_best_prob": float(solvers[0].bandit.best_prob),  # 转换为原生类型避免numpy值序列化问题
    }
    
    # 记录实验配置
    for key, value in config.items():
        wandb.config[key] = value
    
    # 创建一个数据表格记录每一步的结果
    columns = ["step"] + solver_names
    data_table = wandb.Table(columns=columns)
    
    # 获取最长的regrets列表长度
    max_steps = max([len(solver.regrets) for solver in solvers])
    
    # 创建一个字典来存储每个算法的结果曲线
    all_steps_data = {}
    
    # 为每一步记录每个solver的regret值
    for step in range(max_steps):
        row = [step]
        step_data = {"step": step}
        
        for idx, solver in enumerate(solvers):
            # 如果solver的regrets长度小于当前步数，则使用最后一个值
            regret_value = None
            if step < len(solver.regrets):
                regret_value = float(solver.regrets[step])  # 转换为原生类型
            row.append(regret_value)
            
            # 添加到单步数据中
            if regret_value is not None:
                step_data[f"regret/{solver_names[idx]}"] = regret_value
        
        data_table.add_data(*row)
        
        # 一次性记录所有算法在此步骤的regret值
        all_steps_data[step] = step_data
    
    # 按照步骤顺序记录数据
    for step, step_data in sorted(all_steps_data.items()):
        wandb.log(step_data)
    
    # 记录表格
    wandb.log({"regrets_table": data_table})
    
    # 创建折线图 - 使用wandb.log记录的数据会自动生成图表
    
    # 另外将全部数据作为一个完整的数据集也记录下来
    # 这对于后续导出数据或自定义分析很有帮助
    complete_data = {}
    for idx, solver in enumerate(solvers):
        complete_data[solver_names[idx]] = [float(r) for r in solver.regrets]
    
    # 使用wandb.Table存储完整数据
    complete_data_table = wandb.Table(columns=["algorithm", "regrets"])
    for algo_name, regrets in complete_data.items():
        complete_data_table.add_data(algo_name, regrets)
    
    wandb.log({"complete_data": complete_data_table})
    
    # 结束wandb运行 - 这确保每次调用是一个独立的run
    wandb.finish()
