import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import wandb
import argparse

def plot_beta_distributions(alpha_beta_pairs, x_values, project_name, backend):
    """
    生成 Beta 分布数据并将其记录到 W&B 或在本地显示。
    
    当 backend 为 'wandb' 时，以交互式格式记录 Beta 分布数据，
    而不是静态图像。这使您可以在 W&B 界面中直接探索这些分布。
    
    当 backend 为 'local' 时，使用 Matplotlib 在本地显示分布图。

    Parameters:
    alpha_beta_pairs (list of tuples): 一个包含 (alpha, beta) 参数对的列表。
    x_values (numpy.ndarray): 用于计算 PDF 的 x 值数组 (通常在 [0, 1] 区间)。
    project_name (str): W&B 项目名称 (如果 backend='wandb')。
    backend (str): 'wandb' 或 'local'。
    """
    
    if backend == "wandb":
        # 构造描述性运行名称
        run_name_parts = [f'a{a}b{b}' for a,b in alpha_beta_pairs[:2]]
        run_name = f"beta_interactive_{'_'.join(run_name_parts)}"
        if len(alpha_beta_pairs) > 2:
            run_name += "_etc"
        
        # 初始化 W&B 运行
        current_run = wandb.init(project=project_name, name=run_name, job_type="visualization")
        
        if current_run is None:
            print("错误: 无法初始化 W&B 运行。")
            return

        try:
            print(f"记录到 W&B 项目: {project_name}, 运行: {current_run.name}")
            # 记录配置
            current_run.config.update({
                "alpha_beta_pairs": alpha_beta_pairs,
                "backend": backend,
                "project_name_used": project_name,
                "num_x_points": len(x_values)
            })

            # 创建数据表来存储所有分布的数据点
            data = []
            
            # 为每个分布生成数据点并添加到数据表
            for a, b in alpha_beta_pairs:
                y_values = beta.pdf(x_values, a, b)
                
                # 直接记录这个 beta 分布的线图数据
                # 创建带标签的数据字典
                data_dict = {
                    "x": x_values,
                    f"Beta({a}, {b})": y_values
                }
                
                # 对于第一个分布，我们创建一个表格
                if len(data) == 0:
                    data = wandb.Table(data=[[x, y] for x, y in zip(x_values, y_values)],
                                      columns=["x", f"Beta({a}, {b})"])
                else:
                    # 对于后续分布，我们向表格添加新列
                    data.add_column(f"Beta({a}, {b})", [y for y in y_values])
            
            # 使用 wandb.plot 创建交互式线图
            beta_plot = wandb.plot.line(
                data, 
                "x", 
                [f"Beta({a}, {b})" for a, b in alpha_beta_pairs],
                title="Beta Distribution for Different α and β Parameters",
                xname="x (Value of the random variable)"
            )
            
            # 记录交互式线图
            current_run.log({"Beta Distributions": beta_plot})
            
            # 为每个 (a,b) 对记录摘要统计信息
            for a, b in alpha_beta_pairs:
                mean_val = beta.mean(a, b)
                var_val = beta.var(a, b)
                
                mode_val = np.nan
                if a > 1 and b > 1:
                    mode_val = (a - 1) / (a + b - 2)
                
                current_run.log({
                    f"Beta({a},{b})/mean": mean_val,
                    f"Beta({a},{b})/mode": mode_val if not np.isnan(mode_val) else None,
                    f"Beta({a},{b})/variance": var_val
                })
            
            print("Beta 分布数据以交互式格式记录到 W&B。")
        
        except Exception as e:
            print(f"W&B 记录过程中发生错误: {e}")

        finally:
            # 确保 wandb.finish 被调用以关闭运行
            if current_run:
                current_run.finish()
                print("W&B 运行已完成。")

    elif backend == "local":
        # 本地绘图逻辑保持不变
        fig, ax = plt.subplots(figsize=(12, 8))
        for a, b in alpha_beta_pairs:
            y_values = beta.pdf(x_values, a, b)
            ax.plot(x_values, y_values, label=f'Beta({a}, {b})')

        ax.set_title('Beta Distribution for Different α and β Parameters', fontsize=16)
        ax.set_xlabel('x (Value of the random variable)', fontsize=12)
        ax.set_ylabel('Probability Density Function (PDF)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        
        plt.show() # 在本地显示图像
        print("已在本地显示图像。没有 W&B 记录。")
        plt.close(fig) # 显示后显式关闭图像
    else:
        print(f"错误: 未知后端 '{backend}'。请选择 'wandb' 或 'local'。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="可视化 Beta 分布: 将交互式图表记录到 W&B 或在本地显示。")
    parser.add_argument(
        "--project_name",
        type=str,
        default="handson_rl-mab-visualizations",
        help="要记录到的 Weights & Biases 项目名称 (当 backend 为 'wandb' 时使用)。"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["wandb", "local"], 
        default="local",            
        help="输出后端: 'wandb' 将交互式图表记录到 W&B，'local' 仅在本地显示图像。"
    )
    args = parser.parse_args()

    x = np.linspace(0, 1, 500) 
    
    parameter_pairs = [
        (1, 1), (2, 2), (5, 5), (2, 5), (5, 2),       
        (0.5, 0.5), (0.8, 0.8),                       
        (1, 3), (3, 1),                               
        (10, 30), (30, 10)                              
    ]

    plot_beta_distributions(parameter_pairs, x, args.project_name, args.backend)
