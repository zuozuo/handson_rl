import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import wandb
import argparse

def plot_beta_distributions(alpha_beta_pairs, x_values, project_name, backend):
    """
    生成 Beta 分布数据并将其记录到 W&B 或在本地显示。
    
    当 backend 为 'wandb' 时，将 Beta 分布的 PDF 值作为指标直接记录到 W&B，
    这样您可以在 W&B 界面上直接看到所有分布的可视化曲线图。
    
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
        run_name = f"beta_direct_{'_'.join(run_name_parts)}"
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
                "num_x_points": len(x_values),
                "x_range": [float(min(x_values)), float(max(x_values))]
            })
            
            # 计算并记录每个参数对的摘要统计信息
            summary_data = {}
            for a, b in alpha_beta_pairs:
                mean_val = beta.mean(a, b)
                var_val = beta.var(a, b)
                
                mode_val = np.nan
                if a > 1 and b > 1:  # 只有当 a > 1 且 b > 1 时，众数才有定义
                    mode_val = (a - 1) / (a + b - 2)
                
                summary_data[f"Beta({a},{b})/mean"] = mean_val
                summary_data[f"Beta({a},{b})/variance"] = var_val
                if not np.isnan(mode_val):
                    summary_data[f"Beta({a},{b})/mode"] = mode_val
            
            # 一次性记录所有摘要统计信息
            current_run.log(summary_data)
            
            # 对每个 x 值，我们将创建一个新的数据点
            # 这里我们选择一个较低的采样频率，以保持数据量合理
            # 这里选择50个点，足够绘制平滑的曲线而不会产生过多数据点
            sampled_indices = np.linspace(0, len(x_values)-1, 50, dtype=int)
            sampled_x = x_values[sampled_indices]
            
            # 为每个 x 值记录一个数据点，包含不同分布的 PDF 值
            for step, x in enumerate(sampled_x):
                data_point = {"x": float(x)}  # 记录 x 值
                
                # 计算并记录每个分布在当前 x 值的 PDF 值
                for a, b in alpha_beta_pairs:
                    pdf_value = float(beta.pdf(x, a, b))
                    data_point[f"Beta({a},{b})_pdf"] = pdf_value
                
                # 使用当前步骤作为 x 轴的坐标
                current_run.log(data_point, step=step)
                
            print("Beta 分布数据直接记录到 W&B，可在 W&B 界面上查看。")
            print("提示: 在 W&B 界面上，您可以构建自定义图表，选择 '导入指标' 并选择包含 '_pdf' 后缀的指标。")
        
        except Exception as e:
            print(f"W&B 记录过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # 确保 wandb.finish 被调用以关闭运行
            if current_run:
                current_run.finish()
                print("W&B 运行已完成。")
                print(f"W&B 运行 URL: {current_run.url}")

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
    parser = argparse.ArgumentParser(description="可视化 Beta 分布: 将分布数据记录到 W&B 或在本地显示。")
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
        help="输出后端: 'wandb' 将分布数据直接记录到 W&B，'local' 仅在本地显示图像。"
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
