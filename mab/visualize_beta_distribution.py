import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import wandb
import argparse # 新增：用于处理命令行参数

def plot_beta_distributions(alpha_beta_pairs, x_values, project_name):
    """
    绘制不同 alpha 和 beta 参数下的 Beta 分布图，并记录到 W&B。

    参数:
    alpha_beta_pairs (list of tuples): 一个包含 (alpha, beta) 参数对的列表。
    x_values (numpy.ndarray): 用于绘制 PDF 的 x 值数组 (通常在 [0, 1] 区间)。
    project_name (str): Weights & Biases 的项目名称。
    """
    # 初始化 W&B run
    # 使用 "visualize_beta" 作为 run 的名称或类型，可以根据需要调整
    run_name = f"beta_visualization_{'_'.join([f'a{a}b{b}' for a, b in alpha_beta_pairs[:2]])}" # 基于前几个参数对生成一个run name
    if len(alpha_beta_pairs) > 2:
        run_name += "_etc"

    wandb.init(project=project_name, name=run_name, job_type="visualization")
    print(f"Logging to W&B project: {project_name}, run: {wandb.run.name}")

    # 记录参数对本身，方便查阅
    wandb.config.alpha_beta_pairs = alpha_beta_pairs

    fig, ax = plt.subplots(figsize=(12, 8)) # 使用 subplots 以便更好地控制图像对象

    for a, b in alpha_beta_pairs:
        # 计算 Beta 分布的概率密度函数 (PDF)
        y_values = beta.pdf(x_values, a, b)
        ax.plot(x_values, y_values, label=f'Beta({a}, {b})')

    ax.set_title('Beta Distribution for Different α and β Parameters', fontsize=16)
    ax.set_xlabel('x (Value of the random variable)', fontsize=12)
    ax.set_ylabel('Probability Density Function (PDF)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0, 1)  # Beta 分布定义在 [0, 1] 区间
    ax.set_ylim(bottom=0) # PDF 不可能为负

    # 将图像记录到 W&B
    # 'Beta Distributions' 是 W&B UI 中图像的标题
    wandb.log({"Beta Distributions Plot": wandb.Image(fig)})
    print("Plot logged to W&B.")

    plt.show() # 仍然在本地显示图像
    wandb.finish() # 结束 W&B run
    print("W&B run finished.")


if __name__ == '__main__':
    # --- 新增：命令行参数解析 ---
    parser = argparse.ArgumentParser(description="Visualize Beta distributions and log to Weights & Biases.")
    parser.add_argument(
        "--project_name",
        type=str,
        default="handson_rl-mab-visualizations", # 默认的项目名称
        help="Name of the Weights & Biases project to log to."
    )
    args = parser.parse_args()
    # --- 结束：命令行参数解析 ---

    # 定义一组 x 值用于绘图
    x = np.linspace(0, 1, 500) # 在 0 和 1 之间生成 500 个点

    # 定义要可视化的 (alpha, beta) 参数对
    parameter_pairs = [
        (1, 1),
        (2, 2),
        (5, 5),
        (2, 5),
        (5, 2),
        (0.5, 0.5),
        (1, 3),
        (3, 1),
        (10, 30),
        (30, 10),
    ]

    plot_beta_distributions(parameter_pairs, x, args.project_name)
