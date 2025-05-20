import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import wandb
import argparse

def plot_beta_distributions(alpha_beta_pairs, x_values, project_name, backend): # 新增 backend 参数
    """
    绘制不同 alpha 和 beta 参数下的 Beta 分布图。
    根据 backend 参数决定是否记录到 W&B。

    参数:
    alpha_beta_pairs (list of tuples): 一个包含 (alpha, beta) 参数对的列表。
    x_values (numpy.ndarray): 用于绘制 PDF 的 x 值数组 (通常在 [0, 1] 区间)。
    project_name (str): Weights & Biases 的项目名称 (仅当 backend='wandb' 时使用)。
    backend (str): 'wandb' 或 'local'，决定输出目标。
    """
    if backend == "wandb":
        run_name = f"beta_visualization_{'_'.join([f'a{a}b{b}' for a, b in alpha_beta_pairs[:2]])}"
        if len(alpha_beta_pairs) > 2:
            run_name += "_etc"
        wandb.init(project=project_name, name=run_name, job_type="visualization")
        print(f"Logging to W&B project: {project_name}, run: {wandb.run.name}")
        wandb.config.alpha_beta_pairs = alpha_beta_pairs
        wandb.config.backend = backend # 记录 backend 选择
        wandb.config.project_name_used = project_name # 记录实际使用的 project_name

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

    if backend == "wandb":
        wandb.log({"Beta Distributions Plot": wandb.Image(fig)})
        print("Plot logged to W&B.")

    plt.show() # 始终在本地显示图像

    if backend == "wandb":
        wandb.finish()
        print("W&B run finished.")
    elif backend == "local":
        print("Plot displayed locally. No W&B logging.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Beta distributions and optionally log to Weights & Biases.")
    parser.add_argument(
        "--project_name",
        type=str,
        default="handson_rl-mab-visualizations",
        help="Name of the Weights & Biases project to log to (used if backend is 'wandb')."
    )
    # --- 新增：backend 参数 ---
    parser.add_argument(
        "--backend",
        type=str,
        choices=["wandb", "local"], # 限制可选值
        default="local",            # 默认值为 'local'
        help="Choose the output backend: 'wandb' to log to Weights & Biases, 'local' to display locally only."
    )
    args = parser.parse_args()

    x = np.linspace(0, 1, 500)
    parameter_pairs = [
        (1, 1), (2, 2), (5, 5), (2, 5), (5, 2),
        (0.5, 0.5), (1, 3), (3, 1), (10, 30), (30, 10),
    ]

    plot_beta_distributions(parameter_pairs, x, args.project_name, args.backend)
