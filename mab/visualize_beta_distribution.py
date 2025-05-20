import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def plot_beta_distributions(alpha_beta_pairs, x_values):
    """
    绘制不同 alpha 和 beta 参数下的 Beta 分布图。

    参数:
    alpha_beta_pairs (list of tuples): 一个包含 (alpha, beta) 参数对的列表。
    x_values (numpy.ndarray): 用于绘制 PDF 的 x 值数组 (通常在 [0, 1] 区间)。
    """
    plt.figure(figsize=(12, 8))

    for a, b in alpha_beta_pairs:
        # 计算 Beta 分布的概率密度函数 (PDF)
        y_values = beta.pdf(x_values, a, b)
        plt.plot(x_values, y_values, label=f'Beta({a}, {b})')

    plt.title('Beta Distribution for Different α and β Parameters', fontsize=16)
    plt.xlabel('x (Value of the random variable)', fontsize=12)
    plt.ylabel('Probability Density Function (PDF)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 1)  # Beta 分布定义在 [0, 1] 区间
    plt.ylim(bottom=0) # PDF 不可能为负
    plt.show()

if __name__ == '__main__':
    # 定义一组 x 值用于绘图
    x = np.linspace(0, 1, 500) # 在 0 和 1 之间生成 500 个点

    # 定义要可视化的 (alpha, beta) 参数对
    # 这些参数对旨在展示 Beta 分布的不同形状
    parameter_pairs = [
        (1, 1),     # 均匀分布
        (2, 2),     # 对称的单峰，比 (1,1) 更集中
        (5, 5),     # 更集中的对称单峰
        (2, 5),     # 峰值偏向 0
        (5, 2),     # 峰值偏向 1
        (0.5, 0.5), # U 形分布
        (1, 3),     # 从 0 处开始递减
        (3, 1),     # 向 1 处递增
        (10, 30),   # 相对集中的峰值，偏向较小值
        (30, 10),   # 相对集中的峰值，偏向较大值
    ]

    plot_beta_distributions(parameter_pairs, x)

    # 您也可以尝试其他参数组合
    # parameter_pairs_example2 = [
    #     (0.1, 0.1),
    #     (1, 0.5),
    #     (0.5, 1),
    #     (2, 0.5),
    #     (0.5, 2),
    # ]
    # plot_beta_distributions(parameter_pairs_example2, x)
