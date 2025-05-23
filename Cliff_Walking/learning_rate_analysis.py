import numpy as np
import matplotlib.pyplot as plt

def demonstrate_learning_rate_effect():
    """演示固定学习率vs衰减学习率的效果"""
    
    # 模拟Q值更新过程
    episodes = 1000
    true_q_value = -15.0  # 假设真实的Q值
    
    # 初始Q值
    q_fixed = 0.0  # 固定学习率的Q值
    q_decay = 0.0  # 衰减学习率的Q值
    
    # 学习率设置
    alpha_fixed = 0.1
    alpha_start = 0.1
    alpha_end = 0.01
    alpha_decay = 0.995
    
    # 记录更新历史
    q_history_fixed = []
    q_history_decay = []
    alpha_history = []
    td_errors_fixed = []
    td_errors_decay = []
    
    alpha_current = alpha_start
    
    for episode in range(episodes):
        # 模拟一个经历：有时是最优路径，有时是探索
        if np.random.random() < 0.9:  # 90%时间走最优路径
            reward = -1.0  # 正常移动
            next_q = true_q_value + np.random.normal(0, 0.5)  # 加一点噪声
        else:  # 10%时间探索（可能掉悬崖）
            reward = -100.0  # 掉悬崖
            next_q = 0.0  # 回到起点
        
        # 计算TD误差
        td_error_fixed = reward + 0.9 * next_q - q_fixed
        td_error_decay = reward + 0.9 * next_q - q_decay
        
        # 更新Q值
        update_fixed = alpha_fixed * td_error_fixed
        update_decay = alpha_current * td_error_decay
        
        q_fixed += update_fixed
        q_decay += update_decay
        
        # 记录历史
        q_history_fixed.append(q_fixed)
        q_history_decay.append(q_decay)
        alpha_history.append(alpha_current)
        td_errors_fixed.append(abs(td_error_fixed))
        td_errors_decay.append(abs(td_error_decay))
        
        # 衰减学习率
        alpha_current = max(alpha_end, alpha_current * alpha_decay)
    
    return {
        'q_fixed': q_history_fixed,
        'q_decay': q_history_decay,
        'alpha_history': alpha_history,
        'td_errors_fixed': td_errors_fixed,
        'td_errors_decay': td_errors_decay,
        'true_value': true_q_value
    }

def analyze_update_magnitudes():
    """分析不同阶段的更新幅度"""
    
    print("=" * 60)
    print("学习率对Q值更新幅度的影响分析")
    print("=" * 60)
    
    # 模拟不同场景下的TD误差
    scenarios = [
        ("正常移动", -1.0, -15.0, -14.0),  # reward, next_q, current_q
        ("探索掉悬崖", -100.0, 0.0, -14.0),
        ("微小误差", -1.0, -15.1, -14.95),
        ("噪声干扰", -1.0, -14.8, -15.0)
    ]
    
    alpha_fixed = 0.1
    alpha_decay = 0.01
    gamma = 0.9
    
    print(f"{'场景':<12} {'TD误差':<8} {'固定α更新':<10} {'衰减α更新':<10} {'更新比例':<8}")
    print("-" * 60)
    
    for scenario, reward, next_q, current_q in scenarios:
        td_error = reward + gamma * next_q - current_q
        update_fixed = alpha_fixed * td_error
        update_decay = alpha_decay * td_error
        ratio = abs(update_fixed / update_decay) if update_decay != 0 else "∞"
        
        print(f"{scenario:<12} {td_error:>7.2f} {update_fixed:>9.3f} {update_decay:>9.3f} {ratio:>7.1f}x")

def plot_convergence_comparison():
    """绘制收敛过程对比"""
    data = demonstrate_learning_rate_effect()
    
    plt.figure(figsize=(15, 10))
    
    # 子图1：Q值收敛过程
    plt.subplot(2, 3, 1)
    plt.plot(data['q_fixed'], label='固定学习率 (α=0.1)', alpha=0.7)
    plt.plot(data['q_decay'], label='衰减学习率', alpha=0.7)
    plt.axhline(y=data['true_value'], color='red', linestyle='--', label='真实Q值')
    plt.xlabel('Episodes')
    plt.ylabel('Q Value')
    plt.title('Q值收敛过程对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：后期细节（最后200个episode）
    plt.subplot(2, 3, 2)
    start_idx = len(data['q_fixed']) - 200
    plt.plot(range(start_idx, len(data['q_fixed'])), 
             data['q_fixed'][start_idx:], label='固定学习率', alpha=0.7)
    plt.plot(range(start_idx, len(data['q_decay'])), 
             data['q_decay'][start_idx:], label='衰减学习率', alpha=0.7)
    plt.axhline(y=data['true_value'], color='red', linestyle='--', label='真实Q值')
    plt.xlabel('Episodes')
    plt.ylabel('Q Value')
    plt.title('后期收敛细节')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3：学习率衰减过程
    plt.subplot(2, 3, 3)
    plt.plot(data['alpha_history'], label='衰减学习率')
    plt.axhline(y=0.1, color='red', linestyle='--', label='固定学习率')
    plt.xlabel('Episodes')
    plt.ylabel('Learning Rate')
    plt.title('学习率变化过程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4：TD误差幅度对比
    plt.subplot(2, 3, 4)
    window = 50
    td_smooth_fixed = np.convolve(data['td_errors_fixed'], np.ones(window)/window, mode='valid')
    td_smooth_decay = np.convolve(data['td_errors_decay'], np.ones(window)/window, mode='valid')
    
    plt.plot(td_smooth_fixed, label='固定学习率', alpha=0.7)
    plt.plot(td_smooth_decay, label='衰减学习率', alpha=0.7)
    plt.xlabel('Episodes')
    plt.ylabel('Average |TD Error|')
    plt.title('TD误差幅度对比（移动平均）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图5：Q值标准差（稳定性指标）
    plt.subplot(2, 3, 5)
    window = 100
    std_fixed = [np.std(data['q_fixed'][max(0, i-window):i+1]) 
                 for i in range(len(data['q_fixed']))]
    std_decay = [np.std(data['q_decay'][max(0, i-window):i+1]) 
                 for i in range(len(data['q_decay']))]
    
    plt.plot(std_fixed, label='固定学习率', alpha=0.7)
    plt.plot(std_decay, label='衰减学习率', alpha=0.7)
    plt.xlabel('Episodes')
    plt.ylabel('Q Value Std Dev')
    plt.title('Q值稳定性对比（标准差）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图6：更新幅度对比
    plt.subplot(2, 3, 6)
    updates_fixed = [abs(data['q_fixed'][i] - data['q_fixed'][i-1]) 
                     for i in range(1, len(data['q_fixed']))]
    updates_decay = [abs(data['q_decay'][i] - data['q_decay'][i-1]) 
                     for i in range(1, len(data['q_decay']))]
    
    # 移动平均平滑
    updates_smooth_fixed = np.convolve(updates_fixed, np.ones(window)/window, mode='valid')
    updates_smooth_decay = np.convolve(updates_decay, np.ones(window)/window, mode='valid')
    
    plt.plot(updates_smooth_fixed, label='固定学习率', alpha=0.7)
    plt.plot(updates_smooth_decay, label='衰减学习率', alpha=0.7)
    plt.xlabel('Episodes')
    plt.ylabel('Average Update Magnitude')
    plt.title('Q值更新幅度对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 分析更新幅度
    analyze_update_magnitudes()
    
    print("\n" + "=" * 60)
    print("固定学习率导致后期大幅更新的原因")
    print("=" * 60)
    print("1. 探索噪声：即使后期，ε-greedy仍进行10%随机探索")
    print("2. TD误差放大：固定α=0.1将任何TD误差放大10倍")
    print("3. 过度校正：大更新步长导致Q值在真实值附近振荡")
    print("4. 累积误差：连续的大幅更新导致收敛精度下降")
    print("\n解决方案：")
    print("• 学习率衰减：从0.1逐渐减小到0.01")
    print("• 精细调整：后期小步长允许精确收敛")
    print("• 稳定性：减小更新幅度，提高策略稳定性")
    
    # 绘制对比图
    plot_convergence_comparison()
    
    print(f"\n图表已保存为 'learning_rate_analysis.png'")
    print("从图中可以清楚看到固定学习率导致的振荡现象！") 