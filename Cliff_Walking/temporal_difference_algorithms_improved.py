import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CliffWalkingEnv:
    """悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 网格世界的列数
        self.nrow = nrow  # 网格世界的行数
        # 转移矩阵 P[state][action] = [(p, next_state, reward, done)]
        self.P = self._create_P()
        
    def _create_P(self):
        """创建状态转移矩阵"""
        # 初始化转移矩阵
        P = {}
        for s in range(self.nrow * self.ncol):
            P[s] = {}
            for a in range(4):
                P[s][a] = []
        
        # 四种动作: 0-上, 1-下, 2-左, 3-右
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    state = i * self.ncol + j
                    # 判断是否是悬崖或终点
                    if state == self.nrow * self.ncol - 1:
                        # 终点
                        P[state][a].append((1.0, state, 0, True))
                        continue
                    
                    if i == self.nrow - 1 and j > 0 and j < self.ncol - 1:
                        # 悬崖
                        P[state][a].append((1.0, (self.nrow - 1) * self.ncol, -100, True))
                        continue
                    
                    # 计算下一个状态
                    next_i = i + change[a][1]
                    next_j = j + change[a][0]
                    reward = -1.0
                    done = False
                    
                    # 边界处理
                    if next_i < 0 or next_i >= self.nrow or next_j < 0 or next_j >= self.ncol:
                        next_i = i
                        next_j = j
                    
                    next_state = next_i * self.ncol + next_j
                    
                    # 如果下一个状态是悬崖
                    if next_i == self.nrow - 1 and next_j > 0 and next_j < self.ncol - 1:
                        next_state = (self.nrow - 1) * self.ncol
                        reward = -100
                        done = True
                    
                    # 如果下一个状态是终点
                    if next_state == self.nrow * self.ncol - 1:
                        done = True
                    
                    P[state][a].append((1.0, next_state, reward, done))
        
        return P
    
    def reset(self):
        """重置环境"""
        self.state = (self.nrow - 1) * self.ncol  # 起始状态在左下角
        return self.state
    
    def step(self, action):
        """执行动作"""
        # 从转移矩阵中获取下一个状态
        p, next_state, reward, done = self.P[self.state][action][0]
        self.state = next_state
        return next_state, reward, done, {}


class ImprovedSarsa:
    """改进的Sarsa算法实现（带epsilon和学习率衰减）"""
    def __init__(self, ncol, nrow, epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=0.995, 
                 alpha_start=0.1, alpha_end=0.01, alpha_decay=0.995, gamma=0.9, n_actions=4):
        self.Q_table = np.zeros([nrow * ncol, n_actions])  # 初始化Q表
        self.n_actions = n_actions  # 动作个数
        
        # 衰减参数
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.alpha = alpha_start
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_decay = alpha_decay
        
        self.gamma = gamma  # 折扣因子
        
        # 统计信息
        self.episode_count = 0
        self.epsilon_history = []
        self.alpha_history = []
        
    def take_action(self, state):
        """根据epsilon-贪婪策略选择动作"""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def best_action(self, state):
        """选择最优动作"""
        return np.argmax(self.Q_table[state])
    
    def update(self, s0, a0, r, s1, a1):
        """Sarsa更新规则"""
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error
    
    def decay_parameters(self):
        """衰减epsilon和学习率"""
        # epsilon衰减
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # 学习率衰减
        self.alpha = max(self.alpha_end, self.alpha * self.alpha_decay)
        
        # 记录历史
        self.epsilon_history.append(self.epsilon)
        self.alpha_history.append(self.alpha)
    
    def train_episode(self, env):
        """训练一个回合"""
        episode_return = 0
        state = env.reset()
        action = self.take_action(state)
        done = False
        
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = self.take_action(next_state)
            self.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            episode_return += reward
        
        # 回合结束后衰减参数
        self.decay_parameters()
        self.episode_count += 1
        
        return episode_return


class ImprovedQLearning:
    """改进的Q-learning算法实现（带epsilon和学习率衰减）"""
    def __init__(self, ncol, nrow, epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=0.995,
                 alpha_start=0.1, alpha_end=0.01, alpha_decay=0.995, gamma=0.9, n_actions=4):
        self.Q_table = np.zeros([nrow * ncol, n_actions])  # 初始化Q表
        self.n_actions = n_actions  # 动作个数
        
        # 衰减参数
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.alpha = alpha_start
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_decay = alpha_decay
        
        self.gamma = gamma  # 折扣因子
        
        # 统计信息
        self.episode_count = 0
        self.epsilon_history = []
        self.alpha_history = []
        
    def take_action(self, state):
        """根据epsilon-贪婪策略选择动作"""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def best_action(self, state):
        """选择最优动作"""
        return np.argmax(self.Q_table[state])
    
    def update(self, s0, a0, r, s1):
        """Q-learning更新规则"""
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error
    
    def decay_parameters(self):
        """衰减epsilon和学习率"""
        # epsilon衰减
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # 学习率衰减
        self.alpha = max(self.alpha_end, self.alpha * self.alpha_decay)
        
        # 记录历史
        self.epsilon_history.append(self.epsilon)
        self.alpha_history.append(self.alpha)
    
    def train_episode(self, env):
        """训练一个回合"""
        episode_return = 0
        state = env.reset()
        done = False
        
        while not done:
            action = self.take_action(state)
            next_state, reward, done, _ = env.step(action)
            self.update(state, action, reward, next_state)
            state = next_state
            episode_return += reward
        
        # 回合结束后衰减参数
        self.decay_parameters()
        self.episode_count += 1
        
        return episode_return


def train_improved_agent(agent, env, num_episodes):
    """训练改进的智能体"""
    return_list = []
    
    for episode in range(num_episodes):
        episode_return = agent.train_episode(env)
        return_list.append(episode_return)
        
        # 每1000回合打印一次进度
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}: epsilon={agent.epsilon:.4f}, alpha={agent.alpha:.4f}, "
                  f"avg_return={np.mean(return_list[-100:]):.2f}")
    
    return return_list


def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    """打印智能体的策略"""
    print("状态价值：")
    v = agent.Q_table.max(axis=1)
    for i in range(env.nrow):
        for j in range(env.ncol):
            print('%6.2f' % v[i * env.ncol + j], end=' ')
        print()

    print("策略：")
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if k == a else 'o'
                print(pi_str, end=' ')
        print()


def moving_average(a, window_size):
    """计算移动平均"""
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def plot_parameter_decay(agent, algorithm_name):
    """绘制参数衰减曲线"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(agent.epsilon_history)
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title(f'{algorithm_name} Epsilon Decay')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(agent.alpha_history)
    plt.xlabel('Episodes')
    plt.ylabel('Learning Rate (Alpha)')
    plt.title(f'{algorithm_name} Learning Rate Decay')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{algorithm_name.lower()}_parameter_decay.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 设置参数
    ncol, nrow = 12, 4
    env = CliffWalkingEnv(ncol, nrow)
    num_episodes = 5000
    
    # 动作含义
    action_meaning = ['^', 'v', '<', '>']
    
    # 悬崖位置和终点位置
    cliff = list(range(37, 47))
    goal = [47]
    
    print("=" * 60)
    print("改进的Sarsa算法训练（带参数衰减）")
    print("=" * 60)
    
    # 训练改进的Sarsa
    agent_sarsa_improved = ImprovedSarsa(ncol, nrow)
    sarsa_returns_improved = train_improved_agent(agent_sarsa_improved, env, num_episodes)
    
    print(f"\n改进Sarsa最终收敛到的回报：{sarsa_returns_improved[-10:]}")
    print(f"最终epsilon: {agent_sarsa_improved.epsilon:.6f}")
    print(f"最终学习率: {agent_sarsa_improved.alpha:.6f}")
    print("\n改进Sarsa最终策略：")
    print_agent(agent_sarsa_improved, env, action_meaning, cliff, goal)
    
    print("\n" + "=" * 60)
    print("改进的Q-learning算法训练（带参数衰减）")
    print("=" * 60)
    
    # 训练改进的Q-learning
    agent_qlearning_improved = ImprovedQLearning(ncol, nrow)
    qlearning_returns_improved = train_improved_agent(agent_qlearning_improved, env, num_episodes)
    
    print(f"\n改进Q-learning最终收敛到的回报：{qlearning_returns_improved[-10:]}")
    print(f"最终epsilon: {agent_qlearning_improved.epsilon:.6f}")
    print(f"最终学习率: {agent_qlearning_improved.alpha:.6f}")
    print("\n改进Q-learning最终策略：")
    print_agent(agent_qlearning_improved, env, action_meaning, cliff, goal)
    
    # 绘制学习曲线对比
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(moving_average(sarsa_returns_improved, 51), label='Improved Sarsa', linewidth=2)
    plt.plot(moving_average(qlearning_returns_improved, 51), label='Improved Q-learning', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.title('Improved Algorithms Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(agent_sarsa_improved.epsilon_history, label='Sarsa', linewidth=2)
    plt.plot(agent_qlearning_improved.epsilon_history, label='Q-learning', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(agent_sarsa_improved.alpha_history, label='Sarsa', linewidth=2)
    plt.plot(agent_qlearning_improved.alpha_history, label='Q-learning', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Decay Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_algorithms_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("分析总结")
    print("=" * 60)
    print("改进措施：")
    print("1. Epsilon衰减: 从0.1衰减到0.01，减少后期探索")
    print("2. 学习率衰减: 从0.1衰减到0.01，实现精细调整")
    print("3. 策略稳定性: 后期参数接近0，策略趋于稳定")
    print("\n原始算法问题：")
    print("1. 固定epsilon=0.1导致持续探索")
    print("2. 固定学习率=0.1导致Q值持续大幅更新")
    print("3. Sarsa学习探索策略价值，受探索影响") 