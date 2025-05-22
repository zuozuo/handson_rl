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


class Sarsa:
    """Sarsa算法实现"""
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_actions=4):
        self.Q_table = np.zeros([nrow * ncol, n_actions])  # 初始化Q表
        self.n_actions = n_actions  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略参数
        
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


class QLearning:
    """Q-learning算法实现"""
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_actions=4):
        self.Q_table = np.zeros([nrow * ncol, n_actions])  # 初始化Q表
        self.n_actions = n_actions  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略参数
        
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


def train_sarsa(env, agent, num_episodes):
    """训练Sarsa算法"""
    return_list = []
    
    for episode in range(num_episodes):
        episode_return = 0
        state = env.reset()
        action = agent.take_action(state)
        done = False
        
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = agent.take_action(next_state)
            agent.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            episode_return += reward
            
        return_list.append(episode_return)
        
    return return_list


def train_qlearning(env, agent, num_episodes):
    """训练Q-learning算法"""
    return_list = []
    
    for episode in range(num_episodes):
        episode_return = 0
        state = env.reset()
        done = False
        
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            episode_return += reward
            
        return_list.append(episode_return)
        
    return return_list


def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    """打印智能体的策略"""
    print("状态价值：")
    v = agent.Q_table.max(axis=1)
    for i in range(env.nrow):
        for j in range(env.ncol):
            # 为了输出美观，保持输出6个字符
            print('%6.2f' % v[i * env.ncol + j], end=' ')
        print()

    print("策略：")
    for i in range(env.nrow):
        for j in range(env.ncol):
            # 一些特殊的状态，例如悬崖漫步中的悬崖
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


if __name__ == "__main__":
    # 设置参数
    ncol, nrow = 12, 4
    env = CliffWalkingEnv(ncol, nrow)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    num_episodes = 500
    
    # 动作含义
    action_meaning = ['^', 'v', '<', '>']
    
    # 悬崖位置和终点位置
    cliff = list(range(37, 47))  # 悬崖位置
    goal = [47]  # 终点位置
    
    print("=" * 50)
    print("训练Sarsa算法")
    print("=" * 50)
    
    # 训练Sarsa
    agent_sarsa = Sarsa(ncol, nrow, epsilon, alpha, gamma)
    sarsa_returns = train_sarsa(env, agent_sarsa, num_episodes)
    
    print(f"Sarsa算法最终收敛到的回报：{sarsa_returns[-10:]}")
    print("\nSarsa算法最终收敛得到的策略为：")
    print_agent(agent_sarsa, env, action_meaning, cliff, goal)
    
    print("\n" + "=" * 50)
    print("训练Q-learning算法")
    print("=" * 50)
    
    # 训练Q-learning
    agent_qlearning = QLearning(ncol, nrow, epsilon, alpha, gamma)
    qlearning_returns = train_qlearning(env, agent_qlearning, num_episodes)
    
    print(f"Q-learning算法最终收敛到的回报：{qlearning_returns[-10:]}")
    print("\nQ-learning算法最终收敛得到的策略为：")
    print_agent(agent_qlearning, env, action_meaning, cliff, goal)
    
    # 绘制训练过程
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sarsa_returns, label='Sarsa', alpha=0.6)
    plt.plot(moving_average(sarsa_returns, 21), label='Sarsa (Smoothed)', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.title('Sarsa Learning Curve in Cliff Walking')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(qlearning_returns, label='Q-learning', alpha=0.6)
    plt.plot(moving_average(qlearning_returns, 21), label='Q-learning (Smoothed)', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.title('Q-learning Learning Curve in Cliff Walking')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temporal_difference_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 比较两种算法
    print("\n" + "=" * 50)
    print("算法比较")
    print("=" * 50)
    
    # 绘制对比图
    plt.figure(figsize=(10, 6))
    plt.plot(moving_average(sarsa_returns, 21), label='Sarsa', linewidth=2)
    plt.plot(moving_average(qlearning_returns, 21), label='Q-learning', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.title('Sarsa vs Q-learning Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sarsa_vs_qlearning.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Sarsa平均回报: {np.mean(sarsa_returns[-100:]):.2f}")
    print(f"Q-learning平均回报: {np.mean(qlearning_returns[-100:]):.2f}")
    
    print("\n注意：")
    print("1. Sarsa是在线策略算法，学习的是执行策略")
    print("2. Q-learning是离线策略算法，学习的是最优策略")
    print("3. 在训练过程中，Sarsa通常获得更高的回报，因为它更保守")
    print("4. Q-learning学到的策略更激进，沿着悬崖边走，理论上是最优的") 