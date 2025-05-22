import numpy as np
import copy
import matplotlib.pyplot as plt

class FrozenLakeEnv:
    """ 冰湖环境 """
    def __init__(self, ncol=4, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    s = i * self.ncol + j  # 二维坐标转换为一维状态编号
                    
                    # 目标状态
                    if i == self.nrow - 1 and j == self.ncol - 1:
                        P[s][a] = [(1.0, s, 0, True)]
                        continue
                    
                    # 陷阱状态
                    if (i == 1 and j == 1) or (i == 1 and j == 3) or (i == 2 and j == 3) or (i == 3 and j == 0):
                        P[s][a] = [(1.0, s, 0, True)]
                        continue
                    
                    # 正常状态
                    next_x = j
                    next_y = i
                    # 冰湖环境中，我们以0.7的概率滑向正确的方向，0.1的概率滑向左边的方向，
                    # 0.1的概率滑向右边的方向，0.1的概率滑向相反的方向
                    # 计算不同滑行方向
                    slips = []
                    
                    # 向前滑行
                    forward_y = i + change[a][1]
                    forward_x = j + change[a][0]
                    if forward_x < 0 or forward_x >= self.ncol or forward_y < 0 or forward_y >= self.nrow:  # 是否超出边界
                        forward_y, forward_x = i, j
                    forward_s = forward_y * self.ncol + forward_x
                    slips.append((0.7, forward_s))
                    
                    # 向左滑行
                    left_a = (a - 1) % 4
                    left_y = i + change[left_a][1]
                    left_x = j + change[left_a][0]
                    if left_x < 0 or left_x >= self.ncol or left_y < 0 or left_y >= self.nrow:  # 是否超出边界
                        left_y, left_x = i, j
                    left_s = left_y * self.ncol + left_x
                    slips.append((0.1, left_s))
                    
                    # 向右滑行
                    right_a = (a + 1) % 4
                    right_y = i + change[right_a][1]
                    right_x = j + change[right_a][0]
                    if right_x < 0 or right_x >= self.ncol or right_y < 0 or right_y >= self.nrow:  # 是否超出边界
                        right_y, right_x = i, j
                    right_s = right_y * self.ncol + right_x
                    slips.append((0.1, right_s))
                    
                    # 向后滑行
                    back_a = (a + 2) % 4
                    back_y = i + change[back_a][1]
                    back_x = j + change[back_a][0]
                    if back_x < 0 or back_x >= self.ncol or back_y < 0 or back_y >= self.nrow:  # 是否超出边界
                        back_y, back_x = i, j
                    back_s = back_y * self.ncol + back_x
                    slips.append((0.1, back_s))
                    
                    P[s][a] = []
                    for p, next_s in slips:
                        # 陷阱状态
                        is_trap = ((next_s // self.ncol == 1 and next_s % self.ncol == 1) or
                                  (next_s // self.ncol == 1 and next_s % self.ncol == 3) or
                                  (next_s // self.ncol == 2 and next_s % self.ncol == 3) or
                                  (next_s // self.ncol == 3 and next_s % self.ncol == 0))
                        
                        # 目标状态
                        is_goal = (next_s // self.ncol == self.nrow - 1 and next_s % self.ncol == self.ncol - 1)
                        
                        # 如果到达终点或者掉入陷阱，游戏结束
                        done = is_trap or is_goal
                        # 掉入陷阱获得0奖励，到达终点获得1奖励，其他情况获得0奖励
                        reward = 1.0 if is_goal else 0.0
                        
                        P[s][a].append((p, next_s, reward, done))
        return P

class PolicyIteration:
    """ 策略迭代算法 """
    def __init__(self, env, theta=1e-5, gamma=0.9):
        self.env = env
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子
        
        # 状态数量和动作数量
        self.n_states = env.nrow * env.ncol
        self.n_actions = 4
        
        # 初始化随机策略和状态价值
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions  # 随机策略
        self.v = np.zeros(self.n_states)  # 状态价值函数初始化为0
        
    def policy_evaluation(self):
        """ 策略评估 """
        cnt = 0  # 记录迭代次数
        while True:
            cnt += 1
            delta = 0
            for s in range(self.n_states):
                old_v = self.v[s]
                self.v[s] = 0
                # 利用当前策略和贝尔曼期望方程计算状态价值
                for a in range(self.n_actions):
                    for p, next_s, r, done in self.env.P[s][a]:
                        self.v[s] += self.policy[s][a] * p * (r + self.gamma * self.v[next_s] * (1 - done))
                delta = max(delta, abs(old_v - self.v[s]))
            if delta < self.theta:
                break
        print(f"策略评估进行{cnt}轮后完成")
        return self.v
    
    def policy_improvement(self):
        """ 策略提升 """
        policy_stable = True
        for s in range(self.n_states):
            old_policy = self.policy[s].copy()
            # 计算新策略
            q_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for p, next_s, r, done in self.env.P[s][a]:
                    q_values[a] += p * (r + self.gamma * self.v[next_s] * (1 - done))
            
            # 更新为贪婪策略
            best_a = np.argmax(q_values)
            self.policy[s] = np.eye(self.n_actions)[best_a]
            
            # 检查策略是否稳定
            if not np.all(old_policy == self.policy[s]):
                policy_stable = False
        print("策略提升完成")
        return policy_stable
    
    def policy_iteration(self):
        """ 策略迭代 """
        while True:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            if policy_stable:
                break
        return self.policy

class ValueIteration:
    """ 价值迭代算法 """
    def __init__(self, env, theta=1e-5, gamma=0.9):
        self.env = env
        self.theta = theta  # 价值迭代收敛阈值
        self.gamma = gamma  # 折扣因子
        
        # 状态数量和动作数量
        self.n_states = env.nrow * env.ncol
        self.n_actions = 4
        
        # 初始化状态价值和策略
        self.v = np.zeros(self.n_states)  # 状态价值函数初始化为0
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions  # 初始化为随机策略
        
    def value_iteration(self):
        """ 价值迭代 """
        cnt = 0  # 记录迭代次数
        while True:
            cnt += 1
            delta = 0
            for s in range(self.n_states):
                old_v = self.v[s]
                # 计算动作价值
                q_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    for p, next_s, r, done in self.env.P[s][a]:
                        q_values[a] += p * (r + self.gamma * self.v[next_s] * (1 - done))
                
                # 更新状态价值为最大动作价值
                self.v[s] = np.max(q_values)
                delta = max(delta, abs(old_v - self.v[s]))
            
            if delta < self.theta:
                break
        
        print(f"价值迭代一共进行{cnt}轮")
        
        # 根据最终的状态价值计算最优策略
        for s in range(self.n_states):
            q_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for p, next_s, r, done in self.env.P[s][a]:
                    q_values[a] += p * (r + self.gamma * self.v[next_s] * (1 - done))
            
            # 更新为贪婪策略
            best_a = np.argmax(q_values)
            self.policy[s] = np.eye(self.n_actions)[best_a]
        
        return self.v, self.policy

def print_agent(agent, action_meaning, trap_states=[], goal_states=[]):
    """ 打印智能体的策略和价值函数 """
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            s = i * agent.env.ncol + j
            print(f'{agent.v[s]:.3f}', end=' ')
        print()
    
    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            s = i * agent.env.ncol + j
            if s in trap_states:
                print('**** ', end='')
            elif s in goal_states:
                print('EEEE ', end='')
            else:
                a = np.argmax(agent.policy[s])
                pi_str = action_meaning[a]
                print(pi_str + '    ', end='')
        print()

if __name__ == "__main__":
    # FrozenLake环境测试
    env = FrozenLakeEnv()
    
    # 策略迭代
    print("=== 策略迭代算法 ===")
    action_meaning = ['↑', '↓', '←', '→']
    theta = 1e-5
    gamma = 0.9
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()
    # 陷阱状态和目标状态
    trap_states = [5, 7, 11, 12]
    goal_states = [15]
    print_agent(agent, action_meaning, trap_states, goal_states)
    
    print("\n=== 价值迭代算法 ===")
    agent = ValueIteration(env, theta, gamma)
    agent.value_iteration()
    print_agent(agent, action_meaning, trap_states, goal_states) 