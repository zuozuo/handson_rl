from frozen_lake import FrozenLakeEnv, PolicyIteration, ValueIteration, print_agent

def main():
    # 创建FrozenLake环境
    env = FrozenLakeEnv()
    
    # 定义动作含义符号
    action_meaning = ['↑', '↓', '←', '→']
    
    # 算法参数
    theta = 1e-5  # 收敛阈值
    gamma = 0.9   # 折扣因子
    
    # 定义陷阱和目标状态
    trap_states = [5, 7, 11, 12]  # 对应于(1,1), (1,3), (2,3), (3,0)
    goal_states = [15]            # 对应于(3,3)
    
    print("=== FrozenLake环境 ===")
    print("环境大小: 4x4")
    print("目标状态: 右下角(3,3)")
    print("陷阱状态: (1,1), (1,3), (2,3), (3,0)")
    print("动作: ↑(上), ↓(下), ←(左), →(右)")
    print("特性: 存在滑行可能性 - 行动可能导致预期之外的方向移动")
    print("\n")
    
    # 策略迭代
    print("=== 策略迭代算法 ===")
    policy_agent = PolicyIteration(env, theta, gamma)
    policy_agent.policy_iteration()
    print_agent(policy_agent, action_meaning, trap_states, goal_states)
    
    print("\n=== 价值迭代算法 ===")
    value_agent = ValueIteration(env, theta, gamma)
    value_agent.value_iteration()
    print_agent(value_agent, action_meaning, trap_states, goal_states)

if __name__ == "__main__":
    main() 