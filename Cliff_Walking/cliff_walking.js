/**
 * 悬崖漫步环境
 */
class CliffWalkingEnv {
  constructor(ncol = 12, nrow = 4) {
    this.ncol = ncol; // 网格世界的列数
    this.nrow = nrow; // 网格世界的行数
    // 转移矩阵 P[state][action] = [{p, next_state, reward, done}]
    this.P = this.createP();
  }

  /**
   * 创建状态转移矩阵
   */
  createP() {
    // 初始化转移矩阵
    const P = Array(this.nrow * this.ncol).fill().map(() => 
      Array(4).fill().map(() => [])
    );

    // 四种动作: 0-上, 1-下, 2-左, 3-右
    const change = [[0, -1], [0, 1], [-1, 0], [1, 0]];
    
    for (let i = 0; i < this.nrow; i++) {
      for (let j = 0; j < this.ncol; j++) {
        for (let a = 0; a < 4; a++) {
          const state = i * this.ncol + j;
          // 判断是否是悬崖或终点
          if (state === this.nrow * this.ncol - 1) {
            // 终点
            P[state][a].push({
              p: 1.0,
              next_state: state,
              reward: 0,
              done: true
            });
            continue;
          }
          
          if (i === this.nrow - 1 && j > 0 && j < this.ncol - 1) {
            // 悬崖
            P[state][a].push({
              p: 1.0,
              next_state: (this.nrow - 1) * this.ncol,
              reward: -100,
              done: true
            });
            continue;
          }
          
          // 计算下一个状态
          let next_i = i + change[a][1];
          let next_j = j + change[a][0];
          let reward = -1.0;
          let done = false;
          
          // 边界处理
          if (next_i < 0 || next_i >= this.nrow || 
              next_j < 0 || next_j >= this.ncol) {
            next_i = i;
            next_j = j;
          }
          
          let next_state = next_i * this.ncol + next_j;
          
          // 如果下一个状态是悬崖
          if (next_i === this.nrow - 1 && next_j > 0 && next_j < this.ncol - 1) {
            next_state = (this.nrow - 1) * this.ncol;
            reward = -100;
            done = true;
          }
          
          // 如果下一个状态是终点
          if (next_state === this.nrow * this.ncol - 1) {
            done = true;
          }
          
          P[state][a].push({
            p: 1.0,
            next_state: next_state,
            reward: reward,
            done: done
          });
        }
      }
    }
    
    return P;
  }
}

/**
 * 策略迭代算法
 */
class PolicyIteration {
  constructor(env, theta = 1e-5, gamma = 0.9) {
    this.env = env;
    this.theta = theta; // 收敛阈值
    this.gamma = gamma; // 折扣因子
    this.v = Array(env.nrow * env.ncol).fill(0); // 状态价值
    this.pi = Array(env.nrow * env.ncol).fill().map(() => 
      Array(4).fill(0.25)
    ); // 初始策略为均匀随机
  }

  /**
   * 策略评估
   */
  policyEvaluation() {
    let iteration = 0;
    while (true) {
      let delta = 0;
      for (let s = 0; s < this.env.nrow * this.env.ncol; s++) {
        let v = this.v[s];
        let new_v = 0;
        
        for (let a = 0; a < 4; a++) {
          for (const {p, next_state, reward} of this.env.P[s][a]) {
            new_v += this.pi[s][a] * p * (reward + this.gamma * this.v[next_state]);
          }
        }
        
        this.v[s] = new_v;
        delta = Math.max(delta, Math.abs(v - new_v));
      }
      
      iteration++;
      
      if (delta < this.theta) {
        console.log(`策略评估进行${iteration}轮后完成`);
        break;
      }
    }
  }

  /**
   * 策略提升
   */
  policyImprovement() {
    let policy_stable = true;
    
    for (let s = 0; s < this.env.nrow * this.env.ncol; s++) {
      const old_action_probs = [...this.pi[s]];
      
      // 计算Q(s,a)
      const q_sa = Array(4).fill(0);
      
      for (let a = 0; a < 4; a++) {
        for (const {p, next_state, reward} of this.env.P[s][a]) {
          q_sa[a] += p * (reward + this.gamma * this.v[next_state]);
        }
      }
      
      // 找到最优动作
      const best_a = q_sa.indexOf(Math.max(...q_sa));
      
      // 更新策略为确定性策略
      this.pi[s] = Array(4).fill(0);
      this.pi[s][best_a] = 1.0;
      
      // 检查策略是否稳定
      if (JSON.stringify(old_action_probs) !== JSON.stringify(this.pi[s])) {
        policy_stable = false;
      }
    }
    
    console.log("策略提升完成");
    return policy_stable;
  }

  /**
   * 策略迭代
   */
  policyIteration() {
    while (true) {
      this.policyEvaluation();
      const policy_stable = this.policyImprovement();
      
      if (policy_stable) {
        break;
      }
    }
  }
}

/**
 * 价值迭代算法
 */
class ValueIteration {
  constructor(env, theta = 1e-5, gamma = 0.9) {
    this.env = env;
    this.theta = theta; // 收敛阈值
    this.gamma = gamma; // 折扣因子
    this.v = Array(env.nrow * env.ncol).fill(0); // 状态价值
    this.pi = Array(env.nrow * env.ncol).fill().map(() => 
      Array(4).fill(0.25)
    ); // 策略
  }

  /**
   * 价值迭代
   */
  valueIteration() {
    let iteration = 0;
    
    while (true) {
      let delta = 0;
      iteration++;
      
      for (let s = 0; s < this.env.nrow * this.env.ncol; s++) {
        const v = this.v[s];
        
        // 计算每个动作的价值
        const q_sa = Array(4).fill(0);
        
        for (let a = 0; a < 4; a++) {
          for (const {p, next_state, reward} of this.env.P[s][a]) {
            q_sa[a] += p * (reward + this.gamma * this.v[next_state]);
          }
        }
        
        // 更新状态价值为最大的动作价值
        this.v[s] = Math.max(...q_sa);
        
        // 计算最大误差
        delta = Math.max(delta, Math.abs(v - this.v[s]));
      }
      
      if (delta < this.theta) {
        console.log(`价值迭代一共进行${iteration}轮`);
        break;
      }
    }
    
    // 根据价值函数提取最优策略
    for (let s = 0; s < this.env.nrow * this.env.ncol; s++) {
      const q_sa = Array(4).fill(0);
      
      for (let a = 0; a < 4; a++) {
        for (const {p, next_state, reward} of this.env.P[s][a]) {
          q_sa[a] += p * (reward + this.gamma * this.v[next_state]);
        }
      }
      
      // 找到最优动作
      const best_a = q_sa.indexOf(Math.max(...q_sa));
      
      // 更新策略为确定性策略
      this.pi[s] = Array(4).fill(0);
      this.pi[s][best_a] = 1.0;
    }
  }
}

/**
 * 可视化状态价值和策略
 */
function printAgent(agent, actionMeaning, cliffPos, goalPos) {
  const env = agent.env;
  const v_table = [];
  const policy_table = [];
  
  // 构建价值表格
  for (let i = 0; i < env.nrow; i++) {
    const v_row = [];
    for (let j = 0; j < env.ncol; j++) {
      const state = i * env.ncol + j;
      v_row.push(agent.v[state].toFixed(3));
    }
    v_table.push(v_row);
  }
  
  // 构建策略表格
  for (let i = 0; i < env.nrow; i++) {
    const policy_row = [];
    for (let j = 0; j < env.ncol; j++) {
      const state = i * env.ncol + j;
      
      if (cliffPos.includes(state)) {
        policy_row.push('C'); // 悬崖
      } else if (goalPos.includes(state)) {
        policy_row.push('G'); // 目标
      } else {
        // 找到最优动作
        const best_a = agent.pi[state].indexOf(Math.max(...agent.pi[state]));
        policy_row.push(actionMeaning[best_a]);
      }
    }
    policy_table.push(policy_row);
  }
  
  // 打印状态价值
  console.log("状态价值：");
  v_table.forEach(row => {
    console.log(row.join('\t'));
  });
  
  // 打印策略
  console.log("\n策略：");
  policy_table.forEach(row => {
    console.log(row.join('\t'));
  });
}

// 执行算法
function main() {
  const env = new CliffWalkingEnv();
  const actionMeaning = ['^', 'v', '<', '>'];
  const theta = 1e-5;
  const gamma = 0.9;
  
  // 悬崖位置和目标位置
  const cliffPos = Array.from({length: 10}, (_, i) => (env.nrow - 1) * env.ncol + i + 1);
  const goalPos = [env.nrow * env.ncol - 1];
  
  console.log("策略迭代算法：");
  const policyAgent = new PolicyIteration(env, theta, gamma);
  policyAgent.policyIteration();
  printAgent(policyAgent, actionMeaning, cliffPos, goalPos);
  
  console.log("\n价值迭代算法：");
  const valueAgent = new ValueIteration(env, theta, gamma);
  valueAgent.valueIteration();
  printAgent(valueAgent, actionMeaning, cliffPos, goalPos);
}

// 运行主函数
main(); 