import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import wandb
import time
import argparse

# 定义Q网络结构
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# DQN智能体
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, 
                 buffer_capacity=10000, batch_size=64, target_update=10, use_gpu=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子
        self.use_gpu = use_gpu  # 是否使用GPU
        
        # epsilon贪婪策略参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 经验回放参数
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        
        # 训练时间统计
        self.total_update_time = 0.0
        self.update_count_for_timing = 0
        self.data_transfer_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        
        # 创建Q网络和目标网络
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_update = target_update
        self.update_count = 0
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 设备配置 - 支持Mac M系列GPU
        if self.use_gpu:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                print("警告: 请求使用GPU但GPU不可用，将使用CPU")
        else:
            self.device = torch.device("cpu")
            
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        
        print(f"使用设备: {self.device}")
        print(f"GPU使用设置: {'启用' if self.use_gpu else '禁用'}")
        print(f"实际使用设备类型: {self.device.type}")
    
    def select_action(self, state):
        # epsilon贪婪策略选择动作
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def update_epsilon(self):
        # 更新epsilon值
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def store_transition(self, state, action, reward, next_state, done):
        # 存储经验到回放缓冲区
        self.buffer.add(state, action, reward, next_state, done)
    
    def update(self):
        # 如果缓冲区中的样本不足，则不进行更新
        if len(self.buffer) < self.batch_size:
            return
        
        # 记录整个更新过程的开始时间
        update_start_time = time.time()
        
        # 从缓冲区中采样一批经验
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # 数据转换和传输到设备的时间统计
        data_transfer_start = time.time()
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        data_transfer_end = time.time()
        
        # 前向传播时间统计
        forward_start = time.time()
        # 计算当前Q值
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = F.mse_loss(q_values, target_q_values)
        forward_end = time.time()
        
        # 反向传播时间统计
        backward_start = time.time()
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        
        # 记录梯度信息
        total_norm = 0
        for p in self.q_network.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.optimizer.step()
        backward_end = time.time()
        
        # 计算各阶段耗时
        update_end_time = time.time()
        total_update_time = update_end_time - update_start_time
        data_transfer_time = data_transfer_end - data_transfer_start
        forward_time = forward_end - forward_start
        backward_time = backward_end - backward_start
        
        # 累积时间统计
        self.total_update_time += total_update_time
        self.update_count_for_timing += 1
        self.data_transfer_time += data_transfer_time
        self.forward_time += forward_time
        self.backward_time += backward_time
        
        # 记录Q值统计信息
        q_min = q_values.min().item()
        q_max = q_values.max().item()
        q_mean = q_values.mean().item()
        target_q_min = target_q_values.min().item()
        target_q_max = target_q_values.max().item()
        target_q_mean = target_q_values.mean().item()
        
        # 计算平均时间
        avg_update_time = self.total_update_time / self.update_count_for_timing
        avg_data_transfer_time = self.data_transfer_time / self.update_count_for_timing
        avg_forward_time = self.forward_time / self.update_count_for_timing
        avg_backward_time = self.backward_time / self.update_count_for_timing
        
        # 记录到wandb（如果wandb可用）
        try:
            if wandb.run is not None:
                wandb.log({
                    # Q值统计
                    "train/q_min": q_min,
                    "train/q_max": q_max,
                    "train/q_mean": q_mean,
                    "train/target_q_min": target_q_min,
                    "train/target_q_max": target_q_max,
                    "train/target_q_mean": target_q_mean,
                    "train/gradient_norm": total_norm,
                    "train/loss": loss.item(),
                    
                    # 性能统计
                    "performance/device_type": self.device.type,
                    "performance/use_gpu": self.use_gpu,
                    "performance/update_time_ms": total_update_time * 1000,
                    "performance/avg_update_time_ms": avg_update_time * 1000,
                    "performance/data_transfer_time_ms": data_transfer_time * 1000,
                    "performance/avg_data_transfer_time_ms": avg_data_transfer_time * 1000,
                    "performance/forward_time_ms": forward_time * 1000,
                    "performance/avg_forward_time_ms": avg_forward_time * 1000,
                    "performance/backward_time_ms": backward_time * 1000,
                    "performance/avg_backward_time_ms": avg_backward_time * 1000,
                    "performance/updates_per_second": 1.0 / avg_update_time if avg_update_time > 0 else 0,
                    "performance/total_updates": self.update_count_for_timing,
                    
                    # 时间分布百分比
                    "performance/data_transfer_ratio": (data_transfer_time / total_update_time) * 100 if total_update_time > 0 else 0,
                    "performance/forward_ratio": (forward_time / total_update_time) * 100 if total_update_time > 0 else 0,
                    "performance/backward_ratio": (backward_time / total_update_time) * 100 if total_update_time > 0 else 0,
                })
        except:
            pass  # 如果wandb不可用，则忽略
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        if self.update_count_for_timing == 0:
            return {
                "device": str(self.device),
                "use_gpu": self.use_gpu,
                "total_updates": 0,
                "avg_update_time_ms": 0,
                "avg_data_transfer_time_ms": 0,
                "avg_forward_time_ms": 0,
                "avg_backward_time_ms": 0,
                "updates_per_second": 0
            }
        
        avg_update_time = self.total_update_time / self.update_count_for_timing
        avg_data_transfer_time = self.data_transfer_time / self.update_count_for_timing
        avg_forward_time = self.forward_time / self.update_count_for_timing
        avg_backward_time = self.backward_time / self.update_count_for_timing
        
        return {
            "device": str(self.device),
            "use_gpu": self.use_gpu,
            "total_updates": self.update_count_for_timing,
            "total_time_seconds": self.total_update_time,
            "avg_update_time_ms": avg_update_time * 1000,
            "avg_data_transfer_time_ms": avg_data_transfer_time * 1000,
            "avg_forward_time_ms": avg_forward_time * 1000,
            "avg_backward_time_ms": avg_backward_time * 1000,
            "updates_per_second": 1.0 / avg_update_time if avg_update_time > 0 else 0,
            "data_transfer_ratio": (avg_data_transfer_time / avg_update_time) * 100 if avg_update_time > 0 else 0,
            "forward_ratio": (avg_forward_time / avg_update_time) * 100 if avg_update_time > 0 else 0,
            "backward_ratio": (avg_backward_time / avg_update_time) * 100 if avg_update_time > 0 else 0
        }
    
    def reset_performance_stats(self):
        """重置性能统计信息"""
        self.total_update_time = 0.0
        self.update_count_for_timing = 0
        self.data_transfer_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
    
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)
    
    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())
