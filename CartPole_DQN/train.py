import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
import os
import wandb
import argparse
import time
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="DQN for CartPole")
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="环境名称")
    parser.add_argument("--num_episodes", type=int, default=500, help="训练的总episode数")
    parser.add_argument("--max_steps", type=int, default=500, help="每个episode的最大步数")
    parser.add_argument("--batch_size", type=int, default=64, help="批量大小")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="初始epsilon值")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="最小epsilon值")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="epsilon衰减率")
    parser.add_argument("--buffer_capacity", type=int, default=10000, help="经验回放缓冲区容量")
    parser.add_argument("--target_update", type=int, default=10, help="目标网络更新频率")
    parser.add_argument("--eval_interval", type=int, default=20, help="评估间隔")
    parser.add_argument("--render", action="store_true", help="是否渲染环境")
    parser.add_argument("--use_gpu", action="store_true", help="是否使用GPU加速训练")
    parser.add_argument("--no_gpu", action="store_true", help="强制使用CPU训练")
    parser.add_argument("--wandb_project", type=str, default="cartpole-dqn", help="Weights & Biases项目名称")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases实体名称")
    parser.add_argument("--wandb_tags", type=str, default="", help="Weights & Biases标签，用逗号分隔")
    parser.add_argument("--no_wandb", action="store_true", help="禁用Weights & Biases")
    parser.add_argument("--log_model", action="store_true", help="将模型保存到Weights & Biases")
    parser.add_argument("--log_video", action="store_true", help="将环境视频记录到Weights & Biases")
    parser.add_argument("--video_interval", type=int, default=100, help="记录视频的间隔（episode数）")
    return parser.parse_args()

# 评估函数
def evaluate(env, agent, num_episodes=10, render=False, log_video=False, episode_num=None):
    eval_rewards = []
    
    # 如果需要记录视频，设置监视器包装器
    video_env = env
    if log_video and episode_num is not None and wandb.run is not None:
        try:
            import gym.wrappers
            video_path = f"videos/episode_{episode_num}"
            os.makedirs(video_path, exist_ok=True)
            video_env = gym.wrappers.RecordVideo(
                env, 
                video_path,
                episode_trigger=lambda x: True  # 记录所有episode
            )
        except Exception as e:
            print(f"无法创建视频记录器: {e}")
            log_video = False
    
    for episode in range(num_episodes):
        state, _ = video_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if render:
                video_env.render()
            
            # 使用贪婪策略选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
            action = q_values.argmax().item()
            
            # 执行动作
            next_state, reward, done, truncated, _ = video_env.step(action)
            done = done or truncated
            
            state = next_state
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
    
    # 如果记录了视频，上传到wandb
    if log_video and episode_num is not None and wandb.run is not None:
        try:
            import glob
            video_files = glob.glob(f"videos/episode_{episode_num}/*.mp4")
            if video_files:
                wandb.log({f"video/episode_{episode_num}": wandb.Video(video_files[0], fps=30, format="mp4")})
        except Exception as e:
            print(f"上传视频时出错: {e}")
    
    return np.mean(eval_rewards)

# 训练函数
def train(env, agent, num_episodes, max_steps, eval_interval, render, no_wandb, args):
    rewards = []
    avg_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 更新智能体
            loss = agent.update()
            
            # 记录训练步骤信息到wandb
            if loss is not None and not no_wandb:
                wandb.log({"train/step_loss": loss})
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # 更新epsilon
        agent.update_epsilon()
        
        # 记录奖励
        rewards.append(episode_reward)
        avg_reward = np.mean(rewards[-100:])
        avg_rewards.append(avg_reward)
        
        # 打印训练信息
        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # 记录episode信息到wandb
        if not no_wandb:
            wandb.log({
                "train/episode": episode + 1,
                "train/reward": episode_reward,
                "train/avg_reward": avg_reward,
                "train/epsilon": agent.epsilon
            })
        
        # 定期评估和保存模型
        if (episode + 1) % eval_interval == 0:
            # 检查是否需要记录视频
            should_log_video = not no_wandb and args.log_video and (episode + 1) % args.video_interval == 0
            
            eval_reward = evaluate(
                env, 
                agent, 
                render=(episode + 1) == num_episodes,
                log_video=should_log_video,
                episode_num=episode + 1
            )
            print(f"Evaluation at episode {episode+1}: {eval_reward}")
            agent.save(f"models/dqn_cartpole_episode_{episode+1}.pth")
            
            # 记录评估信息到wandb
            if not no_wandb:
                wandb.log({
                    "eval/episode": episode + 1,
                    "eval/reward": eval_reward
                })
                
                # 保存模型到wandb
                if args.log_model:
                    model_path = f"models/dqn_cartpole_episode_{episode+1}.pth"
                    wandb.save(model_path)
    
    # 保存最终模型
    agent.save("models/dqn_cartpole_final.pth")
    
    # 绘制奖励曲线
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Average Reward (last 100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training on CartPole')
    plt.legend()
    plt.savefig('results/reward_curve.png')
    
    # 将奖励曲线上传到wandb
    if not no_wandb:
        wandb.log({"reward_curve": wandb.Image('results/reward_curve.png')})
    
    plt.close()
    
    return rewards, avg_rewards

# 主函数
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 创建保存模型和结果的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 设置超参数
    env_name = args.env_name
    num_episodes = args.num_episodes
    max_steps = args.max_steps
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    gamma = args.gamma
    epsilon_start = args.epsilon_start
    epsilon_end = args.epsilon_end
    epsilon_decay = args.epsilon_decay
    buffer_capacity = args.buffer_capacity
    target_update = args.target_update
    eval_interval = args.eval_interval
    render = args.render
    
    # 处理GPU使用逻辑
    if args.no_gpu:
        use_gpu = False
        print("强制使用CPU训练")
    elif args.use_gpu:
        use_gpu = True
        print("启用GPU加速训练")
    else:
        # 默认行为：如果可用则使用GPU
        use_gpu = True
        print("自动检测GPU设备")
    
    # 生成运行名称
    device_suffix = "gpu" if use_gpu else "cpu"
    run_name = f"{env_name}_{device_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 初始化wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=args.wandb_tags.split(",") if args.wandb_tags else None,
            config={
                "env_name": env_name,
                "num_episodes": num_episodes,
                "max_steps": max_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "epsilon_start": epsilon_start,
                "epsilon_end": epsilon_end,
                "epsilon_decay": epsilon_decay,
                "buffer_capacity": buffer_capacity,
                "target_update": target_update,
                "eval_interval": eval_interval,
                "use_gpu": use_gpu,
                "device_type": "auto" if not args.no_gpu and not args.use_gpu else ("gpu" if use_gpu else "cpu"),
                "gpu_available_mps": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
                "gpu_available_cuda": torch.cuda.is_available(),
            }
        )

    # 初始化环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 初始化DQN智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update=target_update,
        use_gpu=use_gpu
    )

    # 创建视频目录
    if args.log_video:
        os.makedirs('videos', exist_ok=True)
    
    print("=" * 60)
    print("DQN训练配置信息:")
    print(f"环境: {env_name}")
    print(f"训练episodes: {num_episodes}")
    print(f"GPU设置: {'启用' if use_gpu else '禁用'}")
    print(f"实际使用设备: {agent.device}")
    print(f"设备类型: {agent.device.type}")
    if hasattr(torch.backends, 'mps'):
        print(f"MPS可用: {torch.backends.mps.is_available()}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print("=" * 60)
        
    print("Starting DQN training on CartPole...")
    start_time = time.time()
    rewards, avg_rewards = train(env, agent, num_episodes, max_steps, eval_interval, render, args.no_wandb, args)
    training_time = time.time() - start_time
    
    print("Training completed!")
    print(f"总训练时间: {training_time:.2f}秒")
    
    # 获取并打印性能统计
    performance_stats = agent.get_performance_stats()
    print("\n性能统计:")
    print(f"总更新次数: {performance_stats['total_updates']}")
    print(f"平均更新时间: {performance_stats['avg_update_time_ms']:.3f}ms")
    print(f"更新速度: {performance_stats['updates_per_second']:.2f} updates/sec")
    print(f"数据传输时间占比: {performance_stats['data_transfer_ratio']:.1f}%")
    print(f"前向传播时间占比: {performance_stats['forward_ratio']:.1f}%")
    print(f"反向传播时间占比: {performance_stats['backward_ratio']:.1f}%")
    
    # 记录最终性能统计到wandb
    if not args.no_wandb:
        wandb.log({
            "final/training_time_seconds": training_time,
            "final/episodes_per_second": num_episodes / training_time,
            **{f"final/{k}": v for k, v in performance_stats.items()}
        })
    
    # 加载最佳模型并进行最终评估
    agent.load("models/dqn_cartpole_final.pth")
    final_reward = evaluate(
        env, 
        agent, 
        num_episodes=10, 
        render=True,
        log_video=args.log_video and not args.no_wandb,
        episode_num="final"
    )
    print(f"Final evaluation reward: {final_reward}")
    
    # 保存最终模型到wandb
    if not args.no_wandb and args.log_model:
        wandb.save("models/dqn_cartpole_final.pth")
    
    # 关闭环境和wandb
    env.close()
    if not args.no_wandb:
        wandb.finish()
