"""Training script for TOP-TQC with learnable quantile dropping"""

import argparse
import gymnasium as gym
import numpy as np
import torch
from top_tqc_learnable_drop import TOP_TQC_LearnableDrop_Agent
from utils import RunningMeanStd
import os
from torch.utils.tensorboard import SummaryWriter
import time
import sys


def evaluate(env, agent, state_filter, n_episodes=5):
    """Evaluate agent performance"""
    returns = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_return = 0
        while not done:
            action = agent.get_action(state, state_filter, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
        returns.append(episode_return)
    return np.mean(returns), np.std(returns)


def train(args):
    # Create environment
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_lim = float(env.action_space.high[0])
    
    # Create agent
    agent = TOP_TQC_LearnableDrop_Agent(
        seed=args.seed,
        state_dim=state_dim,
        action_dim=action_dim,
        action_lim=action_lim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        batchsize=args.batch_size,
        hidden_size=args.hidden_size,
        update_interval=args.update_interval,
        buffer_size=args.buffer_size,
        target_noise=args.target_noise,
        target_noise_clip=args.target_noise_clip,
        explore_noise=args.explore_noise,
        n_quantiles=args.n_quantiles,
        n_critics=args.n_critics,
        top_quantiles_to_drop=args.top_quantiles_to_drop,
        kappa=args.kappa,
        beta=args.beta,
        bandit_lr=args.bandit_lr,
        learnable_drop=args.learnable_drop,
        drop_lr=args.drop_lr
    )
    
    # State normalization
    state_filter = RunningMeanStd(shape=(state_dim,))
    
    # TensorBoard logger: append seed folder to save_dir, or use default runs structure
    if args.save_dir:
        log_dir = os.path.join(args.save_dir, f"seed_{args.seed}")
    else:
        log_dir = os.path.join('runs', 'top_tqc_learnable_quantiles', args.env, f"seed_{args.seed}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training loop
    state, _ = env.reset(seed=args.seed)
    episode_return = 0
    episode_length = 0
    total_steps = 0
    episode_num = 0
    
    print(f"Training TOP-TQC-LearnableDrop on {args.env} (seed {args.seed})")
    print(f"Learnable drop: {args.learnable_drop}")
    print(f"Initial drop count: {args.top_quantiles_to_drop}")
    
    while total_steps < args.train_steps:
        # Collect experience
        if total_steps < args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state, state_filter, deterministic=False)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        real_done = terminated
        
        # Store transition
        agent.replay_pool.push(state, action, reward, next_state, real_done)
        
        # Update state normalization
        state_filter.update(state)
        
        state = next_state
        episode_return += reward
        episode_length += 1
        total_steps += 1
        
        # Train agent
        if total_steps >= args.start_steps:
            # Get beta from bandit
            beta = agent.TDC.sample()
            
            # Update agent
            q_loss, pi_loss, wd, _, _ = agent.optimize(
                n_updates=args.n_updates,
                beta=beta,
                state_filter=state_filter
            )
            
            # Update bandit (simplified - using episode return as reward)
            if done and episode_num > 0:
                agent.TDC.update(beta, episode_return)
        
        # Episode end
        if done:
            writer.add_scalar('train/episode_return', episode_return, total_steps)
            writer.add_scalar('train/episode_length', episode_length, total_steps)
            
            # Update drop count bandit at end of episode
            if args.learnable_drop:
                agent.update_drop_count_from_bandit(episode_return)
                current_drop_total = agent.get_drop_count(detach=True)
                writer.add_scalar('train/drop_count_total', current_drop_total, total_steps)
                writer.add_scalar('train/drop_count_per_critic', current_drop_total / args.n_critics, total_steps)
            
            episode_num += 1
            episode_return = 0
            episode_length = 0
            state, _ = env.reset()
        
        # Logging
        if total_steps % args.log_interval == 0 and total_steps >= args.start_steps:
            # Evaluate
            eval_mean, eval_std = evaluate(eval_env, agent, state_filter, n_episodes=args.n_evals)
            
            # Log metrics
            writer.add_scalar('eval/mean_return', eval_mean, total_steps)
            writer.add_scalar('eval/std_return', eval_std, total_steps)
            
            if args.learnable_drop:
                current_drop_total = agent.get_drop_count(detach=True)
                current_drop_per_critic = current_drop_total / agent.n_critics if agent.n_critics else current_drop_total
                writer.add_scalar('agent/drop_count_total', current_drop_total, total_steps)
                writer.add_scalar('agent/drop_count_per_critic', current_drop_per_critic, total_steps)
                print(f"Step {total_steps}: Eval Return = {eval_mean:.2f} ± {eval_std:.2f}, Drop Count (total/per_critic) = {current_drop_total}/{current_drop_per_critic}")
            else:
                print(f"Step {total_steps}: Eval Return = {eval_mean:.2f} ± {eval_std:.2f}")
    
    env.close()
    eval_env.close()
    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TOP-TQC with learnable quantile dropping')
    
    # Environment
    parser.add_argument('--env', type=str, default='HalfCheetah-v5', help='Gym environment')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Training
    parser.add_argument('--train_steps', type=int, default=1000000, help='Total training steps')
    parser.add_argument('--start_steps', type=int, default=10000, help='Steps before training starts')
    parser.add_argument('--n_updates', type=int, default=1, help='Number of updates per step')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='Replay buffer size')
    
    # Agent hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--update_interval', type=int, default=2, help='Policy update frequency')
    
    # TQC parameters
    parser.add_argument('--n_quantiles', type=int, default=50, help='Number of quantiles per critic')
    parser.add_argument('--n_critics', type=int, default=5, help='Number of critic networks')
    parser.add_argument('--top_quantiles_to_drop', type=int, default=4, help='Initial quantiles to drop per critic')
    parser.add_argument('--kappa', type=float, default=1.0, help='Huber loss parameter')
    
    # TOP parameters
    parser.add_argument('--beta', type=float, default=0.0, help='Optimism parameter')
    parser.add_argument('--bandit_lr', type=float, default=0.1, help='Bandit learning rate')
    
    # Learnable drop
    parser.add_argument('--learnable_drop', action='store_true', help='Make drop count learnable')
    parser.add_argument('--drop_lr', type=float, default=3e-4, help='Learning rate for drop parameter')
    
    # Exploration
    parser.add_argument('--target_noise', type=float, default=0.2, help='Target policy smoothing noise')
    parser.add_argument('--target_noise_clip', type=float, default=0.5, help='Target noise clip')
    parser.add_argument('--explore_noise', type=float, default=0.1, help='Exploration noise')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=10000, help='Logging interval')
    parser.add_argument('--n_evals', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--save_dir', type=str, default='runs/top_tqc_learnable_drop', help='Save directory')
    
    args = parser.parse_args()

    # Environment-specific default for TQC drop count if not provided on CLI
    if '--top_quantiles_to_drop' not in sys.argv:
        if 'Hopper' in args.env:
            args.top_quantiles_to_drop = 5
        else:
            args.top_quantiles_to_drop = 4

    train(args)
