"""
TD3 using Stable Baselines3 with RL Zoo best hyperparameters
Based on: https://github.com/DLR-RM/rl-baselines3-zoo
"""
import os
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='HalfCheetah-v5')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--total_timesteps', type=int, default=1000000)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--n_evals', type=int, default=5)
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def get_hyperparameters(env_id: str):
    """Get RL Zoo optimized hyperparameters for common MuJoCo environments"""
    # Default hyperparameters work well across MuJoCo tasks
    base_params = {
        'policy': 'MlpPolicy',
        'learning_rate': 1e-3,
        'buffer_size': 1000000,
        'learning_starts': 10000,
        'batch_size': 100,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': (1, "episode"),
        'gradient_steps': -1,  # Do as many gradient steps as steps done in the episode
        'policy_delay': 2,
        'target_policy_noise': 0.2,
        'target_noise_clip': 0.5,
        'policy_kwargs': dict(net_arch=[400, 300]),
    }
    
    # Environment-specific tuning
    env_specific = {
        'HalfCheetah-v5': {
            'learning_rate': 1e-3,
            'batch_size': 100,
            'policy_kwargs': dict(net_arch=[400, 300]),
        },
        'Hopper-v5': {
            'learning_rate': 1e-3,
            'batch_size': 100,
            'policy_kwargs': dict(net_arch=[256, 256]),
        },
        'Walker2d-v5': {
            'learning_rate': 1e-3,
            'batch_size': 100,
            'policy_kwargs': dict(net_arch=[400, 300]),
        },
        'Ant-v5': {
            'learning_rate': 1e-3,
            'batch_size': 100,
            'policy_kwargs': dict(net_arch=[400, 300]),
        },
    }
    
    # Merge base with environment-specific
    if env_id in env_specific:
        base_params.update(env_specific[env_id])
    
    return base_params


def main():
    args = parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env = gym.make(args.env_id)
    env = Monitor(env)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    
    # Create evaluation environment
    eval_env = gym.make(args.env_id)
    eval_env = Monitor(eval_env)
    eval_env.reset(seed=args.seed + 100)
    
    # Setup save directory
    if args.save_dir:
        log_dir = args.save_dir
    else:
        log_dir = f"runs/sb3_td3/{args.env_id}/seed_{args.seed}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Get hyperparameters
    hyperparams = get_hyperparameters(args.env_id)
    
    # Action noise for exploration
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )
    
    # Create model
    model = TD3(
        **hyperparams,
        action_noise=action_noise,
        tensorboard_log=log_dir,
        seed=args.seed,
        device=args.device,
        env=env
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=args.log_interval,
        n_eval_episodes=args.n_evals,
        deterministic=True,
        render=False,
    )
    
    callbacks = CallbackList([eval_callback])
    
    # Train
    print(f"Training TD3 on {args.env_id} for {args.total_timesteps} steps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        log_interval=10,
        tb_log_name="td3",
    )
    
    # Save final model
    model.save(os.path.join(log_dir, "final_model"))
    print(f"Training complete. Model saved to {log_dir}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
