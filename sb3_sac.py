"""
SAC using Stable Baselines3 with RL Zoo best hyperparameters
Based on: https://github.com/DLR-RM/rl-baselines3-zoo
"""
import os
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
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
        'learning_rate': 3e-4,
        'buffer_size': 1000000,
        'learning_starts': 10000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'ent_coef': 'auto',
        'target_update_interval': 1,
        'target_entropy': 'auto',
        'policy_kwargs': dict(
            net_arch=[256, 256],
            log_std_init=-3,
        ),
    }
    
    # Environment-specific tuning from RL Zoo
    env_specific = {
        'HalfCheetah-v5': {
            'learning_rate': 3e-4,
            'batch_size': 256,
            'policy_kwargs': dict(
                net_arch=[256, 256],
                log_std_init=-3,
            ),
        },
        'Hopper-v5': {
            'learning_rate': 3e-4,
            'batch_size': 256,
            'policy_kwargs': dict(
                net_arch=[256, 256],
                log_std_init=-3,
            ),
        },
        'Walker2d-v5': {
            'learning_rate': 3e-4,
            'batch_size': 256,
            'policy_kwargs': dict(
                net_arch=[256, 256],
                log_std_init=-3,
            ),
        },
        'Ant-v5': {
            'learning_rate': 3e-4,
            'batch_size': 256,
            'policy_kwargs': dict(
                net_arch=[256, 256],
                log_std_init=-3,
            ),
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
        log_dir = f"runs/sb3_sac/{args.env_id}/seed_{args.seed}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Get hyperparameters
    hyperparams = get_hyperparameters(args.env_id)
    
    # Create model
    model = SAC(
        **hyperparams,
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
    print(f"Training SAC on {args.env_id} for {args.total_timesteps} steps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        log_interval=10,
        tb_log_name="sac",
    )
    
    # Save final model
    model.save(os.path.join(log_dir, "final_model"))
    print(f"Training complete. Model saved to {log_dir}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
