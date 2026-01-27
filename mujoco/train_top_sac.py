import argparse
import gymnasium
from gymnasium.wrappers import RescaleAction
import numpy as np
import torch
import os
import time
from torch.utils.tensorboard import SummaryWriter
from top_sac import TOP_SAC_Agent
from utils import MeanStdevFilter, Transition


def evaluate(env, agent, state_filter, n_episodes=5):
    returns = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        ret = 0
        while not done:
            action = agent.get_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ret += reward
        returns.append(ret)
    return np.mean(returns), np.std(returns)


def train(args):
    env = gymnasium.make(args.env)
    env = RescaleAction(env, -1, 1)
    eval_env = gymnasium.make(args.env)
    eval_env = RescaleAction(eval_env, -1, 1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_lim = float(env.action_space.high[0])

    agent = TOP_SAC_Agent(seed=args.seed, state_dim=state_dim, action_dim=action_dim,
                          action_lim=action_lim, lr=args.lr, gamma=args.gamma,
                          tau=args.tau, batchsize=args.batch_size, hidden_size=args.hidden_size,
                          n_quantiles=args.n_quantiles, kappa=args.kappa, beta=args.beta,
                          bandit_lr=args.bandit_lr, learnable_beta=args.learnable_beta, beta_lr=args.beta_lr)

    state_filter = MeanStdevFilter(state_dim) if args.use_obs_filter else None

    # Use save_dir and append seed folder, or use default runs structure
    if args.save_dir:
        log_dir = os.path.join(args.save_dir, f"seed_{args.seed}")
    else:
        log_dir = os.path.join('runs', 'top_sac', args.env, f"seed_{args.seed}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    state, _ = env.reset(seed=args.seed)
    episode_return = 0
    total_steps = 0
    episode = 0

    while total_steps < args.train_steps:
        if total_steps < args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state, deterministic=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # `RescaleAction` wrapper (and many gym wrappers) don't expose step_limit.
        # Treat `terminated` (env signalled True) as a real done; if only `truncated`
        # is True (time limit), consider it not a real terminal for bootstrap purposes.
        real_done = terminated
        agent.replay_pool.push(Transition(state, action, reward, next_state, real_done))
        if state_filter:
            state_filter.update(state)
        state = next_state
        episode_return += reward
        total_steps += 1

        if total_steps >= args.start_steps:
            beta = agent.TDC.sample()
            q1_loss, q2_loss, pi_loss = agent.optimize(args.n_updates, beta, state_filter=state_filter)

        if done:
            episode += 1
            writer.add_scalar('train/episode_return', episode_return, total_steps)
            episode_return = 0
            state, _ = env.reset()

        if total_steps % args.log_interval == 0 and total_steps >= args.start_steps:
            eval_mean, eval_std = evaluate(eval_env, agent, state_filter, n_episodes=args.n_evals)
            writer.add_scalar('eval/mean_return', eval_mean, total_steps)
            writer.add_scalar('eval/std_return', eval_std, total_steps)
            print(f"Step {total_steps}: Eval {eval_mean:.2f} ± {eval_std:.2f}")

    writer.close()
    env.close()
    eval_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v5')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--train_steps', type=int, default=1000000)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--n_updates', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_quantiles', type=int, default=50)
    parser.add_argument('--kappa', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--bandit_lr', type=float, default=0.1)
    parser.add_argument('--learnable_beta', action='store_true')
    parser.add_argument('--beta_lr', type=float, default=3e-4)
    parser.add_argument('--use_obs_filter', dest='use_obs_filter', action='store_true')
    parser.add_argument('--log_interval', type=int, default=10000)
    parser.add_argument('--n_evals', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default=None)

    args = parser.parse_args()
    train(args)
