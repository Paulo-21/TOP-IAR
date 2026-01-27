import random
from argparse import ArgumentParser
import sys
import os
from collections import deque

import gymnasium
from gymnasium.wrappers import RescaleAction
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Callable
from tqdm import tqdm

from top_tqc import TOP_TQC_Agent
from utils import MeanStdevFilter, Transition, make_gif

import time

def train_agent_model_free(agent: TOP_TQC_Agent, env, params: Dict) -> None:

    update_timestep = params['update_every_n_steps']
    seed = params['seed']
    log_interval = params.get('log_interval', 1000)
    gif_interval = 10000
    n_random_actions = params['n_random_actions']
    n_evals = params['n_evals']
    n_collect_steps = params['n_collect_steps']
    use_statefilter = params['obs_filter']
    save_model = params['save_model']

    assert n_collect_steps > agent.batchsize, "We must initially collect as many steps as the batch size!"

    avg_length = 0
    time_step = 0
    cumulative_timestep = 0
    cumulative_log_timestep = 0
    n_updates = 0
    i_episode = 0
    log_episode = 0
    samples_number = 0
    episode_rewards = []
    episode_steps = []

    if use_statefilter:
        state_filter = MeanStdevFilter(env.env.observation_space.shape[0])
    else:
        state_filter = None

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env.action_space.seed(seed)

    max_steps = env.spec.max_episode_steps

    file_algo = os.path.splitext(os.path.basename(__file__))[0]
    algo_name = params.get('algorithm') or file_algo
    env_name = params.get('env') or params.get('env_id')
    com = f"{algo_name}_{env_name}_nq{params['n_quantiles']}_nc{params['n_critics']}_drop{params['top_quantiles_to_drop']}_beta{params['bandit_lr']}_seed{seed}"
    if params.get('save_dir'):
        # when a save_dir is provided, append seed folder to it
        run_dir = os.path.join(params.get('save_dir'), f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=run_dir)
    else:
        run_dir = os.path.join('runs', algo_name, env_name, f"seed_{seed}", f"id_{int(time.time())}")
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard'))

    prev_episode_reward = 0
    train_steps = int(params.get('train_steps', int(1e6)))
    with tqdm(total=train_steps, desc="Training TOP-TQC") as pbar:
        while samples_number < train_steps:
            time_step = 0
            episode_reward = 0
            i_episode += 1
            log_episode += 1
            state, info = env.reset(seed=seed)
            if state_filter:
                state_filter.update(state)
            done = False

            # Sample an optimism setting for this episode
            # If learnable_beta=True, use the learned value; otherwise sample from bandit
            if agent.learnable_beta:
                optimism = agent.get_beta(detach=True)
            else:
                optimism = agent.TDC.sample()

            while (not done):
                cumulative_log_timestep += 1
                cumulative_timestep += 1
                time_step += 1
                samples_number += 1
                pbar.update(1)
                if samples_number < n_random_actions:
                    action = env.action_space.sample()
                else:
                    action = agent.get_action(state, state_filter=state_filter)

                nextstate, reward, done, truncated, _ = env.step(action)
                done = done or truncated  # Episode ends on either done or truncated
                # if we hit the time-limit, it's not a 'real' done
                real_done = False if time_step == max_steps else done
                agent.replay_pool.push(Transition(state, action, reward, nextstate, real_done))
                state = nextstate
                if state_filter:
                    state_filter.update(state)
                episode_reward += reward
                # update if it's time
                if cumulative_timestep % update_timestep == 0 and cumulative_timestep > n_collect_steps:
                    q_loss, pi_loss, avg_wd, q1, q2 = agent.optimize(update_timestep, optimism, state_filter=state_filter)
                    n_updates += 1
                # logging
                if cumulative_timestep % log_interval == 0 and cumulative_timestep > n_collect_steps:
                    writer.add_scalar('Loss/Q-func', q_loss, cumulative_timestep)
                    writer.add_scalar('Loss/WD', avg_wd, cumulative_timestep)
                    writer.add_scalar('Distributions/Mean_1', torch.mean(q1), cumulative_timestep)
                    writer.add_scalar('Distributions/Median_1', torch.median(q1), cumulative_timestep)
                    writer.add_scalar('Distributions/Mean_2', torch.mean(q2), cumulative_timestep)
                    writer.add_scalar('Distributions/Median_2', torch.median(q2), cumulative_timestep)

                    # bandit tracking
                    writer.add_scalar('Distributions/optimism', optimism, cumulative_timestep)
                    arm_probs = agent.TDC.get_probs()
                    for i, p in enumerate(arm_probs):
                        writer.add_scalar(f'Distributions/arm{i}', p, cumulative_timestep)

                    # Track beta if learnable (use detach=True for logging)
                    if agent.learnable_beta:
                        writer.add_scalar('Distributions/beta', agent.get_beta(detach=True), cumulative_timestep)

                    if pi_loss:
                        writer.add_scalar('Loss/policy', pi_loss, cumulative_timestep)
                    avg_length = np.mean(episode_steps) if episode_steps else 0
                    running_reward = np.mean(episode_rewards) if episode_rewards else 0
                    
                    # Pause progress bar during evaluation
                    pbar.set_postfix({'status': 'evaluating...'})
                    pbar.refresh()
                    eval_reward = evaluate_agent(env, agent, state_filter, n_starts=n_evals)
                    
                    writer.add_scalar('Reward/Train', running_reward, cumulative_timestep)
                    writer.add_scalar('Reward/Test', eval_reward, cumulative_timestep)
                    pbar.set_postfix({
                        'episode': i_episode,
                        'test_reward': f'{eval_reward:.2f}',
                        'train_reward': f'{running_reward:.2f}',
                        'avg_len': f'{avg_length:.1f}'
                    })
                    episode_steps = []
                    episode_rewards = []
                if cumulative_timestep % gif_interval == 0:
                    # make_gif(agent, env, cumulative_timestep, state_filter, name=com, out_dir=run_dir)
                    pass

            episode_steps.append(time_step)
            episode_rewards.append(episode_reward)

            # Update bandit (for non-learnable beta)
            feedback = episode_reward - prev_episode_reward
            if not agent.learnable_beta:
                agent.TDC.update_dists(feedback)
            
            # For learnable beta: meta-learning from episode returns
            if agent.learnable_beta:
                if len(episode_rewards) > 10:
                    baseline = np.mean(episode_rewards[-10:])
                else:
                    baseline = 0.0
                agent.update_beta_from_return(episode_reward, baseline)
                current_beta = agent.get_beta(detach=True)
                writer.add_scalar('Params/Beta', current_beta, cumulative_timestep)
            
            prev_episode_reward = episode_reward

            # Log per-episode reward to TensorBoard (match CleanRL-style logging)
            try:
                writer.add_scalar('Reward/Episode', episode_reward, cumulative_timestep)
            except Exception:
                pass

    # close writer to flush tensorboard events
    try:
        writer.close()
    except Exception:
        pass


def evaluate_agent(
    env,
    agent: TOP_TQC_Agent,
    state_filter: Callable,
    n_starts: int = 1
) -> float:

    reward_sum = 0
    for _ in range(n_starts):
        done = False
        state, info = env.reset()
        while not done:
            action = agent.get_action(state, state_filter=state_filter, deterministic=True)
            nextstate, reward, done, truncated, _ = env.step(action)
            done = done or truncated  # Episode ends on either done or truncated
            reward_sum += reward
            state = nextstate
    return reward_sum / n_starts


def main():

    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v5')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--use_obs_filter', dest='obs_filter', action='store_true')
    parser.add_argument('--update_every_n_steps', type=int, default=1)
    parser.add_argument('--n_random_actions', type=int, default=25000)
    parser.add_argument('--n_collect_steps', type=int, default=1000)
    parser.add_argument('--n_evals', type=int, default=1)
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--n_quantiles', type=int, default=50)
    parser.add_argument('--n_critics', type=int, default=5)
    parser.add_argument('--top_quantiles_to_drop', type=int, default=4)
    parser.add_argument('--bandit_lr', type=float, default=0.1)
    parser.add_argument('--learnable_beta', dest='learnable_beta', action='store_true')
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--beta_lr', type=float, default=3e-4)
    parser.add_argument('--preserve_distribution', dest='preserve_distribution', action='store_true',
                        help='If set, applies optimism to distribution directly instead of collapsing to scalar')
    parser.add_argument('--log_interval', type=int, default=1000, help='Step interval for periodic logging/evaluation')
    parser.add_argument('--train_steps', type=int, default=1000000, help='Total number of environment steps to collect')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to write logs/checkpoints for this run')
    parser.add_argument('--algorithm', type=str, default=None, help='Algorithm name (used to name run folders)')
    parser.set_defaults(obs_filter=False)
    parser.set_defaults(save_model=False)
    parser.set_defaults(learnable_beta=False)
    parser.set_defaults(preserve_distribution=False)

    args = parser.parse_args()

    # If the user did not explicitly provide --top_quantiles_to_drop on the CLI,
    # set sensible environment-specific defaults.
    # Hopper tends to need more truncation (per-critic drops = 5), otherwise use 4.
    if '--top_quantiles_to_drop' not in sys.argv:
        if 'Hopper' in args.env:
            args.top_quantiles_to_drop = 10
        else:
            args.top_quantiles_to_drop = 4

    params = vars(args)

    seed = params['seed']
    env = gymnasium.make(params['env'])
    env = RescaleAction(env, -1, 1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # initialize agent
    agent = TOP_TQC_Agent(
        seed, state_dim, action_dim,
        n_quantiles=params['n_quantiles'],
        n_critics=params['n_critics'],
        top_quantiles_to_drop=params['top_quantiles_to_drop'],
        bandit_lr=params['bandit_lr'],
        learnable_beta=params['learnable_beta'],
        beta=params['beta'],
        beta_lr=params['beta_lr'],
        preserve_distribution=params.get('preserve_distribution', False)
    )

    # train agent
    train_agent_model_free(agent=agent, env=env, params=params)


if __name__ == '__main__':
    main()
