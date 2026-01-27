# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass, fields
import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# removed dependency on `tyro`; using argparse with dataclass introspection
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    log_interval: int = 1000
    """step interval for periodic logging/evaluation (0 to disable)"""
    save_dir: str = None
    """directory to write logs/checkpoints for this run"""
    algorithm: str = None
    """algorithm name used to group runs (optional)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    n_evals: int = 1
    """number of deterministic evaluation episodes to run at each log interval"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def evaluate_agent(env_id, actor, device, n_evals=1, seed=0):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    total = 0.0
    for _ in range(n_evals):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                _, _, mean = actor.get_action(torch.Tensor(obs).to(device))
                action = mean.detach().cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward
        total += ep_ret
    env.close()
    return total / max(1, n_evals)


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":

    def parse_args_dataclass(cls):
        parser = argparse.ArgumentParser()
        for f in fields(cls):
            name = f.name
            default = f.default
            if isinstance(default, bool):
                # store_true if default is False, else store_false to allow --flag to toggle
                if default is False:
                    parser.add_argument(f"--{name}", action="store_true")
                else:
                    parser.add_argument(f"--{name}", action="store_false")
            else:
                # argparse `type` must be callable; when dataclass default is None
                # `type(default)` is `NoneType` which is not callable. Use `str`
                # for such fields so CLI values are accepted.
                if default is None:
                    parser.add_argument(f"--{name}", type=str, default=default)
                else:
                    parser.add_argument(f"--{name}", type=type(default), default=default)
        ns = parser.parse_args()
        return cls(**vars(ns))

    args = parse_args_dataclass(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    # prepare run directory and writer
    env_name = getattr(args, 'env_id', None) or getattr(args, 'env', None)
    algo_name = args.algorithm or args.exp_name
    if args.save_dir:
        # if a save_dir is provided by the caller, use it directly as the run directory
        run_dir = args.save_dir
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=run_dir)
    else:
        run_dir = os.path.join('runs', algo_name, env_name, f"seed_{args.seed}", f"id_{int(time.time())}")
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard'))

    obs, _ = envs.reset(seed=args.seed)
    with tqdm(total=args.total_timesteps, desc="Training SAC") as pbar:
        for global_step in range(args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None:
                        ep_ret = info['episode']['r']
                        pbar.set_postfix({'episode_return': f"{ep_ret:.2f}", 'sps': int(global_step / (time.time() - start_time))})
                        writer.add_scalar('Reward/Episode', ep_ret, global_step)
                        try:
                            writer.add_scalar('Reward/Train', ep_ret, global_step)
                        except Exception:
                            pass
                        break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            if "final_observation" in infos:
                for idx, trunc in enumerate(truncations):
                    if trunc:
                        real_next_obs[idx] = infos["final_observation"][idx]
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # Update progress bar every step
            pbar.update(1)
            if global_step % 100 == 0:
                pbar.set_postfix({'sps': int(global_step / (time.time() - start_time))})

            # Periodic deterministic evaluation (log_interval)
            if args.log_interval and global_step % args.log_interval == 0 and global_step > args.learning_starts:
                if writer is not None:
                    try:
                        eval_reward = evaluate_agent(args.env_id, actor, device, n_evals=args.n_evals, seed=args.seed)
                        writer.add_scalar('Reward/Test', eval_reward, global_step)
                    except Exception:
                        pass

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf2_a_values = qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # Logging diagnostics similar to TOP-TQC (only at log_interval)
                if writer is not None and args.log_interval and global_step % args.log_interval == 0:
                    try:
                        # Q losses (map to TOP naming)
                        writer.add_scalar('Loss/Q-func_1', qf1_loss.item(), global_step)
                        writer.add_scalar('Loss/Q-func_2', qf2_loss.item(), global_step)
                        writer.add_scalar('Loss/Q', qf_loss.item(), global_step)

                        # Distributional summaries (use mean/median of scalar Qs)
                        mean_q1 = float(qf1_a_values.mean().detach().cpu().numpy())
                        mean_q2 = float(qf2_a_values.mean().detach().cpu().numpy())
                        med_q1 = float(np.median(qf1_a_values.detach().cpu().numpy()))
                        med_q2 = float(np.median(qf2_a_values.detach().cpu().numpy()))
                        writer.add_scalar('Distributions/Mean_1', mean_q1, global_step)
                        writer.add_scalar('Distributions/Median_1', med_q1, global_step)
                        writer.add_scalar('Distributions/Mean_2', mean_q2, global_step)
                        writer.add_scalar('Distributions/Median_2', med_q2, global_step)

                        # Wasserstein distance not applicable for scalar Qs; log 0 for consistency
                        writer.add_scalar('Loss/WD', 0.0, global_step)

                        # Entropy / alpha
                        writer.add_scalar('Entropy/alpha', float(alpha), global_step)
                    except Exception:
                        pass

                if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(args.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = actor.get_action(data.observations)
                        qf1_pi = qf1(data.observations, pi)
                        qf2_pi = qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(data.observations)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

                        # Log policy loss and updated alpha using TOP-style tags (only at log_interval)
                        if writer is not None and args.log_interval and global_step % args.log_interval == 0:
                            try:
                                writer.add_scalar('Loss/policy', actor_loss.item(), global_step)
                                writer.add_scalar('Entropy/alpha', float(alpha), global_step)
                            except Exception:
                                pass

                # update the target networks
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    envs.close()
    # optionally save models
    # if args.save_model:
    #     ckpt_dir = os.path.join(run_dir, 'checkpoints')
    #     os.makedirs(ckpt_dir, exist_ok=True)
    #     torch.save({'actor': actor.state_dict(), 'qf1': qf1.state_dict(), 'qf2': qf2.state_dict()},
    #                os.path.join(ckpt_dir, f'model-{run_name}.pt'))
    writer.close()
