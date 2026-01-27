import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import ReplayPool, calculate_quantile_huber_loss
from bandit import ExpWeights
from typing import Callable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, log_std_min=-10, log_std_max=2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_size, action_dim)
        self.logstd_head = nn.Linear(hidden_size, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs, compute_pi=True, compute_log_pi=True):
        h = self.trunk(obs)
        mu = self.mu_head(h)
        log_std = self.logstd_head(h)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = log_std.exp()
        if compute_pi:
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            noise = None
        log_pi = None
        if compute_log_pi and compute_pi:
            log_pi = (-0.5 * ((noise ** 2) + 2 * log_std + np.log(2 * np.pi))).sum(-1, keepdim=True)
        mu = torch.tanh(mu)
        if pi is not None:
            pi = torch.tanh(pi)
        if log_pi is not None and pi is not None:
            # correction for tanh squashing
            log_pi -= (2 * (np.log(2) - pi - F.softplus(-2 * pi))).sum(-1, keepdim=True)
        return mu, pi, log_pi, log_std


class QuantileQ(nn.Module):
    def __init__(self, state_dim, action_dim, n_quantiles, hidden_size=256):
        super().__init__()
        input_dim = state_dim + action_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_quantiles),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.trunk(x)


class TOP_SAC_Agent:
    """TOP-SAC: SAC actor + distributional quantile critics + TOP optimism"""
    def __init__(
        self,
        seed: int,
        state_dim: int,
        action_dim: int,
        action_lim: float = 1.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        batchsize: int = 256,
        hidden_size: int = 256,
        n_quantiles: int = 50,
        kappa: float = 1.0,
        beta: float = 0.0,
        bandit_lr: float = 0.1,
        learnable_beta: bool = False,
        beta_lr: float = 3e-4,
    ):
        self.gamma = gamma
        self.tau = tau
        self.batchsize = batchsize
        self.action_lim = action_lim
        self.n_quantiles = n_quantiles
        self.kappa = kappa

        torch.manual_seed(seed)

        self.learnable_beta = learnable_beta
        self.beta_min = -10.0
        self.beta_max = 10.0
        if learnable_beta:
            self.log_beta = torch.tensor([np.log(max(abs(beta) + 1e-8, 1e-8))], requires_grad=True, device=device)
            self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=beta_lr)
        else:
            self.beta = beta

        # actor
        self.actor = Actor(state_dim, action_dim, hidden_size=hidden_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # critics (distributional quantiles)
        self.q1 = QuantileQ(state_dim, action_dim, n_quantiles, hidden_size=hidden_size).to(device)
        self.q2 = QuantileQ(state_dim, action_dim, n_quantiles, hidden_size=hidden_size).to(device)
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)
        for p in self.target_q1.parameters():
            p.requires_grad = False
        for p in self.target_q2.parameters():
            p.requires_grad = False

        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)

        # entropy temperature
        self.log_alpha = torch.tensor(np.log(0.2)).to(device)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -action_dim

        # replay
        self.replay_pool = ReplayPool(capacity=int(1e6))
        self._update_counter = 0

        # bandit
        self.TDC = ExpWeights(arms=[-1, 0], lr=bandit_lr, init=0.0, use_std=True)

    def get_beta(self, detach: bool = True):
        if self.learnable_beta:
            beta_tensor = torch.clamp(torch.exp(self.log_beta), self.beta_min, self.beta_max)
            return beta_tensor.item() if detach else beta_tensor
        else:
            return self.beta

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, state: np.ndarray, deterministic: bool = False):
        state_t = torch.FloatTensor(state).view(1, -1).to(device)
        with torch.no_grad():
            mu, pi, log_pi, _ = self.actor(state_t, compute_pi=not deterministic, compute_log_pi=True)
        action = mu if deterministic else pi
        if action is None:
            action = mu
        action = action.clamp(-self.action_lim, self.action_lim)
        return action.squeeze(0).cpu().numpy()

    def update_target(self):
        with torch.no_grad():
            for t, s in zip(self.target_q1.parameters(), self.q1.parameters()):
                t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)
            for t, s in zip(self.target_q2.parameters(), self.q2.parameters()):
                t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def update_critic(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch, beta):
        # target actions
        with torch.no_grad():
            mu_next, pi_next, log_pi_next, _ = self.actor(nextstate_batch, compute_pi=True, compute_log_pi=True)
            pi_next = pi_next.clamp(-self.action_lim, self.action_lim)
            # target quantiles
            q1_t = self.target_q1(nextstate_batch, pi_next)
            q2_t = self.target_q2(nextstate_batch, pi_next)
            # Stack and compute belief distribution per quantile
            quantiles_all = torch.stack([q1_t, q2_t], dim=-1)  # [batch, n_quantiles, 2]
            mu = torch.mean(quantiles_all, dim=-1)  # [batch, n_quantiles]
            sigma = torch.sqrt((torch.pow(q1_t - mu, 2) + torch.pow(q2_t - mu, 2)) / 2 + 1e-4)
            belief = mu + beta * sigma  # [batch, n_quantiles]
            # SAC entropy bonus - broadcast log_pi_next to match belief shape
            target_q = belief - self.alpha.detach() * log_pi_next  # [batch, n_quantiles]
            # Compute scalar target for each batch element (mean over quantiles)
            target_scalar = torch.mean(target_q, dim=1, keepdim=True)  # [batch, 1]
            quantile_target = reward_batch + (1.0 - done_batch) * self.gamma * target_scalar
        # Quantile midpoints for loss
        taus = torch.arange(0, self.n_quantiles + 1, device=device, dtype=torch.float32) / self.n_quantiles
        tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, self.n_quantiles)
        # current quantiles
        q1_pred = self.q1(state_batch, action_batch)  # [batch, n_quantiles]
        q2_pred = self.q2(state_batch, action_batch)  # [batch, n_quantiles]
        # TD errors for quantile regression - target should be broadcast to match pred quantiles
        td1 = quantile_target.unsqueeze(1) - q1_pred.unsqueeze(2)  # [batch, 1, 1] - [batch, n_quantiles, 1] = [batch, n_quantiles, 1]
        td2 = quantile_target.unsqueeze(1) - q2_pred.unsqueeze(2)
        loss1 = calculate_quantile_huber_loss(td1, tau_hats, kappa=self.kappa)
        loss2 = calculate_quantile_huber_loss(td2, tau_hats, kappa=self.kappa)
        return loss1, loss2

    def update_actor_and_alpha(self, state_batch):
        mu, pi, log_pi, _ = self.actor(state_batch, compute_pi=True, compute_log_pi=True)
        pi = pi.clamp(-self.action_lim, self.action_lim)
        # get quantile estimates for pi
        q1_pi = self.q1(state_batch, pi)
        q2_pi = self.q2(state_batch, pi)
        # aggregate mean across quantiles and critics
        q_pi = 0.5 * (q1_pi + q2_pi).mean(dim=-1, keepdim=True)
        actor_loss = (self.alpha.detach() * log_pi - q_pi).mean()
        # alpha loss
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return actor_loss, alpha_loss

    def optimize(self, n_updates: int, beta: float, state_filter: Callable = None):
        q1_loss_total, q2_loss_total, pi_loss_total = 0.0, 0.0, 0.0
        for i in range(n_updates):
            samples = self.replay_pool.sample(self.batchsize)
            if state_filter:
                state_batch = torch.FloatTensor(state_filter(samples.state)).to(device)
                nextstate_batch = torch.FloatTensor(state_filter(samples.nextstate)).to(device)
            else:
                state_batch = torch.FloatTensor(samples.state).to(device)
                nextstate_batch = torch.FloatTensor(samples.nextstate).to(device)
            action_batch = torch.FloatTensor(samples.action).to(device)
            reward_batch = torch.FloatTensor(samples.reward).to(device).unsqueeze(1)
            done_batch = torch.FloatTensor(samples.real_done).to(device).unsqueeze(1)

            current_beta = self.get_beta(detach=True) if self.learnable_beta else beta

            # critic update
            loss1, loss2 = self.update_critic(state_batch, action_batch, reward_batch, nextstate_batch, done_batch, current_beta)
            q1_loss_total += loss1.detach().item()
            q2_loss_total += loss2.detach().item()

            self.q1_optimizer.zero_grad()
            self.q2_optimizer.zero_grad()
            (loss1 + loss2).backward()
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=1.0)
            self.q1_optimizer.step()
            self.q2_optimizer.step()

            self._update_counter += 1

            # delayed policy/alpha/beta update
            if self._update_counter % 2 == 0:
                # actor/alpha
                actor_loss, alpha_loss = self.update_actor_and_alpha(state_batch)
                self.actor_optimizer.zero_grad()
                self.alpha_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # beta update if learnable (compute separately to avoid graph issues)
                if self.learnable_beta:
                    # Recompute actor loss with beta for gradient flow
                    state_rep = self.actor.get_representation(states)
                    action_dist, _ = self.actor.forward_dist(state_rep)
                    new_actions = action_dist.rsample()
                    log_probs = action_dist.log_prob(new_actions).sum(dim=-1, keepdim=True)
                    
                    # Compute Q-values with current beta
                    beta_tensor = self.get_beta(detach=False)
                    q1_mu, q1_sigma = self.q1.get_mu_sigma(states, new_actions)
                    q2_mu, q2_sigma = self.q2.get_mu_sigma(states, new_actions)
                    q1_belief = q1_mu + beta_tensor * q1_sigma
                    q2_belief = q2_mu + beta_tensor * q2_sigma
                    q_min_belief = torch.min(q1_belief, q2_belief)
                    
                    # Actor loss for beta gradient: maximize Q - alpha * log_pi
                    alpha_detached = self.get_alpha(detach=True)
                    actor_loss_beta = -(q_min_belief - alpha_detached * log_probs).mean()
                    
                    # Add regularization to prevent beta from hitting bounds
                    beta_reg = 0.001 * (beta_tensor ** 2)
                    total_beta_loss = actor_loss_beta + beta_reg
                    
                    self.beta_optimizer.zero_grad()
                    total_beta_loss.backward()
                    torch.nn.utils.clip_grad_norm_([self.log_beta], max_norm=1.0)
                    if not (torch.isnan(total_beta_loss) or torch.isinf(total_beta_loss)):
                        self.beta_optimizer.step()

                self.update_target()
                pi_loss_total += actor_loss.detach().item()

        return q1_loss_total, q2_loss_total, pi_loss_total

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(),
                    'q1': self.q1.state_dict(),
                    'q2': self.q2.state_dict()}, path)

    def load(self, path):
        d = torch.load(path)
        self.actor.load_state_dict(d['actor'])
        self.q1.load_state_dict(d['q1'])
        self.q2.load_state_dict(d['q2'])
