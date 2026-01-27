# TOP-TQC with Learnable Quantile Dropping
# Instead of learning beta (optimism), this learns how many quantiles to drop for conservatism

import copy
import numpy as np
import torch
import torch.nn.functional as F

from utils import ReplayPool, calculate_quantile_huber_loss, compute_wd_quantile
from bandit import ExpWeights
from typing import Callable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QuantileCritic(torch.nn.Module):
    """Single quantile critic network"""
    def __init__(self, state_dim: int, action_dim: int, n_quantiles: int, hidden_size: int = 256):
        super().__init__()
        self.n_quantiles = n_quantiles
        input_dim = state_dim + action_dim
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_quantiles),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class TQCCritics(torch.nn.Module):
    """Multiple quantile critics for TQC"""
    def __init__(self, state_dim: int, action_dim: int, n_quantiles: int, n_critics: int, hidden_size: int = 256):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n_critics = n_critics
        self.critics = torch.nn.ModuleList([
            QuantileCritic(state_dim, action_dim, n_quantiles, hidden_size) for _ in range(n_critics)
        ])

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        # Returns list of quantile predictions from each critic
        return [critic(state, action) for critic in self.critics]


class Policy(torch.nn.Module):
    """Actor policy network"""
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_dim),
        )
        self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = self.tanh(x)
        return x


class TOP_TQC_LearnableDrop_Agent:
    def __init__(
        self,
        seed: int,
        state_dim: int,
        action_dim: int,
        action_lim: int = 1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        batchsize: int = 256,
        hidden_size: int = 256,
        update_interval: int = 2,
        buffer_size: int = int(1e6),
        target_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        explore_noise: float = 0.1,
        n_quantiles: int = 25,
        n_critics: int = 5,
        top_quantiles_to_drop: int = 2,
        kappa: float = 1.0,
        beta: float = 0.0,
        bandit_lr: float = 0.1,
        learnable_drop: bool = False,
        drop_lr: float = 3e-4
    ) -> None:
        """
        Initialize TOP-TQC agent with learnable quantile dropping.

        Args:
            learnable_drop (bool): whether to make the number of dropped quantiles learnable
            drop_lr (float): learning rate for drop parameter if learnable
        """
        self.gamma = gamma
        self.tau = tau
        self.batchsize = batchsize
        self.update_interval = update_interval
        self.action_lim = action_lim

        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.explore_noise = explore_noise

        torch.manual_seed(seed)
        
        # Learnable drop parameter (how many quantiles to drop)
        self.learnable_drop = learnable_drop
        self.n_critics = n_critics
        if learnable_drop:
            # Start with initial drop count
            self.current_drop_count = top_quantiles_to_drop * n_critics
            # Bandit for adjusting drop count: arms = [-1, 0, +1] (remove drop, stay, add drop)
            self.drop_bandit = ExpWeights(arms=[-1, 0, 1], lr=drop_lr, init=0.0, use_std=True)
            # Bounds: drop at least 0, at most 50% of total quantiles
            self.min_drop_count = 0
            self.max_drop_count = int(0.5 * n_quantiles * n_critics)
        else:
            self.top_quantiles_to_drop = top_quantiles_to_drop
        
        self.beta = beta  # Fixed beta for optimism

        # TQC critics
        self.q_funcs = TQCCritics(state_dim, action_dim, n_quantiles, n_critics, hidden_size).to(device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # Actor
        self.policy = Policy(state_dim, action_dim, hidden_size=hidden_size).to(device)
        self.target_policy = copy.deepcopy(self.policy)
        for p in self.target_policy.parameters():
            p.requires_grad = False

        # TQC parameters
        self.n_quantiles = n_quantiles
        self.kappa = kappa
        
        # Quantile midpoints
        taus = torch.arange(0, n_quantiles + 1, device=device, dtype=torch.float32) / n_quantiles
        self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, n_quantiles)

        # Bandit top-down controller
        self.TDC = ExpWeights(arms=[-1, 0], lr=bandit_lr, init=0.0, use_std=True)

        # Optimizers
        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.replay_pool = ReplayPool(capacity=int(buffer_size))
        self._update_counter = 0

    def get_drop_count(self, detach: bool = True):
        """Get the current number of quantiles to drop
        
        Args:
            detach: if True, return int for logging; if False, return int (no gradient)
        
        Returns:
            int: total number of quantiles to drop across all critics
        """
        if self.learnable_drop:
            return int(self.current_drop_count)
        else:
            return int(self.top_quantiles_to_drop * self.n_critics)
    
    def update_drop_count_from_bandit(self, episode_return: float):
        """Update drop count based on bandit's learned adjustment
        
        Args:
            episode_return: Total reward from episode (used as bandit feedback)
        """
        if not self.learnable_drop:
            return
        
        # Sample adjustment from bandit: -1 (drop fewer), 0 (stay), +1 (drop more)
        adjustment = self.drop_bandit.sample()
        
        # Apply adjustment with bounds
        new_drop_count = self.current_drop_count + adjustment
        self.current_drop_count = np.clip(new_drop_count, self.min_drop_count, self.max_drop_count)
        
        # Update bandit with feedback (episode return)
        self.drop_bandit.update(adjustment, episode_return)

    def reallocate_replay_pool(self, new_size: int) -> None:
        """Reset buffer"""
        assert new_size != self.replay_pool.capacity, "Error, you've tried to allocate a new pool which has the same length"
        new_replay_pool = ReplayPool(capacity=new_size)
        new_replay_pool.initialise(self.replay_pool)
        self.replay_pool = new_replay_pool

    def get_action(
        self,
        state: np.ndarray,
        state_filter: Callable = None,
        deterministic: bool = False
    ) -> np.ndarray:
        """Given the current state, produce an action"""
        if state_filter:
            state = state_filter(state)
        state = torch.Tensor(state).view(1, -1).to(device)
        with torch.no_grad():
            action = self.policy(state)
        if not deterministic:
            action += self.explore_noise * torch.randn_like(action)
        action.clamp_(-self.action_lim, self.action_lim)
        return np.atleast_1d(action.squeeze().cpu().numpy())

    def update_target(self) -> None:
        """Moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)
            for target_pi_param, pi_param in zip(self.target_policy.parameters(), self.policy.parameters()):
                target_pi_param.data.copy_(self.tau * pi_param.data + (1.0 - self.tau) * target_pi_param.data)

    def update_q_functions(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        nextstate_batch: torch.Tensor,
        done_batch: torch.Tensor,
        beta: float,
        drop_count: int
    ):
        """Compute quantile losses for TQC critics with learned quantile dropping"""
        with torch.no_grad():
            # Get next action from target network
            nextaction_batch = self.target_policy(nextstate_batch)
            # Add noise
            target_noise = self.target_noise * torch.randn_like(nextaction_batch)
            target_noise.clamp_(-self.target_noise_clip, self.target_noise_clip)
            nextaction_batch += target_noise
            nextaction_batch.clamp_(-self.action_lim, self.action_lim)
            
            # Get all quantiles from all target critics (TQC)
            next_quantiles_list = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # Stack: [batch_size, n_critics * n_quantiles]
            next_quantiles_all = torch.cat(next_quantiles_list, dim=1)
            
            # TQC: Apply truncation using current drop count
            sorted_quantiles, _ = torch.sort(next_quantiles_all, dim=1)
            if drop_count > 0:
                truncated_quantiles = sorted_quantiles[:, :-drop_count]
            else:
                truncated_quantiles = sorted_quantiles
            
            # Compute mean and std of truncated quantiles
            mu = torch.mean(truncated_quantiles, dim=1, keepdim=True)
            sigma = torch.std(truncated_quantiles, dim=1, keepdim=True) + 1e-4
            
            # Apply TOP optimism
            belief_scalar = mu + beta * sigma  # [batch_size, 1]
            
            # Broadcast to n_quantiles for target -> shape [batch_size, n_quantiles]
            quantile_target = reward_batch + (1.0 - done_batch) * self.gamma * belief_scalar.squeeze(1)
            # quantile_target now has shape [batch_size, n_quantiles]
        
        # Get current Q estimates from each critic
        quantiles_list = self.q_funcs(state_batch, action_batch)
        
        total_loss = 0
        for quantiles in quantiles_list:
            # quantiles: [batch_size, n_quantiles]
            # quantile_target: [batch_size, n_quantiles]
            # Compute TD errors with shape [batch_size, N, N_dash]
            td_errors = quantile_target.unsqueeze(1) - quantiles.unsqueeze(2)  # [batch, n_pred, n_target]
            # Compute quantile loss
            loss = calculate_quantile_huber_loss(td_errors, self.tau_hats, weights=None, kappa=self.kappa)
            total_loss += loss
        
        return total_loss, quantiles_list

    def update_policy(self, state_batch: torch.Tensor, beta: float, drop_count: int) -> torch.Tensor:
        """Update the actor with TOP optimism
        
        Args:
            state_batch: batch of states
            beta: optimism parameter
            drop_count: number of quantiles to drop (integer, no gradient)
        """
        action_batch = self.policy(state_batch)
        quantiles_list = self.q_funcs(state_batch, action_batch)
        
        # Stack and sort all quantiles
        quantiles_all = torch.cat(quantiles_list, dim=1)  # [batch, n_critics * n_quantiles]
        sorted_quantiles, _ = torch.sort(quantiles_all, dim=1)
        
        # Drop top quantiles (conservative)
        if drop_count > 0:
            truncated_quantiles = sorted_quantiles[:, :-drop_count]
        else:
            truncated_quantiles = sorted_quantiles
        
        # Weighted mean (equal weights for now)
        truncated_quantiles_weighted = torch.mean(truncated_quantiles, dim=1, keepdim=True)
        
        # Optimism
        sigma = torch.std(sorted_quantiles, dim=1, keepdim=True) + 1e-4
        belief = truncated_quantiles_weighted + beta * sigma
        
        # DPG loss
        policy_loss = (-belief).mean()
        
        return policy_loss

    def optimize(
        self,
        n_updates: int,
        beta: float,
        state_filter: Callable = None
    ):
        """Sample transitions from buffer and update parameters"""
        q_loss_total, pi_loss, wd = 0, 0, 0
        
        for i in range(n_updates):
            samples = self.replay_pool.sample(self.batchsize)
            if state_filter:
                state_batch = torch.FloatTensor(state_filter(samples.state)).to(device)
                nextstate_batch = torch.FloatTensor(state_filter(samples.nextstate)).to(device)
            else:
                state_batch = torch.FloatTensor(samples.state).to(device)
                nextstate_batch = torch.FloatTensor(samples.nextstate).to(device)
            # Convert list-of-arrays to a single numpy array first (faster tensor creation)
            action_batch = torch.FloatTensor(np.array(samples.action)).to(device)
            reward_batch = torch.FloatTensor(samples.reward).to(device).unsqueeze(1)
            done_batch = torch.FloatTensor(samples.real_done).to(device).unsqueeze(1)

            # Get current drop count (detached for critic update)
            current_drop_count = self.get_drop_count(detach=True) if self.learnable_drop else int(self.top_quantiles_to_drop * self.n_critics)

            # Update critics
            q_loss_step, quantiles_list = self.update_q_functions(
                state_batch, action_batch, reward_batch, nextstate_batch, done_batch, beta, current_drop_count
            )

            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.q_funcs.parameters(), max_norm=1.0)
            self.q_optimizer.step()

            q_loss_total += q_loss_step.detach().item()
            self._update_counter += 1
            
            # Check for NaN in critic losses
            if torch.isnan(q_loss_step) or torch.isinf(q_loss_step):
                print(f"WARNING: NaN/Inf detected in critic loss at update {self._update_counter}. Skipping policy update.")
                continue

            # Update actor and targets
            if self._update_counter % self.update_interval == 0:
                for p in self.q_funcs.parameters():
                    p.requires_grad = False

                # Always use detached drop count - no gradient through drop parameter
                policy_drop_count = self.get_drop_count(detach=True) if self.learnable_drop else int(self.top_quantiles_to_drop * self.n_critics)
                pi_loss_step = self.update_policy(state_batch, beta, policy_drop_count)

                self.policy_optimizer.zero_grad()
                pi_loss_step.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.policy_optimizer.step()

                for p in self.q_funcs.parameters():
                    p.requires_grad = True

                self.update_target()
                pi_loss += pi_loss_step.detach().item()

        # Compute final quantiles for logging
        final_quantiles = quantiles_list[0] if quantiles_list else None
        return q_loss_total, pi_loss, wd / n_updates, final_quantiles, final_quantiles
