# TOP-TQC: Combining TOP (Thompson Sampling Over Pessimism) with TQC (Truncated Quantile Critics)
# This combines the optimism control from TOP with the truncated quantile approach from TQC

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


class TOP_TQC_Agent:
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
        learnable_beta: bool = False,
        beta_lr: float = 3e-4,
        preserve_distribution: bool = False
    ) -> None:
        """
        Initialize TOP-TQC agent combining TOP optimism with TQC truncation.

        Args:
            seed (int): random seed
            state_dim (int): state dimension
            action_dim (int): action dimension
            action_lim (int): max action value
            lr (float): learning rate
            gamma (float): discount factor
            tau (float): mixing rate for target nets
            batchsize (int): batch size
            hidden_size (int): hidden layer size
            update_interval (int): delay for actor, target updates
            buffer_size (int): size of replay buffer
            target_noise (float): smoothing noise for target action
            target_noise_clip (float): limit for target
            explore_noise (float): noise for exploration
            n_quantiles (int): number of quantiles per critic
            n_critics (int): number of critic networks
            top_quantiles_to_drop (int): number of top quantiles to drop
            kappa (float): constant for Huber loss
            beta (float): initial optimism parameter
            bandit_lr (float): bandit learning rate
            learnable_beta (bool): whether to make beta learnable
            beta_lr (float): learning rate for beta if learnable
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
        
        # Beta parameter (optimism)
        self.learnable_beta = learnable_beta
        if learnable_beta:
            self.log_beta = torch.tensor([np.log(max(beta, 1e-8))], requires_grad=True, device=device)
            self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=beta_lr)
            # Beta bounds to prevent extreme values that cause NaN
            self.beta_min = -10.0
            self.beta_max = 10.0
        else:
            self.beta = beta

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
        self.n_critics = n_critics
        self.top_quantiles_to_drop = top_quantiles_to_drop
        self.kappa = kappa
        self.preserve_distribution = preserve_distribution
        
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

    def get_beta(self, detach: bool = True):
        """Get the current beta value
        
        Args:
            detach: if True, return float for logging; if False, return tensor for gradient
        """
        if self.learnable_beta:
            beta_tensor = torch.clamp(torch.exp(self.log_beta), self.beta_min, self.beta_max)
            return beta_tensor.item() if detach else beta_tensor
        else:
            return self.beta

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
        preserve_distribution: bool = False
    ):
        """Compute quantile losses for TQC critics with TOP optimism
        
        Args:
            preserve_distribution: If True, applies optimism to distribution directly.
                                  If False (default), uses scalar belief (original behavior).
        """
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
            
            # TQC: Apply truncation (drop highest quantiles for conservatism)
            sorted_quantiles, _ = torch.sort(next_quantiles_all, dim=1)
            n_quantiles_to_drop = self.top_quantiles_to_drop * self.n_critics
            if n_quantiles_to_drop > 0:
                truncated_quantiles = sorted_quantiles[:, :-n_quantiles_to_drop]  # [batch_size, remaining_quantiles]
            else:
                truncated_quantiles = sorted_quantiles
            
            if preserve_distribution:
                # NEW: Apply optimism while preserving distribution
                sigma = torch.std(truncated_quantiles, dim=1, keepdim=True) + 1e-4
                optimistic_quantiles = truncated_quantiles + beta * sigma  # [batch_size, remaining_quantiles]
                
                # Resample/interpolate to match critic's n_quantiles
                if optimistic_quantiles.shape[1] != self.n_quantiles:
                    # Linear interpolation to n_quantiles
                    indices = torch.linspace(0, optimistic_quantiles.shape[1] - 1, self.n_quantiles, device=device)
                    indices_floor = indices.long()
                    indices_ceil = (indices_floor + 1).clamp(max=optimistic_quantiles.shape[1] - 1)
                    weight = indices - indices_floor.float()
                    belief_dist = optimistic_quantiles[:, indices_floor] * (1 - weight) + \
                                 optimistic_quantiles[:, indices_ceil] * weight
                else:
                    belief_dist = optimistic_quantiles
                
                # Compute targets
                quantile_target = reward_batch[..., None] + (1.0 - done_batch[..., None]) * \
                                self.gamma * belief_dist[:, None, :]  # [batch_size, 1, n_quantiles]
            else:
                # ORIGINAL: Collapse to scalar (current behavior)
                mu = torch.mean(truncated_quantiles, dim=1, keepdim=True)  # [batch_size, 1]
                sigma = torch.std(truncated_quantiles, dim=1, keepdim=True) + 1e-4  # [batch_size, 1]
                
                # Apply TOP's optimism control to TQC's conservative estimate
                belief_value = mu + beta * sigma  # [batch_size, 1]
                
                # Expand to match quantile dimension for target computation
                belief_dist = belief_value.expand(-1, self.n_quantiles)  # [batch_size, n_quantiles]
                
                # Compute targets
                quantile_target = reward_batch[..., None] + (1.0 - done_batch[..., None]) * \
                                self.gamma * belief_dist[:, None, :]  # [batch_size, 1, n_quantiles]

        # Get current quantiles from each critic
        current_quantiles_list = self.q_funcs(state_batch, action_batch)
        
        # Compute loss for each critic
        total_loss = 0
        for current_quantiles in current_quantiles_list:
            # Compute td errors
            td_errors = quantile_target - current_quantiles[..., None]  # [batch_size, n_quantiles, n_quantiles]
            loss = calculate_quantile_huber_loss(td_errors, self.tau_hats, weights=None, kappa=self.kappa)
            total_loss += loss

        return total_loss, current_quantiles_list

    def update_policy(self, state_batch: torch.Tensor, beta) -> torch.Tensor:
        """Update the actor with TOP optimism
        
        Args:
            state_batch: batch of states
            beta: optimism parameter (float or tensor if learnable for gradient flow)
        """
        action_batch = self.policy(state_batch)
        quantiles_list = self.q_funcs(state_batch, action_batch)
        
        # Stack quantiles from all critics
        quantiles_all = torch.stack(quantiles_list, dim=-1)  # [batch_size, n_quantiles, n_critics]
        mu = torch.mean(quantiles_all, dim=-1)  # [batch_size, n_quantiles]
        sigma = torch.std(quantiles_all, dim=-1) + 1e-4  # [batch_size, n_quantiles]
        
        # Apply optimism (beta keeps gradient if tensor)
        belief_dist = mu + beta * sigma  # [batch_size, n_quantiles]
        
        # DPG loss
        qval_batch = torch.mean(belief_dist, dim=-1)
        policy_loss = (-qval_batch).mean()
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
            action_batch = torch.FloatTensor(samples.action).to(device)
            reward_batch = torch.FloatTensor(samples.reward).to(device).unsqueeze(1)
            done_batch = torch.FloatTensor(samples.real_done).to(device).unsqueeze(1)

            # Get current beta (detached for critic update)
            current_beta = self.get_beta(detach=True) if self.learnable_beta else beta

            # Update critics
            q_loss_step, quantiles_list = self.update_q_functions(
                state_batch, action_batch, reward_batch, nextstate_batch, done_batch, current_beta, self.preserve_distribution
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

                # Get beta for policy update (keep gradient!)
                policy_beta = self.get_beta(detach=False) if self.learnable_beta else beta
                pi_loss_step = self.update_policy(state_batch, policy_beta)

                self.policy_optimizer.zero_grad()
                if self.learnable_beta:
                    self.beta_optimizer.zero_grad()

                pi_loss_step.backward()
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                if self.learnable_beta:
                    torch.nn.utils.clip_grad_norm_([self.log_beta], max_norm=1.0)
                
                self.policy_optimizer.step()

                if self.learnable_beta:
                    if torch.isnan(pi_loss_step) or torch.isinf(pi_loss_step):
                        print(f"WARNING: NaN/Inf in policy loss at update {self._update_counter}. Skipping beta update.")
                    else:
                        self.beta_optimizer.step()
                        # Log warning if beta is hitting limits
                        current_beta_val = self.get_beta(detach=True)
                        if abs(current_beta_val - self.beta_max) < 0.1 or abs(current_beta_val - self.beta_min) < 0.1:
                            print(f"WARNING: Beta near limit: {current_beta_val:.4f} (range: [{self.beta_min}, {self.beta_max}])")

                for p in self.q_funcs.parameters():
                    p.requires_grad = True

                self.update_target()
                pi_loss += pi_loss_step.detach().item()

        return q_loss_total, pi_loss, wd / n_updates, quantiles_list[0], quantiles_list[1] if len(quantiles_list) > 1 else quantiles_list[0]
