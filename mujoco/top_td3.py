# TOP-TD3: Combining TOP (quantile-based) optimism control with TD3's training approach
# Uses full distributional RL with quantiles, not simplified point estimates

import copy
import numpy as np
import torch
import torch.nn.functional as F

from utils import ReplayPool, Transition, calculate_quantile_huber_loss, compute_wd_quantile
from networks import QuantileDoubleQFunc, Policy
from bandit import ExpWeights
from typing import Callable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TOP_TD3_Agent:
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
        n_quantiles: int = 100,
        kappa: float = 1.0,
        beta: float = 0.0,
        bandit_lr: float = 0.1,
        learnable_beta: bool = False,
        beta_lr: float = 3e-4
    ) -> None:
        """
        Initialize TOP-TD3 agent using distributional RL with quantiles.

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
            n_quantiles (int): number of quantiles for distributional RL
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

        # Init quantile critics (distributional)
        self.q_funcs = QuantileDoubleQFunc(state_dim, action_dim, n_quantiles=n_quantiles, hidden_size=hidden_size).to(device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # Init actor
        self.policy = Policy(state_dim, action_dim, hidden_size=hidden_size).to(device)
        self.target_policy = copy.deepcopy(self.policy)
        for p in self.target_policy.parameters():
            p.requires_grad = False

        # Set distributional parameters
        taus = torch.arange(0, n_quantiles + 1, device=device, dtype=torch.float32) / n_quantiles
        self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, n_quantiles)
        self.n_quantiles = n_quantiles
        self.kappa = kappa

        # Bandit top-down controller
        self.TDC = ExpWeights(arms=[-1, 0], lr=bandit_lr, init=0.0, use_std=True)

        # Init optimizers
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
        beta: float
    ) -> tuple:
        """
        Compute quantile losses for critics with TOP optimism (distributional).
        This is the official TOP-TD3 formulation using full quantile regression.
        """
        with torch.no_grad():
            # Get next action from target network
            nextaction_batch = self.target_policy(nextstate_batch)
            # Add TD3-style target policy smoothing
            target_noise = self.target_noise * torch.randn_like(nextaction_batch)
            target_noise.clamp_(-self.target_noise_clip, self.target_noise_clip)
            nextaction_batch += target_noise
            nextaction_batch.clamp_(-self.action_lim, self.action_lim)
            
            # Get quantiles at (s', a') from both target critics
            quantiles_t1, quantiles_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            
            # TOP: Compute mean and std across critics for each quantile
            quantiles_all = torch.stack([quantiles_t1, quantiles_t2], dim=-1)  # [batch_size, n_quantiles, 2]
            mu = torch.mean(quantiles_all, dim=-1)  # [batch_size, n_quantiles]
            sigma = torch.std(quantiles_all, dim=-1) + 1e-4  # [batch_size, n_quantiles]
            
            # Construct belief distribution with optimism
            belief_dist = mu + beta * sigma  # [batch_size, n_quantiles]
            
            # Compute targets
            quantile_target = reward_batch[..., None] + (1.0 - done_batch[..., None]) * \
                            self.gamma * belief_dist[:, None, :]  # [batch_size, 1, n_quantiles]
        
        # Get quantiles at (s, a)
        quantiles_1, quantiles_2 = self.q_funcs(state_batch, action_batch)
        
        # Compute pairwise TD errors
        td_errors_1 = quantile_target - quantiles_1[..., None]  # [batch_size, n_quantiles, n_quantiles]
        td_errors_2 = quantile_target - quantiles_2[..., None]
        
        # Compute quantile Huber losses
        loss_1 = calculate_quantile_huber_loss(td_errors_1, self.tau_hats, weights=None, kappa=self.kappa)
        loss_2 = calculate_quantile_huber_loss(td_errors_2, self.tau_hats, weights=None, kappa=self.kappa)

        return loss_1, loss_2, quantiles_1, quantiles_2

    def update_policy(self, state_batch: torch.Tensor, beta) -> torch.Tensor:
        """Update the actor with TOP optimism
        
        Args:
            state_batch: batch of states
            beta: optimism parameter (float or tensor if learnable for gradient flow)
        """
        # Get actions
        action_batch = self.policy(state_batch)
        
        # Compute quantiles (s, a)
        quantiles_b1, quantiles_b2 = self.q_funcs(state_batch, action_batch)
        
        # Construct belief distribution (beta keeps gradient if tensor)
        quantiles_all = torch.stack([quantiles_b1, quantiles_b2], dim=-1)  # [batch_size, n_quantiles, 2]
        mu = torch.mean(quantiles_all, dim=-1)  # [batch_size, n_quantiles]
        eps1, eps2 = 1e-4, 1.1e-4  # Small constants for stability
        sigma = torch.sqrt((torch.pow(quantiles_b1 + eps1 - mu, 2) + torch.pow(quantiles_b2 + eps2 - mu, 2)) + eps1)
        belief_dist = mu + beta * sigma  # [batch_size, n_quantiles]
        
        # DPG loss (use scalar loss)
        qval_batch = torch.mean(belief_dist, dim=-1)
        policy_loss = -torch.mean(qval_batch)
        return policy_loss

    def optimize(
        self,
        n_updates: int,
        beta: float,
        state_filter: Callable = None
    ):
        """Sample transitions from buffer and update parameters"""
        q1_loss_total, q2_loss_total, pi_loss = 0.0, 0.0, 0.0
        wd = 0.0
        
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

            # Get beta (detached for critic update)
            current_beta = self.get_beta(detach=True) if self.learnable_beta else beta

            # Update quantile critics
            q1_loss_step, q2_loss_step, quantiles1_step, quantiles2_step = self.update_q_functions(
                state_batch, action_batch, reward_batch, nextstate_batch, done_batch, current_beta
            )
            q_loss_step = q1_loss_step + q2_loss_step

            # Measure Wasserstein distance between critics
            wd_step = compute_wd_quantile(quantiles1_step, quantiles2_step)
            wd += wd_step.detach().item()

            # Take gradient step for critics
            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.q_funcs.parameters(), max_norm=1.0)
            self.q_optimizer.step()

            self._update_counter += 1
            q1_loss_total += q1_loss_step.detach().item()
            q2_loss_total += q2_loss_step.detach().item()
            
            # Check for NaN in critic losses
            if torch.isnan(q_loss_step) or torch.isinf(q_loss_step):
                print(f"WARNING: NaN/Inf detected in critic loss at update {self._update_counter}. Skipping policy update.")
                continue

            # Update actor and targets every update_interval steps (TD3-style delayed updates)
            if self._update_counter % self.update_interval == 0:
                # Update policy
                for p in self.q_funcs.parameters():
                    p.requires_grad = False

                # Get beta for policy (keep gradient!)
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

                # Update target policy and Q-functions using Polyak averaging
                self.update_target()
                pi_loss += pi_loss_step.detach().item()

        avg_wd = wd / n_updates if n_updates > 0 else wd
        return q1_loss_total, q2_loss_total, pi_loss, avg_wd, quantiles1_step, quantiles2_step
