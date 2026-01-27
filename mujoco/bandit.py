import numpy as np 
import pdb
from typing import List, Dict

class ExpWeights(object):
    
    def __init__(self, 
                 arms: List = [-1, 0, 1],
                 lr: float = 0.2,
                 window: int = 5, 
                 decay: float = 0.9,
                 init: float = 0.0,
                 use_std: bool = True) -> None:
        """Initialize bandit.

        Args:
            arms (List, optional): Arm values. Defaults to [-1, 0, 1].
            lr (float, optional): Learning rate. Defaults to 0.2.
            window (int, optional): Window to normalize over. Defaults to 5.
            decay (float, optional): Decay rate for probability. Defaults to 0.9.
            init (float, optional): Weight initialization. Defaults to 0.0.
            use_std (bool, optional): Use std to normalize feedback. Defaults to True.
        """
        
        self.arms = arms
        self.w = {i:init for i in range(len(self.arms))}
        self.arm = 0
        self.value = self.arms[self.arm]
        self.error_buffer = []
        self.window = window
        self.lr = lr
        self.decay = decay
        self.use_std = use_std
        
        self.choices = [self.arm]
        self.data = []
        
    def sample(self) -> float:
        """Sample from distribution. 

        Returns:
            float: The value of the sampled arm.
        """
        p = [np.exp(x) for x in self.w.values()]
        p /= np.sum(p) # normalize to make it a distribution
        self.arm = np.random.choice(range(0,len(p)), p=p)

        self.value = self.arms[self.arm]
        self.choices.append(self.arm)
        
        return self.value

    def get_probs(self) -> List:
        """Get arm probabilities. 

        Returns:
            List: probabilities for each arm. 
        """
        p = [np.exp(x) for x in self.w.values()]
        p /= np.sum(p) # normalize to make it a distribution
        return p

        
    def update_dists(self, feedback: float, norm: float = 1.0) -> None:
        """Update distribution over arms. 

        Args:
            feedback (float): Feedback signal. 
            norm (float, optional): Normalization factor. Defaults to 1.0.
        """

        # Since this is non-stationary, subtract mean of previous self.window errors. 
        self.error_buffer.append(feedback)
        self.error_buffer = self.error_buffer[-self.window:]
        
        # normalize
        feedback -= np.mean(self.error_buffer)
        if self.use_std and len(self.error_buffer) > 1: norm = np.std(self.error_buffer); 
        feedback /= norm 
        
        # update arm weights
        self.w[self.arm] *= self.decay
        self.w[self.arm] += self.lr * (feedback/max(np.exp(self.w[self.arm]), 0.0001))
        
        self.data.append(feedback)

    def update(self, arm_value: float, reward: float) -> None:
        """Compatibility wrapper used by trainers.

        Trainers call `update(beta, episode_return)`; interpret `arm_value`
        as the chosen arm's value, set the internal `arm` index accordingly,
        and forward the `reward` as feedback to `update_dists`.
        """
        # Find the closest arm index for the provided arm_value
        try:
            # exact match first
            idx = self.arms.index(arm_value)
        except ValueError:
            # fallback to nearest numeric match
            arr = np.array(self.arms, dtype=float)
            idx = int(np.argmin(np.abs(arr - float(arm_value))))

        self.arm = idx
        self.value = self.arms[self.arm]
        # Use reward as feedback (trainers provide episode return)
        self.update_dists(float(reward))