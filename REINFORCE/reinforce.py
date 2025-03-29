import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import numpy as np

from .policy_net import PolicyNet





class REINFORCE:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 1e-3,
                 gamma: float = 0.99
                 ) -> None:
        self.policy = PolicyNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []
        self.eps = np.finfo(np.float32).eps.item()


    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs = self.policy(state)
        distribution = Categorical(probs)
        action = distribution.sample()
        self.saved_log_probs.append(distribution.log_prob(action))
        return action.item()
    

    def update_policy(self):
        R = 0
        policy_loss = []
        returns = deque()
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.appendleft(R)
        returns = torch.tensor(returns)
        returns = (returns-returns.mean())/(returns.std()+self.eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob*R)
        
        # print(policy_loss)
        # print(f"length of policy loss : {len(policy_loss)}")
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.saved_log_probs[:]


