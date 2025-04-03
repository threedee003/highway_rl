import torch
import torch.nn as nn
import numpy as np
from typing import Union

class ActorCriticNet(nn.Module):
    '''
    a feedforward neural net for actor and critic network.


    input:
         x (torch.Tensor or numpy.ndarray) : observation
    output
         actor_out (torch.Tensor) : action
         critic_out (torch.Tensor) :action value
         

    
    '''
    def __init__(self,
                 input_dim : int, 
                 action_dim : int,
                 hidden_dim: int,
                 dropout_rate : float,
                 continuous_action_space: bool,
                 device: torch.device,
                 action_std_init: float = 0.0
    ) -> None:
        super(ActorCriticNet, self).__init__()

        self.continuous_action_space = continuous_action_space
        self.device = device


        self.feedfwd = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        ).to(device)
        if continuous_action_space:
            self.action_var = nn.Parameter(torch.full(size=(action_dim,), fill_value=action_std_init * action_std_init)).to(device)
            self.actor_head = nn.Linear(hidden_dim, action_dim, dtype=torch.float32).to(device)
        else:
            self.actor_head = nn.Sequential(
                nn.Linear(hidden_dim, action_dim, dtype=torch.float32),
                nn.Softmax(dim=-1)
            ).to(device)

        

    def forward(self, x : Union[np.ndarray, torch.Tensor]):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype = torch.float32).to(device=self.device)
        features = self.feature_extractor(x)
        actor_out = self.actor_head(features)
        critic_out = self.critic_head(features)

        return actor_out, critic_out
    


    def select_action(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype = torch.float32).to(device=self.device)


        if obs.dim() == 1:
            obs = obs.unsqueeze(0)


        with torch.no_grad():
            action, value = self.forward(obs)

            if self.continuous_action_space:
                action_cov = torch.diag(self.action_var)

                dist = torch.distributions.MultivariateNormal(action, action_cov)
            else:
                dist = torch.distributions.Categorical(action)


            action = dist.sample()
            action_logprob = dist.log_prob(action)



            if self.continuous_action_space:
                if action.dim() == 2 and action.shape[0] == 1:
                    action = action.squeeze(0).cpu().numpy()
                else:
                    action = action.item()


        return action, action_logprob.cpu().numpy(), value.item()
    

    def eval_actions(self, states, actions):
        action, value = self.forward(states)

        if self.continuous_action_space:
            action_cov = torch.diag(self.action_var)
            dist  = torch.distributions.MultivariateNormal(action, action_cov)
            action_logprobs = dist.log_prob(actions)
        
        else:
            dist = torch.distributions.Categorical(action)
            action_logprobs = dist.log_prob(actions)

        dist_entropy = dist.entropy()

        return value.squeeze(), action_logprobs, dist_entropy








     