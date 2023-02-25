from RL.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from RL.infrastructure import sac_utils
from RL.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        # super().__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        entropy = self.log_alpha.exp()
        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        action_distribution = self(observation)
        
        if sample:
          action = action_distribution.sample()
        else:
          action = action_distribution.mean
        action = torch.clip(action, self.action_range[0], self.action_range[1])
       # assert action.ndim == 2 and action.shape[0] == 1
        return ptu.to_numpy(action)

    # This function defines the forward pass of the network.
    def forward(self, observation: torch.FloatTensor):
      #Thisn function takes observation inputs and return the action dist
      batch_mean = self.mean_net(observation)
      scale_tril = torch.exp(torch.clip(self.logstd, self.log_std_bounds[0], self.log_std_bounds[1]))
      
      action_distribution = sac_utils.SquashedNormal(batch_mean, scale_tril)
      
      return action_distribution

    def update(self, obs, critic):
        # TODO Train the actor network 
        observation = ptu.from_numpy(obs)
        action_distribution = self(observation)
        action = action_distribution.rsample()
        log_prob = action_distribution.log_prob(action).sum(-1, keepdim = True)
        assert log_prob.shape == (256,1)
        actor_Qs = critic(observation, action)
        

        actor_Q = torch.min(*actor_Qs)
        assert actor_Q.shape == (256,1)
        actor_loss = torch.mean(self.alpha.detach() * log_prob - actor_Q)
        #Optimizer actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        #Now update entropy
        
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, self.alpha