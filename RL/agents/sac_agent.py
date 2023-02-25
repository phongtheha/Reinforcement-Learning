from collections import OrderedDict

from RL.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from RL.infrastructure.replay_buffer import ReplayBuffer
from RL.infrastructure.utils import *
from RL.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import gym
from RL.policies.sac_policy import MLPPolicySAC
from RL.critics.sac_critic import SACCritic
import RL.infrastructure.pytorch_util as ptu

from RL.infrastructure import sac_utils
import torch

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

        

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        #This function does the following:
        # 1. Compute the target Q value. 
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)

        #Get next action and logprob
        act_dist = self.actor(next_ob_no)
        #print("Act dist shape: "+ str(list(act_dist.event_shape)))
        next_ac_na = act_dist.rsample()

        log_prob = act_dist.log_prob(next_ac_na).sum(-1, keepdim=True)
        assert log_prob.shape == (256,1)
        #Get target Qs
        target_qs = self.critic_target(next_ob_no, next_ac_na)
        Qtarget = torch.min(*target_qs)
        assert Qtarget.shape == (256,1)
        #is this the right way to get alpha?
        target = re_n.unsqueeze(1) + self.gamma*(1-terminal_n.unsqueeze(1))*(Qtarget - self.actor.alpha.detach()*log_prob)
        #Get current Qs

        q1, q2 = self.critic(ob_no, ac_na)
  
        critic_loss = self.critic.loss(q1, target.detach()) + self.critic.loss(q2, target.detach())

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()


        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # Pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # 2. Softly update the target every critic_target_update_frequency

        # 3. 
        # If need to update actor:
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        # 4. gather losses for logging

        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
          critic_loss = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
          if self.training_step % self.critic_target_update_frequency == 0:
                sac_utils.soft_update_params(self.critic, self.critic_target,self.critic_tau)

        #Update actor freq?
        actor_loss, alpha_loss, alpha = None, None, None
        if self.training_step % self.actor_update_frequency == 0:
          for i in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss, alpha_loss, alpha = self.actor.update(ob_no, self.critic)

        self.training_step += 1
          
        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['Alpha_Loss'] = alpha_loss
        loss['Temperature'] = alpha

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
