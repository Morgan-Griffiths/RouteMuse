import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys


sys.path.append('Agents/DDPG')
from Buffers.replay_buffer import ReplayBuffer
from Buffers.PER import PriorityReplayBuffer
from Buffers.priority_tree import PriorityTree
from Agents.DDPG.noise import OUnoise
# sys.path.append('Agents/DDPG')
from models import hard_update,Actor,Critic
from adaptive_noise import AdaptiveParamNoise

class Agent():
    def __init__(self, nS, nA,indicies,config):
        self.nS = nS
        self.nA = nA
        self.indicies = indicies
        self.vector_size = self.indicies[-1][1]
        self.grade_mask = config.grade_technique_keys
        self.terrain_mask = config.terrain_technique_keys
        self.action_low = config.action_low
        self.action_high = config.action_high
        self.seed = config.seed

        self.clip_norm = config.clip_norm
        self.tau = config.tau
        self.gamma = config.gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.L2 = config.L2
        self.SGD_epoch = config.SGD_epoch
        # noise
        self.noise = OUnoise(nA,config.seed)
        self.noise_scale = 1.0
        self.noise_decay = config.noise_decay

        # Priority Replay Buffer
        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size
        self.alpha = config.ALPHA
        self.beta = self.start_beta = config.START_BETA
        self.end_beta = config.END_BETA

        # actors networks
        self.actor = Actor(self.seed,nS, nA,self.grade_mask,self.terrain_mask,indicies).to(self.device)
        self.actor_target = Actor(self.seed,nS, nA,self.grade_mask,self.terrain_mask,indicies).to(self.device)

        # Param noise
        self.param_noise = AdaptiveParamNoise()
        self.actor_perturbed = Actor(self.seed,nS, nA,self.grade_mask,self.terrain_mask,indicies).to(self.device)

        # critic networks
        self.critic = Critic(self.seed,nS, nA).to(self.device)
        self.critic_target = Critic(self.seed,nS, nA).to(self.device)

        # Copy the weights from local to target
        hard_update(self.critic,self.critic_target)
        hard_update(self.actor,self.actor_target)

        # optimizer
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=self.L2)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=self.L2)

        # replay buffer
        self.PER = PriorityReplayBuffer(self.buffer_size, self.batch_size,self.seed,alpha=self.alpha,device=self.device)

        # reset agent for training
        self.reset_episode()
        self.it = 0

    def save_weights(self,path):
        params = {}
        params['actor'] = self.actor.state_dict()
        params['critic'] = self.critic.state_dict()
        torch.save(params, path)

    def load_weights(self,path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic'])

    def reset_episode(self):
        self.noise.reset()

    def ddpg_distance_metric(self,actions1,actions2):
        """
        TODO
        Necessary for param noise
        Computes distance between actions taken by two different policies
        Expects numpy arrays
        """
        diff = actions1-actions2
        mean_diff = np.mean(np.square(diff),axis=0)
        dist = np.sqrt(np.mean(mean_diff))
        return dist

    def norm_action(self,action):
        for index in self.indicies:
            action[index[0]:index[1]] = action[index[0]:index[1]] / np.sum(action[index[0]:index[1]])
        return action

    def act(self, state):
        with torch.no_grad():
            action = self.actor(self.tensor(state)).cpu().numpy()
        action += np.random.rand(self.indicies[-1][1]) * self.noise_scale
        self.noise_scale = max(self.noise_scale * self.noise_decay, 0.01)
        self.actor.train()
        action = self.norm_action(action)
        return action

    def act_perturbed(self,state):
        """
        TODO
        """
        with torch.no_grad():
            action = self.actor_perturbed(self.tensor(state)).cpu().numpy()
        return action

    def perturbed_update(self):
        """
        TODO
        """
        hard_update(self.actor,self.actor_perturbed)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            random = torch.randn(param.shape).to(self.device)
            param += random * self.param_noise.current_stddev
            

    def evaluate(self,state):
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(self.tensor(state)).cpu().numpy()
        return action

    def step(self, obs, actions, rewards, next_obs):
        # cast as torch tensors
        next_obs = torch.from_numpy(next_obs.reshape(self.vector_size)).float().to(self.device)
        obs = torch.from_numpy(obs.reshape(self.vector_size)).float().to(self.device)
        actions = torch.from_numpy(actions.reshape(self.vector_size)).float().to(self.device)
        # Calc TD error
        next_action = self.actor(next_obs)
        next_value = self.critic_target(next_obs,next_action)
        target = rewards + self.gamma * next_value 
        local = self.critic(obs,actions)
        TD_error = (target - local).squeeze(0)
        self.PER.add(obs, actions, rewards, next_obs, TD_error)
        for _ in range(self.SGD_epoch):
            samples,indicies,importances = self.PER.sample()
            self.learn(samples,indicies,importances)

    def add_replay_warmup(self,obs,actions,rewards,next_obs):
        next_obs = torch.from_numpy(next_obs.reshape(self.vector_size)).float().to(self.device)
        obs = torch.from_numpy(obs.reshape(self.vector_size)).float().to(self.device)
        actions = torch.from_numpy(actions.reshape(self.vector_size)).float().to(self.device)
        # Calculate TD_error
        next_action = self.actor(next_obs)
        next_value = self.critic_target(next_obs,next_action)
        target = np.max(rewards) + self.gamma * next_value
        local = self.critic(obs,actions)
        TD_error = (target - local).squeeze(0)
        self.PER.add(obs,actions,np.max(rewards),next_obs,TD_error)

    def learn(self,samples,indicies,importances):
        
        states, actions, rewards, next_states = samples

        with torch.no_grad():
              target_actions = self.actor_target(next_states)
        next_values = self.critic_target(next_states,target_actions)
        y_target = rewards + self.gamma * next_values
        y_current = self.critic(states, actions)
        TD_error = y_current - y_target
        # update critic
        critic_loss = ((torch.tensor(importances).to(self.device)*TD_error)**2).mean()
        self.critic.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(),self.clip_norm)
        self.critic_opt.step()

        # update actor
        local_actions = self.actor(states)
        actor_loss = -self.critic(states, local_actions).mean()
        self.actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.clip_norm)
        self.actor_opt.step()

        # Update PER
        TD_errors = TD_error.squeeze(1).detach().cpu().numpy()
        self.PER.sum_tree.update_priorities(TD_errors,indicies)

        # soft update networks
        self.soft_update()

    def soft_update(self):
        """Soft update of target network
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

    def tensor(self, x):
        return torch.from_numpy(x).float().to(self.device)
