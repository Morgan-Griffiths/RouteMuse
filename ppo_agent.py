import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np 
import os

from Network import PPO_net

class PPO(object):
    def __init__(self,nS,nA,indicies,config):
        self.nS = nS
        self.nA = nA
        self.seed = config.seed
        self.indicies = indicies
        self.lr = config.lr

        self.gradient_clip = config.gradient_clip
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.start_epsilon = self.epsilon = config.epsilon
        self.start_beta = self.beta = config.beta
        self.SGD_epoch = config.SGD_epoch
        self.batch_size = config.batch_size

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.policy = PPO_net(nA,self.seed,indicies).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(),lr=1e-4,weight_decay=config.L2)

    def load_weights(self,path):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()

    def save_weights(self,path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.policy.state_dict(), path)

    def reset_hyperparams(self):
        self.discount = self.start_discount
        self.epsilon = self.start_epsilon
        self.beta = self.start_beta

    def step_hyperparams(self):
        self.epsilon *= 0.999
        self.beta *= 0.995

    def step(self,trajectory):
        N = len(trajectory)
        # decay beta,epsilon
        self.step_hyperparams()
        
        states,actions,values,log_probs,rewards,next_states = self.unwrap(trajectory)
        # Normalize the rewards
        # rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        last_value = self.policy(self.tensor(next_states[-1]))[-1].unsqueeze(1).cpu().detach().numpy()
        values = np.vstack(values + [last_value])
        advs = self.gae(values,rewards)
        returns = self.n_step_returns(rewards)

        
        states,actions,log_probs,returns,advs = self.bulk_tensor(states,actions,log_probs,returns,advs)
        for indicies in self.minibatch(N):
            states_b = states[indicies]
            actions_b = actions[indicies]
            log_probs_b = log_probs[indicies]
            advs_b = advs[indicies]
            returns_b = returns[indicies]
            self.learn(states_b,actions_b,log_probs_b,advs_b,returns_b)
        
    def bulk_tensor(self,states,actions,log_probs,returns,advs):
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        log_probs = torch.from_numpy(log_probs).float().to(self.device)
        advs = torch.from_numpy(advs).float().to(self.device)
        # TO fix negative stride
        returns = torch.flip(torch.from_numpy(np.flip(returns,axis=0).copy()),dims=(0,)).float().to(self.device)
        return states,actions,log_probs,returns,advs

    def unwrap(self,trajectory):
        # states = torch.from_numpy(np.vstack([e.state for e in trajectory])).float().to(self.device)
        # values = torch.from_numpy(np.vstack([e.value for e in trajectory])).float().to(self.device)
        # log_probs = torch.from_numpy(np.vstack([e.log_prob for e in trajectory])).float().to(self.device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in trajectory])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in trajectory])).float().to(self.device)
        
        states = np.vstack([e.state for e in trajectory])
        values = [e.value for e in trajectory]
        actions = np.vstack([e.action for e in trajectory])
        log_probs = np.vstack([e.log_prob for e in trajectory])
        rewards = np.vstack([e.reward for e in trajectory])
        next_states = np.vstack([e.next_state for e in trajectory])
        
        return states,actions,values,log_probs,rewards,next_states

    def gae(self,values,rewards):
        """
        Generalized Advantage Estimate

        1d arrays
        """
        N = rewards.shape[0]
        combined = self.gamma*self.gae_lambda
        vs = values[:-1]
        next_vs = values[1:]
        TD_errors = rewards + next_vs - vs
        advs = np.zeros(rewards.shape)
        for index in reversed(range(len(TD_errors))):
            discounts = combined**np.arange(0,N-index)
            advs[index] = np.sum(TD_errors[index:] * discounts)
        return advs

    def n_step_returns(self,rewards):
        N = rewards.shape[0]
        discounts = self.gamma**np.arange(N)
        discounted_returns = rewards * discounts.reshape(N,1)
        returns = discounted_returns.cumsum()[::-1].reshape(N,1)
        return returns

    def learn(self,states,actions,log_probs,advs,returns):
        """
        Learn on batches from trajectory
        """
        
        new_actions,new_log_probs,new_values = self.policy(states)
        
        # ratio = (new_actions - actions)**2
        ratio = new_log_probs / log_probs
        clip = torch.clamp(ratio,1-self.epsilon,1+self.epsilon)
        clipped_surrogate = torch.min(clip*advs,ratio*advs)

        self.optimizer.zero_grad()
        actor_loss = clipped_surrogate.mean()
        critic_loss = F.smooth_l1_loss(returns,new_values)
        loss = (actor_loss + critic_loss)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(),self.gradient_clip)
        self.optimizer.step()

    def act(self,state):
        state = self.tensor(state)
        route,log_prob,value = self.policy(state)
        return route.detach().cpu().numpy(),log_prob.detach().cpu().numpy(),value.detach().cpu().numpy()

    def minibatch(self,N):
        index = np.arange(N)
        for _ in range(self.SGD_epoch):
            indicies = np.random.choice(index,self.batch_size)
            yield indicies

    def tensor(self,state):
        return torch.tensor(state).float().to(self.device)
