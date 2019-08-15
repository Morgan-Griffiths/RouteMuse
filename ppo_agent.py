import torch
import torch.optim as optim
import numpy as np 

from Network import PPO_net

class PPO(object):
    def __init__(self,nS,nA,indicies,config):
        self.nS = nS
        self.nA = nA
        self.seed = config.seed
        self.indicies = indicies

        self.gradient_clip = config.gradient_clip
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.start_epsilon = self.epsilon = config.epsilon
        self.start_beta = self.beta = config.beta
        self.SGD_epoch = config.SGD_epoch
        self.batch_size = config.batch_size

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.policy = PPO_net(nA,self.seed,indicies).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(),lr=1e-4)

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
        
        states,values,log_probs,rewards,next_states = self.unwrap(trajectory)
        last_value = self.policy(next_states[-1])
        values = values + [last_value]
        advs = self.gae(values,rewards)
        returns = self.n_step_returns(rewards)
        for indicies in self.minibatch:
            states_b = states[indicies]
            values_b = values[indicies]
            log_probs_b = log_probs[indicies]
            rewards_b = rewards[indicies]
            next_states_b = next_states[indicies]
            advs_b = advs[indicies]
            returns_b = returns[indicies]
            self.learn(states_b,values_b,log_probs_b,rewards_b,next_states_b,advs_b,returns_b)
        

    def unwrap(self,trajectory):
        states = torch.from_numpy(np.vstack([e.state for e in trajectory])).to(self.device)
        values = torch.from_numpy(np.vstack([e.value for e in trajectory])).to(self.device)
        log_probs = torch.from_numpy(np.vstack([e.log_prob for e in trajectory])).to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in trajectory])).to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in trajectory])).to(self.device)
        states = torch.from_numpy(np.vstack([e.state for e in trajectory])).to(self.device)
        return states,values,log_probs,rewards,next_states

    def gae(self,values,rewards):
        """
        Generalized Advantage Estimate

        1d arrays
        """
        N = rewards.shape[0]
        combined = self.gamma*self.gae_lambda
        vs = values[:-1]
        next_vs = values[:1]
        TD_errors = rewards + next_vs - vs
        advs = np.zeros(rewards.shape)
        for index in reversed(range(len(TD_errors))):
            discounts = combined**np.arange(0,N-index)
            advs[index] = np.sum(TD_errors[index:] * discounts)
        return advs

    def n_step_returns(self,rewards):
        N = rewards.shape[0]
        discounts = self.gamma**np.arange(N)
        discounted_returns = rewards
        returns = discounted_returns.cumsum()[::-1]
        return returns

    def learn(self,states,values,log_probs,rewards,next_states,advs,returns):
        """
        Learn on batches from trajectory
        """
        
        _,new_log_probs,new_values = self.policy(states,actions)
        
        ratio = (new_log_probs - old_log_probs)**2
        clip = torch.clamp(ratio,1-self.epsilon,1+self.epsilon)
        clipped_surrogate = torch.min(clip*advs,ratio*advs)

        self.optimizer.zero_grad()
        actor_loss = clipped_surrogate.mean()
        critic_loss = F.smooth_l1_loss(returns,new_values)
        loss = (actor_loss + critic_loss)
        loss.backward()
        self.optimizer.step()

    def act(self,state):
        state = self.tensor(state)
        route,log_prob,value = self.policy(state)
        return route,log_prob,value

    def minibatch(self,N):
        index = np.arange(N)
        for _ in range(self.SGD_epoch):
            indicies = np.random.choice(indicies,self.batch_size)
            yield indicies

    def tensor(self,state):
        return torch.tensor(state).to(self.device)
