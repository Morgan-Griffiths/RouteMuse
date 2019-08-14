import torch
import numpy as np 

from Network import PPO_net

class PPO(object):
    def __init__(self,nS,nA,seed,indicies):
        self.nS = nS
        self.nA = nA
        self.seed = seed
        self.indicies = indicies

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.policy = PPO_net(nS,nA,seed,indicies)

    def step(self):
        pass

    def learn(self):
        pass

    def act(self,state):
        pass

    def tensor(self,state):
        return torch.tensor(state).to(device)
