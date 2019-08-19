import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions.distribution as dist
import torch.distributions.categorical as Categorical

"""
PyTorch implementation of Actor Critic class for PPO. 
Combined torso with dual output 

2 Options:
use categorical for each slice
use softmax and torch.log for whole
"""

class PPO_net(nn.Module):
    def __init__(self,nA,seed,indicies,hidden_dims=(256,128)):
        super(PPO_net,self).__init__()
        self.nS = nA
        self.nA = nA
        self.seed = torch.manual_seed(seed)
        self.indicies = indicies
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        
        # Layers
        self.input_layer = nn.Linear(self.nS,hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        self.hidden_batches = nn.ModuleList()
        for i in range(1,len(hidden_dims)):
            hidden_batch = nn.BatchNorm1d(hidden_dims[i-1])
            hidden_layer = nn.Linear(hidden_dims[i-1],hidden_dims[i])
            self.hidden_layers.append(hidden_layer)
            self.hidden_batches.append(hidden_batch)

        
        # Action outputs, we softmax over the various classes for 1 per class (can change this for multi class)
        self.action_outputs = nn.ModuleList()
        for index in self.indicies:
            field = nn.Linear(hidden_dims[-1],index[1]-index[0])
            self.action_outputs.append(field)

        self.value_output = nn.Linear(hidden_dims[-1],1)

    def forward(self,state):
        """
        Expects state to be a torch tensor

        Outputs Action,log_prob, entropy and (state,action) value
        """
        assert isinstance(state,torch.Tensor)
        x = F.relu(self.input_bn(self.input_layer(state)))
        for i,hidden_layer in enumerate(self.hidden_layers):
            x = F.relu(self.hidden_batches[i](hidden_layer(x)))

        actions = []
        for action_layer in self.action_outputs:
            selection = F.softmax(action_layer(x),dim=0)
            actions.append(selection)
        if len(actions[0].size()) > 1:
            action = torch.cat(actions,dim=1)
        else:
            action = torch.cat(actions,dim=0)

        # a = self.action_output(x)
        # a = dist(a)
        # action = a.sample()
        # log_prob = a.log_prob(action)
        # entropy = a.entropy()

        v = self.value_output(x)
        return action,torch.log(action),v
        