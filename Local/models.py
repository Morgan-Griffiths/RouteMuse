import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions.distribution as dist
import torch.distributions.categorical as Categorical
import numpy as np

"""
PyTorch implementation of Actor Critic class for PPO. 
Combined torso with dual output 

2 Options:
use categorical for each slice
use softmax and torch.log for whole

Inputs:
Active routes
Historical routes (for novelty)
Current distance (minus stripped routes)
Can use the mask in the foward pass to auto restrict which techniques to suggest.
"""

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def hard_update(source,target):
    for target_param,param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
    
class Critic(nn.Module):
    def __init__(self,seed,nS,nA,hidden_dims=(256,128)):
        super(Critic,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.nS = nS
        self.nA = nA
        
        self.input_layer = nn.Linear(nS,hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(hidden_dims[0]+nA,hidden_dims[1]))
        for i in range(1,len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        # self.fc1 = nn.Linear(hidden_dims[0]+nA,hidden_dims[1])
        # self.fc1_bn = nn.BatchNorm1d(hidden_dims[1])
        self.output_layer = nn.Linear(hidden_dims[-1],1)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.output_layer.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,obs,action):
        # With batchnorm
        # xs = self.input_bn(F.relu(self.input_layer(state)))
        # x = torch.cat((xs,action),dim=1)
        # x = self.fc1_bn(F.relu(self.fc1(x)))
        xs = F.relu(self.input_layer(obs))
        x = torch.cat((xs, action), dim=-1)
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.output_layer(x)

class Actor(nn.Module):
    def __init__(self,seed,nS,nA,grade_mask,terrain_mask,indicies,hidden_dims=(256,128)):
        super(Actor,self).__init__()

        self.seed = torch.manual_seed(seed)
        self.nS = nS
        self.nA = nA
        self.terrain_mask = terrain_mask
        self.grade_mask = grade_mask
        self.indicies = indicies
        self.std = nn.Parameter(torch.zeros(1, nA))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_layer = nn.Linear(nS,hidden_dims[0])
        self.fc1 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.output_layer = nn.Linear(hidden_dims[1],nA)

        # Action layers
        self.action_outputs = nn.ModuleList()
        for index in self.indicies:
            field = nn.Linear(hidden_dims[-1],index[1]-index[0])
            self.action_outputs.append(field)

        self.reset_parameters()
        
    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.output_layer.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,state):
        assert isinstance(state,torch.Tensor)
        # x = state
        # if not isinstance(state,torch.Tensor):
        #     x = torch.tensor(x,dtype=torch.float32,device = self.device) #device = self.device,
        #     x = x.unsqueeze(0)
        x = F.relu(self.input_layer(state))
        x = F.relu(self.fc1(x))
        # torch.clamp(torch.tanh(self.output_layer(x)), -1,1)
        actions = []
        for action_layer in self.action_outputs:
            selection = F.softmax(action_layer(x),dim=0)
            actions.append(selection)
        # assert len(actions[0].size()) > 1
        axis = len(actions[0].size()) -1
        action = torch.cat(actions,dim=axis)
        return action

class PPO_net(nn.Module):
    def __init__(self,nA,seed,indicies,grade_mask,terrain_mask,hidden_dims=(256,128)):
        super(PPO_net,self).__init__()
        self.nS = nA
        self.nA = nA
        self.seed = torch.manual_seed(seed)
        self.indicies = indicies
        self.hidden_dims = hidden_dims
        self.grade_mask = grade_mask
        self.terrain_mask = terrain_mask
        # TODO implement own batchnorm function
        # self.batch = manual_batchnorm()
        
        # Layers
        self.input_layer = nn.Linear(self.nS,hidden_dims[0])
        # self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        # self.hidden_batches = nn.ModuleList()
        for i in range(1,len(hidden_dims)):
            # hidden_batch = nn.BatchNorm1d(hidden_dims[i-1])
            hidden_layer = nn.Linear(hidden_dims[i-1],hidden_dims[i])
            self.hidden_layers.append(hidden_layer)
            # self.hidden_batches.append(hidden_batch)

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
        x = F.relu(self.input_layer(state))
        for i,hidden_layer in enumerate(self.hidden_layers):
            x = F.relu(hidden_layer(x))

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
        
class manual_batchnorm(object):
    def __init__(self,size):
        self.size = size
        self.epsilon = 1e-7
        self.running_mean = 0

    def compute(self,tensor):
        assert tensor.size == self.size
        if self.running_mean == 0:
            self.running_mean = tensor.mean(dim = 0)
        else:
            # tensor = 
            pass
        return tensor
        # return tensor.mean(dim = 0) / tensor.std(dim = 0) + self.epsilon
        # y = (x - mean(x)) / std + eps