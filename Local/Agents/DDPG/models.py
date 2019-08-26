import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

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
    def __init__(self,seed,nS,nA,hidden_dims=(256,128)):
        super(Actor,self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.nS = nS
        self.nA = nA
        self.std = nn.Parameter(torch.zeros(1, nA))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_layer = nn.Linear(nS,hidden_dims[0])
        self.fc1 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.output_layer = nn.Linear(hidden_dims[1],nA)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.output_layer.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,state):
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32,device = self.device) #device = self.device,
            x = x.unsqueeze(0)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc1(x))
        return torch.clamp(torch.tanh(self.output_layer(x)), -1,1)