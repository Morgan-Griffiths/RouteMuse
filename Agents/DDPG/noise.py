import numpy as np
import random
import copy

def initialize_N(T):
    """
    Orstein Uhlenbeck process
    theta = 0.15
    sigma = 0.2
    mu = 0
    dX = theta(mu-X) dt + sigma * dW
    """
    theta = 0.15
    sigma = 0.1
    mu = 0
    tau = 1
    dt = 1
    n = int(T / dt)
    
    t = np.linspace(0.,T,n)
    sigma_bis = sigma * np.sqrt(2. / tau)
    sqrtdt = np.sqrt(dt)
    x = np.zeros(n)
    
    for i in range(1,n):
        x[i] = x[i-1] + dt * (-(x[i-1] - mu)/tau) + \
            sigma_bis * sqrtdt * np.random.randn()
        
    return x
#     N = theta(-X) * dt + sigma * W
    
#     X += dt * (-(X - mu) / tau) + \
#         sigma * np.random.randn(ntrials)

class GaussianNoise(object):
    def __init__(self, dimension, num_epochs, mu=0.0, var=1):
        self.mu = mu
        self.var = var
        self.dimension = dimension
        self.epochs = 0
        self.num_epochs = num_epochs
        self.min_epsilon = 0.0 # minimum exploration probability
        self.epsilon = 0.5
        self.decay_rate = .9999#5.0/num_epochs # exponential decay rate for exploration prob
        self.iter = 0

    def sample(self):
        x = self.epsilon * np.random.normal(self.mu, self.var, size=self.dimension)
        return x

    def step(self):
        self.epsilon = max(self.min_epsilon,self.decay_rate*self.epsilon)

class OUnoise(object):
    def __init__(self,size,seed,mu=0,theta=0.15,sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x+dx
        return self.state
