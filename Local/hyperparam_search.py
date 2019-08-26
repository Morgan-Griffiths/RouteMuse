import numpy as np
from ppo_agent import PPO
from config import Config
"""
Tune the hyperparameters of the PPO agent

learning rates:     1e-4,2e-4,5e-5
gradient clipping:  5,10,15
epsilon:            0.2,0.3,0.1
network size:       (256,256),(256,256,128),(256,128)
"""

search_grid = np.array([
    [1e-4,2e-4,5e-5],
    [5,10,15],
    [0.2,0.3,0.1],
    [(256,256),(256,256,128),(256,128)]
])

def param_search(search_grid):
    np.scores = np.zeros(search_grid.shape)
    config = Config()
    agent = PPO
    scores = train(agent)
