import numpy as np 
import time
import sys
import os

sys.path.append('/home/shuza/Code/Udacity_multiplayer')

from train_ddpg import seed_replay_buffer,train_ddpg
from ddpg_agent import Agent
from utils.unity_env import UnityEnv
from utils.config import Config
# from config import Config
# from train import train
# from unity_env import UnityEnv
# from env_wrapper import MultiEnv
# from maddpg import MultiAgent
from models import Actor, Critic

"""
Instantiates config, ddpg agent, expands state and action spaces to include both players.
"""
def main(algo):
    # Load the ENV
    ### For running in VSCode ###
    # env = UnityEnv(env_file='Environments/Tennis_Linux/Tennis.x86_64',no_graphics=True)
    ### For running from terminal ###
    env = UnityEnv(env_file='../Environments/Tennis_Linux/Tennis.x86_64',no_graphics=True)

    # number of agents
    num_agents = env.num_agents
    print('Number of agents:', num_agents)

    # size of each action
    action_size = env.action_size*num_agents

    # examine the state space 
    state_size = env.state_size*num_agents
    print('Size of each action: {}, Size of the state space {}'.format(action_size/num_agents,state_size/num_agents))
    
    ddpg_config = Config(algo)

    agent = Agent(state_size, action_size,Actor,Critic,ddpg_config)
    # Fill buffer with random actions up to min buffer size
    seed_replay_buffer(env, agent, ddpg_config.min_buffer_size)
    # Train agent
    train_ddpg(env, agent, ddpg_config)

if __name__ == '__main__':
    algo='ddpg'
    main(algo)