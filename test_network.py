import numpy as np 
import sys
import os
from collections import namedtuple,deque
import time
import pickle

from plot import plot,plot_episode
from Network import PPO_net
from gym import Gym
from utils import Utilities
from config import Config
from ppo_agent import PPO
# sys.path.append('/Users/morgan/Code/RouteMuse/test')
sys.path.append('/home/kenpachi/Code/RouteMuse/test')
print('path',os.getcwd())
from test_data import build_data

"""
Generate training data and train on it

TODO
Generate a unique hash for each route
"""

def main():
	# Instantiate objects
	config = Config()
	fields = build_data()
	utils = Utilities(fields,config)
	agent = PPO(utils.total_fields,utils.total_fields,utils.field_indexes,config)
	# train on data
	verify_network(agent,utils,config)

def verify_network(agent,utils,config):
    """
    For plotting - plot the math mean error versions along with the agent mean
    """
    experience = namedtuple('experience',field_names=['state','value','log_prob','action','reward','next_state'])
    # Create two instances of gyms
    gym_network = Gym(config.total_routes,config.num_reset_routes,utils)
    gym_math = Gym(config.total_routes,config.num_reset_routes,utils)
    # tic = time.time()
    # Load weights of trained model
    agent.load_weights(config.checkpoint_path)

    # Collections
    math_means = []
    math_stds = []
    means = []
    stds = []
    scores_window = deque(maxlen=100)
    math_window = deque(maxlen=100)
    for e in range(1,11):
        trajectory = []
        rewards = []
        math_rewards = []
        state = gym_network.reset()
        math_state = gym_math.reset()
        math_loss = [gym_math.loss]
        network_loss = [gym_network.loss]
        for t in range(config.tmax):
            suggestion,log_prob,value = agent.act(state)
            route = utils.route_from_suggestion(suggestion)
            next_state,reward = gym_network.step(route)
            # Compare with math
            math_route = utils.route_from_distance(math_state)
            math_next_state, math_reward = gym_math.step(math_route)
            math_rewards.append(math_reward)
            math_loss.append(gym_math.loss)
            # Record (s,a,r,s)
            exp = experience(state,value,log_prob,suggestion,reward,next_state)
            trajectory.append(exp)
            rewards.append(reward)
            network_loss.append(gym_network.loss)
            state = next_state
        scores_window.append(np.sum(rewards))
        means.append(np.mean(scores_window))
        stds.append(np.std(scores_window))
        math_window.append(np.sum(math_rewards))
        math_means.append(np.mean(math_window))
        math_stds.append(np.std(math_window))
        # Compare network vs math
        if e == 1:
            plot_episode(math_loss,name="Math single episode")
            plot_episode(network_loss,name="Network single episode")
        if e % 5 == 0:
            plot(means,stds,name=config.name,game='RouteMuse')
            plot(math_means,math_stds,name='Math',game='RouteMuse')

if __name__ == '__main__':
    main()