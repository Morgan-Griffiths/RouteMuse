import numpy as np 
import sys
import os
from collections import namedtuple

from Network import PPO_net
from gym import Gym
from utils import Utilities
from config import Config
from ppo_agent import PPO
# sys.path.append('/Users/morgan/Code/RouteMuse/')
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
	utils = Utilities(fields,config.keys)
	agent = PPO(utils.total_fields,utils.total_fields,utils.field_indexes,config)
	# train on data
	train_network(agent,utils,config)

def train_network(agent,utils,config):
	experience = namedtuple('experience',field_names=['state','value','log_prob','action','reward','next_state'])
	gym = Gym(config.total_routes,config.num_reset_routes,utils)
	total_rewards = []
	for e in range(config.episodes):
		trajectory = []
		rewards = []
		state = gym.reset()
		for t in range(config.tmax):
			suggestion,log_prob,value = agent.act(state)
			route = utils.route_from_suggestion(suggestion)
			next_state,reward = gym.step(config.num_reset_routes)
			# Record (s,a,r,s)
			exp = experience(state,value,log_prob,suggestion,reward,next_state)
			trajectory.append(exp)
			rewards.append(reward)
			state = next_state
		agent.step(trajectory)
		total_rewards.append(sum(rewards))
		print('mean score',np.mean(total_rewards))

if __name__ == '__main__':
    main()