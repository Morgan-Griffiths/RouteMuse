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
"""

def main():
	# Instantiate objects
	config = Config()
	fields = build_data()
	utils = Utilities(fields,config.keys)
	agent = PPO(utils.total_fields,utils.total_fields,utils.field_indexes,config)
	# train on data
	train_network(agent,utils,config)

def generate_data(utils,episodes,N):
	"""
	Generator that returns new random gym
	"""
	for _ in range(episodes):
		new_goals = utils.gen_random_goals(N)
		new_routes = utils.gen_random_routes(N)
		new_gym = Gym(new_routes,new_goals)
		yield new_gym

def train_network(agent,utils,config):
	experience = namedtuple('experience',field_names=['state','value','log_prob','action','reward','next_state'])
	for gym in generate_data(utils,config.episodes,config.total_routes):
		trajectory = []
		inital_loss = gym.loss
		state = gym.return_masked_distance(config.num_reset_routes)
		for t in range(config.tmax):
			suggestion,log_prob,value = agent.act(state)
			route = utils.route_from_suggestion(suggestion)
			gym.add_route(route)
			loss = gym.loss
			reward = loss - inital_loss
			next_state = gym.return_masked_distance(config.num_reset_routes)
			# Record (s,a,r,s)
			exp = experience(state,value,log_prob,route,reward,next_state)
			trajectory.append(exp)

			state = next_state
			inital_loss = loss
		agent.step(trajectory)

if __name__ == '__main__':
    main()