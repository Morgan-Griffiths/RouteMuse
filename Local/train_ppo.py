import numpy as np 
import sys
import os
from collections import namedtuple,deque
import time
import pickle

from plots.plot import plot
from Network import PPO_net
from gym import Gym
from utils import Utilities
from config import Config
from Agents.ppo_agent import PPO
sys.path.append('/Users/morgan/Code/RouteMuse/test')
# sys.path.append('/home/kenpachi/Code/RouteMuse/test')
print('path',os.getcwd())
from test_data import build_data

"""
Generate training data and train on it

TODO
Generate a unique hash for each route
"""

def main():
	# Instantiate objects
	agent_name = 'PPO'
	config = Config(agent_name)
	fields = build_data()
	utils = Utilities(fields,config)
	agent = PPO(utils.total_fields,utils.total_fields,utils.field_indexes,config)
	# train on data
	train_network(agent,utils,config)

def train_network(agent,utils,config):
	"""
	For plotting - plot the math mean error versions along with the agent mean
	"""
	experience = namedtuple('experience',field_names=['state','value','log_prob','action','reward','next_state'])
	# Compared training with math baseline
	gym = Gym(config.total_routes,config.num_reset_routes,utils)
	math_gym = Gym(config.total_routes,config.num_reset_routes,utils)
	tic = time.time()
	# Collections
	max_mean = 0
	means = []
	stds = []
	mins = []
	maxes = []
	stds = []
	steps = []
	math_means = []
	math_stds = []
	math_window = deque(maxlen=100)
	scores_window = deque(maxlen=100)
	for e in range(1,config.episodes):
		trajectory = []
		rewards = []
		math_rewards = []
		state = gym.reset()
		math_state = math_gym.reset()
		for t in range(config.tmax):
			# Normalize state for network
			normalized_state = normalize(state)
			suggestion,log_prob,value = agent.act(normalized_state)
			route = utils.route_from_suggestion(suggestion)
			next_state,reward = gym.step(route)
			# math comparison
			math_route = utils.deterministic_route(math_state)
			math_next_state,math_reward = math_gym.step(math_route)
			math_rewards.append(math_reward)
			math_state = math_next_state
			# Record (s,a,r,s)
			exp = experience(state,value,log_prob,suggestion,reward,next_state)
			trajectory.append(exp)
			rewards.append(reward)
			state = next_state
		agent.step(trajectory)
		steps.append(t)
		math_window.append(sum(math_rewards))
		math_means.append(np.mean(math_window))
		math_stds.append(np.std(math_window))
		scores_window.append(sum(rewards))
		means.append(np.mean(scores_window))
		mins.append(np.min(scores_window))
		maxes.append(np.max(scores_window))
		stds.append(np.std(scores_window))
		if e % 10 == 0:
			toc = time.time()
			r_mean = np.mean(scores_window)
			r_max = max(scores_window)
			r_min = min(scores_window)
			r_std = np.std(scores_window)
			plot(math_means,math_stds,name="Math",game="RouteMuse")
			plot(means,stds,name=config.name,game='RouteMuse')
			print("\rEpisode: {} out of {}, Steps {}, Mean steps {:.2f}, Rewards: mean {:.2f}, min {:.2f}, max {:.2f}, std {:.2f}, Elapsed {:.2f}".format(e,config.episodes,np.sum(steps),np.mean(steps),r_mean,r_min,r_max,r_std,(toc-tic)/60))
			# save scores
			if r_mean > max_mean:
				pickle.dump([means,maxes,mins], open(str(config.name)+'_scores.p', 'wb'))
				# save policy
				agent.save_weights(config.checkpoint_path)
				max_mean = r_mean

def normalize(arr):
	return (arr - arr.min()) / (arr.max() - arr.min())
	

if __name__ == '__main__':
	# print('path',sys.path[0])
    main()