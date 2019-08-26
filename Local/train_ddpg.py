import numpy as np 
import sys
import os
from collections import namedtuple,deque
import time
import pickle

from plots.plot import plot
from gym import Gym
from utils import Utilities
from config import Config
print('path',os.getcwd())
from Agents.DDPG.ddpg_agent import Agent
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
    agent_name = 'ddpg'
    config = Config(agent_name)
    fields = build_data()
    gym = Gym(fields,config)
    math_gym = Gym(fields,config)
    agent = Agent(gym.total_fields,gym.total_fields,gym.field_indexes,config)
    # Fill replay buffer
    seed_replay_buffer(gym,agent,config.min_buffer_size)
    # train on data
    train_network(agent,gym,math_gym,config)

def seed_replay_buffer(gym, agent, min_buffer_size):
    obs = gym.reset()
    while len(agent.PER) < min_buffer_size:
        # Random actions between 1 and -1
        normalized_state = normalize(obs)
        actions = agent.act(normalized_state)
        actions = gym.route_from_suggestion(actions)
        next_obs,rewards = gym.step(actions)
        normalized_next_state = normalize(obs)
        # reshape
        agent.add_replay_warmup(normalized_state,actions,rewards,normalized_next_state)
        obs = next_obs
    print('finished replay warm up')

def train_network(agent,gym,math_gym,config):
    """
    For plotting - plot the math mean error versions along with the agent mean
    """
    # Compared training with math baseline
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
        rewards = []
        math_rewards = []
        state = gym.reset()
        math_state = math_gym.reset()
        for t in range(config.tmax):
            # Normalize state for network
            normalized_state = normalize(state)
            suggestion = agent.act(normalized_state)
            next_state,reward = gym.novelty_step(suggestion)
            normalized_next_state = normalize(next_state)
            # Step agent
            agent.step(normalized_state, suggestion, reward, normalized_next_state)

            # math comparison
            math_route = math_gym.deterministic_route(math_state)
            math_next_state,math_reward = math_gym.step(math_route)
            math_rewards.append(math_reward)
            math_state = math_next_state
            rewards.append(reward)
            state = next_state
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
            print("\rEpisode: {} out of {}, Steps {}, Noise {:.2f} Rewards: mean {:.2f}, min {:.2f}, max {:.2f}, std {:.2f}, Elapsed {:.2f}".format(e,config.episodes,np.sum(steps),agent.noise_scale,r_mean,r_min,r_max,r_std,(toc-tic)/60))
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