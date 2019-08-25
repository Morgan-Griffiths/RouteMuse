import numpy as np 
import sys
import os
from collections import namedtuple,deque
import time
import pickle

from plots.plot import plot
from gym import Gym
from config import Config
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
    config = Config('math')
    fields = build_data()
    gym_math = Gym(fields,config)
    # train on data
    test_math(gym_math,config)

def test_math(gym_math,config):
    """
    For plotting - plot the math mean error versions along with the agent mean
    """
    # Create two instances of gyms
    # tic = time.time()

    # Collections
    math_means = []
    math_stds = []
    math_window = deque(maxlen=100)
    for e in range(1,11):
        math_rewards = []
        math_state = gym_math.reset()
        math_loss = [gym_math.loss]
        for t in range(config.tmax):
            # Compare with math
            mean_hist_route = gym_math.historical_grade_route(0)
            mean_grade_route = gym_math.mean_grade_route(0)
            mean_loc_route = gym_math.mean_location_route(0)

            math_route = gym_math.probabilistic_route(math_state)
            math_next_state, math_reward = gym_math.step(math_route)
            math_rewards.append(math_reward)
            math_loss.append(gym_math.loss)
            math_state = math_next_state
        math_window.append(np.sum(math_rewards))
        math_means.append(np.mean(math_window))
        math_stds.append(np.std(math_window))
        # Compare network vs math

if __name__ == '__main__':
    main()