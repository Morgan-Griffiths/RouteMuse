import numpy as np 
import sys

from Network import PPO_net
from gym import Gym
from utils import Utilities
from config import Config
from ppo_agent import PPO
sys.path.append('/Users/morgan/Code/RouteMuse/')
from test.test_data import build_data

"""
Generate training data and train on it
"""

def main():
    # Instantiate objects
    config = Config()
    fields = build_data()
    utils = Utilities(fields,keys)
    agent = PPO(utils.total_fields,config.seed,utils.field_indexes)
    # train on data
    train_network()

def generate_data():
    pass

def train_network():
    pass

if __name__ == '__main__':
    main()