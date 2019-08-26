import unittest
import sys
import numpy as np

from test_data import build_data
sys.path.append('/Users/morgan/Code/RouteMuse/')
# sys.path.append('/home/kenpachi/Code/RouteMuse/')
from utils import Utilities
from config import Config

"""
Unit tests for all functions

seed the Utility class with the relevant data
"""

config = Config()
keys = config.keys
fields = build_data()
utils = Utilities(fields,keys)

class TestConversion(unittest.TestCase):
    def test(self):
        print('utils route_array',utils.route_array)
        self.assertEqual(len(utils.field_indexes), 12)

class TestRandomRoutes(unittest.TestCase):
    def test(self):
        ran_routes = utils.gen_random_routes(5)
        self.assertEqual(ran_routes.shape[0], 5)

class TestReadable(unittest.TestCase):
    def test(self):
        ran_routes = utils.gen_random_routes(5)
        readable_routes = [utils.convert_route_to_readable(route) for route in ran_routes]
        print('readable_routes',readable_routes)

class TestRandomNum(unittest.TestCase):
    def test(self):
        # test random num gen for goals
        styles = utils.random_num(5,5)
        self.assertAlmostEqual(np.sum(styles),5)


class TestDefaultGoals(unittest.TestCase):
    def test(self):
        goals = utils.make_default_goals(5)
        print(goals)

class TestRandomGoals(unittest.TestCase):
    def test(self):
        num_routes = 5
        ran_goals = utils.gen_random_goals(num_routes)
        for index in utils.field_indexes:
            self.assertAlmostEqual(np.sum(ran_goals[index[0]:index[1]]),num_routes)
        

if __name__ == '__main__':
    unittest.main()