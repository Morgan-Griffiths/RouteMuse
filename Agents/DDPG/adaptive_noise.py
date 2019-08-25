import numpy as np 

"""
from github.com/L5shi
"""

class AdaptiveParamNoise(object):
    def __init__(self, initial_stddev=0.1,desired_action_stddev=0.2,adaptation_coefficient=1.01):

        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self,distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev':self.current_stddev,
        }
        return stats
    
    def __repr__(self):
        fmt = "AdaptiveNoiseParam(initial_stddev={},desired_action_stddev={},adaptation_coefficient={})"
        return fmt.format(self.initial_stddev,self.desired_action_stddev,self.adaptation_coefficient)

    
