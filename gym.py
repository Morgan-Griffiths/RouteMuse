import numpy as np

"""
Gym class for generating and training on fictitious data
------------------------
Routes are of type np.ndarray
Goals are of type np.ndarray
------------------------
The distance is Goals - Routes
The loss is MSE(distance)
------------------------
Contains the ability to calculate:
1.distance
2.loss
3.update current routes
4.delete routes

When the network or math is looking at what route to make, 
we should subtract the routes that will be stripped first,
before calculating the distance.

The historical routes are populated by stripped routes over time
They are initially seeded by the same number of routes and the active routes.
Can be used for determining the novelty factor.

The novelty factor contains a few variables:
The age of the compared routes (The importance will decay with age)
The distance between the categories (mean?)
the weight of each category (how important is it to be distant?)
"""

class Gym(object):
    def __init__(self,num_routes,num_reset,utils):
        self.num_routes = num_routes
        self.num_reset = num_reset
        self.utils = utils
        self.routes = None
        self.goals = None
        self.route_shape = None
        self.prev_loss = None
        self.reset_index = 1
        self.set_index = 0
        self.keys = {
                    0:'style',
                    1:'techniques',
                    2:'height_friendly',
                    3:'finish_location',
                    4:'start_location',
                    5:'locations',
                    6:'risk',
                    7:'intensity',
                    8:'complexity',
                    9:'intra_difficulty',
                    10:'hold_sets',
                    11:'grade'
                }
    def reset(self):
        """
        Resets env.
        regenerates new routes and new goals
        resets the indexes
        """
        self.goals = self.utils.gen_random_goals(self.num_routes)
        self.routes = self.utils.gen_random_routes(self.num_routes)
        self.historical_routes = self.utils.gen_random_routes(self.num_routes)
        self.route_shape = self.routes.shape[1]
        self.reset_index = 0
        self.set_index = 0
        self.prev_loss = self.loss
        self.bulk_strip(self.num_reset)
        return self.distance

    def step(self,routes):
        if len(routes.shape) > 1:
            self.bulk_update(routes)
            reward = self.prev_loss - self.loss
            self.bulk_strip(self.num_routes)
        else:
            self.set_route(routes)
            reward = self.prev_loss - self.loss
            self.strip_route()

        self.prev_loss = self.loss
        return self.distance,reward

    def set_route(self,route):
        self.routes[self.set_index][:] = route
        self.update_set_index()

    def strip_route(self):
        empty_route = np.zeros(self.route_shape)
        self.historical_routes[self.reset_index][:] = self.routes[self.reset_index][:]
        self.routes[self.reset_index][:] = empty_route
        self.update_reset_index() 

    def bulk_update(self,routes):
        for i in routes.shape[0]:
            self.routes[self.set_index][:] = routes[i][:]
            self.update_reset_index() 
    
    def bulk_strip(self,num_routes):
        """
        num_routes : int - number of routes to be reset
        route_indicies : np.array - indicies of the routes to be replaced.

        delete routes that are to be stripped and calc distance. More informative than the pure distance
        """
        empty_route = np.zeros(self.route_shape)
        for _ in range(num_routes):
            self.historical_routes[self.reset_index][:] = self.routes[self.reset_index][:]
            self.routes[self.reset_index] = empty_route
            self.update_reset_index()

    def update_set_index(self):
        self.set_index = (self.set_index + 1) % len(self)

    def update_reset_index(self):
        self.reset_index = (self.reset_index + 1) % len(self)
        
    @property
    def distance(self):
        # mean columns in routes
        # if isinstance(routes,(np.ndarray,np.generic)):
        #     current_dist = np.sum(routes,axis = 0)
        #     nd_distance = self.goals - current_dist
        #     return nd_distance
        # else:
        current_dist = np.sum(self.routes,axis = 0)
        nd_distance = self.goals - current_dist
        return nd_distance
    
    @property
    def get_index(self):
        """
        Current resetting index, for stripping routes
        and set index for setting routes
        """
        return self.reset_index,self.set_index
        
    @property
    def loss(self):
        current_dist = np.sum(self.routes,axis = 0)
        nd_distance = self.goals - current_dist
        return np.mean(np.abs(nd_distance))
    
    def __len__(self):
        # Return number of routes
        return self.routes.shape[0]
