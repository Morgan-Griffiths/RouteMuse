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
"""

class Gym(object):
    def __init__(self,routes,goals):
        self.routes = routes
        self.route_shape = routes.shape[1]
        self.goals = goals
        self.index = 0
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
    
    def update_route(self,route):
        self.routes[self.index][:] = route

    def delete_route(self,index):
        empty_route = np.zeros(self.route_shape)
        self.routes[index][:] = empty_route
    
    def setting_session(self,num_routes):
        """
        num_routes : int - number of routes to be reset
        route_indicies : np.array - indicies of the routes to be replaced.

        delete routes that are to be stripped and calc distance. More informative than the pure distance
        """
        empty_route = np.zeros(self.route_shape)
        for _ in range(num_routes):
            self.routes[self.index] = empty_route
            self.update_index()
        return self.distance()

    def update_index(self):
        self.index = (self.index + 1) % len(self)
    
    @property
    def get_index(self):
        """
        Current setting index, for stripping routes
        """
        return self.index
        
    @property
    def distance(self):
        # mean columns in routes
        current_dist = np.sum(self.routes,axis = 0)
        nd_distance = self.goals - current_dist
        return nd_distance
    
    @property
    def loss(self):
        current_dist = np.sum(self.routes,axis = 0)
        nd_distance = self.goals - current_dist
        return np.mean((nd_distance)**2)
    
    def __len__(self):
        # Return number of routes
        return self.routes.shape[0]
