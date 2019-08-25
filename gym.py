import numpy as np
from utils import Utilities

"""
Gym class for generating and training on fictitious data. Is a combination of the Utilities class and gym class
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

class Gym(Utilities):
    def __init__(self,json_obj,config):
        super(Utilities,self)
        ### Utility init ###
        self.keys = config.keys
        self.reverse_keys = {v:k for k,v in self.keys.items()}
        self.route_array,self.field_indexes,self.field_lengths,self.field_dictionary = self.return_field_objects(json_obj)
        self.grade_technique_mask = config.grade_technique_keys
        self.terrain_technique_mask = config.terrain_technique_keys
        self.num_fields = len(self.field_indexes)
        self.importance_weights = config.importance_weights
        assert self.num_fields == 12
        self.total_fields = self.field_indexes[-1][-1]
        print('total number of fields', self.total_fields)
        self.epsilon = 1e-7 
        
        ### Gym init ###
        self.num_routes = config.total_routes
        self.num_reset = config.num_reset_routes
        self.routes = None
        self.goals = None
        self.route_shape = None
        self.prev_loss = None
        self.reset_index = 1
        self.set_index = 0

    def reset(self):
        """
        Resets env.
        regenerates new routes and new goals
        resets the indexes
        """
        self.goals = self.gen_random_goals(self.num_routes)
        self.routes = self.gen_random_routes(self.num_routes)
        self.historical_routes = self.gen_random_routes(self.num_routes)
        self.route_shape = self.routes.shape[1]
        self.reset_index = 0
        self.set_index = 0
        self.prev_loss = self.loss
        self.bulk_strip(self.num_reset)
        return self.distance

    def step(self,suggestion):
        routes = self.route_from_suggestion(suggestion)
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

    def novelty_distance(self,suggestion,routes):
        assert len(routes.shape) == 1
        # Distance from all routes
        mean_distance = np.abs(suggestion - self.mean_route())
        # Distance among the location
        location_num = self.reverse_keys['location']
        location_index = self.field_indexes[location_num]
        location_mask = np.where(routes[location_index[0]:location_index[1]] == 1)[0]
        mean_loc_distance = np.abs(suggestion - self.mean_location_route(location_mask))
        # Distance from grade
        grade_location = self.reverse_keys['grade']
        grade_index = self.field_indexes[grade_location]
        grade_mask = np.where(routes[grade_index[0]:grade_index[1]] == 1)[0]
        mean_grade_distance = np.abs(suggestion - self.mean_grade_route(grade_mask))
        # Distance from historical grade
        hist_grade_location = self.reverse_keys['grade']
        hist_grade_index = self.field_indexes[hist_grade_location]
        hist_grade_mask = np.where(routes[hist_grade_index[0]:hist_grade_index[1]] == 1)[0]
        mean_hist_grade_distance = np.abs(suggestion - self.historical_grade_route(hist_grade_mask))
        temp = self.field_sum(mean_hist_grade_distance) 
        novelty_dist = temp * self.importance_weights
        return np.sum(novelty_dist)

    def field_sum(self,route):
        fields = np.zeros(len(self.field_indexes))
        for i,index in enumerate(self.field_indexes):
            fields[i] = np.sum(route[index[0]:index[1]])
        return fields

    def novelty_step(self,suggestion):
        routes = self.route_from_suggestion(suggestion)
        # minus to maximize distance
        novelty_reward = -self.novelty_distance(suggestion,routes)
        if len(routes.shape) > 1:
            self.bulk_update(routes)
            distance_reward = self.loss - self.prev_loss
            self.bulk_strip(self.num_routes)
        else:
            self.set_route(routes)
            distance_reward = self.loss - self.prev_loss
            self.strip_route()
        reward = distance_reward + novelty_reward
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

    ### Novelty Portion ###
    def mean_route(self):
        return np.mean(self.routes,axis=0)
    def mean_grade_route(self,grade):
        # Select only that grade
        grade_location = self.reverse_keys['grade']
        grade_index = self.field_indexes[grade_location]
        grade_start = grade_index[0]
        grade_mask = np.where(self.routes[:,grade_start+grade] == 1)[0]
        desired_routes = self.routes[grade_mask,:]
        return np.mean(desired_routes,axis=0)
    def mean_location_route(self,location):
        # Select only that location
        location_num = self.reverse_keys['location']
        location_index = self.field_indexes[location_num]
        location_start = location_index[0]
        location_mask = np.where(self.routes[:,location_start+location] == 1)[0]
        desired_routes = self.routes[location_mask,:]
        return np.mean(desired_routes,axis=0)
    def historical_grade_route(self,grade):
        # Select only that grade
        grade_location = self.reverse_keys['grade']
        grade_index = self.field_indexes[grade_location]
        grade_mask = np.where(self.routes[:,grade_index[0]+grade] == 1)[0]
        hist_grade_mask = np.where(self.historical_routes[:,grade_index[0]+grade] == 1)[0]
        active_routes = self.routes[grade_mask,:]
        inactive_routes = self.historical_routes[hist_grade_mask,:]
        desired_routes = np.concatenate([active_routes,inactive_routes],axis = 0)
        return np.mean(desired_routes,axis=0)

    # Route generation with novelty. Uses Utilities
    def novel_probabilistic_route(self):
        route = np.zeros(self.total_fields)
        mask = []
        for key,index in enumerate(self.field_indexes):
            field_location = distance[index[index[0]:index[1]]]
            if self.keys[key] == 'techniques':
                combined_mask = loc_mask * grade_mask
                field_location *= combined_mask
                field_location = field_location / np.sum(field_location)
            # We add the min to turn all values >= 0
            selection_prob = Utilities.softmax(distance[field_location]+np.abs(np.min(field_location)))
            # Make sure no probability is 0
            # selection_prob = (selection_prob + self.epsilon)
            field_choice = np.random.choice(np.arange(index[0],index[1]),p=selection_prob)
            # Get mask for terrain type and grade
            if self.keys[key] == 'grade':
                # Grade
                grade_mask = self.grade_technique_mask[field_choice-index[0]]
                # Now we have the grade, we can get the novelty of that grade
                grade = field_choice + index[0]
                mean_grade_route = self.mean_grade_route(grade)
            elif self.keys[key] == 'terrain_type':
                # terrain type
                loc_mask = self.terrain_technique_mask[field_choice-index[0]]

            mask.append(field_choice + index[0])
        mask = np.array(Utilities.flatten_list(mask))
        route[mask] = 1
        return route
    def novel_deterministic_route(self):
        pass

        
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
