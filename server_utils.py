import numpy as np
import json
"""
Needs to be loaded with 
1. json object of gym data
2. key dictionary akin to 
keys = {
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
    10:'terrain_types',
    11:'grade'
}


field_dictionary : (dictionary) Field : subfields e.g. "grade" : k:0 , v:blue
field_key_dictionary : (dictionary) Field : subfields e.g. "grade" : k:'blue' , v:0
keys : (dictionary) of keys [1,len(fields)] to field names
reverse_keys : (dictionary) of keys [field names] to field index
route_array : (array) of strings by sub field name (blue is a subfield of grade)
field_lengths : (array) of the lengths of each field
field_indexes : (array) A collection of tuples that hold the start and end of each field
reverse_field_index = (dictionary) key (int) value (field)
grade_technique_mask : (array) a binary vector that zeros out techniques based on whether they are available in that grade
terrain_technique_mask : (array) binary vector that zeros out techniques based on whether they are available in that terraintype
num_fields : (int) number of fields
total_fields : (int) number of subfields
novelty_weights : (array) weights for how important novelty is in those fields
goal_weights : (array) weights for how important the goals are in those fields
"""


class ServerUtilities(object):
    def __init__(self, routes,historical_routes,fields, config):
        self.field_dictionary,self.reverse_field_dictionary = self.parse_fields(fields)
        self.field_indexes, self.field_lengths = self.return_indicies()
        self.keys,self.reverse_keys,self.route_array = self.return_keys()
        self.num_fields = len(self.field_indexes)
        self.total_fields = self.field_indexes[-1][-1]
        print('total number of fields', self.total_fields)
        # The following should all come from the database
        self.grade_technique_mask = config.grade_technique_keys
        self.terrain_technique_mask = config.terrain_technique_keys
        self.novelty_weights = config.novelty_weights
        self.goal_weights = config.goal_weights
        self.num_routes = config.total_routes
        self.num_reset = config.num_reset_routes
        self.route_type = config.route_type
        self.epsilon = 1e-7 
        
        self.route_shape = None
        self.prev_loss = None
        
        ### Route function ###
        if self.route_type == 'math':
            self.create_route = self.route_from_network
        elif self.route_type == 'rl':
            self.create_route = self.route_from_probabilities
        else:
            raise ValueError("route_type not understood. Must be 'math' or 'rl'")

    def parse_fields(self,fields):
        """
        Returns 
        keys : (dictionary) Field (ex. grade) {0:subfield(ex. 'blue')}
        reverse keys : (dictionary) Field (ex. grade) {subfield(ex. 'blue'):0}
        """
        keys = {}
        for field in fields.find():
            # This pulls all the values and skips fields like notes and setters
            # Skip primary holds for now
            try:
                if field['label'] != 'Primary Hold Set':
                    values = field['values']
                    indicies = np.arange(len(values))
                    keys[field['label']] = {indicies[i]:values[i] for i in range(len(values))}
            except:
                pass
        reverse_keys = {}
        for key,value in keys.items():
            subdict = {}
            for k,v in value.items():
                subdict[v] = int(k)
            reverse_keys[key] = subdict
        return keys,reverse_keys

    def return_indicies(self):
        """
        Returns 
        the indicies of each field type. E.G. Styles goes from 0-18
        the length of each field type (num occurances)
        """
        length = len(self.field_dictionary.keys())                                # Number of fields
        field_lengths = [len(self.field_dictionary[k].values()) for k,v in self.field_dictionary.items()]    # Size of each fields
        field_indexes = [(0, field_lengths[0])]
        running_index = field_lengths[0]
        for i in range(1, len(field_lengths)):
            field_indexes.append(
                (running_index, field_lengths[i]+running_index))
            running_index += field_lengths[i]
        return field_indexes, field_lengths

    def return_keys(self):
        items = []
        for key,value in self.field_dictionary.items():
            for k,v in value.items():
                items.append(v)
        indicies = np.arange(len(items))
        keys = {indicies[i]:items[i] for i in range(len(items))}
        reverse_keys = {items[i]:indicies[i] for i in range(len(items))}
        route_array = np.array(items)
        return keys,reverse_keys,route_array

    def display_route(self, route):
        """
        Returns human readable routes
        """
        mask = np.where(route > 0)[0]
        return self.route_array[mask]

    @staticmethod
    def flatten_list(nd_list):
        flat_list = [item for sublist in nd_list for item in sublist]
        return flat_list

    ### Math portion ###
    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def deterministic_goal_route(self, distance):
        """
        Deterministic route creation
        """
        route = np.zeros(self.total_fields)
        mask = []
        for key, index in enumerate(self.field_indexes):
            location = np.where(distance[index[0]:index[1]] == np.max(
                distance[index[0]:index[1]]))[0]
            mask.append(location+index[0])
        if isinstance(mask[0], list):
            mask = np.array(Utilities.flatten_list(mask))
        route[np.array(mask)] = 1
        return route

    def probabilistic_goal_route(self, distance):
        probs = np.zeros(self.total_fields)
        mask = []
        for key, index in enumerate(self.field_indexes):
            field_location = distance[index[0]:index[1]]
            selection_prob = field_location+np.abs(np.min(field_location)*2)
            selection_prob /= np.sum(selection_prob)
            probs[index[0]:index[1]] = selection_prob
        return probs

    def route_from_probabilities(self, probabilities):
        """
        For converting network outputs to actual routes
        """
        route = np.zeros(self.total_fields)
        mask = []
        for index in self.field_indexes:
            field_location = probabilities[index[0]:index[1]]
            field_choice = np.random.choice(
                np.arange(index[0], index[1]), p=field_location, replace=False)
            mask.append(field_choice)
        # mask = np.array(Utilities.flatten_list(mask))
        route[mask] = 1
        return route

    ### For the network ###

    def route_from_network(self, suggestion):
        """
        For converting network outputs to actual routes
        """
        route = np.zeros(self.total_fields)

        mask = []
        for key, index in enumerate(self.field_indexes):
            field_location = suggestion[index[0]:index[1]]
            # Zero out the possibilities
            if self.keys[key] == 'techniques':
                # Apply masks
                combined_mask = loc_mask * grade_mask
                field_location *= combined_mask

                # renorm probabilities
                assert np.sum(field_location) != 0
                field_location = field_location / np.sum(field_location)
            if np.isnan(field_location).any():
                print('nan')
            field_choice = np.random.choice(
                np.arange(index[0], index[1]), p=field_location, replace=False)
            # Get mask for terrain type and grade
            if self.keys[key] == 'grade':
                # Grade
                grade_mask = self.grade_technique_mask[field_choice-index[0]]
            elif self.keys[key] == 'terrain_type':
                # terrain type
                loc_mask = self.terrain_technique_mask[field_choice-index[0]]

            mask.append(field_choice)
        # mask = np.array(Utilities.flatten_list(mask))
        route[mask] = 1
        return route

def step(self,suggestion):
        """
        Converts a series of probabilities to a route. 
        Sets the route. 
        Returns the gain/loss from the route
        strips the next route
        """
        routes = self.create_route(suggestion)
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
    
    def field_sum(self,route):
        fields = np.zeros(len(self.field_indexes))
        for i,index in enumerate(self.field_indexes):
            fields[i] = np.sum(route[index[0]:index[1]])
        return fields
    
    def norm_fields(self,route):
        for index in self.field_indexes:
            field = route[index[0]:index[1]]
            route[index[0]:index[1]] = field / np.sum(field)
        return route

    # novelty_type : string. One of ['mean','active','hist']
    # field : string. ex. 'grade'
    def novelty_step(self,suggestion,novelty_type,field):
        routes = self.create_route(suggestion)
        # minus to maximize distance
        novelty_reward,novelty_dist = self.novelty_distance(suggestion,routes,novelty_type,field)
        if len(routes.shape) > 1:
            self.bulk_update(routes)
            distance_reward = self.loss - self.prev_loss
            self.bulk_strip(self.num_routes)
        else:
            self.set_route(routes)
            distance_reward = self.loss - self.prev_loss
            self.strip_route()
        reward = distance_reward - np.mean(novelty_reward)
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
        
    def return_field_mask(self,field,routes):
        location_num = self.reverse_keys[field]
        location_index = self.field_indexes[location_num]
        location_mask = np.where(routes[location_index[0]:location_index[1]] == 1)[0]
        return location_mask

    ### Novelty Portion ###
    
    def novelty_distance(self,suggestion,routes,novelty_type,field=None):
        assert len(routes.shape) == 1
        if novelty_type == 'mean':
            # Distance from all routes
            novelty_dist = np.abs(suggestion - self.mean_active_route())
            novelty_loss = np.sum(novelty_dist)
        elif novelty_type == 'active':
            # Distance among the field
            mask = return_field(field,routes)
            novelty_dist = np.abs(suggestion - self.active_mean_field_route(mask))
            novelty_loss = np.sum(novelty_dist)
        elif novelty_type == 'hist':
            # Distance from historical grade
            hist_grade_mask = return_field(field,self.historical_routes)
            mean_hist_grade_distance = np.abs(suggestion - self.historical_mean_field_route(hist_grade_mask))
            temp = self.field_sum(mean_hist_grade_distance) 
            novelty_dist = temp * self.importance_weights
            novelty_loss = np.sum(novelty_dist)
        return novelty_dist,novelty_loss
    
    def mean_active_route(self):
        return np.mean(self.routes,axis=0)
    
    def active_mean_field_route(self,field_entry,field_choice):
        # Select only that field
        field_location = self.reverse_keys[field_choice]
        field_index = self.field_indexes[field_location]
        field_start = field_index[0]
        field_mask = np.where(self.routes[:,field_start+field_entry] == 1)[0]
        assert field_mask.any()
        desired_routes = self.routes[field_mask,:]
        return np.mean(desired_routes,axis=0)
    
    def historical_mean_field_route(self,field_entry,field_choice):
        # Select only that field
        field_location = self.reverse_keys[field_choice]
        field_index = self.field_indexes[field_location]
        field_mask = np.where(self.routes[:,field_index[0]+field_entry] == 1)[0]
        hist_field_mask = np.where(self.historical_routes[:,field_index[0]+field_entry] == 1)[0]
        # check if there are any such routes
        assert field_mask.any()
        assert hist_field_mask.any()
        active_routes = self.routes[field_mask,:]
        inactive_routes = self.historical_routes[hist_field_mask,:]
        desired_routes = np.concatenate([active_routes,inactive_routes],axis = 0)
        return np.mean(desired_routes,axis=0)
    
    def novel_probabilistic_route(self,field_choice):
        target = self.field_from_distance(field_choice)
        field_index = self.field_indexes[self.reverse_keys[field_choice]]
        target_mean = self.historical_mean_field_route(target,field_choice)
        
        route = np.zeros(self.total_fields)
        mask = np.arange(field_index[1],self.total_fields,dtype=np.int)
        inverse = 1 - target_mean
        # Rescale inverse
        inverse = self.norm_fields(inverse)
        route[mask] = inverse[mask]
        return route
        
    def novel_deterministic_route(self,field_choice):
        # Skip grade
        target = self.field_from_distance(field_choice)
        target_mean = self.historical_mean_field_route(target,field_choice)
        route = np.zeros(self.total_fields)
        for index in self.field_indexes:
            field = target_mean[index[0]:index[1]]
            loc = np.where(field == np.min(field))[0]
            selection = np.random.choice(loc)
            route[selection+index[0]] = 1
        return route
    
    def field_from_distance(self,field):
        # Deterministic field from distance
        index = self.field_indexes[self.reverse_keys[field]]
        field = self.distance[index[0]:index[1]]
        field_selection = np.where(field == np.max(field))[0]
        return field_selection
    
    def apply_weights(self,route,weights):
        for i,index in enumerate(self.field_indexes):
            route[index[0]:index[1]] *= weights[i]
        return route
        
    def deterministic_novel_goal_route(self,field):
        goal_route = self.deterministic_goal_route(self.distance)
        novel_route = self.novel_deterministic_route(field)
        goal_route = self.apply_weights(goal_route,self.goal_weights)
        novel_route = self.apply_weights(novel_route,self.novelty_weights)
        combined_route = goal_route + novel_route
        # Probabilistically select each field
        final_route = self.route_from_probabilities(combined_route)
        return final_route
        
    def probabilistic_novel_goal_route(self,field):
        """
        Creates a route probabilistically from the combination of novel probabilities and goal probabilities.
        """
        goal_route = self.probabilistic_goal_route(self.distance)
        novel_route = self.novel_probabilistic_route(field)
        goal_route = self.apply_weights(goal_route,self.goal_weights)
        novel_route = self.apply_weights(novel_route,self.novelty_weights)
        combined_route = goal_route + novel_route
#         print('combined_route',combined_route)
        # Probabilistically select each field
        final_route = self.route_from_probabilities(combined_route)
        return final_route
        
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
