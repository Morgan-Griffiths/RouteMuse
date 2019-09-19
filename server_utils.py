import numpy as np
import time
import datetime
import json
"""
Needs to be loaded with 
1. json object of gym data
2. key dictionary akin to 
keys = {
    0: 'grade',
    1: 'terrain_type',
    2: 'height_friendly',
    3: 'finish_location',
    4: 'start_location',
    5: 'locations',
    6: 'risk',
    7: 'intensity',
    8: 'complexity',
    9: 'intra_difficulty',
    10: 'style',
    11: 'teaches' # TODO turn into techniques
}

Converts all routes into array format on initialization

Put the stripped routes into historical? Or a special category.

pass locations to strip. mask out all of those routes -> put in historical. Instantiate new routes with locations taken from the stripped routes.
Later this will change when locations contain the number of routes we need to set.
Populate the routes one at a time. To take account of the changing reality created by putting up new routes.
Mask the routes with terrain types and technique by grade.
locations that have multiple terrain types will be additive for the mask. So that its the union between the two. As both styles will be possible.

Stored Variables:
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

Functions:
parse_fields : Takes in the field object from DB and returns field_dictionary,reverse_field_dictionary,field_names,goal_names,reverse_field_names
return_indicies : Returns indicies of each field type
return_keys :
display_route : displays the route in human readable form
### Math ###
deterministic_goal_route :
probabilistic_goal_route :
route_from_probabilities : Takes in a route in the form of probabilities and returns a route of 0s and 1s
### Network ###
"""


class ServerUtilities(object):
    def __init__(self, active_routes,historical_routes,fields, config):
        self.fields_to_ignore = config.fields_to_ignore
        self.goalfields_to_ignore = config.goalfields_to_ignore
        self.terrain_types = config.terrain_types
        self.field_dictionary,self.reverse_field_dictionary,self.field_names,self.goal_names,self.reverse_field_names = self.parse_fields(fields)
        self.field_indexes, self.field_lengths = self.return_indicies()
        self.keys,self.reverse_keys,self.route_array = self.return_keys()
        self.num_fields = len(self.field_indexes)
        self.total_fields = self.field_indexes[-1][-1]
        # print('total number of fields', self.total_fields)
        # print('num_fields', self.num_fields)
        # print('reverse_keys', self.reverse_keys)
        # print('reverse_field_dictionary', self.reverse_field_dictionary)
        # print('field_names', self.field_names)
        # print('self.route_array',self.route_array)
        # The following should all come from the database
        self.grade_technique_mask = config.grade_technique_keys
        self.terrain_technique_mask = config.terrain_technique_keys
        self.novelty_weights = config.novelty_weights
        self.goal_weights = config.goal_weights
        self.num_routes = config.total_routes
        self.num_reset = config.num_reset_routes
        self.route_type = config.route_type
        self.epsilon = 1e-7 
        
        self.route_shape = self.total_fields
        # print('self.field_lengths',self.field_lengths)
        self.goal_shape = self.total_fields - self.field_lengths[0]
        self.prev_loss = None

        self.active_routes = self.convert_to_array(active_routes)
        self.historical_routes = self.convert_to_array(historical_routes)
        # print(self.display_route(self.active_routes[-1]))
        # print(self.display_route(self.active_routes[-2]))
        # print(active_routes[:2])
        # print(self.display_route(self.convert_to_array([active_routes[0]])))
        # print(self.display_route(self.convert_to_array([active_routes[1]])))
        self.max_grade = None
        
        ### Route function ###
        if self.route_type == 'math':
            self.create_route = self.route_from_network
        elif self.route_type == 'rl':
            self.create_route = self.route_from_probabilities
        else:
            raise ValueError("route_type not understood. Must be 'math' or 'rl'")

    def update_max_grade(self,grade):
        self.max_grade = np.ones(self.field_lengths[1])
        if self.field_lengths[1] != grade:
            self.max_grade[grade+1:] = 0

    def parse_fields(self,fields):
        """
        Returns 
        keys : (dictionary) Field (ex. grade) {0:subfield(ex. 'blue')}
        reverse keys : (dictionary) Field (ex. grade) {subfield(ex. 'blue'):0}
        """
        keys = {}
        field_names = []
        goal_names = []
        reverse_field_names = {}
        for field in fields.find():
            # This pulls all the values and skips fields like notes and setters
            # Skip primary holds for now
            try:
                if field['label'] not in self.fields_to_ignore:
                    if field['label'] not in self.goalfields_to_ignore:
                        goal_names.append(field['name'])
                    values = field['values']
                    field_names.append(field['name'])
                    indicies = np.arange(len(values))
                    keys[field['name']] = {indicies[i]:values[i] for i in range(len(values))}
            except:
                pass
        reverse_keys = {}
        i = 0
        j = 0
        for key,value in keys.items():
            subdict = {}
            for k,v in value.items():
                if isinstance(v,float):
                    v = int(v)
                subdict[v] = i
                i += 1
            reverse_keys[key] = subdict
            reverse_field_names[key] = j
            j += 1
        return keys,reverse_keys,field_names,goal_names,reverse_field_names

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
        print(self.route_array[mask])
        return self.route_array[mask]

    def display_all(self, routes):
        """
        routes : np.array
        Returns human readable routes
        """
        for i in range(routes.shape[0]):
            self.display_route(routes[i,:])

    ### Math portion ###
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
        probs = np.zeros(self.goal_shape)
        mask = []
        location_length = self.field_lengths[0]
        for i in range(1,len(self.field_indexes)):
            index = self.field_indexes[i]
            start = index[0]-location_length
            end = index[1]-location_length
            field_location = distance[start:end]
            selection_prob = field_location+np.abs(np.min(field_location)*2)
            selection_prob /= np.sum(selection_prob)
            probs[start:end] = selection_prob
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
            if self.keys[key] == 'teaches':
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
        self.active_routes[self.set_index][:] = route
        self.update_set_index()

    def strip_route(self):
        empty_route = np.zeros(self.route_shape)
        self.historical_routes[self.reset_index][:] = self.active_routes[self.reset_index][:]
        self.active_routes[self.reset_index][:] = empty_route
        self.update_reset_index() 

    def bulk_update(self,routes):
        for i in routes.shape[0]:
            self.active_routes[self.set_index][:] = routes[i][:]
            self.update_reset_index() 
    
    def bulk_strip(self,num_routes):
        """
        num_routes : int - number of routes to be reset
        route_indicies : np.array - indicies of the routes to be replaced.

        delete routes that are to be stripped and calc distance. More informative than the pure distance
        """
        empty_route = np.zeros(self.route_shape)
        for _ in range(num_routes):
            self.historical_routes[self.reset_index][:] = self.active_routes[self.reset_index][:]
            self.active_routes[self.reset_index] = empty_route
            self.update_reset_index()
    @staticmethod
    def asvoid(arr):
        arr = np.ascontiguousarray(arr)
        if np.issubdtype(arr.dtype,np.floating):
            arr += 0
        return arr.view(np.dtype((np.void,arr.dtype.itemsize * arr.shape[-1])))
    @staticmethod
    def inNd(a,b,assume_unique=False):
        a = ServerUtilities.asvoid(a)
        b = ServerUtilities.asvoid(b)
        return np.in1d(a,b,assume_unique)

    def bulk_strip_by_route(self,routes):
        arr_routes = self.convert_to_array(routes)
        stripped_mask = []
        for i in range(arr_routes.shape[0]):
            stripped_mask.append(np.where((self.active_routes == arr_routes[i]).all(axis=1))[0][0])
        stripped_mask = np.array(stripped_mask)
        print('stripped_mask',stripped_mask)
        self.stripped_routes = self.active_routes[stripped_mask,:]
        print('self.active_routes[stripped_mask,:]',self.active_routes[stripped_mask,:])
        print('self.stripped_routes.shape',self.stripped_routes.shape)
        self.suggested_routes = self.stripped_routes
        self.suggested_routes[self.field_lengths[self.reverse_field_names['location']]:] = 0
        routes_remaining = np.ones(self.active_routes.shape[0],dtype=bool)
        routes_remaining[stripped_mask] = 0
        self.active_routes = self.active_routes[routes_remaining]
        self.historical_routes = np.vstack((self.stripped_routes,self.historical_routes))
        self.display_all(self.stripped_routes)

    
    def bulk_strip_by_location(self,locations):
        """
        num_routes : int - number of routes to be reset
        route_indicies : np.array - indicies of the routes to be replaced.

        first convert locations into array
        move currently active routes in that location to historical
        """
        stripped_locations = np.zeros(self.route_shape)
        for location in locations:
            stripped_locations[self.reverse_field_dictionary['location'][location['name']]] = 1
        stripped_mask = np.where(stripped_locations == 1)[0]
        route_mask = np.where(self.active_routes[:,stripped_mask] == 1)[0]
        routes_remaining = np.ones(self.active_routes.shape[0],dtype=bool)
        routes_remaining[route_mask] = 0
        self.stripped_routes = self.active_routes[route_mask,:]
        self.suggested_routes = self.stripped_routes
        self.suggested_routes[self.field_lengths[self.reverse_field_names['location']]:] = 0
        self.active_routes = self.active_routes[routes_remaining]
        # If we want to add them to our historical routes
        self.historical_routes = np.vstack((self.stripped_routes,self.historical_routes))
        # self.display_all(self.stripped_routes)
        # self.display_all(self.active_routes)

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
        return np.mean(self.active_routes,axis=0)
    
    def active_mean_field_route(self,field_entry,field_choice):
        # Select only that field
        field_location = self.reverse_field_names[field_choice]
        field_index = self.field_indexes[field_location]
        field_start = field_index[0]
        field_mask = np.where(self.active_routes[:,field_start+field_entry] == 1)[0]
        assert field_mask.any()
        desired_routes = self.active_routes[field_mask,:]
        return np.mean(desired_routes,axis=0)
    
    def historical_mean_field_route(self,field_entry,field_choice):
        # Select only that field
        field_location = self.reverse_field_names[field_choice]
        field_index = self.field_indexes[field_location]
        field_mask = np.where(self.active_routes[:,field_index[0]+field_entry] == 1)[0]
        hist_field_mask = np.where(self.historical_routes[:,field_index[0]+field_entry] == 1)[0]
        # check if there are any such routes
        assert field_mask.any()
        assert hist_field_mask.any()
        active_routes = self.active_routes[field_mask,:]
        inactive_routes = self.historical_routes[hist_field_mask,:]
        desired_routes = np.concatenate([active_routes,inactive_routes],axis = 0)
        return np.mean(desired_routes,axis=0)
    
    def novel_probabilistic_route(self,field_choice):
        target = self.field_from_distance(field_choice)
        field_index = self.field_indexes[self.reverse_field_names[field_choice]]
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
        index = self.field_indexes[self.reverse_field_names[field]]
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
        location_mask = self.field_lengths[self.reverse_field_names['location']]
        novel_route[location_mask:] += goal_route
        combined_route = novel_route
        # Probabilistically select each field
        final_route = self.route_from_probabilities(combined_route)
        return final_route
        
    def probabilistic_novel_goal_route(self,location_route,field):
        """
        Creates a route probabilistically from the combination of novel probabilities and goal probabilities.
        """
        goal_route = self.probabilistic_goal_route(self.distance)
        novel_route = self.novel_probabilistic_route(field)
        goal_route = self.apply_weights(goal_route,self.goal_weights)
        novel_route = self.apply_weights(novel_route,self.novelty_weights)
        location_mask = self.field_lengths[self.reverse_field_names['location']]
        novel_route[location_mask:] += goal_route
        novel_route[:location_mask] = location_route[:location_mask]
        grade_index = self.field_indexes[1]
        # Mask techniques with grades and terrain types TODO (Need techniques to be trimmed down to final version. And grade mask)
        # Mask grades to avoid generating grades higher than max setting ability
        novel_route[grade_index[0]:grade_index[1]] *= self.max_grade
        for i,index in enumerate(self.field_indexes):
            assert np.sum(novel_route[index[0]:index[1]]) != 0
            novel_route[index[0]:index[1]] = novel_route[index[0]:index[1]] / np.sum(novel_route[index[0]:index[1]])
        # Probabilistically select each field
        final_route = self.route_from_probabilities(novel_route)
        return final_route
        
    # For processing from DB

    def convert_goals(self,dbgoals):
        """
        goals is a container from DB

        Have to convert the goals back from goal format to actual values.
        """
        goals = np.empty(self.goal_shape,dtype=float)
        for field in self.goal_names:
            for dbvalue in dbgoals[field]:
                key = dbvalue['key']
                value = float(dbvalue['value'])
                if key.isdigit():
                    key = int(dbvalue['key'])
                goals[self.reverse_field_dictionary[field][key] - self.field_lengths[0]] = self.num_routes / value
        self.goals = goals

    def return_suggestions(self):
        suggestions = np.array([])
        field_choice = 'grade'
        for i in range(self.suggested_routes.shape[0]):
            final_route = self.probabilistic_novel_goal_route(self.suggested_routes[i,:],field_choice)
            # Add to current active routes
            self.active_routes = np.vstack((final_route,self.active_routes))
            if suggestions.size == 0:
                suggestions = final_route
            else:
                suggestions = np.vstack((final_route,suggestions))
        # print all suggestions
        # for suggestion in suggestions:
        #     self.display_route(suggestion)
        # Convert routes back to readable
        readable_suggestions = self.convert_routes_from_array(suggestions)
        return suggestions,readable_suggestions

    def convert_routes_from_array(self,suggestions):
        now = datetime.datetime.now().isoformat()
        readable_routes = []
        for suggestion in suggestions:
            default_route = {"primary_hold_set":[],"teaches":[],"hold_set":[],"location":'',"setter":'',"image":'',"grade":"","intra_difficulty":4,"risk":5,"intensity":3,"complexity":3,"height_friendly":"Average","set_screwed":'',"stripped":'',"start_location":'',"finish_location":'',"date":now,"notes":''}
        
            mask = np.where(suggestion > 0)[0]
            print(self.route_array[mask])
            for i,name in enumerate(self.field_names):
                value = self.route_array[mask][i]
                if name == 'teaches':
                    default_route[name].append(value)
                else:
                    default_route[name] = value
            readable_routes.append(default_route)
        print(readable_routes)
        return readable_routes

    def convert_to_array(self,routes):
        arr_routes = np.array([])
        for route in routes:
            new_route = np.zeros(self.route_shape)
            for field in self.field_names:
                try:
                    new_route[self.reverse_field_dictionary[field][route[field]]] = 1
                except:
                    pass
            if arr_routes.size == 0:
                arr_routes = new_route
            else:
                arr_routes = np.vstack((new_route,arr_routes))
        return arr_routes

    @staticmethod
    def flatten_list(nd_list):
        flat_list = [item for sublist in nd_list for item in sublist]
        return flat_list

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @property
    def distance(self):
        # mean columns in routes
        # Goals doesn't use locations, so we maskout locations before subtraction
        current_dist = np.sum(self.active_routes[self.field_lengths[0]:],axis = 0)
        nd_distance = self.goals - current_dist[self.field_lengths[0]:]
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
        current_dist = np.sum(self.active_routes[self.field_lengths[0]:],axis = 0)
        nd_distance = self.goals - current_dist
        return np.mean(np.abs(nd_distance))
    
    def __len__(self):
        # Return number of routes
        return self.active_routes.shape[0]
