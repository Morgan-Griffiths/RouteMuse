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

Field dictionary is the opposite of keys. Key is string, value is position.
3. grade_technique_mask is a matrix of values, 1s where the technique is possible
 [[
     1,0,0,0
     0,0,0,1
 ]]
4. location_technique_maskis a matrix of values, 1s where the technique is possible
 [[
     1,0,0,0
     0,0,0,1
 ]]
 They will be supplied by the DB in the form of a matrix. each row will correspond to a particular
 grade or terrain type. Sorted in order of how they are stored. That way they can be easily accessed.
 The way they will be used:
 The network will output a vector of probabilities. Given the location/terrain type, we will zero
 out some of the technique possibilities. And given the grade, we will also do the same. Then we
 willl renormalize the category and select the techniques from then on. If we want to sample
 without replacement, then add replace=False
"""


class Utilities(object):
    def __init__(self, json_obj, config):
        self.keys = config.keys
        self.route_array, self.field_indexes, self.field_lengths, self.field_dictionary = self.return_field_objects(
            json_obj)
        self.grade_technique_mask = config.grade_technique_keys
        self.terrain_technique_mask = config.terrain_technique_keys
        self.num_fields = len(self.field_indexes)
        self.total_fields = self.field_indexes[-1][-1]
        print('total number of fields', self.total_fields)
        self.epsilon = 1e-7

    def return_field_objects(self, j_fields):
        """
        Expects fields to be a json object

        Takes in the current fields, outputs array, and indexes of those fields

        Returns:
        Array of the fields (for quick indexing)
        indicies for each field type
        """
        fields, field_dictionary = self.convert_json(j_fields)
        field_indexes, field_lengths = self.return_indicies(fields)
        flat_fields = Utilities.flatten_list(fields)
        ROUTE_ARRAY = np.array(flat_fields)
        return ROUTE_ARRAY, field_indexes, field_lengths, field_dictionary

    def convert_json(self, json_obj):
        """
        Loads json_obj, converts resulting dictionary to list
        """
        j_dictionary = json.loads(json_obj)
        fields = [j_dictionary[key] for key in self.keys.values()]
        fields_list = [list(field.values()) for field in fields]
        return fields_list, j_dictionary

    def return_indicies(self, fields):
        """
        Returns 
        the indicies of each field type. E.G. Styles goes from 0-18
        the length of each field type (num occurances)
        """
        length = len(fields)                                # Number of fields
        field_lengths = [len(field)
                         for field in fields]    # Size of each fields
        field_indexes = [(0, field_lengths[0])]
        running_index = field_lengths[0]
        for i in range(1, len(field_lengths)):
            field_indexes.append(
                (running_index, field_lengths[i]+running_index))
            running_index += field_lengths[i]
        return field_indexes, field_lengths

    def gen_random_routes(self, num_routes):
        """
        Gen ints between the field indicies.

        can gen multiple in a given category for instance multiple styles
        """
        routes = np.zeros((num_routes, self.field_indexes[-1][-1]))
        for n in range(num_routes):
            mask = [np.random.choice(np.arange(start, end))
                    for start, end in self.field_indexes]
            routes[n][mask] = 1
        return routes

    def display_route(self, route):
        """
        Returns human readable routes
        """
        mask = np.where(route > 0)[0]
        return self.route_array[mask]

    ### Goals ###

    def make_default_goals(self, num_routes):
        """
        Returns evenly distributed goals
        """
        goals = np.zeros(self.total_fields)
        defaults = [num_routes /
                    field_length for field_length in self.field_lengths]
        for i, default in enumerate(defaults):
            goals[self.field_indexes[i][0]:self.field_indexes[i]
                  [1]] = np.full(self.field_lengths[i], default)
        return goals

    def random_num(self, gym_routes, N):
        """
        gym_routes : Int 
        N : Int

        For the generation of random goals. Makes sure all the sub fields sum to the total gym routes.

        Styles and techniques can be > 1 occurances per route
        Can scale style and technique suggestions with the route length

        For now i'm doing 1 per route. Will change later
        """
        arr = np.zeros(N)
        arr[0] = np.random.random() * gym_routes
        gym_routes -= arr[0]
        for i in range(1, N-1):
            arr[i] = np.random.random() * gym_routes
            gym_routes -= arr[i]
        arr[-1] = gym_routes
        return arr

    def gen_random_goals(self, num_routes):
        # goals = np.zeros(self.total_fields)

        values = [self.random_num(num_routes, index[1]-index[0])
                  for index in (self.field_indexes)]
        goals = np.array(Utilities.flatten_list(values))
        return goals

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
