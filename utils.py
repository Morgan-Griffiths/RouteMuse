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
    10:'hold_sets',
    11:'grade'
}
"""
class Utilities(object):
    def __init__(self,json_obj,keys):
        self.keys = keys
        self.route_array,self.field_indexes,self.field_lengths = self.return_field_objects(json_obj)
        self.num_fields = len(self.field_indexes)
        assert self.num_fields == 12
        self.total_fields = self.field_indexes[-1][-1]
        assert self.total_fields == 120

    def return_field_objects(self,j_fields):
        """
        Expects fields to be a json object

        Takes in the current fields, outputs array, and indexes of those fields

        Returns:
        Array of the fields (for quick indexing)
        indicies for each field type
        """
        fields = self.convert_json(j_fields)
        field_indexes,field_lengths = self.return_indicies(fields)
        flat_fields = Utilities.flatten_list(fields)
        ROUTE_ARRAY = np.array(flat_fields)
        return ROUTE_ARRAY,field_indexes,field_lengths

    def return_indicies(self,fields):
        """
        Returns 
        the indicies of each field type. E.G. Styles goes from 0-18
        the length of each field type (num occurances)
        """
        length = len(fields)                                # Number of fields
        field_lengths = [len(field) for field in fields]    # Size of each fields
        field_indexes = [(0,field_lengths[0])]
        running_index = field_lengths[0]
        for i in range(1,len(field_lengths)):
            field_indexes.append((running_index,field_lengths[i]+running_index))
            running_index += field_lengths[i]
        return field_indexes,field_lengths

    def convert_json(self,json_obj):
        """
        Loads json_obj, converts resulting dictionary to list
        """
        j_dictionary = json.loads(json_obj)
        fields = [j_dictionary[key] for key in self.keys.values()]
        fields_list = [list(field.values()) for field in fields]
        return fields_list

    def gen_random_routes(self,num_routes):
        """
        Gen ints between the field indicies.

        can gen multiple in a given category for instance multiple styles
        """
        routes = np.zeros((num_routes,self.field_indexes[-1][-1]))
        for n in range(num_routes):
            mask = [np.random.choice(np.arange(start,end)) for start,end in self.field_indexes]
            routes[n][mask] = 1
        return routes

    def convert_route_to_readable(self,route):
        """
        Returns human readable routes
        """
        mask = np.where(route > 0)[0]
        return self.route_array[mask]

    ### Goals ###

    def make_default_goals(self,num_routes):
        """
        Returns evenly distributed goals
        """
        goals = np.zeros(self.total_fields)
        defaults = [num_routes / field_length for field_length in self.field_lengths]
        for i,default in enumerate(defaults):
            goals[self.field_indexes[i][0]:self.field_indexes[i][1]] = np.full(self.field_lengths[i],default)
        return goals

    def random_num(self,gym_routes,N):
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
        for i in range(1,N-1):
            arr[i] = np.random.random() * gym_routes
            gym_routes -= arr[i]
        arr[-1] = gym_routes
        return arr

    def gen_random_goals(self,num_routes):
        # goals = np.zeros(self.total_fields)
        
        values = [self.random_num(num_routes,index[1]-index[0]) for index in (self.field_indexes)]
        goals = np.array(Utilities.flatten_list(values))
        print('goals',goals.shape)
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

    def route_from_distance(self,distance):
        route = np.zeros(self.total_fields)
        mask = []
        for index in self.field_indexes:
            location = np.where(distance[index[0]:index[1]] == np.max(distance[index[0]:index[1]]))
            mask.append(location + index[0])
        mask = np.array(Utilities.flatten_list(mask))
        route[mask] = 1
        return route

    def probabilistic_route(self,distance):
        route = np.zeros(self.total_fields)
        mask = []
        for index in self.field_indexes:
            field_location = distance[index[index[0]:index[1]]]
            # We add the min to turn all values >= 0
            selection_prob = Utilities.softmax(distance[field_location]+np.abs(np.min(field_location)))
            field_choice = np.random.choice(np.arange(index[0],index[1]),p=selection_prob)
            mask.append(field_choice + index[0])
        mask = np.array(Utilities.flatten_list(mask))
        route[mask] = 1
        return route

    ### For the network ###

    def route_from_suggestion(self,suggestion):
        """
        For converting network outputs to actual routes
        """
        route = np.zeros(self.total_fields)
        
        mask = []
        for index in self.field_indexes:
            field_location = suggestion[index[0]:index[1]]
            field_choice = np.random.choice(np.arange(index[0],index[1]),p=field_location)
            mask.append(field_choice)
        # mask = np.array(Utilities.flatten_list(mask))
        route[mask] = 1
        return route