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

Information passed into math or network to output the relevant route:
distance vector made by subtracting reality from goals
novelty vector made by grade or location
"""


class Gym(Utilities):
    def __init__(self, json_obj, config):
        super(Utilities, self)
        ### Utility init ###
        self.keys = config.keys
        self.reverse_keys = {v: k for k, v in self.keys.items()}
        self.route_array, self.field_indexes, self.field_lengths, self.field_dictionary = self.return_field_objects(
            json_obj)
        self.grade_technique_mask = config.grade_technique_keys
        self.terrain_technique_mask = config.terrain_technique_keys
        self.num_fields = len(self.field_indexes)
        self.novelty_weights = config.novelty_weights
        self.goal_weights = config.goal_weights
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

    def display_routes(self):
        for route in self.routes:
            print(self.convert_route_to_readable(route))

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

    def step(self, suggestion):
        """
        Converts a series of probabilities to a route. 
        Sets the route. 
        Returns the gain/loss from the route
        strips the next route
        """
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
        return self.distance, reward

    def novelty_distance(self, suggestion, routes):
        assert len(routes.shape) == 1
        # Distance from all routes
        mean_distance = np.abs(suggestion - self.mean_route())
        # Distance among the location
        location_num = self.reverse_keys['location']
        location_index = self.field_indexes[location_num]
        location_mask = np.where(
            routes[location_index[0]:location_index[1]] == 1)[0]
        mean_loc_distance = np.abs(
            suggestion - self.mean_location_route(location_mask))
        # Distance from grade
        grade_location = self.reverse_keys['grade']
        grade_index = self.field_indexes[grade_location]
        grade_mask = np.where(routes[grade_index[0]:grade_index[1]] == 1)[0]
        mean_grade_distance = np.abs(
            suggestion - self.mean_grade_route(grade_mask))
        # Distance from historical grade
        hist_grade_location = self.reverse_keys['grade']
        hist_grade_index = self.field_indexes[hist_grade_location]
        hist_grade_mask = np.where(
            routes[hist_grade_index[0]:hist_grade_index[1]] == 1)[0]
        mean_hist_grade_distance = np.abs(
            suggestion - self.historical_grade_route(hist_grade_mask))
        temp = self.field_sum(mean_hist_grade_distance)
        novelty_dist = temp * self.importance_weights
        novelty_loss = np.sum(novelty_dist)
        return novelty_dist, novelty_loss

    def field_sum(self, route):
        fields = np.zeros(len(self.field_indexes))
        for i, index in enumerate(self.field_indexes):
            fields[i] = np.sum(route[index[0]:index[1]])
        return fields

    def norm_fields(self, route):
        for index in self.field_indexes:
            field = route[index[0]:index[1]]
            route[index[0]:index[1]] = field / np.sum(field)
        return route

    def novelty_step(self, suggestion):
        routes = self.route_from_suggestion(suggestion)
        # minus to maximize distance
        novelty_reward, novelty_dist = self.novelty_distance(
            suggestion, routes)
        if len(routes.shape) > 1:
            self.bulk_update(routes)
            distance_reward = self.loss - self.prev_loss
            self.bulk_strip(self.num_routes)
        else:
            self.set_route(routes)
            distance_reward = self.loss - self.prev_loss
            self.strip_route()
        reward = distance_reward - novelty_reward
        self.prev_loss = self.loss
        return self.distance, reward

    def set_route(self, route):
        self.routes[self.set_index][:] = route
        self.update_set_index()

    def strip_route(self):
        empty_route = np.zeros(self.route_shape)
        self.historical_routes[self.reset_index][:] = self.routes[self.reset_index][:]
        self.routes[self.reset_index][:] = empty_route
        self.update_reset_index()

    def bulk_update(self, routes):
        for i in routes.shape[0]:
            self.routes[self.set_index][:] = routes[i][:]
            self.update_reset_index()

    def bulk_strip(self, num_routes):
        """
        num_routes : int - number of routes to be reset
        route_indicies : np.array - indicies of the routes to be replaced.

        delete routes that are to be stripped and calc distance. More informative than the pure distance
        """
        empty_route = np.zeros(self.route_shape)
        for _ in range(num_routes):
            self.historical_routes[self.reset_index][:
                                                     ] = self.routes[self.reset_index][:]
            self.routes[self.reset_index] = empty_route
            self.update_reset_index()

    def update_set_index(self):
        self.set_index = (self.set_index + 1) % len(self)

    def update_reset_index(self):
        self.reset_index = (self.reset_index + 1) % len(self)

    ### Novelty Portion ###
    def mean_active_route(self):
        return np.mean(self.routes, axis=0)

    def active_mean_field_route(self, field_entry, field_choice):
        # Select only that field
        field_location = self.reverse_keys[field_choice]
        field_index = self.field_indexes[field_location]
        field_start = field_index[0]
        field_mask = np.where(self.routes[:, field_start+field_entry] == 1)[0]
        assert field_mask.any()
        desired_routes = self.routes[field_mask, :]
        return np.mean(desired_routes, axis=0)

    def historical_mean_field_route(self, field_entry, field_choice):
        # Select only that field
        field_location = self.reverse_keys[field_choice]
        field_index = self.field_indexes[field_location]
        field_mask = np.where(
            self.routes[:, field_index[0]+field_entry] == 1)[0]
        hist_field_mask = np.where(
            self.historical_routes[:, field_index[0]+field_entry] == 1)[0]
        # check if there are any such routes
        assert field_mask.any()
        assert hist_field_mask.any()
        active_routes = self.routes[field_mask, :]
        inactive_routes = self.historical_routes[hist_field_mask, :]
        desired_routes = np.concatenate(
            [active_routes, inactive_routes], axis=0)
        return np.mean(desired_routes, axis=0)

    def novel_probabilistic_route(self, field_choice):
        target = self.field_from_distance(field_choice)
        field_index = self.field_indexes[self.reverse_keys[field_choice]]
        target_mean = self.historical_mean_field_route(target, field_choice)

        route = np.zeros(self.total_fields)
        mask = np.arange(field_index[1], self.total_fields, dtype=np.int)
        inverse = 1 - target_mean
        # Rescale inverse
        inverse = self.norm_fields(inverse)
        route[mask] = inverse[mask]
        return route

    def novel_deterministic_route(self, field_choice):
        # Skip grade
        target = self.field_from_distance(field_choice)
        target_mean = self.historical_mean_field_route(target, field_choice)
        route = np.zeros(self.total_fields)
        for index in self.field_indexes:
            field = target_mean[index[0]:index[1]]
            loc = np.where(field == np.min(field))[0]
            selection = np.random.choice(loc)
            route[selection+index[0]] = 1
        return route

    def field_from_distance(self, field):
        # Deterministic field from distance
        index = self.field_indexes[self.reverse_keys[field]]
        field = self.distance[index[0]:index[1]]
        field_selection = np.where(field == np.max(field))[0]
        return field_selection

    def apply_weights(self, route, weights):
        for i, index in enumerate(self.field_indexes):
            route[index[0]:index[1]] *= weights[i]
        return route

    def deterministic_novel_goal_route(self, field):
        goal_route = self.deterministic_goal_route(self.distance)
        novel_route = self.novel_deterministic_route(field)
        print('goal_route', goal_route)
        print('novel_route', novel_route)
        goal_route = self.apply_weights(goal_route, self.goal_weights)
        novel_route = self.apply_weights(novel_route, self.novelty_weights)
        print('goal_route', goal_route)
        print('novel_route', novel_route)
        combined_route = goal_route + novel_route
        print('combined_route', combined_route)
        # Probabilistically select each field
        final_route = self.route_from_probabilities(combined_route)
        return final_route

    def probabilistic_novel_goal_route(self, field):
        """
        Creates a route probabilistically from the combination of novel probabilities and goal probabilities.
        """
        goal_route = self.probabilistic_goal_route(self.distance)
        novel_route = self.novel_probabilistic_route(field)
        goal_route = self.apply_weights(goal_route, self.goal_weights)
        novel_route = self.apply_weights(novel_route, self.novelty_weights)
        combined_route = goal_route + novel_route
        print('combined_route', combined_route)
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
        current_dist = np.sum(self.routes, axis=0)
        nd_distance = self.goals - current_dist
        return nd_distance

    @property
    def get_index(self):
        """
        Current resetting index, for stripping routes
        and set index for setting routes
        """
        return self.reset_index, self.set_index

    @property
    def loss(self):
        current_dist = np.sum(self.routes, axis=0)
        nd_distance = self.goals - current_dist
        return np.mean(np.abs(nd_distance))

    def __len__(self):
        # Return number of routes
        return self.routes.shape[0]
