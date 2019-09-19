from flask import Flask
from flask import request
from flask import jsonify
import datetime
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
import pprint
import pymongo
import json
import itertools
from pymongo import MongoClient

from server_utils import ServerUtilities
from Utils.config import Config

app = Flask(__name__)

"""
App passes Plan ID to flask server
flask accesses the plan -> retrives the locations and setters
from DB
- fields
- routes (minus the routes in the locations)
- setters
- grades
- goals

convert routes from objs to arrays

passes the fields,grades, routes (historical and active, minus the active location routes) 
    to the suggestion engine

"""
@app.route('/plan')
def data():
    config = Config('math')

    # TODO
    # Get plan, get locations, get setters
    # Pull the goal data
    # Return routes according to loss and novelty.
    #request.args.get('data')
    
    db = connect()
    # Pull the route and plan data
    routes,fields,grades,plan,goals,settings = get_collections(db)
    historical_routes,active_routes = separate_routes(routes)
    # Restrict historical routes to the previous 6 months. 
    now = date.today()
    six_months = now - relativedelta(months=+6)
    six_months_routes = restrict_history_by_date(historical_routes,six_months)
    # Restrict historical routes to the previous N routes (sorted according to date)
    N_historical_routes = historical_routes[-config.total_routes:]
    # Max grade to be suggested by engine
    setters = plan['setters']
    # Instantiate the utils class - this converts routes into arrays and stores them locally in the utils
    utils = ServerUtilities(active_routes,six_months_routes,fields,config)
    utils.convert_goals(goals)
    max_setting,setting_time,setter_nicknames,relative_time,setting_mask,num_grades,max_grade,grade_index = get_setter_attributes(setters,utils)
    # Set max grade available to set
    utils.update_max_grade(max_grade)
    # Two setting styles -> By Location, By Route
    # Two climbing disciplines -> Bouldering and Roped Climbing
    # if plan['discipline'] == 'Bouldering':
    if plan['byLocation']:
        print('by location')
        locations = plan['locations']
        # Change this based on location flag. To take account of when they set by route.
        # Find all the routes we are about to strip
        stripping_routes = return_stripped(active_routes,locations)
        # update config based on settings and goals. Update tehcnique mask, grade mask, novelty weights, routes by location. terrain types.
        update_config(config,goals,settings,stripping_routes)
        utils.bulk_strip_by_location(locations)
        
        routes,readable_routes = utils.return_suggestions()
    else:
        print('by route')
        routes = plan['routes']
        update_config(config,goals,settings,routes)
        utils.bulk_strip_by_route(routes)
        routes,readable_routes = utils.return_suggestions()

    # Distribute the routes among the setters
    distributed_routes = distribute_routes(routes,readable_routes,max_setting,setting_time,setter_nicknames,relative_time,setting_mask,num_grades,grade_index)
    update_plan(db,distributed_routes)
    # else:
    #     raise ValueError("{} is not supported... die".format(plan['discipline']))
    # location_routes = get_routes_by_location(routes,locations)
    # return jsonify(json_obj)
    return 'Donezors!'

def get_setter_attributes(setters,utils):
    num_grades = utils.field_lengths[1]
    grade_index = utils.field_indexes[1]
    num_setters = len(setters)
    setter_nicknames = []

    max_setting = np.zeros((num_setters,num_grades))
    setting_time = np.zeros((num_setters,1))

    for i,setter in enumerate(setters):
        max_setting[i,int(setter['max_b_setting_ability'])] = 1
        setter_nicknames.append(setter['nickname'])
        # setter['terrain_preference']
        # setter['hold_preference']
        # setter['b_comfort_zone']
        # setter['max_rc_setting_ability']
        # setter['rc_comfort_zone']
        # setter['total_routes']
        # setter['assignments']
    setter_nicknames = np.array(setter_nicknames)    
    setting_mask = np.where(max_setting == 1)[1]
    relative_time = np.zeros((num_setters,num_grades))
    max_grade = 0
    for i in range(num_setters):
        max_setter_grade = np.where(max_setting[i] == 1)[0][0]
        # print('max_setter_grade',max_setter_grade,i)
        relative_time[i,:max_setter_grade+1] = np.linspace(0.2,1,max_setter_grade+1)
        max_grade = max_setter_grade if max_setter_grade > max_grade else max_grade

    # print('relative_time',relative_time)
    # print('max_setting',max_setting)
    # print('max_grade',max_grade)
    return max_setting,setting_time,setter_nicknames,relative_time,setting_mask,num_grades,max_grade,grade_index

def update_plan(db,routes):
    plan = db['plans'].find()[0]
    db['plans'].update_one({'_id':plan['_id']},{"$set":{"suggestions":routes}})

def update_config(config,goals,settings,stripping_routes):
    """
    Need to update config with the following
    total_routes : Total number of routes in the gym
    num_reset_routes : Number of routes to reset

    novelty_weights : How important novelty is in that field.
    terrain_technique_keys : techniques available based on the terrain type
    grade_technique_keys : techniques available based on grade
    set_novelty_weights(new Weights)
    """
    config.total_routes = goals['totalGymRoutes']
    config.num_reset_routes = len(stripping_routes)
    # Update Keys and Weights
    # config.set_novelty_weights(weights from settings)
    # config.terrain_technique_keys = terrain_technique_keys
    # config.grade_technique_keys = grade_technique_keys

def connect(): 
    client = MongoClient('localhost',27017)
    db = client.setting
    return db

def get_collections(db):
    plan = db['plans'].find()[0]
    goals = db['goals'].find()[0]
    settings = db['settings'].find()[0]
    routes = db['routes']
    # setters = db['setters']
    fields = db['fields']
    grades = db['grades']
    return routes,fields,grades,plan,goals,settings

def return_stripped(routes,locations):
    names = []
    routes_to_strip = []
    for location in locations:  
        names.append(location['name'])
        # location['terrain_type']
    for route in routes:
        if route['location'] in names:
            routes_to_strip.append(route)
    return routes_to_strip

def separate_routes(routes):
    historical_routes = []
    active_routes = []
    for route in routes.find():
        if route['stripped'] == False:
            active_routes.append(route)
        else:
            historical_routes.append(route)
    return historical_routes,active_routes

def restrict_history_by_date(historical_routes,cutoff):
    date_routes = []
    for route in historical_routes:
        # print('route',route)
        try:
            if route['date'].date() > cutoff:
                date_routes.append(route)
        except:
            # print('Route has no date')
            pass
    return date_routes

def get_routes_by_location(routes,locations):
    masked_routes = []
    for location in locations:
        for route in routes.find({'location':location['name']}):
            # pprint.pprint(route)
            masked_routes.append(route)
    return masked_routes

def distribute_routes(*args):
    routes,readable_routes,max_setting,setting_time,setter_nicknames,relative_time,setting_mask,num_grades,grade_index = args
    # Whenever we distribute a route. We must update the actual route setter. I need to take into account the fact that there could be multiple routes
    for difficulty in reversed(range(0,num_grades)):
        # We need to bump the location of grade difficulty by the index of grade start
        grade_mask = np.where(routes[:,grade_index[0]+difficulty] == 1)[0]
        print('grade_mask',grade_mask,'difficulty',difficulty)
        if grade_mask.size > 0:
            ability_mask = setting_mask >= difficulty
            possible_setters = setting_mask[ability_mask]
            print('possible_setters',possible_setters,type(possible_setters),possible_setters.shape)
            print('ability_mask',ability_mask)
            if possible_setters.size > 1:
                # Distribute the grades across setters. 
                for route in grade_mask:
                    lowest_time = np.min(setting_time[ability_mask])
                    time_mask = np.where(setting_time[ability_mask] == lowest_time)[0]
                    print('lowest_time',lowest_time,'all times',setting_time)
                    print('time_mask',time_mask)
                    if time_mask.size > 1:
                        # Distribute to best setter if available, else equally
                        print('time tied')
                        best_setter = np.max(possible_setters)
                        best_mask = np.where(possible_setters == best_setter)[0]
                        print('best_setter',best_setter)
                        print('best_mask',best_mask)
                        print('relative_time[best_mask,difficulty]',relative_time[best_mask,difficulty])
                        if best_mask.size > 1:
                            # Random across options
                            print('random')
                            choice = np.random.choice(best_mask)
                            setting_time[choice] += relative_time[choice,difficulty]
                            readable_routes[route]['setter'] = setter_nicknames[choice]
                            print('choice',choice)
                            print('setter_nicknames[choice]',setter_nicknames[choice])
                        else:
                            setting_time[best_mask] += relative_time[best_mask,difficulty]
                            readable_routes[route]['setter'] = setter_nicknames[best_mask[0]]
                            print('setter_nicknames[best_mask]',setter_nicknames[best_mask[0]])
                    else:
                        # Give to lowest time
                        setting_time[time_mask[0]] += relative_time[time_mask[0],difficulty]
                        readable_routes[route]['setter'] = setter_nicknames[time_mask[0]]
                        print('setter_nicknames[time_mask[0]]',setter_nicknames[time_mask[0]])
            elif possible_setters.size == 1:
                # Distribute all routes to this setter
                setting_time[ability_mask] += relative_time[ability_mask,difficulty]
                for route in grade_mask:
                    readable_routes[route]['setter'] = setter_nicknames[ability_mask]
                    print('setter_nicknames[ability_mask]',setter_nicknames[ability_mask])
    print(setting_time)
    return readable_routes

# def return_terrain_types():
#     terrains = ['slab','verticle','overhung','roof']
#     terrain_categories = {}
#     category = 0
#     for L in range(1,5):
#         for subset in itertools.combinations(terrains,L):
#             terrain_categories[subset] = category
#             category += 1
#     print(terrain_categories)
#     return terrain_categories

if __name__ == '__main__':
    app.run()