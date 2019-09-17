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
    setters,locations = deconstruct_plan(plan)

    historical_routes,active_routes = separate_routes(routes)
    # Find all the routes we are about to strip
    stripping_routes = return_stripped(active_routes,locations)
    # update config based on settings and goals
    update_config(config,goals,settings,stripping_routes)
    # Convert locations into machine readable
    # convert_route_to_array(active_routes[0])
    # Convert goals into machine readable
    # print('len active',len(active_routes))
    # print('stripping_routes',len(stripping_routes))
    # print(goals)
    # print(settings)

    # Restrict historical routes to the previous 6 months. 
    now = date.today()
    six_months = now - relativedelta(months=+6)
    six_months_routes = restrict_history_by_date(historical_routes,six_months)
    # Restrict historical routes to the previous N routes (sorted according to date)
    N_historical_routes = historical_routes[-config.total_routes:]

    # Instantiate the utils class
    utils = ServerUtilities(active_routes,six_months_routes,fields,config)
    utils.bulk_strip_by_location(locations)
    utils.convert_goals(goals)
    routes = utils.return_suggestions()
    
    # locations = ['Titan North','Great Roof']
    # location_routes = get_routes_by_location(routes,locations)
    # Mask the fields in those locations
    # create N routes given the novelty and goals
    
    # return jsonify(json_obj)
    return 'Donezors!'


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
    # Find all the routes in the locations to strip
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

def deconstruct_plan(plan):
    plan_setters = plan['setters']
    plan_locations = plan['locations']
    print('plan_setters',plan_setters)
    print('plan_locations',plan_locations)
    return plan_setters,plan_locations

def distribute_routes(setters):
    pass
    # for setter in setters:
    #     setter['terrain_preference']
    #     setter['hold_preference']
    #     setter['max_b_setting_ability']
    #     setter['b_comfort_zone']
    #     setter['max_b_setting_ability']
    #     setter['total_routes']
    #     setter['assignments']

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

def convert_route_to_array(route):
    print(route)


def convert_routes_to_array(routes):
    pass

if __name__ == '__main__':
    app.run()