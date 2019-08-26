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
    N = 189
    config = Config('math')

    plan = request.args.get('data')
    # TODO
    # Get plan, get locations, get setters
    # Pull the goal data
    # Return routes according to loss and novelty.
    
    db = connect()
    # Pull the route data
    routes,setters,fields,grades = get_collections(db)
    historical_routes,active_routes = separate_routes(routes)

    # Instantiate the utils class
    utils = ServerUtilities(routes,historical_routes,fields,config)

    # Restrict historical routes to the previous 6 months. 
    now = date.today()
    six_months = now - relativedelta(months=+6)
    six_months_routes = restrict_history_by_date(historical_routes,six_months)
    # Restrict historical routes to the previous N routes (sorted according to date)
    N_historical_routes = historical_routes[-N:]
    
    locations = ['Titan North','Great Roof']
    location_routes = get_routes_by_location(routes,locations)
    # Mask the fields in those locations
    # create N routes given the novelty and goals
    
    # return jsonify(json_obj)
    return 'Donezors!'



def connect(): 
    client = MongoClient('localhost',27017)
    db = client.setting
    return db

def get_collections(db):
    routes = db['routes']
    setters = db['setters']
    fields = db['fields']
    grades = db['grades']
    return routes,setters,fields,grades

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
        for route in routes.find({'location':location}):
            pprint.pprint(route)
            masked_routes.append(route)
    return masked_routes

def convert_routes_to_array(routes):
    pass

if __name__ == '__main__':
    app.run()