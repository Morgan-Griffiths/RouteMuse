# RouteMuse

Suggestion engine for rock climbing routes

## Overview

---

**The Goal**

---

### Training Data

Create the ability to artificially generate routes for training a network and to demonstrate the capability and performance of the various methods of suggesting routes.
**Two**

### Algorithms

Create the Network and algorithms that perform well on the training data.

## Description of the problem

Each Gym contains a number of walls. Each wall has its own unique characteristics. In RouteMuse each wall is labeled with a category - Roof,Overhung,Verticle,Slab. Some walls are combinations of these categories. The wall type can dictate what techiniques and styles are possible on the wall. For example its impossible to have a balance problem on a roof. Likewise its impossible to have a compus problem on a slab.

A routes success is based on the enjoyment of those who climb it. One aspect of well recieved routes is novelty. Therefore taking into account the historical setting preferences of the gym and making sure to reward novelty is important. In Reenforcement Learning terms, the environment is none-stationary, because the user tastes will change over time, by getting familiar with the type of routes that are set. However, long term route styles can be recycled with no detriment to user happiness, because its more about the recency bias than climbing a similar route 10 years ago.

Our task is to use historical and present route data, taken in consideration with the gym's setting goals, to output the types of routes that will close the gap between the reality of the gym and the goal of the gym. Supplementary goals are creating inspiration and challenge for the setters, and guiding the setters to understand the wishes of the people who climb their problems. By Suggesting routes that are likely to be highly rated.

## The Algorithm

#### **There are two main approaches to solving this:**

1. Math
2. Neural Nets with RL

We will seek to compare the two and select whichever one performs the best, or is best suited to our applications.

#### _Loss functions_

- L1 distance = Goals - Current Routes.
- Loss = Huber(distance)
  _Huber loss is MSE when the loss > 1, absolute difference otherwise_

## Math

1. Take the max, deterministically populate the desired routes
2. Probabilistically output routes in a distribution

## RL

_Proximal Policy Optimization_

#### **Input**

- distance

#### **Possible Inputs**

- All historical routes - for calculation of distance between suggested routes and historical routes
- Current routes - for calculation of distance between suggested routes and current routes

#### **Outputs**

Route suggestions (Array of routes with 1s in the appropriate categories)

This will then be converted into JSON format to be fed back into the database as a planning document.

#### _Additional terms_

- Novelty term: Increase the value of novel routes. This could be a distance from other routes. When distance is high, the factor should be possitive. When distance is low, the factor should be subtractive.

- Entropy penalty: Penalize stationary route generation (similarity between previous and current routes). More for keeping the weights moving. High when the distance between weights is small.

#### **Restrictions**

- Wall type is a hard restriction on styles.
- Number of styles can vary by route.
- Number of techniques can vary by route.

## Setting the suggested routes

After the routes are created then we move on to the distribution of those routes across the setting team. The goal is to inspire and motivate the setters to close the gap to the goals, while improving their own setting skill.

#### **Setter Attributes**

Historical route distribution.
Time taken to create routes.
Bias in route creation.

We can use the bias in two ways:

1. To get routes quickly up on the wall if the route is aligned with their personal preferences.
2. To challenge the setter to work outside their comfort zone and grow their skills. When the route does not conform to their bias.

This will be up to the head setter, and will vary from gym to gym and potentially setting session to setting session. Perhaps a slider that goes from challenge to comfort.
