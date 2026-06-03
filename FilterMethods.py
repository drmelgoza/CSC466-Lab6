import pandas as pd
import numpy as np

"""
methods to use (replace filler names when finalized):
1. Mean utility
2. weighted sum
3. adjusted sum 
4. adjusted NN weighted sum
"""

def mean_utility(user_id: int, item_id: int, ratings: pd.DataFrame):
    actual_rating = None
    if ratings.loc[user_id, item_id] != 99.00:
        actual_rating = ratings.loc[user_id, item_id]
        ratings.loc[user_id, item_id] = 99.00

    reviewers = ratings[ratings.loc[:item_id] != 99.00].loc[:, item_id]

    prediction = reviewers.mean()

    if actual_rating:
        ratings.loc[user_id, item_id] = actual_rating

    return prediction

def method_2():
    ...

def method_3():
    ...

def method_4():
    ...

