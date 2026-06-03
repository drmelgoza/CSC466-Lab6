import pandas as pd
import numpy as np

"""
methods to use (replace filler names when finalized):
1. Mean utility
2. weighted sum
3. adjusted sum 
4. adjusted NN weighted sum
"""

def method_1(user_id: int, item_id: int, ratings: pd.DataFrame):
    reviewers = ratings[ratings.loc[:item_id] != 99.00].loc[:, item_id]

    prediction = reviewers.mean()
    if ratings[iloc(user_id)]

    return

def method_2():
    ...

def method_3():
    ...

def method_4():
    ...

