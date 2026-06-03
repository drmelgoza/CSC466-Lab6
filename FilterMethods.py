import pandas as pd
import numpy as np

"""
methods to use (replace filler names when finalized):
1. Mean utility
2. weighted sum
3. adjusted sum 
4. adjusted NN weighted sum
"""

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    numerator = np.dot(a, b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return numerator / denominator


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

def weighted_sum(user_id: int, item_id: int, ratings: pd.DataFrame):
    actual_rating = None
    if ratings.loc[user_id, item_id] != 99.00:
        actual_rating = ratings.loc[user_id, item_id]
        ratings.loc[user_id, item_id] = 99.00

    #get the targeted user
    user = ratings.loc[user_id]

    #get set of all users that aren't the target
    not_user = ratings.copy().drop(user_id, axis=0).to_numpy()

    #get similarities between target and all other users
    sims = [cosine_similarity(user, other_user) for other_user in not_user]

    # get normalization factor k
    k = 1 / np.sum(np.abs(sims))

    #get the ratings for the target item for all other users
    not_user_ratings = not_user[:, item_id]

    #sum up similarities multiplied by rankings.
    x = np.sum(sims * not_user_ratings)

    if actual_rating:
        ratings.loc[user_id, item_id] = actual_rating

    return k * x

def method_3():
    ...

def method_4():
    ...

