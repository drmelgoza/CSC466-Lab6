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

#weighted sum: based on multiplying ratings by the similarity between the target user and all other user vectors.
#This product for all other users is summed and multiplied by a normalization factor
def weighted_sum(user_id: int, item_id: int, ratings: pd.DataFrame):
    actual_rating = None
    if ratings.loc[user_id, item_id] != 99.00:
        actual_rating = ratings.loc[user_id, item_id]
        ratings.loc[user_id, item_id] = 99.00

    #get the targeted user's ratings row
    target_user = ratings[user_id]

    #get set of all user rating rows that aren't the target
    other_users = ratings.copy().drop(user_id, axis=0).to_numpy()

    # get the ratings for the target item for all other users
    other_ratings = other_users[:, item_id]

    # filter down other user ratings and users to those who have actually rated the target item.
    mask = (other_ratings != 99.00)
    valid_users = other_users[mask]
    valid_ratings = other_ratings[mask]

    #get similarities between target and all other users who've rated the target item
    sims = [cosine_similarity(target_user, other_user) for other_user in valid_users]

    # get normalization factor k
    k = 1 / np.sum(np.abs(sims))

    #sum up similarities multiplied by rankings.
    x = np.sum(sims * valid_ratings)

    if actual_rating:
        ratings.loc[user_id, item_id] = actual_rating

    return k * x

def adjusted_weighted_sum(user_id: int, item_id: int, ratings: pd.DataFrame):
    actual_rating = None
    if ratings.loc[user_id, item_id] != 99.00:
        actual_rating = ratings.loc[user_id, item_id]
        ratings.loc[user_id, item_id] = 99.00

    target_user = ratings[user_id]
    target_average_rating = np.mean(target_user[target_user != 99.00])

    other_users = ratings.copy().drop(user_id, axis=0).to_numpy()

    other_ratings = other_users[:, item_id]

    mask = (other_ratings != 99.00)
    valid_users = other_users[mask]
    valid_ratings = other_ratings[mask]

    sims = [cosine_similarity(target_user, other_user) for other_user in valid_users]

    k = 1 / np.sum(np.abs(sims))

    x = np.sum(sims * (valid_ratings - target_average_rating))

    if actual_rating:
        ratings.loc[user_id, item_id] = actual_rating

    return (target_average_rating + k) * x


def method_4():
    ...

