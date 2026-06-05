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
    a_prime = np.copy(a)
    b_prime = np.copy(b)
    a_mask = a_prime != 99.00
    b_mask = b_prime != 99.00
    mask = a_mask & b_mask
    a_prime = a_prime[mask]
    b_prime = b_prime[mask]

    numerator = np.dot(a_prime, b_prime)
    denominator = np.linalg.norm(a_prime) * np.linalg.norm(b_prime)
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
#This product is summed and multiplied by a normalization factor to get the final result.
def weighted_sum(user_id: int, item_id: int, ratings: pd.DataFrame):
    actual_rating = None
    if ratings.loc[user_id, item_id] != 99.00:
        actual_rating = ratings.loc[user_id, item_id]
        ratings.loc[user_id, item_id] = 99.00

    #get the targeted user's ratings row
    target_user = ratings.loc[user_id]

    #get set of all user rating rows that aren't the target
    other_users = ratings.drop(user_id, axis=0).to_numpy()

    # get the ratings for the target item for all other users
    other_ratings = other_users[:, item_id]

    # filter down other user ratings and users to those who have actually rated the target item.
    mask = (other_ratings != 99.00)
    valid_users = other_users[mask]
    valid_ratings = other_ratings[mask]

    #get similarities between target and all other users who've rated the target item
    sims = np.array([cosine_similarity(target_user, other_user) for other_user in valid_users])

    # get normalization factor k
    k = 1 / np.sum(np.abs(sims))

    #sum up similarities multiplied by rankings.
    x = np.sum(sims * valid_ratings)

    if actual_rating:
        ratings.loc[user_id, item_id] = actual_rating

    return k * x


#similar to weighted sum, but the calculation now keeps in mind that the target user has their own unique range of review values.
#Use the average in the target user's range to "adjust" the weighted sum to be accurate to the target user's ratings.
def adjusted_weighted_sum(user_id: int, item_id: int, ratings: pd.DataFrame):
    #replace already existing rating for user-item pair if it already exists.
    actual_rating = None
    if ratings.loc[user_id, item_id] != 99.00:
        actual_rating = ratings.loc[user_id, item_id]
        ratings.loc[user_id, item_id] = 99.00

    #get target user row and their average rating for all valid ratings (no 99's)
    target_user = ratings.loc[user_id]
    target_average_rating = np.mean(target_user[target_user != 99.00])

    #get the set of all other users minus the target user, and retrieve their ratings for the target item
    other_users = ratings.drop(user_id, axis=0).to_numpy()
    other_ratings = other_users[:, item_id]

    #filter the users to those who have actually reviewed the target item and filter the ratings accordingly
    mask = (other_ratings != 99.00)
    valid_users = other_users[mask]
    valid_ratings = other_ratings[mask]
    average_valid_ratings = [np.mean(user[user != 99.00]) for user in valid_users]

    # get the similarity value between the target user and all other users.
    sims = np.array([cosine_similarity(target_user, other_user) for other_user in valid_users])

    # set the normalization factor using the similarity values
    k = 1 / np.sum(np.abs(sims))

    # calculate the summation
    x = np.sum(sims * (valid_ratings - average_valid_ratings))

    # return the actual rating to the matrix for later use
    if actual_rating:
        ratings.loc[user_id, item_id] = actual_rating

    #return the summation multiplied by the normalization factor, adding the average to "adjust" the predicted rating.
    return target_average_rating + (k * x)


def adjusted_weighted_nearest_neighbors_sum(user_id: int, item_id: int, ratings: pd.DataFrame, k=5):
    # replace already existing rating for user-item pair if it already exists.
    actual_rating = None
    if ratings.loc[user_id, item_id] != 99.00:
        actual_rating = ratings.loc[user_id, item_id]
        ratings.loc[user_id, item_id] = 99.00

    # get target user row and their average rating for all valid ratings (no 99's)
    target_user = ratings.loc[user_id]
    target_average_rating = np.mean(target_user[target_user != 99.00])

    # get the set of all other users minus the target user, and retrieve their ratings for the target item
    other_users = ratings.drop(user_id, axis=0).to_numpy()
    other_ratings = other_users[:, item_id]

    #filter the users to those who have actually reviewed the target item and filter the ratings accordingly
    mask = (other_ratings != 99.00)
    valid_users = other_users[mask]
    valid_ratings = other_ratings[mask]

    # get the similarity value between the target user and all other users
    sims = np.array([cosine_similarity(target_user, other_user) for other_user in valid_users])

    #get the k-nearest users who are most similar to the target user.
    #np.argsort() organizes in ascending order, so this array need to be flipped using np.flip().
    sims_for_k = np.abs(np.copy(sims))
    k_nearest = np.flip(np.argsort(sims_for_k))[0:k]

    #get the k nearest similarity values and k nearest ratings
    k_nearest_users = valid_users[k_nearest]
    k_nearest_sims = sims[k_nearest]
    k_nearest_ratings = valid_ratings[k_nearest]
    k_average_ratings = [np.mean(user[user != 99.00]) for user in k_nearest_users]

    # set the normalization factor using the k-nearest similarity values
    k = 1 / np.sum(np.abs(k_nearest_sims))

    # calculate the summation using the k-nearest similarities and ratings
    x = np.sum(k_nearest_sims * (k_nearest_ratings - k_average_ratings))

    # return the actual rating to the matrix for later use
    if actual_rating:
        ratings.loc[user_id, item_id] = actual_rating

    # return the summation multiplied by the normalization factor, adding the average to "adjust" the predicted rating.
    return target_average_rating + (k * x)

