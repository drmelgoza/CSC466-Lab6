# CSC466-Lab6

## Team

* Diego Melgoza (drmelgoz@calpoly.edu)
* Summer Ortega (sorteg16@calpoly.edu)

## Running the Programs
The two executable programs for this assignment are `EvaluateCFRandom.py` and `EvaluateCFList.py`

The two programs utilize methods defined in `Parser.py`, `FilterMethods.py`, and `evalutils.py` to run the calculations needed to predict ratings and create recommendations.

To run `EvaluateCFRandom.py` use the following terminal command: `python3 EvaluateCFRandom.py <method> <size> [<repeats>]`

To run `EvaluateCFList.py` use the following terminal command: `python3 EvaluateCFList.py <method> <filename>`

The program parameters are defined as so:
1. `method` is an integer from 1 to 4 defining which collaborative filtering method will be used by the program. The numbering for this parameter is the same as the listing below in `List of Methods Used`.
2. `size` is used by `EvaluateCFRandom.py` to determine the # of random user-item pairs to assess for each individual repetition.
3. `repeats` is used by `EvaluateCFRandom.py` to determine the # of repetitions to execute during the program.
4. `filename` is used by `EvaluateCFList.py` to identify the .csv file used to hold the user-item pairs to be assessed by the program.

# List of Methods Used
All methods used are located within the `FilterMethods.py` file.

1. Mean Utility; `mean_utility()`
2. Weighted Sum by User; `weighted_sum()`
3. Adjusted Weighted Sum by User; `adjusted_weighted_sum()`
4. K-Nearest Adjusted Weighted Sum by User; `k_nearest_adjusted_weighted_sum()`