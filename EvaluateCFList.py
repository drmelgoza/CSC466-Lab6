import sys
import pandas as pd

from Parser import parse_data
from evalutils import *


def main(method: int, filename: str):
    jokes_rated, ratings_df = parse_data()

    test_cases = pd.read_csv(filename, header=None, names=["user_id", "item_id"])

    actuals = []
    predicted = []
    results = []

    for _, row in test_cases.iterrows():
        user_id = int(row["user_id"])
        item_id = int(row["item_id"])

        actual_rating = ratings_df.loc[user_id, item_id]

        if actual_rating == 99:
            continue

        actual_rating = float(actual_rating)
        pred_rating = predict(method, user_id, item_id, ratings_df)

        delta = actual_rating - pred_rating

        actuals.append(actual_rating)
        predicted.append(pred_rating)

        results.append([user_id, item_id, actual_rating, pred_rating, delta])

    results_df = pd.DataFrame(
        results,
        columns=["userID", "itemID", "Actual_Rating", "Predicted_Rating", "Delta_Rating"]
    )

    print(results_df.to_string(index=False))

    metrics = evaluate_predictions(actuals, predicted)
    print_metrics(metrics)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Collaborative filtering methods (pick a number):")
        for method_id, description in METHODS:
            print(f"{method_id}. {description}")

    elif len(sys.argv) == 3:
        main(int(sys.argv[1]), sys.argv[2])

    else:
        print("Usage: python3 EvaluateCFList.py <method> <filename>")