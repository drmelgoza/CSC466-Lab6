import sys
import pandas as pd
import numpy as np

from Parser import parse_data
from evalutils import *
def main(method:int, size: int, repeats: int=1):
    jokes_rated, ratings_df = parse_data()

    valid_pairs = ratings_df.stack().reset_index()
    valid_pairs.columns = ["user_id", "item_id", "actual_rating"]
    valid_pairs = valid_pairs[valid_pairs.iloc[:,2] != 99]

    maes = []

    for _ in range(repeats):
        pairs = valid_pairs.sample(n=size)

        actuals = []
        predicted = []

        results = []

        for _, row in pairs.iterrows():
            user_id = int(row["user_id"])
            item_id = int(row["item_id"])
            actual_rating = float(row["actual_rating"])

            pred_rating = predict(method, user_id, item_id, ratings_df)

            delta = actual_rating - pred_rating

            actuals.append(actual_rating)
            predicted.append(pred_rating)

            results.append([user_id, item_id, actual_rating, pred_rating, delta])

        results_df = pd.DataFrame(results, columns=["userID", "itemID", "Actual_Rating",
                                                    "Predicted_Rating", "Delta_Rating"])
        metrics = evaluate_predictions(actuals, predicted)
        maes.append(metrics["mae"])

        print(results_df.to_string(index=False))
        print_metrics(metrics)

    if repeats > 1:
        print(f"\nMean MAE: {np.mean(maes):.4f}")
        print(f"Std MAE: {np.std(maes):.4f}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Collaborative filtering methods (pick a number):")
        for method_id, description in METHODS:
            print(f"{method_id}. {description}")
    elif len(sys.argv) == 3:
        main(int(sys.argv[1]), int(sys.argv[2]))
    elif len(sys.argv) == 4:
        main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    else:
        print("Usage: python3 EvaluateCFRandom.py <method> <size> [<repeats>]")
