from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, mean_absolute_error
from FilterMethods import *
import pandas as pd
import numpy as np

METHODS = [
    (1, "Mean utility (user-based)"),
    (2, "Weighted sum (user-based)"),
    (3, "Adjusted sum (user-based)"),
    (4, "Adjusted NN weighted sum (user-based)"),
]

THRESHOLD = 5
def print_metrics(metrics):
    print(f"\nMAE: {metrics['mae']:.4f}\n")

    print("Confusion Matrix:")
    print(f"{metrics['confusion_matrix']}\n")

    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
def evaluate_predictions(actuals, preds):
    mae = mean_absolute_error(actuals, preds)

    actual_labels = [a >= THRESHOLD for a in actuals]
    pred_labels = [p >= THRESHOLD for p in preds]

    cm = confusion_matrix(actual_labels, pred_labels, labels=[False, True])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Rec", "Actual: Rec"],
        columns=["Predicted: No Rec", "Predicted: Rec"]
    )

    precision = precision_score(actual_labels, pred_labels, zero_division=0)
    recall = recall_score(actual_labels, pred_labels, zero_division=0)
    f1 = f1_score(actual_labels, pred_labels, zero_division=0)
    accuracy = accuracy_score(actual_labels, pred_labels)

    return {
        "mae": mae,
        "confusion_matrix": cm_df,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
def predict(method: int, user_id: int, item_id: int, ratings_df):
    if method == 1:
        return mean_utility(user_id, item_id, ratings_df)
    elif method == 2:
        return weighted_sum(user_id, item_id, ratings_df)
    elif method == 3:
        return adjusted_weighted_sum(user_id, item_id, ratings_df)
    elif method == 4:
        return k_nearest_adjusted_weighted_sum(user_id, item_id, ratings_df)
    else:
        raise ValueError(f"Unknown method: {method}")
