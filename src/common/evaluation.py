import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

model_results = []

def calculate_metrics(
    model_name: str,
    y_true,
    y_pred,
    training_time: float
) -> dict:
    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "Training Time": training_time
    }

    model_results.append(results)
    return results

def compare_models(
    results: list[dict]
) -> pd.DataFrame:

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    ).reset_index(drop=True)

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    results_df[metric_columns] = (
        results_df[metric_columns].round(4)
    )
    results_df["Training Time"] = (
        results_df["Training Time"].round(2)
    )

    return results_df

def display_best_model(
    results_df: pd.DataFrame
) -> None:
    if results_df.empty:
        print("No model results found.")
        return

    best_model = (results_df.iloc[0])

    print("Best Model")
    print("=" * 40)
    print(f"Model: {best_model['Model']}")
    print(f"Accuracy: {best_model['Accuracy']}")
    print(f"Precision: {best_model['Precision']}")
    print(f"Recall: {best_model['Recall']}")
    print(f"F1 Score: {best_model['F1 Score']}")
    print(
        f"Training Time: "
        f"{best_model['Training Time']} seconds"
    )