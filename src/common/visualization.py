import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

def plot_training_history(
    history: tf.keras.callbacks.History,
    model_name: str
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")

    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    display.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.show()

def visualize_model_comparison(
    results_df: pd.DataFrame
) -> None:
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
    comparison_plot = results_df.set_index("Model")[metrics]
    comparison_plot.plot(kind="bar", figsize=(14, 6))

    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.show()

# compare training time
def visualize_training_time(
    results_df: pd.DataFrame
) -> None:
    plt.figure(figsize=(12, 6))
    plt.bar(results_df["Model"], results_df["Training Time"])
    plt.title("Model Training Time Comparison")
    plt.xlabel("Model")
    plt.ylabel("Training Time (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()