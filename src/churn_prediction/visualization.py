import pandas as pd
import matplotlib.pyplot as plt

def visualize_target_distribution(data: pd.DataFrame, target_column: str) -> None:
    target_counts = (data[target_column].value_counts())
    plt.figure(figsize=(7, 5))
    plt.bar(target_counts.index, target_counts.values)
    plt.title("Target Distribution")
    plt.xlabel(target_column)
    plt.ylabel("Number of Customers")
    plt.show()

def visualize_numerical_features(data: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        plt.figure(figsize=(7, 4))
        plt.hist(data[column].dropna(), bins=30)
        plt.title(f"{column} Distribution")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

def visualize_categorical_feature(data: pd.DataFrame, column: str) -> None:
    counts = (data[column].value_counts())
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.title(f"{column} Distribution")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_feature_vs_target(data: pd.DataFrame, feature: str, target: str) -> None:
    comparison = pd.crosstab(data[feature], data[target], normalize="index")
    comparison.plot(kind="bar", figsize=(9, 5))
    plt.title(f"{target} by {feature}")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()