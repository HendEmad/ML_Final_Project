import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    print("Dataset loaded successfully.")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")
    return data

def check_missing_values(data: pd.DataFrame) -> pd.Series:
    missing_values = (data.isnull().sum())
    return missing_values[missing_values > 0]

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    data["TotalCharges"] = (pd.to_numeric(data["TotalCharges"], errors="coerce"))
    data["TotalCharges"] = (data["TotalCharges"].fillna(data["TotalCharges"].median()))
    data["Churn"] = (data["Churn"].map({"No": 0, "Yes": 1}))
    return data

def separate_features_target(data: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    x = data.drop(columns=[target_column])
    y = data[target_column]
    return x, y

def split_data(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y)

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train, y_train, test_size=validation_size, random_state=random_state, stratify=y_train)

    return (x_train, x_validation, x_test, y_train, y_validation, y_test)