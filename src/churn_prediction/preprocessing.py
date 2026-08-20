import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder)

def get_feature_types(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    numerical_columns = (data.select_dtypes(include=np.number).columns.tolist())
    categorical_columns = (data.select_dtypes(exclude=np.number).columns.tolist())
    return (numerical_columns,categorical_columns)

def create_preprocessor(numerical_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[("numerical", StandardScaler(), numerical_columns),
                      ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns)
        ]
    )
    return preprocessor