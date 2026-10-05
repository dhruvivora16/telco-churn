import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (CATEGORICAL, NUMERIC, PROCESSED_CSV, SEED,
                        SERVICE_COLS, TARGET)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shared by training and the Streamlit app so they never drift apart."""
    df = df.copy()
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    return df


def load_xy():
    df = pd.read_csv(PROCESSED_CSV)
    return df[CATEGORICAL + NUMERIC], df[TARGET]


def split_data():
    X, y = load_xy()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    num = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [("num", num, NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        sparse_threshold=0,
    )


def clean_name(name: str) -> str:
    """'cat__Contract_Month-to-month' -> 'Contract = Month-to-month'."""
    name = name.split("__", 1)[-1]
    for col in CATEGORICAL:
        if name.startswith(col + "_"):
            return f"{col} = {name[len(col) + 1:]}"
    return name
