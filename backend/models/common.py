from typing import Tuple, Dict, Any, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "Label" # dependent variable in CICIDS datasets

def load_cicids_sample(root: str) -> pd.DataFrame:
    # mirror the notebook’s data loading; point to repo’s /data subset
    # e.g., read CSVs and concat
    df = pd.read_csv(f"{root}/data/CICIDS2017_sample.csv")  # example path/name
    return df

def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y

def train_val_test(X, y, test_size=0.2, val_size=0.2, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size,
                                                      stratify=y, random_state=random_state)
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=rel_val,
                                                      stratify=y_temp, random_state=random_state)
    return X_train, y_train, X_val, y_val, X_test, y_test

class TabularPreprocessor:
    """reproduce notebook feature handling (scaling/encodings)."""
    def __init__(self, numeric_cols: List[str]):
        self.numeric_cols = numeric_cols
        self.scaler = StandardScaler()  # only if notebook scales numeric features for some models

    def fit(self, X: pd.DataFrame):
        if self.numeric_cols:
            self.scaler.fit(X[self.numeric_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.numeric_cols:
            X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        return X
