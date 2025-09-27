from __future__ import annotations
from typing import Tuple, List, Dict
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(cfg: Dict) -> pd.DataFrame:
    # implement exactly how the notebook loads data
    # example: one CSV
    path = f"{cfg['data_root']}/CICIDS2017_sample.csv"
    return pd.read_csv(path)

def select_features(df, target_col):
    features = [c for c in df.columns if c != target_col]
    return df[features], df[target_col], features


def split_sets(X, y, split_cfg: Dict):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=split_cfg["test_size"], stratify=y, random_state=split_cfg["random_state"])
    rel_val = split_cfg["val_size"] / (1.0 - split_cfg["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=rel_val, stratify=y_temp, random_state=split_cfg["random_state"])
    return X_train, y_train, X_val, y_val, X_test, y_test
