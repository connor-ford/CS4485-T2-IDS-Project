from __future__ import annotations
from typing import Tuple, List, Dict
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_dataset(cfg):
    root = Path(cfg["data_root"])
    # example: single CSV; if you have multiple, concat them
    df = pd.read_csv(root / "CICIDS2017_sample.csv")
    # ensure numeric dtypes where applicable
    for c in df.columns:
        if c != cfg["target_col"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0)
    return df


def select_features(df, target_col):
    features = [c for c in df.columns if c != target_col]
    return df[features], df[target_col], features


def split_sets(X, y, split_cfg):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        stratify=y,
        random_state=split_cfg["random_state"],
    )
    rel_val = split_cfg["val_size"] / (1.0 - split_cfg["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=rel_val,
        stratify=y_temp,
        random_state=split_cfg["random_state"],
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
