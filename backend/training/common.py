from __future__ import annotations
import os, json, joblib
from pathlib import Path
from typing import Dict
from sklearn.metrics import classification_report

def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_artifacts(out_dir: str, **artifacts):
    out = ensure_dir(out_dir)
    for name, obj in artifacts.items():
        joblib.dump(obj, out / f"{name}.pkl")

def dump_report(y_true, y_pred, out_path: str):
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=2)
