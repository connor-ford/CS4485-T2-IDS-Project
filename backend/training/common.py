from __future__ import annotations
from pathlib import Path
from typing import Dict, Union
import json, joblib
from sklearn.metrics import classification_report

PathLike = Union[str, Path]


def ensure_dir(p: PathLike) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_artifacts(out_dir: PathLike, **artifacts) -> None:
    out = ensure_dir(out_dir)
    for name, obj in artifacts.items():
        joblib.dump(obj, out / f"{name}.pkl")


def dump_report(y_true, y_pred, out_path: PathLike) -> None:
    out_path = Path(out_path)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    out_path.write_text(json.dumps(rep, indent=2))
