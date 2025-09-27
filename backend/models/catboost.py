from typing import Any, Dict, List
import os, joblib, numpy as np, pandas as pd
from .base import BaseModelRunner

MODEL_NAME = "catboost"


def _art_dir() -> str:
    # default to repo-local artifacts/; override via env
    default = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
    return os.getenv("IDSML_ARTIFACTS_DIR", default)


class _CatRunner(BaseModelRunner):
    def __init__(self):
        art = _art_dir()
        self.model = joblib.load(os.path.join(art, "catboost.pkl"))
        self.features: List[str] = joblib.load(os.path.join(art, "features.pkl"))
        self.classes: List[str] = joblib.load(os.path.join(art, "classes.pkl"))

    def validate(self, inputs: Dict[str, Any]) -> None:
        missing = [f for f in self.features if f not in inputs]
        if missing:
            raise ValueError(f"missing features: {missing}")

    def _predict_impl(self, inputs: Dict[str, Any]) -> Any:
        x = pd.DataFrame([{f: inputs[f] for f in self.features}])
        proba = self.model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        return {"label": self.classes[idx], "proba": proba.tolist()}


RUNNER = _CatRunner()
