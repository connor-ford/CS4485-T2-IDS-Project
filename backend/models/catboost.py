from typing import Any, Dict, List
import joblib, numpy as np, pandas as pd
from .base import BaseModelRunner

MODEL_NAME = "catboost"


class _CatRunner(BaseModelRunner):
    def __init__(self):
        self.model = joblib.load("/app/artifacts/catboost.pkl")
        self.features: List[str] = joblib.load("/app/artifacts/features.pkl")
        self.classes: List[str] = joblib.load("/app/artifacts/classes.pkl")

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
