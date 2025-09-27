from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
from .base import BaseModelRunner

MODEL_NAME = "xgb"


class _XGBRunner(BaseModelRunner):
    def __init__(self):
        self.model = joblib.load("/app/artifacts/xgb.pkl")
        self.features: List[str] = joblib.load("/app/artifacts/features.pkl")
        self.classes: List[str] = joblib.load("/app/artifacts/classes.pkl")  # <-- add

    def validate(self, inputs: Dict[str, Any]) -> None:
        missing = [f for f in self.features if f not in inputs]
        if missing:
            raise ValueError(f"missing features: {missing}")

    def _predict_impl(self, inputs: Dict[str, Any]) -> Any:
        x = pd.DataFrame([{f: inputs[f] for f in self.features}])
        proba = self.model.predict_proba(x)[0]  # shape (num_classes,)
        idx = int(np.argmax(proba))
        label = self.classes[idx]  # map ID -> original string
        return {"label": str(label), "proba": proba.tolist()}


RUNNER = _XGBRunner()
