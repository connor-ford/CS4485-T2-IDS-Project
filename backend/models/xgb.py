from typing import Any, Dict, Tuple, List
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from .base import BaseModelRunner
from .common import TabularPreprocessor

MODEL_NAME = "xgb"

FEATURES: List[str] = []  # set exactly to the notebook’s feature columns

class _XGBRunner(BaseModelRunner):
    def __init__(self):
        self.model = joblib.load("/app/artifacts/xgb.pkl")
        self.prep: TabularPreprocessor = joblib.load("/app/artifacts/xgb_prep.pkl")
        self.features = FEATURES

    def validate(self, inputs: Dict[str, Any]) -> None:
        missing = [f for f in self.features if f not in inputs]
        if missing:
            raise ValueError(f"missing features: {missing}")

    def _predict_impl(self, inputs: Dict[str, Any]) -> Any:
        x = pd.DataFrame([{f: inputs[f] for f in self.features}])
        x = self.prep.transform(x)
        proba = self.model.predict_proba(x)[0]
        pred = self.model.classes_[int(np.argmax(proba))]
        return {"label": str(pred), "proba": proba.tolist()}

RUNNER = _XGBRunner()
