from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from .base import BaseModelRunner
from .common import TabularPreprocessor

MODEL_NAME = "catboost"
FEATURES: List[str] = []


class _CatRunner(BaseModelRunner):
    def __init__(self):
        self.model = joblib.load("/app/artifacts/catboost.pkl")
        self.prep: TabularPreprocessor = joblib.load("/app/artifacts/catboost_prep.pkl")
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


RUNNER = _CatRunner()
