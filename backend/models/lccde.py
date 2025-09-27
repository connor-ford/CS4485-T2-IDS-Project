# backend/models/lccde.py
from typing import Any, Dict, List
import os, joblib, numpy as np, pandas as pd
from .base import BaseModelRunner

MODEL_NAME = "lccde"


def _art_dir() -> str:
    default = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
    return os.getenv("IDSML_ARTIFACTS_DIR", default)


class _LCCDERunner(BaseModelRunner):
    def __init__(self):
        art = _art_dir()
        self.xgb = joblib.load(os.path.join(art, "xgb.pkl"))
        self.lgbm = joblib.load(os.path.join(art, "lgbm.pkl"))
        self.cat = joblib.load(os.path.join(art, "catboost.pkl"))
        self.features: List[str] = joblib.load(os.path.join(art, "features.pkl"))
        self.classes: List[str] = joblib.load(os.path.join(art, "classes.pkl"))
        self.meta: Dict[str, Any] = joblib.load(os.path.join(art, "lccde_meta.pkl"))

    def validate(self, inputs: Dict[str, Any]) -> None:
        missing = [f for f in self.features if f not in inputs]
        if missing:
            raise ValueError(f"missing features: {missing}")

    def _predict_impl(self, inputs: Dict[str, Any]) -> Any:
        x = pd.DataFrame([{f: inputs[f] for f in self.features}])
        px = self.xgb.predict_proba(x)[0]
        pl = self.lgbm.predict_proba(x)[0]
        pc = self.cat.predict_proba(x)[0]

        # choose per-class leader from meta
        leaders = self.meta["leaders"]  # {label: "xgb"|"lgbm"|"cat"}
        probs = np.zeros_like(px, dtype=float)
        for idx, label in enumerate(self.classes):
            m = leaders.get(label, "xgb")
            probs[idx] = {"xgb": px, "lgbm": pl, "cat": pc}[m][idx]

        pred_idx = int(np.argmax(probs))
        return {"label": self.classes[pred_idx], "proba": probs.tolist()}


RUNNER = _LCCDERunner()
