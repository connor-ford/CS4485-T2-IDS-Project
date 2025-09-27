from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from .base import BaseModelRunner

MODEL_NAME = "lccde"

class _LCCDERunner(BaseModelRunner):
    """
    LCCDE:
      - leader per class is chosen from {xgb, lgbm, catboost} using validation metrics.
      - at inference, use model probabilities, respecting leader assignment per class.
    """
    def __init__(self):
        # load base models + preprocessors
        self.xgb = joblib.load("/app/artifacts/xgb.pkl")
        self.lgbm = joblib.load("/app/artifacts/lgbm.pkl")
        self.cat = joblib.load("/app/artifacts/catboost.pkl")
        self.prep = joblib.load("/app/artifacts/lccde_prep.pkl")   # share a single prep if they match
        self.features: List[str] = joblib.load("/app/artifacts/features.pkl")

        # mapping: class_label -> {"leader": "xgb"|"lgbm"|"cat", "classes": [ordered labels]}
        self.meta: Dict[str, Any] = joblib.load("/app/artifacts/lccde_meta.pkl")

        # ensure aligned class order for each model
        self.xgb_classes = list(self.xgb.classes_)
        self.lgbm_classes = list(self.lgbm.classes_)
        self.cat_classes = list(self.cat.classes_)

    def validate(self, inputs: Dict[str, Any]) -> None:
        missing = [f for f in self.features if f not in inputs]
        if missing:
            raise ValueError(f"missing features: {missing}")

    def _proba(self, model, classes, x: pd.DataFrame) -> Dict[str, float]:
        p = model.predict_proba(x)[0]
        return {str(classes[i]): float(p[i]) for i in range(len(classes))}

    def _predict_impl(self, inputs: Dict[str, Any]) -> Any:
        x = pd.DataFrame([{f: inputs[f] for f in self.features}])
        x = self.prep.transform(x)

        px = self._proba(self.xgb, self.xgb_classes, x)
        pl = self._proba(self.lgbm, self.lgbm_classes, x)
        pc = self._proba(self.cat, self.cat_classes, x)

        # raw best across all (model,class)
        all_scores = []
        for c, s in px.items(): all_scores.append(("xgb", c, s))
        for c, s in pl.items(): all_scores.append(("lgbm", c, s))
        for c, s in pc.items(): all_scores.append(("cat", c, s))
        model_w, class_w, score_w = max(all_scores, key=lambda t: t[2])

        # enforce leader-by-class
        leader = self.meta["leaders"][class_w]  # e.g., "lgbm"
        if leader == "xgb":
            final_score = px[class_w]
        elif leader == "lgbm":
            final_score = pl[class_w]
        else:
            final_score = pc[class_w]

        # optional thresholding / tie-breaks could go here
        return {"label": class_w, "leader": leader, "score": final_score,
                "proba": {"xgb": px.get(class_w, 0.0),
                          "lgbm": pl.get(class_w, 0.0),
                          "cat": pc.get(class_w, 0.0)}}

RUNNER = _LCCDERunner()
