from __future__ import annotations
import joblib, json
from collections import defaultdict
from typing import Dict
from sklearn.metrics import classification_report
from ..datasets import load_dataset, select_features, split_sets
from ..common import save_artifacts

def build_lccde(cfg: Dict, out_root: str):
    out_dir = f"{out_root}/{cfg['dataset']}"
    xgb = joblib.load(f"{out_dir}/xgb.pkl")
    lgbm = joblib.load(f"{out_dir}/lgbm.pkl")
    cat  = joblib.load(f"{out_dir}/catboost.pkl")
    features = joblib.load(f"{out_dir}/features.pkl")

    df = load_dataset(cfg)
    X, y, _ = select_features(df, cfg["target_col"], cfg.get("feature_include", []), cfg.get("feature_exclude", []))
    _, _, Xva, yva, _, _ = split_sets(X, y, cfg["split"])

    def per_class_f1(model):
        rep = classification_report(yva, model.predict(Xva[features]), output_dict=True, zero_division=0)
        return {k: v["f1-score"] for k, v in rep.items() if k not in ("accuracy","macro avg","weighted avg")}

    fx, fl, fc = per_class_f1(xgb), per_class_f1(lgbm), per_class_f1(cat)
    classes = sorted(set(fx)|set(fl)|set(fc))
    leaders = {}
    for c in classes:
        winners = max([("xgb", fx.get(c,0.0)), ("lgbm", fl.get(c,0.0)), ("cat", fc.get(c,0.0))], key=lambda t: t[1])
        leaders[c] = winners[0]

    meta = {"leaders": leaders, "classes": classes}
    save_artifacts(out_dir, lccde_meta=meta)
