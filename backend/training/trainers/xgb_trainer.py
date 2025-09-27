from __future__ import annotations
from typing import Dict
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from ..datasets import load_dataset, select_features, split_sets
from ..common import save_artifacts, dump_report, ensure_dir


def train_xgb(cfg: Dict, out_root: str) -> None:
    df = load_dataset(cfg)
    X, y, features = select_features(df, cfg["target_col"])
    Xtr, ytr, Xva, yva, Xte, yte = split_sets(X, y, cfg["split"])

    # encode string labels -> ints
    le = LabelEncoder().fit(y)
    ytr_enc = le.transform(ytr)
    yva_enc = le.transform(yva)
    yte_enc = le.transform(yte)

    mcfg = cfg.get("models", {}).get("xgb", {})
    params = {
        "n_estimators": mcfg.get("n_estimators", 500),
        "max_depth": mcfg.get("max_depth", 8),
        "learning_rate": mcfg.get("learning_rate", 0.1),
        "subsample": mcfg.get("subsample", 0.8),
        "colsample_bytree": mcfg.get("colsample_bytree", 0.8),
        "tree_method": mcfg.get("tree_method", "hist"),
        "eval_metric": mcfg.get("eval_metric", "mlogloss"),
        "objective": "multi:softprob",
        "random_state": cfg["split"]["random_state"],
        "n_jobs": mcfg.get("n_jobs", -1),
        # optional but explicit:
        "num_class": len(le.classes_),
    }

    clf = XGBClassifier(**params)
    clf.fit(Xtr, ytr_enc)

    # human-readable reports: inverse-transform predictions to original string labels
    import numpy as np

    val_pred_enc = clf.predict(Xva)
    test_pred_enc = clf.predict(Xte)
    val_pred = le.inverse_transform(val_pred_enc.astype(int))
    test_pred = le.inverse_transform(test_pred_enc.astype(int))

    out_dir = ensure_dir(out_root)
    dump_report(yva, val_pred, f"{out_dir}/xgb_val_report.json")
    dump_report(yte, test_pred, f"{out_dir}/xgb_test_report.json")

    # save estimator, features, and classes (for inference mapping)
    save_artifacts(out_dir, xgb=clf, features=features, classes=list(le.classes_))
