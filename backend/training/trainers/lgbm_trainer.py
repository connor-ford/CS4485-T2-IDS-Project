from __future__ import annotations
from typing import Dict
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

from ..datasets import load_dataset, select_features, split_sets
from ..common import save_artifacts, dump_report, ensure_dir


def train_lgbm(cfg: Dict, out_root: str) -> None:
    df = load_dataset(cfg)
    X, y, features = select_features(df, cfg["target_col"])
    Xtr, ytr, Xva, yva, Xte, yte = split_sets(X, y, cfg["split"])

    # encode string labels -> ints
    le = LabelEncoder().fit(y)
    ytr_enc = le.transform(ytr)
    yva_enc = le.transform(yva)
    yte_enc = le.transform(yte)

    mcfg = cfg.get("models", {}).get("lgbm", {})
    params = {
        "n_estimators": mcfg.get("n_estimators", 800),
        "learning_rate": mcfg.get("learning_rate", 0.05),
        "subsample": mcfg.get("subsample", 0.8),
        "colsample_bytree": mcfg.get("colsample_bytree", 0.8),
        "reg_lambda": mcfg.get("reg_lambda", 1.0),
        "max_depth": mcfg.get("max_depth", -1),
        "objective": "multiclass",
        "random_state": cfg["split"]["random_state"],
        "verbosity": -1,
        "num_class": len(le.classes_),
    }

    clf = LGBMClassifier(**params)
    clf.fit(Xtr, ytr_enc)

    # reports (inverse-transform to original labels)
    import numpy as np

    val_pred = le.inverse_transform(clf.predict(Xva).astype(int))
    test_pred = le.inverse_transform(clf.predict(Xte).astype(int))

    out_dir = f"{out_root}/{cfg['dataset']}"
    ensure_dir(out_dir)
    dump_report(yva, val_pred, f"{out_dir}/lgbm_val_report.json")
    dump_report(yte, test_pred, f"{out_dir}/lgbm_test_report.json")

    save_artifacts(out_dir, lgbm=clf, features=features, classes=list(le.classes_))
