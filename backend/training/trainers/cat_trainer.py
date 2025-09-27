from __future__ import annotations
from typing import Dict
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

from ..datasets import load_dataset, select_features, split_sets
from ..common import save_artifacts, dump_report, ensure_dir


def train_cat(cfg: Dict, out_root: str) -> None:
    df = load_dataset(cfg)
    X, y, features = select_features(df, cfg["target_col"])
    Xtr, ytr, Xva, yva, Xte, yte = split_sets(X, y, cfg["split"])

    # encode string labels -> ints
    le = LabelEncoder().fit(y)
    ytr_enc = le.transform(ytr)
    yva_enc = le.transform(yva)
    yte_enc = le.transform(yte)

    mcfg = cfg.get("models", {}).get("cat", {})
    params = {
        "iterations": mcfg.get("iterations", 800),
        "depth": mcfg.get("depth", 8),
        "learning_rate": mcfg.get("learning_rate", 0.1),
        "loss_function": mcfg.get("loss_function", "MultiClass"),
        "random_state": cfg["split"]["random_state"],
        "verbose": False,
        "allow_writing_files": False,
    }

    clf = CatBoostClassifier(**params)
    # if you have categorical feature indices, add cat_features=[...]
    clf.fit(Xtr, ytr_enc, eval_set=(Xva, yva_enc), use_best_model=False)

    # reports (inverse-transform)
    import numpy as np

    val_pred = le.inverse_transform(clf.predict(Xva).astype(int).ravel())
    test_pred = le.inverse_transform(clf.predict(Xte).astype(int).ravel())

    out_dir = f"{out_root}/{cfg['dataset']}"
    ensure_dir(out_dir)
    dump_report(yva, val_pred, f"{out_dir}/catboost_val_report.json")
    dump_report(yte, test_pred, f"{out_dir}/catboost_test_report.json")

    save_artifacts(out_dir, catboost=clf, features=features, classes=list(le.classes_))
