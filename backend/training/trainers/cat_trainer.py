from __future__ import annotations
from typing import Dict
from catboost import CatBoostClassifier

from ..datasets import load_dataset, select_features, split_sets
from ..common import save_artifacts, dump_report, ensure_dir


def train_cat(cfg: Dict, out_root: str) -> None:
    """
    Train a CatBoost multi-class classifier using the dataset/config provided.
    Artifacts saved to: {out_root}/{cfg['dataset']}/
      - catboost.pkl
      - features.pkl
      - catboost_val_report.json
      - catboost_test_report.json
    """
    # load & prepare data
    df = load_dataset(cfg)
    X, y, features = select_features(df, cfg["target_col"])
    Xtr, ytr, Xva, yva, Xte, yte = split_sets(X, y, cfg["split"])

    # model params from YAML (with defaults if missing)
    mcfg = cfg.get("models", {}).get("cat", {})
    params = {
        "iterations":     mcfg.get("iterations", 800),
        "depth":          mcfg.get("depth", 8),
        "learning_rate":  mcfg.get("learning_rate", 0.1),
        "loss_function":  mcfg.get("loss_function", "MultiClass"),
        "random_state":   cfg["split"]["random_state"],
        "verbose":        False,
    }

    # if you have categorical columns, pass their indices via cat_features=...
    clf = CatBoostClassifier(**params)
    clf.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=False)

    # reports + artifacts
    out_dir = f"{out_root}/{cfg['dataset']}"
    ensure_dir(out_dir)
    dump_report(yva, clf.predict(Xva), f"{out_dir}/catboost_val_report.json")
    dump_report(yte, clf.predict(Xte), f"{out_dir}/catboost_test_report.json")

    save_artifacts(out_dir, catboost=clf, features=features)
