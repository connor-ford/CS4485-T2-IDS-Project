from __future__ import annotations
from typing import Dict
from xgboost import XGBClassifier

from ..datasets import load_dataset, select_features, split_sets
from ..common import save_artifacts, dump_report, ensure_dir


def train_xgb(cfg: Dict, out_root: str) -> None:
    """
    Train an XGBoost multi-class classifier using the dataset/config provided.
    Artifacts saved to: {out_root}/{cfg['dataset']}/
      - xgb.pkl
      - features.pkl
      - xgb_val_report.json
      - xgb_test_report.json
    """
    # load & prepare data
    df = load_dataset(cfg)
    X, y, features = select_features(df, cfg["target_col"])
    Xtr, ytr, Xva, yva, Xte, yte = split_sets(X, y, cfg["split"])

    # model params from YAML (with defaults if missing)
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
    }

    clf = XGBClassifier(**params)
    clf.fit(Xtr, ytr)

    # reports + artifacts
    out_dir = f"{out_root}/{cfg['dataset']}"
    ensure_dir(out_dir)
    dump_report(yva, clf.predict(Xva), f"{out_dir}/xgb_val_report.json")
    dump_report(yte, clf.predict(Xte), f"{out_dir}/xgb_test_report.json")

    save_artifacts(out_dir, xgb=clf, features=features)
