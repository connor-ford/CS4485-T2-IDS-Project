from __future__ import annotations
from typing import Dict
from lightgbm import LGBMClassifier

from ..datasets import load_dataset, select_features, split_sets
from ..common import save_artifacts, dump_report, ensure_dir


def train_lgbm(cfg: Dict, out_root: str) -> None:
    """
    Train a LightGBM multi-class classifier using the dataset/config provided.
    Artifacts saved to: {out_root}/{cfg['dataset']}/
      - lgbm.pkl
      - features.pkl
      - lgbm_val_report.json
      - lgbm_test_report.json
    """
    # load & prepare data
    df = load_dataset(cfg)
    X, y, features = select_features(df, cfg["target_col"])
    Xtr, ytr, Xva, yva, Xte, yte = split_sets(X, y, cfg["split"])

    # model params from YAML (with defaults if missing)
    mcfg = cfg.get("models", {}).get("lgbm", {})
    params = {
        "n_estimators":     mcfg.get("n_estimators", 800),
        "learning_rate":    mcfg.get("learning_rate", 0.05),
        "subsample":        mcfg.get("subsample", 0.8),
        "colsample_bytree": mcfg.get("colsample_bytree", 0.8),
        "reg_lambda":       mcfg.get("reg_lambda", 1.0),
        "max_depth":        mcfg.get("max_depth", -1),
        "objective":        "multiclass",
        "random_state":     cfg["split"]["random_state"],
        "verbosity":        -1,
    }

    clf = LGBMClassifier(**params)
    clf.fit(Xtr, ytr)

    # reports + artifacts
    out_dir = f"{out_root}/{cfg['dataset']}"
    ensure_dir(out_dir)
    dump_report(yva, clf.predict(Xva), f"{out_dir}/lgbm_val_report.json")
    dump_report(yte, clf.predict(Xte), f"{out_dir}/lgbm_test_report.json")

    save_artifacts(out_dir, lgbm=clf, features=features)
