from __future__ import annotations
import joblib
import numpy as np
from typing import Dict
from sklearn.metrics import classification_report

from ..datasets import load_dataset, select_features, split_sets
from ..common import save_artifacts


def build_lccde(cfg: Dict, out_root: str) -> None:
    """
    Build LCCDE meta by picking a leader model per class using validation F1.
    Assumes all base models were trained with the same label encoder and
    that features.pkl/classes.pkl exist in artifacts/<dataset>/.
    """
    out_dir = f"{out_root}/{cfg['dataset']}"

    # load base models + schema
    xgb = joblib.load(f"{out_dir}/xgb.pkl")
    lgbm = joblib.load(f"{out_dir}/lgbm.pkl")
    cat = joblib.load(f"{out_dir}/catboost.pkl")
    features = joblib.load(f"{out_dir}/features.pkl")
    classes = joblib.load(f"{out_dir}/classes.pkl")  # string labels in encoded order

    # load data & make validation split once
    df = load_dataset(cfg)
    X, y, _ = select_features(df, cfg["target_col"])
    Xtr, ytr, Xva, yva, Xte, yte = split_sets(X, y, cfg["split"])

    # encode yva using the same class order saved during training
    label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
    yva_enc = np.array([label_to_idx[str(lbl)] for lbl in yva], dtype=int)

    # restrict to training feature schema
    Xva = Xva[features]

    # predictions from each model (encoded int classes)
    px = xgb.predict(Xva).astype(int)
    pl = lgbm.predict(Xva).astype(int)
    pc = cat.predict(Xva).astype(int).ravel()

    # build per-class F1 with human-readable class names
    def per_class_f1(y_true_enc, y_pred_enc):
        rep = classification_report(
            y_true_enc,
            y_pred_enc,
            labels=list(range(len(classes))),
            target_names=list(classes),
            output_dict=True,
            zero_division=0,
        )
        return {
            k: v["f1-score"]
            for k, v in rep.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        }

    fx = per_class_f1(yva_enc, px)
    fl = per_class_f1(yva_enc, pl)
    fc = per_class_f1(yva_enc, pc)

    # pick leader per class
    leaders: Dict[str, str] = {}
    for cls in classes:
        winners = max(
            [
                ("xgb", fx.get(cls, 0.0)),
                ("lgbm", fl.get(cls, 0.0)),
                ("cat", fc.get(cls, 0.0)),
            ],
            key=lambda t: t[1],
        )
        leaders[cls] = winners[0]

    meta = {"leaders": leaders, "classes": list(classes)}
    save_artifacts(out_dir, lccde_meta=meta)
