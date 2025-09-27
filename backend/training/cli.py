from __future__ import annotations
import argparse, yaml
from .trainers.xgb_trainer import train_xgb
from .trainers.lgbm_trainer import train_lgbm
from .trainers.cat_trainer import train_cat
from .ensemble.build_lccde import build_lccde

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["train-xgb","train-lgbm","train-cat","build-lccde","train-all"])
    p.add_argument("--config", required=True, help="YAML config path (e.g., training/config/cicids.yaml)")
    p.add_argument("--out", default="artifacts", help="artifacts root dir")
    args = p.parse_args()

    cfg = load_cfg(args.config)

    if args.task in ("train-xgb","train-all"):
        train_xgb(cfg, args.out)
    if args.task in ("train-lgbm","train-all"):
        train_lgbm(cfg, args.out)
    if args.task in ("train-cat","train-all"):
        train_cat(cfg, args.out)
    if args.task in ("build-lccde","train-all"):
        build_lccde(cfg, args.out)

if __name__ == "__main__":
    main()
