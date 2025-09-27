#!/usr/bin/env python3
"""
Demo client for the IDS-ML IDE backend.

- Fetches /schema to get feature names and class labels.
- Builds a valid payload either with zeros or from a row in a CSV.
- Sends POST /predict and prints the response.

Usage examples:
  python demo.py --model xgb --mode zero
  python demo.py --model lccde --mode row --csv backend/data/CICIDS2017_sample.csv --row 0
  python demo.py --model catboost --mode row --csv path/to/your.csv --row 123 --pretty
"""

import argparse
import json
import math
import sys
from typing import Dict, List, Any

import requests

try:
    import pandas as pd  # only needed for --mode row
except Exception:
    pd = None


def fetch_schema(base_url: str, timeout: float) -> Dict[str, Any]:
    r = requests.get(f"{base_url}/schema", timeout=timeout)
    r.raise_for_status()
    return r.json()


def zero_payload(features: List[str]) -> Dict[str, float]:
    return {f: 0.0 for f in features}


def row_payload(features: List[str], csv_path: str, row_idx: int) -> Dict[str, float]:
    if pd is None:
        raise RuntimeError("pandas is required for --mode row. pip install pandas")
    df = pd.read_csv(csv_path)
    if row_idx < 0 or row_idx >= len(df):
        raise IndexError(f"row index {row_idx} out of range (0..{len(df)-1})")
    row = df.iloc[row_idx].to_dict()

    # ensure all required features exist
    missing = [f for f in features if f not in row]
    if missing:
        raise KeyError(
            f"CSV is missing required columns: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    # build inputs in schema order, coerce to float, sanitize
    inputs: Dict[str, float] = {}
    for f in features:
        v = row[f]
        try:
            fv = float(v)
        except Exception:
            fv = 0.0
        if math.isinf(fv) or math.isnan(fv):
            fv = 0.0
        inputs[f] = fv
    return inputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo client for IDS-ML IDE backend")
    ap.add_argument("--host", default="http://127.0.0.1:8000", help="Server base URL")
    ap.add_argument(
        "--model",
        default="xgb",
        choices=["xgb", "lgbm", "catboost", "lccde"],
        help="Model to use",
    )
    ap.add_argument(
        "--mode",
        default="zero",
        choices=["zero", "row"],
        help="zero = all zeros; row = take from CSV",
    )
    ap.add_argument("--csv", help="CSV path (required for --mode row)")
    ap.add_argument("--row", type=int, default=0, help="Row index for --mode row")
    ap.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout seconds")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON response")
    ap.add_argument(
        "--print-payload",
        action="store_true",
        help="Print the JSON payload before sending",
    )
    ap.add_argument("--save-payload", help="Path to write payload JSON (optional)")
    args = ap.parse_args()

    try:
        schema = fetch_schema(args.host, args.timeout)
    except requests.RequestException as e:
        print(
            f"[error] failed to fetch schema from {args.host}/schema: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    features = schema.get("features") or []
    classes = schema.get("classes") or []
    if not features:
        print(
            "[error] schema.features is empty — training artifacts not found/loaded.",
            file=sys.stderr,
        )
        sys.exit(1)

    # build inputs
    try:
        if args.mode == "zero":
            inputs = zero_payload(features)
        else:
            if not args.csv:
                print("[error] --csv is required for --mode row", file=sys.stderr)
                sys.exit(2)
            inputs = row_payload(features, args.csv, args.row)
    except Exception as e:
        print(f"[error] failed to build inputs: {e}", file=sys.stderr)
        sys.exit(2)

    payload = {"model": args.model, "inputs": inputs}

    if args.print_payload or args.save_payload:
        payload_str = json.dumps(payload, indent=2)
        if args.print_payload:
            print(payload_str)
        if args.save_payload:
            with open(args.save_payload, "w") as f:
                f.write(payload_str)

    # POST /predict
    try:
        r = requests.post(f"{args.host}/predict", json=payload, timeout=args.timeout)
        if args.pretty:
            try:
                print(json.dumps(r.json(), indent=2))
            except Exception:
                print(r.text)
        else:
            print(r.text)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"[error] request failed: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
