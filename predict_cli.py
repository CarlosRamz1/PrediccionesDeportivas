from __future__ import annotations
import json
import argparse
import pandas as pd

from src.model import predict_proba_from_row

def main():
    parser = argparse.ArgumentParser(description="CLI para predecir (home/draw/away).")
    parser.add_argument("--payload", type=str, required=True, help="Ruta a un JSON con las entradas.")
    args = parser.parse_args()

    with open(args.payload, "r", encoding="utf-8") as f:
        payload = json.load(f)

    row = pd.Series(payload)
    probs = predict_proba_from_row(row)
    print(json.dumps(probs, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
