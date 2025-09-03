from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from src.model import train_and_eval
from src.features import EXPECTED_COLUMNS

DATA_PATH = Path("data/matches.csv")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("No existe data/matches.csv. Ejecuta primero train.py.")

    df = pd.read_csv(DATA_PATH)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {DATA_PATH}: {missing}")

    metrics = train_and_eval(df)
    print("[RE-ENTRENAMIENTO COMPLETO]")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
