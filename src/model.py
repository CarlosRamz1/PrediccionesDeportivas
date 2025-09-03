from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.model_selection import train_test_split

from .features import split_X_y, INV_TARGET_MAP

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"


def build_pipeline() -> Pipeline:
    # Pipeline sencillo: escalado + regresión logística multinomial
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, multi_class="multinomial")),
    ])
    return pipe


def train_and_eval(df: pd.DataFrame) -> Dict[str, Any]:
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)
    y_pred = y_prob.argmax(axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, y_prob)),
        "classes": {str(k): v for k, v in INV_TARGET_MAP.items()},
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    joblib.dump(pipe, MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modelo no entrenado. Ejecuta train.py primero.")
    return joblib.load(MODEL_PATH)


def predict_proba_from_row(row: pd.Series) -> Dict[str, float]:
    """
    row: Serie con columnas de entrada (por equipo).
    Devuelve un dict con probabilidades por clase (home_win/draw/away_win).
    """
    model = load_model()
    import pandas as pd
    from .features import make_features, INV_TARGET_MAP

    df = pd.DataFrame([row])
    X = make_features(df)
    probs = model.predict_proba(X)[0]

    return {
        INV_TARGET_MAP[i]: float(probs[i])
        for i in range(len(probs))
    }
