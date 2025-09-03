from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict

import pandas as pd

from .model import predict_proba_from_row, load_model

app = FastAPI(title="Sports Predictor API", version="0.1.0")


class MatchInput(BaseModel):
    # Promedios recientes (o ratings), por equipo. Adapta estos campos a tu fuente real de datos.
    goals_scored_avg_home: float = Field(..., ge=0)
    goals_scored_avg_away: float = Field(..., ge=0)
    goals_conceded_avg_home: float = Field(..., ge=0)
    goals_conceded_avg_away: float = Field(..., ge=0)
    shots_avg_home: float = Field(..., ge=0)
    shots_avg_away: float = Field(..., ge=0)
    shots_on_target_avg_home: float = Field(..., ge=0)
    shots_on_target_avg_away: float = Field(..., ge=0)
    possession_avg_home: float = Field(..., ge=0, le=100)
    possession_avg_away: float = Field(..., ge=0, le=100)
    rating_home: float
    rating_away: float


@app.get("/health")
def health():
    # Verifica que el modelo esté cargable
    try:
        load_model()
        return {"status": "ok", "model": "loaded"}
    except Exception as e:
        return {"status": "warning", "detail": str(e)}


@app.post("/predict")
def predict(payload: MatchInput) -> Dict[str, float]:
    # Convertimos el payload a una serie equivalente a una fila del CSV de entrenamiento
    row = pd.Series(payload.dict())
    probs = predict_proba_from_row(row)
    return probs
