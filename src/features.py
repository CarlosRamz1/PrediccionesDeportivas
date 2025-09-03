from __future__ import annotations
import pandas as pd

# Columnas esperadas en data/matches.csv (sintético)
EXPECTED_COLUMNS = [
    # inputs por equipo (promedios recientes o rating)
    "goals_scored_avg_home", "goals_scored_avg_away",
    "goals_conceded_avg_home", "goals_conceded_avg_away",
    "shots_avg_home", "shots_avg_away",
    "shots_on_target_avg_home", "shots_on_target_avg_away",
    "possession_avg_home", "possession_avg_away",
    "rating_home", "rating_away",
    # etiqueta
    "result",  # valores: "home_win", "draw", "away_win"
]


TARGET_MAP = {"home_win": 0, "draw": 1, "away_win": 2}
INV_TARGET_MAP = {v: k for k, v in TARGET_MAP.items()}


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de columnas por equipo, genera variables de diferencia (home - away).
    Esto ayuda a que el modelo sea invariante al nombre de los equipos.
    """
    f = pd.DataFrame(index=df.index)
    f["goals_scored_diff"] = df["goals_scored_avg_home"] - df["goals_scored_avg_away"]
    f["goals_conceded_diff"] = df["goals_conceded_avg_home"] - df["goals_conceded_avg_away"]
    f["shots_diff"] = df["shots_avg_home"] - df["shots_avg_away"]
    f["shots_on_target_diff"] = df["shots_on_target_avg_home"] - df["shots_on_target_avg_away"]
    f["possession_diff"] = df["possession_avg_home"] - df["possession_avg_away"]
    f["rating_diff"] = df["rating_home"] - df["rating_away"]
    # Podrías añadir interacciones, no-linealidades, etc.
    return f


def split_X_y(df: pd.DataFrame):
    X = make_features(df)
    y = df["result"].map(TARGET_MAP).astype(int)
    return X, y
