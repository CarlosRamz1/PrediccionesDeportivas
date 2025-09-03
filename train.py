from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.model import train_and_eval
from src.features import EXPECTED_COLUMNS

DATA_PATH = Path("data/matches.csv")


def maybe_generate_synthetic_data(path: Path, n: int = 600, seed: int = 7):
    if path.exists():
        return

    rng = np.random.default_rng(seed)
    # Genera datos plausibles (no reales)
    def clip(a, lo, hi): return np.minimum(np.maximum(a, lo), hi)

    # Home stronger on average to embed a meaningful pattern
    goals_scored_avg_home = clip(rng.normal(1.6, 0.6, n), 0.2, 4.5)
    goals_scored_avg_away = clip(rng.normal(1.3, 0.6, n), 0.1, 4.0)

    goals_conceded_avg_home = clip(rng.normal(1.2, 0.5, n), 0.1, 3.5)
    goals_conceded_avg_away = clip(rng.normal(1.4, 0.6, n), 0.1, 3.8)

    shots_avg_home = clip(rng.normal(12, 4, n), 2, 25)
    shots_avg_away = clip(rng.normal(10, 4, n), 2, 25)

    shots_on_target_avg_home = clip(shots_avg_home * clip(rng.normal(0.35, 0.08, n), 0.15, 0.6), 1, 12)
    shots_on_target_avg_away = clip(shots_avg_away * clip(rng.normal(0.32, 0.08, n), 0.10, 0.55), 1, 10)

    possession_avg_home = clip(rng.normal(52, 8, n), 30, 70)
    possession_avg_away = 100 - possession_avg_home + rng.normal(0, 2, n)

    rating_home = clip(rng.normal(72, 8, n), 50, 95)   # rating ficticio tipo Elo
    rating_away = clip(rng.normal(70, 8, n), 50, 95)

    # Regla latente para resultado con ruido
    # score = w1*diff_goals_scored - w2*diff_goals_conceded + w3*rating_diff + w4*shots_diff + w5*possession_diff + ruido
    score = (
        0.9 * (goals_scored_avg_home - goals_scored_avg_away)
        - 0.8 * (goals_conceded_avg_home - goals_conceded_avg_away)
        + 0.04 * (rating_home - rating_away)
        + 0.05 * (shots_avg_home - shots_avg_away)
        + 0.02 * (possession_avg_home - possession_avg_away)
        + rng.normal(0, 0.8, n)
    )

    # Convertimos score continuo a clases 0/1/2 con umbrales
    # > 0.5 -> home_win ; < -0.5 -> away_win ; en medio -> draw
    result = np.where(score > 0.5, "home_win", np.where(score < -0.5, "away_win", "draw"))

    df = pd.DataFrame({
        "goals_scored_avg_home": goals_scored_avg_home,
        "goals_scored_avg_away": goals_scored_avg_away,
        "goals_conceded_avg_home": goals_conceded_avg_home,
        "goals_conceded_avg_away": goals_conceded_avg_away,
        "shots_avg_home": shots_avg_home,
        "shots_avg_away": shots_avg_away,
        "shots_on_target_avg_home": shots_on_target_avg_home,
        "shots_on_target_avg_away": shots_on_target_avg_away,
        "possession_avg_home": possession_avg_home,
        "possession_avg_away": possession_avg_away,
        "rating_home": rating_home,
        "rating_away": rating_away,
        "result": result,
    })
    df.to_csv(path, index=False)
    print(f"[OK] Datos sintéticos generados en {path} ({len(df)} filas).")


def main():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    maybe_generate_synthetic_data(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    # Validar columnas
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {DATA_PATH}: {missing}")

    metrics = train_and_eval(df)
    print("[MÉTRICAS]")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
