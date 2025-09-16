Sports Predictor (Starter)

Modelo inicial de **clasificación multiclase** (local/empate/visitante) con capacidad de **re-entrenamiento**. 
Incluye:
- `FastAPI` para servir el modelo por HTTP.
- `scikit-learn` con `LogisticRegression` (multinomial).
- Datos sintéticos para que puedas entrenar enseguida (reemplázalos por datos reales después).
- Scripts: entrenar, predecir, re-entrenar.

> ⚠️ Este starter es educativo. Conectaremos una **API deportiva** para reemplazar los datos sintéticos por datos reales y programa re-entrenamientos periódicos.

```

---

## Estructura del proyecto
```
sports-predictor-starter/
├── data/
│   └── matches.csv                 # datos sintéticos (reemplaza por datos reales)
├── examples/
│   └── payload.json                # ejemplo de datos para /predict
├── models/
│   ├── model.joblib                # modelo entrenado
│   └── metrics.json                # métricas del último entrenamiento
├── src/
│   ├── __init__.py
│   ├── features.py                 # ingeniería de características
│   ├── model.py                    # cargar/guardar/entrenar/predecir
│   └── service.py                  # API FastAPI
├── predict_cli.py
├── retrain.py
├── train.py
├── requirements.txt
└── README.md
```

---

## Siguientes pasos recomendados
- Sustituir `data/matches.csv` por datos reales (desde una API).
- Agregar más variables: lesiones, localía, clima, rachas, Elo, etc.
- Probar modelos: `RandomForest`, `GradientBoosting`, `XGBoost`.
- Validación por **temporalidad** (train en pasado, test en futuro).
- Programar re-entrenamiento semanal (cron) y guardar todas las predicciones con su resultado real para medir **calibración** y **log-loss**.
