# Sports Predictor (Starter)

Modelo inicial de **clasificación multiclase** (local/empate/visitante) con capacidad de **re-entrenamiento**. 
Incluye:
- `FastAPI` para servir el modelo por HTTP.
- `scikit-learn` con `LogisticRegression` (multinomial).
- Datos sintéticos para que puedas entrenar enseguida (reemplázalos por datos reales después).
- Scripts: entrenar, predecir, re-entrenar.

> ⚠️ Este starter es educativo. Conecta una **API deportiva** para reemplazar los datos sintéticos por datos reales y programa re-entrenamientos periódicos.

---

## 1) Requisitos previos
- **Python 3.10+** instalado.
- (Opcional) **Git** instalado.

## 2) Crear entorno virtual
### Windows (PowerShell)
```powershell
cd sports-predictor-starter
py -3.10 -m venv .venv
. .venv\Scripts\Activate.ps1
```
> Si te bloquea la ejecución de scripts: abre PowerShell como Administrador y ejecuta: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### macOS / Linux (bash/zsh)
```bash
cd sports-predictor-starter
python3 -m venv .venv
source .venv/bin/activate
```

## 3) Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Entrenar el modelo
```bash
python train.py
```
Salva el modelo en `models/model.joblib` y muestra métricas.

## 5) Levantar la API
```bash
uvicorn src.service:app --reload --port 8000
```
Prueba en `http://127.0.0.1:8000/docs` o con:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @examples/payload.json
```

## 6) Hacer una predicción por CLI
```bash
python predict_cli.py --payload examples/payload.json
```

## 7) Re-entrenar con un nuevo partido
1. Agrega una nueva fila a `data/matches.csv` (ver columnas en esa misma tabla).
2. Ejecuta:
```bash
python retrain.py
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
