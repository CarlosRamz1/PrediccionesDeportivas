import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

TOKEN = "8292197038:AAEtibgkf7an5Skmvtwx7fnb9WjDFz"
API_URL = "http://127.0.0.1:8000/predict"

# Comando /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("¡Hola! Soy tu bot predictor deportivo 🤖⚽")

# Comando /predict
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    payload = {
        "goals_scored_avg_home": 1.8,
        "goals_scored_avg_away": 1.1,
        "goals_conceded_avg_home": 1.0,
        "goals_conceded_avg_away": 1.4,
        "shots_avg_home": 13.5,
        "shots_avg_away": 10.2,
        "shots_on_target_avg_home": 4.8,
        "shots_on_target_avg_away": 3.3,
        "possession_avg_home": 55.0,
        "possession_avg_away": 45.0,
        "rating_home": 75.0,
        "rating_away": 70.0
    }
    response = requests.post(API_URL, json=payload)
    data = response.json()
    msg = f"Predicciones:\nHome Win: {data['home_win']*100:.1f}%\nDraw: {data['draw']*100:.1f}%\nAway Win: {data['away_win']*100:.1f}%"
    await update.message.reply_text(msg)

# Crear aplicación
app = ApplicationBuilder().token(TOKEN).build()

# Registrar comandos
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("predict", predict))

# Correr bot
app.run_polling()

