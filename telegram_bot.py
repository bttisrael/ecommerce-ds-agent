import os
import logging
import pickle
import numpy as np
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import anthropic

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

predictions_df = None
final_model = None

ACCURACY = 0.9609
ROWS = 500_000
TARGET = "event_type"
PROBLEM = "classification"
MODEL_NAME = "XGBoost"
PREDICTION_CLASSES = ["view"]
DISTRIBUTION = {"view": 500000}
TOP_FEATURES = [
    ("feat_interact", 0.3487),
    ("price_per_product", 0.2725),
    ("price", 0.1910),
    ("feat_product", 0.1878),
    ("log_price", 0.0000),
    ("sq_price", 0.0000),
]
HYPOTHESES = [
    "Specific brands are associated with higher purchase conversion rates, suggesting brand loyalty effects are significant.",
    "The log-transformed price (log_price) shows a stronger linear relationship with event type than raw price.",
]


def load_artifacts():
    global predictions_df, final_model
    logger.info("Loading df4_predictions.parquet ...")
    predictions_df = pd.read_parquet("df4_predictions.parquet")
    logger.info("Loaded predictions dataframe with shape: %s", predictions_df.shape)
    logger.info("Loading final_model.pkl ...")
    with open("final_model.pkl", "rb") as f:
        final_model = pickle.load(f)
    logger.info("Model loaded successfully.")


def ask_claude(prompt: str) -> str:
    message = anthropic_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user.first_name or "there"
    text = (
        f"Hello, {user}! Welcome to the Event Type Classification Bot.\n\n"
        f"This bot gives you insights about an XGBoost model trained to predict "
        f"'{TARGET}' with {ACCURACY*100:.2f}% accuracy on {ROWS:,} rows.\n\n"
        "Available commands:\n"
        "/start - Welcome message\n"
        "/stats - Dataset and model statistics\n"
        "/top_features - Top predictive features\n"
        "/hypotheses - Verified hypotheses about the data\n"
        "/predict - Make a sample prediction\n"
        "/insights - AI-generated insights via Claude\n"
        "/help - Show this help message"
    )
    await update.message.reply_text(text)


async def stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global predictions_df
    if predictions_df is None:
        await update.message.reply_text("Artifacts not loaded yet. Please try again shortly.")
        return

    pred_col = None
    for col in ["predicted_event_type", "prediction", "predicted", "y_pred"]:
        if col in predictions_df.columns:
            pred_col = col
            break

    dist_text = "\n".join(
        f"  - {cls}: {cnt:,}" for cls, cnt in DISTRIBUTION.items()
    )

    if pred_col:
        pred_counts = predictions_df[pred_col].value_counts()
        pred_text = "\n".join(
            f"  - {cls}: {cnt:,}" for cls, cnt in pred_counts.items()
        )
    else:
        pred_text = "  Prediction column not identified in dataframe."

    cols_preview = ", ".join(predictions_df.columns[:10].tolist())
    if len(predictions_df.columns) > 10:
        cols_preview += f" ... (+{len(predictions_df.columns)-10} more)"

    text = (
        "Model and Dataset Statistics\n\n"
        f"Model: {MODEL_NAME}\n"
        f"Problem: {PROBLEM}\n"
        f"Target: {TARGET}\n"
        f"Accuracy: {ACCURACY*100:.2f}%\n"
        f"Total Rows: {ROWS:,}\n\n"
        f"Class Distribution:\n{dist_text}\n\n"
        f"Predictions Dataframe Shape: {predictions_df.shape[0]:,} rows x {predictions_df.shape[1]} cols\n"
        f"Columns Preview: {cols_preview}\n\n"
        f"Predicted Distribution (from parquet):\n{pred_text}"
    )
    await update.message.reply_text(text)


async def top_features_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = ["Top Predictive Features (XGBoost Feature Importance):\n"]
    for rank, (feat, score) in enumerate(TOP_FEATURES, start=1):
        bar_len = int(score * 30)
        bar = "#" * bar_len
        lines.append(f"{rank}. {feat}\n   Score: {score:.4f}  [{bar}]")
    text = "\n".join(lines)
    await update.message.reply_text(text)


async def hypotheses_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = ["Verified Hypotheses (Status: TRUE):\n"]
    for i, hyp in enumerate(HYPOTHESES, start=1):
        lines.append(f"{i}. {hyp}\n   Status: TRUE\n")
    await update.message.reply_text("\n".join(lines))


async def predict_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global predictions_df, final_model

    if predictions_df is None or final_model is None:
        await update.message.reply_text("Artifacts not loaded yet. Please try again shortly.")
        return

    await update.message.reply_text(
        "Running a sample prediction using a random row from the dataset..."
    )

    feature_names = None
    if hasattr(final_model, "feature_names_in_"):
        feature_names = list(final_model.feature_names_in_)
    elif hasattr(final_model, "get_booster"):
        try:
            feature_names = final_model.get_booster().feature_names
        except Exception:
            feature_names = None

    if feature_names is None:
        known_features = [f for f, _ in TOP_FEATURES]
        available = [c for c in known_features if c in predictions_df.columns]
        if not available:
            numeric_cols = predictions_df.select_dtypes(include=[np.number]).columns.tolist()
            target_variants = [TARGET, "event_type", "label", "y", "target"]
            available = [c for c in numeric_cols if c not in target_variants][:6]
        feature_names = available

    usable_features = [f for f in feature_names if f in predictions_df.columns]

    if not usable_features:
        await update.message.reply_text(
            "Could not identify usable feature columns in the predictions dataframe for a live prediction.\n"
            f"Model expects features: {feature_names}\n"
            f"Dataframe columns: {list(predictions_df.columns[:15])}"
        )
        return

    sample_row = predictions_df[usable_features].dropna().sample(1, random_state=np.random.randint(0, 10000))
    sample_values = sample_row.iloc[0]

    try:
        pred = final_model.predict(sample_row)[0]
        proba = None
        if hasattr(final_model, "predict_proba"):
            proba_arr = final_model.predict_proba(sample_row)[0]
            proba = max(proba_arr)
    except Exception as e:
        await update.message.reply_text(f"Prediction error: {e}")
        return

    feature_display = "\n".join(
        f"  {col}: {val:.4f}" if isinstance(val, float) else f"  {col}: {val}"
        for col, val in sample_values.items()
    )

    text = (
        "Sample Prediction Result\n\n"
        f"Input Features:\n{feature_display}\n\n"
        f"Predicted Class: {pred}\n"
    )
    if proba is not None:
        text += f"Confidence: {proba*100:.2f}%\n"

    await update.message.reply_text(text)


async def insights_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Generating AI insights via Claude... Please wait.")

    prompt = (
        f"You are a data science expert. Here is a summary of an ML project:\n\n"
        f"- Target variable: {TARGET}\n"
        f"- Problem type: {PROBLEM}\n"
        f"- Model: {MODEL_NAME}\n"
        f"- Accuracy: {ACCURACY*100:.2f}%\n"
        f"- Dataset size: {ROWS:,} rows\n"
        f"- Class distribution: all rows are 'view' events (500,000 samples)\n"
        f"- Top features by importance:\n"
        + "\n".join(f"  {i+1}. {f}: {s:.4f}" for i, (f, s) in enumerate(TOP_FEATURES))
        + f"\n\n- Verified hypotheses:\n"
        + "\n".join(f"  - {h}" for h in HYPOTHESES)
        + "\n\nProvide 3 to 5 concise, actionable business and technical insights "
        "based on this information. Focus on what the feature importances and hypotheses "
        "tell us about user behavior and model improvement opportunities."
    )

    try:
        insight_text = ask_claude(prompt)
    except Exception as e:
        await update.message.reply_text(f"Claude API error: {e}")
        return

    await update.message.reply_text(f"AI-Generated Insights:\n\n{insight_text}")


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Event Type Classification Bot - Help\n\n"
        "Commands:\n"
        "/start - Welcome message and bot overview\n"
        "/stats - View dataset and model statistics\n"
        "/top_features - See the most important predictive features\n"
        "/hypotheses - View verified data hypotheses\n"
        "/predict - Run a sample prediction from the dataset\n"
        "/insights - Get AI-powered insights from Claude\n"
        "/help - Show this help message\n\n"
        f"Model: {MODEL_NAME} | Accuracy: {ACCURACY*100:.2f}% | Rows: {ROWS:,}"
    )
    await update.message.reply_text(text)


def main():
    load_artifacts()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("stats", stats_handler))
    app.add_handler(CommandHandler("top_features", top_features_handler))
    app.add_handler(CommandHandler("hypotheses", hypotheses_handler))
    app.add_handler(CommandHandler("predict", predict_handler))
    app.add_handler(CommandHandler("insights", insights_handler))
    app.add_handler(CommandHandler("help", help_handler))

    logger.info("Bot is starting...")
    app.run_polling()


if __name__ == "__main__":
    main()