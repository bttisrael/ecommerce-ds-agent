import os
import logging
import pickle
import numpy as np
import pandas as pd
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

df = None
model = None

PREDICT_FEAT_RATIO = 0
PREDICT_SHIPPING_EFFICIENCY = 1
PREDICT_FEAT_INTERACT = 2
PREDICT_FEAT_DIFF = 3
PREDICT_CONFIRM = 4

TOP_FEATURES = [
    ("feat_ratio", 0.9998),
    ("shipping_efficiency", 0.0000),
    ("feat_interact", 0.0000),
    ("feat_diff", 0.0000),
    ("days_for_shipping_real", 0.0000),
    ("feat_sum", 0.0000),
    ("days_for_shipment_scheduled", 0.0000),
]

FEATURE_EXPLANATIONS = {
    "feat_ratio": "Ratio between actual and scheduled shipping days. The single most important signal — if actual shipping far exceeds scheduled, late delivery risk spikes dramatically.",
    "shipping_efficiency": "A composite score measuring how efficiently orders move through the supply chain. Low efficiency strongly correlates with delays.",
    "feat_interact": "Interaction term between shipping features. Captures non-linear relationships between delivery time components.",
    "feat_diff": "Difference between actual and scheduled shipping days (days_for_shipping_real - days_for_shipment_scheduled). Positive values indicate late shipments.",
    "days_for_shipping_real": "Actual number of days taken to ship the order. Longer real shipping times directly increase late delivery probability.",
    "feat_sum": "Sum of key shipping timing features. Acts as a combined magnitude indicator for overall shipment duration.",
    "days_for_shipment_scheduled": "Number of days originally scheduled for shipment. Sets the baseline expectation — tighter schedules raise risk when delays occur.",
}

HYPOTHESES = [
    {
        "title": "Orders where days_for_shipping_real exceeds days_for_shipment_scheduled tend to have higher late delivery risk",
        "explanation": "When the actual shipping time surpasses the scheduled window, late delivery risk rises sharply. This validates that schedule adherence is the primary driver — the model's top feature (feat_ratio) directly encodes this relationship.",
    },
    {
        "title": "Orders placed in certain category_names (e.g., bulky or high-volume categories) carry elevated late delivery risk",
        "explanation": "Product categories — particularly bulky items or high-demand categories — experience systematic shipping delays. This likely reflects warehouse handling complexity and carrier capacity constraints for specific item types.",
    },
]


def load_data():
    global df, model
    try:
        df = pd.read_parquet("df4_predictions.parquet")
        logger.info(f"Loaded dataframe with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load parquet file: {e}")
        df = pd.DataFrame()

    try:
        with open("final_model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded model successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = (
            "👋 *Welcome to the Data Science Assistant Bot!*\n\n"
            "I'm your intelligent companion for exploring the *E-commerce Late Delivery Risk* ML project.\n\n"
            "📦 *About this project:*\n"
            "We analyse 285M user events from an e-commerce platform to predict whether an order will be delivered late. "
            "The model helps optimise logistics, improve customer satisfaction, and reduce operational costs.\n\n"
            "🤖 *Model:* XGBoost | *Accuracy:* 97.45% | *Records:* 180,519\n\n"
            "📋 *Available Commands:*\n"
            "/start — Show this welcome message\n"
            "/stats — Dataset & model summary\n"
            "/top\\_features — Top 7 most important features\n"
            "/hypotheses — Validated business hypotheses\n"
            "/predict — Get a prediction for new data\n"
            "/insights — AI-generated business insights\n"
            "/help — List all commands\n\n"
            "Type any command to get started! 🚀"
        )
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in /start: {e}")
        await update.message.reply_text("An error occurred. Please try again.")


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if df is None or df.empty:
            await update.message.reply_text("⚠️ Dataset not available.")
            return

        total_records = len(df)

        pred_counts = {}
        pred_pcts = {}
        if "prediction" in df.columns:
            vc = df["prediction"].value_counts()
            for cls in ["0.0", "1.0"]:
                cnt = int(vc.get(cls, 0))
                pred_counts[cls] = cnt
                pred_pcts[cls] = (cnt / total_records * 100) if total_records > 0 else 0.0

        avg_confidence = None
        if "prediction_proba" in df.columns:
            avg_confidence = float(df["prediction_proba"].mean())

        text = (
            "📊 *Dataset & Model Summary*\n\n"
            f"📁 *Total Records:* {total_records:,}\n"
            f"🤖 *Model:* XGBoost\n"
            f"🎯 *Accuracy:* 97.45%\n"
            f"🔖 *Target:* late\\_delivery\\_risk (Binary Classification)\n\n"
            "📈 *Prediction Class Distribution:*\n"
            f"  ✅ No Late Delivery (0.0): {pred_counts.get('0.0', 77119):,} ({pred_pcts.get('0.0', 42.7):.1f}%)\n"
            f"  🚨 Late Delivery Risk (1.0): {pred_counts.get('1.0', 103400):,} ({pred_pcts.get('1.0', 57.3):.1f}%)\n\n"
        )

        if avg_confidence is not None:
            text += f"🔮 *Average Confidence Score:* {avg_confidence:.4f} ({avg_confidence*100:.2f}%)\n\n"

        text += (
            "📌 *Key Data Columns:*\n"
            "  • event\\_type: view, cart, purchase\n"
            "  • product\\_id: 28,548 unique products\n"
            "  • category\\_code: 123 categories\n"
            "  • brand: 1,850 unique brands\n"
            "  • price: Product price (float)\n"
            "  • user\\_id: 36,159 unique users\n"
        )

        if len(text) > 4096:
            text = text[:4090] + "..."

        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in /stats: {e}")
        await update.message.reply_text("⚠️ An error occurred while fetching stats. Please try again.")


async def top_features(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = "🔍 *Top 7 Most Important Features*\n\n"
        text += "These features drive the XGBoost model's predictions:\n\n"

        for i, (feat, score) in enumerate(TOP_FEATURES, 1):
            explanation = FEATURE_EXPLANATIONS.get(feat, "No explanation available.")
            escaped_feat = feat.replace("_", "\\_")
            text += f"*{i}. {escaped_feat}* — Importance: `{score:.4f}`\n"
            text += f"   💡 {explanation}\n\n"

        text += (
            "⚠️ *Note:* feat\\_ratio dominates with ~99.98% importance, "
            "meaning the ratio between actual and scheduled shipping days is by far "
            "the strongest predictor of late delivery risk. All other features contribute marginally."
        )

        if len(text) > 4096:
            text = text[:4090] + "..."

        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in /top_features: {e}")
        await update.message.reply_text("⚠️ An error occurred. Please try again.")


async def hypotheses(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = "✅ *Validated Business Hypotheses*\n\n"
        text += "The following hypotheses were tested and confirmed TRUE by the analysis:\n\n"

        for i, hyp in enumerate(HYPOTHESES, 1):
            text += f"*{i}. {hyp['title']}*\n"
            text += f"   📌 {hyp['explanation']}\n\n"

        text += (
            "💼 *Business Implications:*\n"
            "• Implement real-time monitoring of shipping\\_real vs scheduled to flag at-risk orders early.\n"
            "• Apply category-specific buffer times for bulky/high-volume product logistics planning.\n"
            "• Use feat\\_ratio threshold alerts to trigger proactive customer communication.\n"
        )

        if len(text) > 4096:
            text = text[:4090] + "..."

        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in /hypotheses: {e}")
        await update.message.reply_text("⚠️ An error occurred. Please try again.")


async def predict_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        context.user_data.clear()
        text = (
            "🔮 *Late Delivery Risk Predictor*\n\n"
            "I'll ask you for values of the top features to predict late delivery risk.\n\n"
            "📥 *Step 1 of 4:*\n"
            "Enter the *feat\\_ratio* value.\n\n"
            "This is the ratio of actual shipping days to scheduled shipping days.\n"
            "• Value < 1.0 → shipped faster than scheduled\n"
            "• Value = 1.0 → shipped exactly on schedule\n"
            "• Value > 1.0 → shipped slower than scheduled\n\n"
            "Example: `1.25`\n\n"
            "Type /cancel to exit at any time."
        )
        await update.message.reply_text(text, parse_mode="Markdown")
        return PREDICT_FEAT_RATIO
    except Exception as e:
        logger.error(f"Error in predict_start: {e}")
        await update.message.reply_text("⚠️ An error occurred. Please try again.")
        return ConversationHandler.END


async def predict_feat_ratio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = update.message.text.strip()
        value = float(user_input)
        if value < 0:
            raise ValueError("feat_ratio cannot be negative.")
        context.user_data["feat_ratio"] = value

        text = (
            "✅ feat\\_ratio recorded.\n\n"
            "📥 *Step 2 of 4:*\n"
            "Enter the *shipping\\_efficiency* score.\n\n"
            "This is a composite score (typically 0.0 to 1.0) measuring supply chain efficiency.\n"
            "• Higher values (close to 1.0) → more efficient shipping\n"
            "• Lower values (close to 0.0) → less efficient, higher risk\n\n"
            "Example: `0.75`"
        )
        await update.message.reply_text(text, parse_mode="Markdown")
        return PREDICT_SHIPPING_EFFICIENCY
    except ValueError as e:
        logger.warning(f"Invalid feat_ratio input: {e}")
        await update.message.reply_text(
            "⚠️ Invalid input. Please enter a valid non-negative number for feat\\_ratio.\nExample: `1.25`",
            parse_mode="Markdown",
        )
        return PREDICT_FEAT_RATIO
    except Exception as e:
        logger.error(f"Error in predict_feat_ratio: {e}")
        await update.message.reply_text("⚠️ An error occurred. Please try again.")
        return ConversationHandler.END


async def predict_shipping_efficiency(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = update.message.text.strip()
        value = float(user_input)
        context.user_data["shipping_efficiency"] = value

        text = (
            "✅ shipping\\_efficiency recorded.\n\n"
            "📥 *Step 3 of 4:*\n"
            "Enter the *feat\\_interact* value.\n\n"
            "This is an interaction term between shipping features capturing non-linear relationships.\n"
            "It's typically computed as the product of key shipping metrics.\n\n"
            "Example: `2.50`"
        )
        await update.message.reply_text(text, parse_mode="Markdown")
        return PREDICT_FEAT_INTERACT
    except ValueError:
        await update.message.reply_text(
            "⚠️ Invalid input. Please enter a valid number for shipping\\_efficiency.\nExample: `0.75`",
            parse_mode="Markdown",
        )
        return PREDICT_SHIPPING_EFFICIENCY
    except Exception as e:
        logger.error(f"Error in predict_shipping_efficiency: {e}")
        await update.message.reply_text("⚠️ An error occurred. Please try again.")
        return ConversationHandler.END


async def predict_feat_interact(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = update.message.text.strip()
        value = float(user_input)
        context.user_data["feat_interact"] = value

        text = (
            "✅ feat\\_interact recorded.\n\n"
            "📥 *Step 4 of 4:*\n"
            "Enter the *feat\\_diff* value.\n\n"
            "This is the difference: days\\_for\\_shipping\\_real minus days\\_for\\_shipment\\_scheduled.\n"
            "• Negative values → arrived early\n"
            "• 0 → arrived exactly on time\n"
            "• Positive values → arrived late\n\n"
            "Example: `2` (2 days late)"
        )
        await update.message.reply_text(text, parse_mode="Markdown")
        return PREDICT_FEAT_DIFF
    except ValueError:
        await update.message.reply_text(
            "⚠️ Invalid input. Please enter a valid number for feat\\_interact.\nExample: `2.50`",
            parse_mode="Markdown",
        )
        return PREDICT_FEAT_INTERACT
    except Exception as e:
        logger.error(f"Error in predict_feat_interact: {e}")
        await update.message.reply_text("⚠️ An error occurred. Please try again.")
        return ConversationHandler.END


async def predict_feat_diff(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = update.message.text.strip()
        value = float(user_input)
        context.user_data["feat_diff"] = value

        feat_ratio = context.user_data.get("feat_ratio", 0.0)
        shipping_efficiency = context.user_data.get("shipping_efficiency", 0.0)
        feat_interact = context.user_data.get("feat_interact", 0.0)
        feat_diff = value

        feat_sum = feat_ratio + shipping_efficiency + feat_interact + feat_diff
        days_for_shipping_real = feat_ratio * 3.0
        days_for_shipment_scheduled = 3.0

        input_features = {
            "feat_ratio": feat_ratio,
            "shipping_efficiency": shipping_efficiency,
            "feat_interact": feat_interact,
            "feat_diff": feat_diff,
            "days_for_shipping_real": days_for_shipping_real,
            "feat_sum": feat_sum,
            "days_for_shipment_scheduled": days_for_shipment_scheduled,
        }

        prediction_label = None
        prediction_proba = None
        used_model = False

        if model is not None:
            try:
                feature_names = [f[0] for f in TOP_FEATURES]
                input_values = [input_features.get(f, 0.0) for f in feature_names]
                input_array = np.array([input_values], dtype=np.float32)

                try:
                    import xgboost as xgb
                    dmatrix = xgb.DMatrix(input_array, feature_names=feature_names)
                    proba = model.predict(dmatrix)
                    if hasattr(proba, '__len__'):
                        proba = float(proba[0])
                    prediction_proba = proba
                    prediction_label = 1 if proba >= 0.5 else 0
                    used_model = True
                except Exception:
                    try:
                        input_df = pd.DataFrame([input_features])
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(input_df)[0]
                            prediction_proba = float(proba[1])
                            prediction_label = int(model.predict(input_df)[0])
                        else:
                            pred = model.predict(input_df)
                            prediction_label = int(pred[0])
                            prediction_proba = float(prediction_label)
                        used_model = True
                    except Exception as inner_e:
                        logger.warning(f"Model prediction failed: {inner_e}")
            except Exception as e:
                logger.warning(f"Model prediction attempt failed: {e}")

        if not used_model:
            if feat_ratio > 1.1:
                prediction_proba = min(0.95, 0.5 + (feat_ratio - 1.0) * 0.8)
            elif feat_ratio < 0.9:
                prediction_proba = max(0.05, 0.5 - (1.0 - feat_ratio) * 0.8)
            else:
                prediction_proba = 0.5 + feat_diff * 0.05
            prediction_proba = max(0.0, min(1.0, prediction_proba))
            prediction_label = 1 if prediction_proba >= 0.5 else 0

        if prediction_label == 1:
            risk_icon = "🚨"
            risk_label = "LATE DELIVERY RISK"
            risk_desc = "This order is likely to be delivered late."
            action = (
                "📋 *Recommended Actions:*\n"
                "• Notify the customer proactively about potential delay\n"
                "• Escalate to logistics team for priority handling\n"
                "• Consider expedited shipping options\n"
                "• Check carrier capacity and handoff status"
            )
        else:
            risk_icon = "✅"
            risk_label = "ON-TIME DELIVERY"
            risk_desc = "This order is likely to be delivered on time."
            action = (
                "📋 *Recommended Actions:*\n"
                "• No immediate intervention needed\n"
                "• Standard monitoring applies\n"
                "• Continue normal logistics workflow"
            )

        confidence_pct = (prediction_proba * 100) if prediction_label == 1 else ((1 - prediction_proba) * 100)
        model_note = "" if used_model else "\n⚠️ _Note: Using rule-based estimate (model unavailable)_\n"

        summary = (
            f"{risk_icon} *Prediction Result*\n\n"
            f"*Outcome:* {risk_label}\n"
            f"*Confidence:* {confidence_pct:.1f}%\n"
            f"*Risk Probability:* {prediction_proba:.4f}\n"
            f"{model_note}\n"
            f"📊 *Input Summary:*\n"
            f"  • feat\\_ratio: `{feat_ratio}`\n"
            f"  • shipping\\_efficiency: `{shipping_efficiency}`\n"
            f"  • feat\\_interact: `{feat_interact}`\n"
            f"  • feat\\_diff: `{feat_diff}`\n\n"
            f"🔎 *Interpretation:* {risk_desc}\n\n"
            f"{action}\n\n"
            "Use /predict to run another prediction."
        )

        if len(summary) > 4096:
            summary = summary[:4090] + "..."

        await update.message.reply_text(summary, parse_mode="Markdown")
        context.user_data.clear()
        return ConversationHandler.END

    except ValueError:
        await update.message.reply_text(
            "⚠️ Invalid input. Please enter a valid number for feat\\_diff.\nExample: `2`",
            parse_mode="Markdown",
        )
        return PREDICT_FEAT_DIFF
    except Exception as e:
        logger.error(f"Error in predict_feat_diff: {e}")
        await update.message.reply_text("⚠️ An error occurred during prediction. Please try again.")
        context.user_data.clear()
        return ConversationHandler.END


async def predict_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "❌ Prediction cancelled. Use /predict to start again.",
        parse_mode="Markdown",
    )
    return ConversationHandler.END


async def insights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not ANTHROPIC_API_KEY:
            await update.message.reply_text(
                "⚠️ ANTHROPIC\\_API\\_KEY is not configured. Cannot fetch AI insights.",
                parse_mode="Markdown",
            )
            return

        await update.message.reply_text("🧠 Generating AI insights... Please wait a moment.")

        total_records = len(df) if df is not None and not df.empty else 180519
        late_risk_count = 103400
        on_time_count = 77119
        accuracy = 97.45

        prompt = (
            "You are a senior data scientist and business analyst reviewing an ML project for an e-commerce platform.\n\n"
            "Project Summary:\n"
            f"- Platform: E-commerce with 285 million user events (view, cart, purchase behaviors)\n"
            f"- Goal: Predict late delivery risk for orders using XGBoost classification\n"
            f"- Dataset: {total_records:,} records\n"
            f"- Model Accuracy: {accuracy}%\n"
            "- Target: late_delivery_risk (binary: 0 = on-time, 1 = late)\n"
            f"- Class Distribution: {late_risk_count:,} late delivery predictions ({late_risk_count/(total_records)*100:.1f}%) vs {on_time_count:,} on-time ({on_time_count/(total_records)*100:.1f}%)\n\n"
            "Top Features (by importance):\n"
            "1. feat_ratio (0.9998) — ratio of actual to scheduled shipping days\n"
            "2. shipping_efficiency (0.0000)\n"
            "3. feat_interact (0.0000)\n"
            "4. feat_diff (0.0000)\n"
            "5. days_for_shipping_real (0.0000)\n"
            "6. feat_sum (0.0000)\n"
            "7. days_for_shipment_scheduled (0.0000)\n\n"
            "Validated Hypotheses:\n"
            "- Orders where days_for_shipping_real exceeds days_for_shipment_scheduled have higher late delivery risk (TRUE)\n"
            "- Certain product categories (bulky/high-volume) have elevated late delivery risk (TRUE)\n\n"
            "Please provide 2-3 paragraphs of actionable business insights covering:\n"
            "1. What the model reveals about the business's operational challenges\n"
            "2. How the company can use these predictions to improve customer experience and reduce costs\n"
            "3. Strategic recommendations for logistics optimization\n\n"
            "Keep the language clear and business-focused, avoiding heavy technical jargon."
        )

        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        insight_text = message.content[0].text

        response = (
            "🧠 *AI Business Insights*\n"
            "_(Powered by Claude AI)_\n\n"
            f"{insight_text}\n\n"
            "─────────────────────\n"
            "💡 Use /hypotheses to see validated business findings.\n"
            "📊 Use /stats for dataset overview."
        )

        if len(response) > 4096:
            response = response[:4090] + "..."

        await update.message.reply_text(response, parse_mode="Markdown")

    except ImportError:
        logger.error("anthropic package not installed.")
        await update.message.reply_text(
            "⚠️ The `anthropic` package is not installed. Please install it to use /insights."
        )
    except Exception as e:
        logger.error(f"Error in /insights: {e}")
        await update.message.reply_text(
            "⚠️ Failed to generate AI insights. Please check your API key and try again."
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = (
            "🤖 *Data Science Assistant — Command Reference*\n\n"
            "Here are all available commands:\n\n"
            "📌 */start* — Welcome message and project overview\n\n"
            "📊 */stats* — Dataset & model summary including:\n"
            "   • Total