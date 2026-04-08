import os
import logging
import pickle
import pandas as pd
import anthropic
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

df = None
model = None


def load_data():
    global df, model
    try:
        df = pd.read_parquet("df4_predictions.parquet")
        logger.info("Loaded df4_predictions.parquet with shape %s", df.shape)
    except Exception as e:
        logger.error("Failed to load parquet file: %s", e)
        df = None
    try:
        with open("final_model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded final_model.pkl successfully")
    except Exception as e:
        logger.error("Failed to load model file: %s", e)
        model = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        msg = (
            "Welcome to the E-commerce ML Assistant!\n\n"
            "This bot gives you insights into a 5M-row XGBoost classification model\n"
            "that predicts user purchase behavior on an e-commerce platform.\n\n"
            "Available commands:\n"
            "/start - Show this welcome message\n"
            "/stats - Dataset and model summary\n"
            "/top_features - Top features with importance scores\n"
            "/hypotheses - Validated business hypotheses\n"
            "/insights - AI-generated business insights (Claude)\n"
            "/help - List all commands with descriptions\n\n"
            "Use any command to get started!"
        )
        await update.message.reply_text(msg)
    except Exception as e:
        logger.error("Error in /start handler: %s", e)
        await update.message.reply_text("An error occurred. Please try again.")


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if df is None:
            await update.message.reply_text("Data not loaded. Please check server logs.")
            return

        n_rows = len(df)
        n_cols = len(df.columns)
        n_users = df["user_id"].nunique() if "user_id" in df.columns else "N/A"
        n_products = df["product_id"].nunique() if "product_id" in df.columns else "N/A"
        n_categories = df["category_id"].nunique() if "category_id" in df.columns else "N/A"

        view_count = 4999736
        cart_count = 264
        cart_pct = round(cart_count / (view_count + cart_count) * 100, 4)
        view_pct = round(view_count / (view_count + cart_count) * 100, 4)

        price_mean = "N/A"
        price_min = "N/A"
        price_max = "N/A"
        if "price" in df.columns:
            price_mean = round(df["price"].mean(), 2)
            price_min = round(df["price"].min(), 2)
            price_max = round(df["price"].max(), 2)

        msg = (
            "[DATASET & MODEL SUMMARY]\n\n"
            "[DATA]\n"
            f"- Total rows: {n_rows:,}\n"
            f"- Total columns: {n_cols}\n"
            f"- Unique users: {n_users:,}\n"
            f"- Unique products: {n_products:,}\n"
            f"- Unique category IDs: {n_categories:,}\n\n"
            "[PRICE STATS]\n"
            f"- Mean price: ${price_mean}\n"
            f"- Min price: ${price_min}\n"
            f"- Max price: ${price_max}\n\n"
            "[MODEL]\n"
            "- Algorithm: XGBoost Classifier\n"
            "- Target: event_type (view vs cart)\n"
            "- Accuracy: 97.24%\n"
            "- Training rows: 5,000,000\n\n"
            "[PREDICTION DISTRIBUTION]\n"
            f"- view: {view_count:,} ({view_pct}%)\n"
            f"- cart: {cart_count:,} ({cart_pct}%)\n\n"
            "[BUSINESS GOAL]\n"
            "- Predict purchase intent from browsing behavior\n"
            "- Enable product recommendations and conversion targeting"
        )
        await update.message.reply_text(msg)
    except Exception as e:
        logger.error("Error in /stats handler: %s", e)
        await update.message.reply_text("An error occurred while fetching stats.")


async def top_features(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        msg = (
            "[TOP 7 FEATURES BY IMPORTANCE]\n\n"
            "1. feat_interact (0.3809)\n"
            "   - Interaction feature combining user and product signals.\n"
            "   - Captures how a specific user behaves with a specific product.\n"
            "   - Strongest predictor of cart intent.\n\n"
            "2. price (0.2633)\n"
            "   - Raw product price in USD.\n"
            "   - Higher or lower prices strongly shift purchase probability.\n"
            "   - Second most influential factor overall.\n\n"
            "3. feat_product (0.1537)\n"
            "   - Engineered feature summarizing product-level behavior.\n"
            "   - Reflects how often a product leads to cart events historically.\n\n"
            "4. price_tier (0.1259)\n"
            "   - Bucketed price category (e.g. budget, mid, premium).\n"
            "   - Segment-level pricing context improves predictions.\n\n"
            "5. user_segment (0.0762)\n"
            "   - Cluster or group the user belongs to.\n"
            "   - Captures behavioral patterns across similar users.\n\n"
            "6. log_price (0.0000)\n"
            "   - Log-transformed price. Low importance in final model.\n"
            "   - Likely redundant given price and price_tier are included.\n\n"
            "[KEY TAKEAWAY]\n"
            "- Interaction effects and price are dominant signals.\n"
            "- Product-level history matters more than user-level alone.\n"
            "- Price segmentation (tiers) adds value beyond raw price."
        )
        await update.message.reply_text(msg)
    except Exception as e:
        logger.error("Error in /top_features handler: %s", e)
        await update.message.reply_text("An error occurred while fetching features.")


async def hypotheses(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        msg = (
            "[VALIDATED BUSINESS HYPOTHESES]\n\n"
            "All hypotheses below were tested against the dataset and confirmed TRUE.\n\n"
            "1. Brand influence on behavior [TRUE]\n"
            "   - Products associated with a specific brand tend to have significantly\n"
            "     different event-type distributions.\n"
            "   - Implication: Brand is a meaningful segmentation axis for targeting.\n"
            "   - Recommendation models should incorporate brand affinity signals.\n\n"
            "2. Category code drives purchase rate [TRUE]\n"
            "   - Products belonging to specific category_code values tend to have\n"
            "     a higher purchase event_type rate.\n"
            "   - Implication: Some product categories convert far better than others.\n"
            "   - Budget allocation and promotions should prioritize high-converting\n"
            "     categories such as electronics and appliances.\n\n"
            "3. feat_ratio predicts purchase intent [TRUE]\n"
            "   - Events with a higher feat_ratio tend to have a higher purchase\n"
            "     event_type rate.\n"
            "   - Implication: The engineered ratio feature is a reliable signal.\n"
            "   - This validates the feature engineering pipeline and confirms\n"
            "     that ratio-based features should be retained in production.\n\n"
            "[SUMMARY]\n"
            "- 3 of 3 tested hypotheses confirmed TRUE.\n"
            "- Brand, category, and engineered ratio features all carry\n"
            "  real predictive and business signal."
        )
        await update.message.reply_text(msg)
    except Exception as e:
        logger.error("Error in /hypotheses handler: %s", e)
        await update.message.reply_text("An error occurred while fetching hypotheses.")


async def insights(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await update.message.reply_text("Generating AI insights via Claude... please wait.")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            await update.message.reply_text("ANTHROPIC_API_KEY is not set in environment.")
            return

        client = anthropic.Anthropic(api_key=api_key)

        prompt = (
            "You are a senior data scientist advising an e-commerce business. "
            "Here is the context for a machine learning project:\n\n"
            "- Platform: E-commerce with 285 million user events\n"
            "- Goal: Predict whether a user will add a product to cart based on browsing\n"
            "- Model: XGBoost classifier, accuracy 97.24%\n"
            "- Target classes: view vs cart\n"
            "- Class distribution: 4,999,736 view events, only 264 cart predictions\n"
            "- Top features: feat_interact (0.38), price (0.26), feat_product (0.15), "
            "price_tier (0.13), user_segment (0.08)\n"
            "- Validated: brand affects behavior, category_code drives purchase rate, "
            "feat_ratio predicts purchase intent\n\n"
            "Write 2-3 paragraphs of actionable business insights. "
            "Focus on revenue impact, product recommendations, and conversion strategy. "
            "Use plain English. No bullet points. No markdown."
        )

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=600,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        raw = message.content[0].text.strip()

        if len(raw) > 3800:
            raw = raw[:3800] + "...\n[Truncated for Telegram limit]"

        response = "[AI BUSINESS INSIGHTS - Claude]\n\n" + raw
        await update.message.reply_text(response)

    except anthropic.APIConnectionError as e:
        logger.error("Anthropic connection error: %s", e)
        await update.message.reply_text("Could not connect to Claude API. Check your network.")
    except anthropic.AuthenticationError as e:
        logger.error("Anthropic auth error: %s", e)
        await update.message.reply_text("Invalid ANTHROPIC_API_KEY. Please check configuration.")
    except anthropic.RateLimitError as e:
        logger.error("Anthropic rate limit: %s", e)
        await update.message.reply_text("Claude API rate limit reached. Please try again later.")
    except Exception as e:
        logger.error("Error in /insights handler: %s", e)
        await update.message.reply_text("An error occurred while generating insights.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        msg = (
            "[HELP - ALL COMMANDS]\n\n"
            "/start\n"
            "  - Show the welcome message and command overview.\n\n"
            "/stats\n"
            "  - Display dataset statistics and model performance summary.\n"
            "  - Includes row count, unique users, products, price stats,\n"
            "    accuracy, and prediction class distribution.\n\n"
            "/top_features\n"
            "  - List the top 7 features ranked by XGBoost importance score.\n"
            "  - Each feature includes a plain-language explanation of\n"
            "    what it represents and why it matters.\n\n"
            "/hypotheses\n"
            "  - Show the 3 business hypotheses that were validated as TRUE.\n"
            "  - Includes brand influence, category conversion rates,\n"
            "    and the feat_ratio signal.\n\n"
            "/insights\n"
            "  - Call the Claude AI API to generate 2-3 paragraphs of\n"
            "    actionable business insights from the model results.\n"
            "  - Requires ANTHROPIC_API_KEY to be set.\n\n"
            "/help\n"
            "  - Show this help message.\n\n"
            "[NOTE]\n"
            "- All data is based on 5M e-commerce events.\n"
            "- Model: XGBoost | Accuracy: 97.24% | Target: view vs cart"
        )
        await update.message.reply_text(msg)
    except Exception as e:
        logger.error("Error in /help handler: %s", e)
        await update.message.reply_text("An error occurred while fetching help.")


def main():
    load_data()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is not set in environment variables.")
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is required.")

    application = ApplicationBuilder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("top_features", top_features))
    application.add_handler(CommandHandler("hypotheses", hypotheses))
    application.add_handler(CommandHandler("insights", insights))
    application.add_handler(CommandHandler("help", help_command))

    logger.info("Bot is starting with run_polling()...")
    application.run_polling()


if __name__ == "__main__":
    main()