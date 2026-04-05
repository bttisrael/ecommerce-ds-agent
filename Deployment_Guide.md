# Telegram Bot Deployment Guide

## Setup

### 1. Create your Telegram bot
1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the instructions
3. Copy the token you receive

### 2. Add token to .env
```
TELEGRAM_BOT_TOKEN=your_token_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the bot
```bash
python telegram_bot.py
```

## Available Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message and command list |
| `/stats` | Dataset and model summary (Accuracy: 0.9702) |
| `/top_features` | Top 7 predictive features with business explanation |
| `/hypotheses` | Validated TRUE business hypotheses |
| `/predict` | Interactive prediction — enter feature values via chat |
| `/insights` | AI-generated business insight powered by Claude |
| `/help` | List all commands |

## Model Info
- **Model:** XGBoost
- **Target:** `event_type` (classification)
- **Accuracy:** 0.9702
- **Rows in df4_predictions.parquet:** 2,000,000

## Deploy to a Server (keep bot running 24/7)
```bash
# Option 1: nohup (Linux/Mac)
nohup python telegram_bot.py &

# Option 2: systemd service (Linux)
# Option 3: Railway, Render, or Fly.io (free tier available)
# Option 4: AWS Lambda + polling (serverless)
```
