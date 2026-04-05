# Telegram Bot Deployment Guide

## 1. Get your bot token
Open Telegram → @BotFather → /newbot → copy the token

## 2. Add to .env
```
TELEGRAM_BOT_TOKEN=your_token_here
ANTHROPIC_API_KEY=your_anthropic_key
```

## 3. Install + run
```bash
pip install -r requirements.txt
python telegram_bot.py
```

## Commands: /start /stats /top_features /hypotheses /predict /insights /help
Model: XGBoost | Accuracy: 0.9609 | Rows: 500,000
