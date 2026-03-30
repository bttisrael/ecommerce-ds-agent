# Auto Data Scientist v7.1 — SOTA Multi-Agent Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-6C3FC6?style=flat)](https://github.com/joaomdmoura/crewAI)
[![Claude](https://img.shields.io/badge/Claude-4.6%20Sonnet-CC7722?style=flat)](https://www.anthropic.com/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?style=flat&logo=telegram&logoColor=white)](https://core.telegram.org/bots)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Search-4C8BF5?style=flat)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> # Executive Summary

Late delivery is a silent revenue killer in e-commerce — eroding customer trust, inflating support costs, and increasing churn. This project builds a predictive system to identify shipment orders at risk of late delivery **before they occur**, enabling logistics teams to intervene proactively and protect customer satisfaction at scale across a dataset of 180,519 orders.

The solution is powered by a **multi-agent, AI-driven pipeline** that automates the full data science lifecycle: ingesting raw data, auto-detecting the target variable (`late_delivery_risk`), generating and statistically validating business hypotheses, and running a competitive model benchmarking process — requiring minimal human configuration from start to finish.

**XGBoost emerged as the top-performing model with 97.45% test accuracy.** Two findings translate directly into operational action: (1) orders where actual shipping time *exceeds* the scheduled window are the strongest predictor of lateness — tightening carrier SLA monitoring can reduce risk immediately; (2) specific high-volume product categories show disproportionately high late-delivery rates, suggesting those fulfillment workflows warrant dedicated capacity planning. Together, these insights give supply chain and operations teams a clear, data-backed roadmap for reducing delivery failures.

---

## Table of Contents
1. [Project Result](#1-project-result)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Dataset](#3-dataset)
4. [Data Quality & Imputation](#4-data-quality--imputation)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Feature Engineering](#6-feature-engineering)
7. [Business Hypothesis Validation](#7-business-hypothesis-validation)
8. [Model Training & Selection](#8-model-training--selection)
9. [Error Analysis](#9-error-analysis)
10. [Deployment — Telegram Bot](#10-deployment--telegram-bot)
11. [Output Files](#11-output-files)
12. [How to Reproduce](#12-how-to-reproduce)
13. [Agent Architecture Reference](#13-agent-architecture-reference)
14. [Limitations & Next Steps](#14-limitations--next-steps)

---

## 1. Project Result

| | |
|---|---|
| **Target variable** | `late_delivery_risk` |
| **Problem type** | Classification |
| **Best model** | XGBoost |
| **Accuracy (test set)** | **97.45%** |
| **Optimized parameters** | `{"n_estimators": 63, "learning_rate": 0.023386742875571638, "max_depth": 7, "subsample": 0.5830823325796838}` |
| **CV strategy** | 2-fold StratifiedKFold + Optuna (3 trials) + Stacking |
| **Features used** | 23 (Boruta-selected from 13 engineered) |
| **Dataset** | 67,501,979 rows × 9 columns → 180,519 rows × 32 ML-ready |
| **Predictions generated** | 180519 rows in `df4_predictions.parquet` |

### AI-Identified Target Justification
> *Auto-selected fallback: 'late_delivery_risk' chosen from actual dataset columns.*

### Top Dataset Insights (by Claude)
1. Dataset has 180,519 rows × 53 columns. Target auto-detected as 'late_delivery_risk'.

---

## 2. Pipeline Architecture

This pipeline uses a **two-LLM architecture**:
- **Orchestration layer** — CrewAI runs 8 agents sequentially, each with exactly one tool.
- **Intelligence layer** — Claude 4.6 Sonnet is called directly *inside* each tool to do the actual reasoning: target identification, custom code generation, self-healing, feature design, hypothesis generation, model narrative, and Telegram bot authoring.

```
Kaggle Dataset
      │
      ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Ingestor   │───▶│    Analyst       │───▶│  Feature Engineer   │
│  (dl+clean) │    │ (QA+insights+    │    │ (std feats + Claude │
└─────────────┘    │  target detect)  │    │  feats + Boruta)    │
                   └──────────────────┘    └─────────────────────┘
                          │ Claude calls          │ Claude calls
                          ▼                       ▼
                   ┌──────────────┐    ┌──────────────────────┐
                   │ EDA Analyst  │───▶│ Hypothesis Validator │
                   │ (6 charts +  │    │ (10 hyps, TRUE/FALSE │
                   │  Cramér's V) │    │  verdict per Claude) │
                   └──────────────┘    └──────────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │   ML Scientist   │
                                    │  CV+Optuna+Stack │
                                    │  +error analysis │
                                    └──────────────────┘
                                              │ Claude interprets
                                              ▼
                                    ┌──────────────────┐    ┌──────────────────┐
                                    │    Deployer      │───▶│ Notebook Writer  │
                                    │ (predictions +   │    │  (.ipynb, GitHub │
                                    │  Telegram bot)   │    │   renders)       │
                                    └──────────────────┘    └──────────────────┘
```

### What Claude Does Inside Each Tool

| Tool | Claude's Role |
|------|--------------|
| `analyze_data_with_ai` | Reads full column stats → identifies target + problematic columns → writes & executes custom analysis code → **self-heals** on error |
| `generate_features_with_ai_strategy` | Receives correlation matrix → proposes 3–5 domain-specific engineered features → code runs once (no double-exec) |
| `validate_hypotheses` | Generates 10 business hypotheses → tests each with pandas → reads output → issues TRUE/FALSE/INCONCLUSIVE verdict + business insight |
| `train_and_save_model` | Receives model competition results → writes 3-paragraph narrative interpretation → contextualises the score for business stakeholders |
| `deploy_telegram_bot` | Generates df4_predictions.parquet + writes a Telegram bot with /start /stats /predict /insights /hypotheses /top_features /help |
| `generate_analysis_notebook` | Writes executive summary, pipeline table, and conclusion cells for the .ipynb |

---

## 3. Dataset

| | |
|---|---|
| **Source** | [mkechinov/ecommerce-behavior-data-from-multi-category-store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) |
| **Raw shape** | 67,501,979 rows × 9 columns |
| **ML-ready shape** | 180,519 rows × 32 columns |
| **Target** | `late_delivery_risk` (classification) |
| **Business context** | # business_context.txt
echo "E-commerce platform with 285M user events. Goal: predict whether a user 
will purchase a product based on their browsing behavior (view, cart, purchase). 
Key business questions: which products to recommend, which users are likely to 
convert, and which product categories drive the most revenue." |

![Dataset Sample](dataset_sample.png)

---

## 4. Data Quality & Imputation

- **Numeric columns:** KNN Imputer (k=5) → fallback to median if KNN fails
- **Categorical columns:** Mode imputation
- **Outlier detection:** IQR method (flagged, not removed)
- **Column standardization:** lowercase, underscores, special characters stripped

→ Full report: [Quality_Report.md](Quality_Report.md)

---

## 5. Exploratory Data Analysis

Six charts generated automatically:

| Chart | Description |
|-------|-------------|
| ![](target_dist.png) | **Target distribution** — class balance or value spread |
| ![](distributions.png) | **Feature distributions** — histograms for all numeric columns |
| ![](boxplots.png) | **Boxplots** — outlier visualisation per feature |
| ![](categoricals.png) | **Categorical features** — top-15 value counts per column |
| ![](correlation_matrix.png) | **Pearson correlation matrix** — numeric associations |
| ![](cramers_v_matrix.png) | **Cramér's V matrix** — categorical association strength |

AI analysis chart (Claude-generated code):

![AI Analysis](intelligent_analysis.png)

---

## 6. Feature Engineering

### Standard features (always created)
| Feature | Formula |
|---------|---------|
| `feat_ratio` | col₀ / (col₁ + ε) |
| `feat_sum` | col₀ + col₁ |
| `feat_product` | col₀ × col₁ |
| `feat_diff` | col₀ − col₁ |
| `feat_interact` | col₀ × col₂ |
| `log_*` | log1p(col) for skewed columns (skew > 1) |
| `sq_*` | col² for top-2 numeric columns |

### AI-generated features
Claude proposed the following custom features based on the actual correlation structure of this dataset:
- `shipping_delay`
- `shipping_efficiency`
- `shipping_time_interaction`
- `revenue_efficiency`
- `shipping_pressure`

### Boruta feature selection
After engineering, Boruta (Random Forest shadow features) selected **23 features** from 13 total engineered features.
Selected: `days_for_shipping_real, days_for_shipment_scheduled, feat_ratio, feat_sum, feat_product, feat_diff, feat_interact, sq_days_for_shipping_real, sq_days_for_shipment_scheduled, shipping_delay...`

→ Full log: [feature_strategy.json](feature_strategy.json)

---

## 7. Business Hypothesis Validation

Claude generated 10 business hypotheses about `late_delivery_risk`, tested each with real pandas code, and issued a verdict.

**Summary:** ✅ 2 TRUE · ❌ 7 FALSE · ⚪ 1 INCONCLUSIVE

| ID | Verdict | Hypothesis | Business Insight |
|----|---------|-----------|-----------------|
| H1 | ✅ **TRUE** | Orders where days_for_shipping_real exceeds days_for_shipment_scheduled tend to have highe | Orders that take longer than their scheduled shipping window are highly likely to be flagg |
| H2 | ❌ **FALSE** | Orders with a higher days_for_shipping_real tend to have higher late_delivery_risk, since  | The business should focus monitoring efforts on orders expected to ship in the 4.8-6.0 day |
| H3 | ❌ **FALSE** | Orders with a lower days_for_shipment_scheduled tend to have higher late_delivery_risk, as | Shipments scheduled with very tight windows (around 0.8-1.6 days) pose the greatest late d |
| H4 | ❌ **FALSE** | Orders shipped to certain markets (e.g., LATAM or Africa) tend to have higher late_deliver | Since late delivery risk is uniformly high (~55%) across all markets, the root cause is li |
| H5 | ❌ **FALSE** | Orders belonging to certain order types (e.g., 'DEBIT' or 'TRANSFER') tend to have higher  | Since TRANSFER orders actually have the lowest late delivery risk, payment type is not a r |
| H6 | ❌ **FALSE** | Orders from certain department_names (e.g., large/heavy item departments like 'Outdoors' o | Late delivery risk is fairly uniform across all departments (ranging only from ~54.7% to ~ |
| H7 | ❌ **FALSE** | Orders placed by customers in certain customer_segments (e.g., 'Consumer' vs 'Corporate')  | Since late delivery risk is uniformly high (~55%) across all customer segments, the fulfil |
| H8 | ⚪ **INCONCLUSIVE** | Orders shipped to distant or remote order_countries tend to have higher late_delivery_risk | The business should investigate logistics partnerships and fulfillment strategies for cons |
| H9 | ❌ **FALSE** | Orders with lower benefit_per_order tend to have higher late_delivery_risk, as low-margin  | Fulfillment and delivery delays appear to be driven by operational or logistical factors r |
| H10 | ✅ **TRUE** | Orders placed in certain category_names (e.g., bulky or high-volume categories) tend to ha | The business should prioritize targeted logistics improvements and buffer inventory for hi |


![Hypothesis Validation](hypothesis_validation.png)

→ Full results: [Hypothesis_Validation.md](Hypothesis_Validation.md) · [hypothesis_results.json](hypothesis_results.json)

---

## 8. Model Training & Selection

### Competition protocol
1. **Baseline CV** — all candidates scored with 2-fold cross-validation
2. **Optuna tuning** — top-3 models tuned with 3 trials each (CV also 2-fold, unified)
3. **Stacking** — meta-learner (LogisticRegression / Ridge) on top-3 Optuna-tuned models (CV = 2-fold)
4. **Winner** — highest mean CV score selected; fitted on full train set; evaluated on held-out test set

### Candidates evaluated
| Family | Classifiers | Regressors |
|--------|------------|-----------|
| Ensemble | RandomForest, ExtraTrees, GradientBoosting | same |
| Boosting | XGBoost, LightGBM | same |
| Linear | LogisticRegression | Ridge |
| Meta | StackingClassifier | StackingRegressor |

### Result
**Winner: `XGBoost`** · Accuracy on test set: **97.45%**

Best Optuna parameters: `{"n_estimators": 63, "learning_rate": 0.023386742875571638, "max_depth": 7, "subsample": 0.5830823325796838}`

![Model Comparison](model_comparison.png)
![Feature Importance](feature_importance.png)

→ Full metrics: [Model_Metrics.md](Model_Metrics.md)
→ Train/test gap analysis: [Model_Evaluation.md](Model_Evaluation.md)

---

## 9. Error Analysis

4-panel diagnostic chart:

![Error Analysis](error_analysis.png)

# Error Analysis

## Model: `XGBoost` | Target: `late_delivery_risk`

**Overall failure rate:** 0.0255 (2.6% of test samples misclassified)

## Classification Report
```
              precision    recall  f1-score   support

           0       1.00      0.94      0.97     16308
           1       0.96      1.00      0.98     19796

    accuracy                           0.97     36104
   macro avg       0.98      0.97      0.97     36104
weighted avg       0.98      0.97      0.97     36104

```

## Error Analysis Chart
See `error_analysis.png` for confusion matrix and per-class accuracy.


→ Full report: [Error_Analysis.md](Error_Analysis.md)

---

## 10. Deployment — Telegram Bot

Claude wrote a complete Telegram bot (`telegram_bot.py`) tailored to this specific dataset.

**4 tabs:**
- **Overview** — KPI cards: total records, Accuracy score, prediction distribution, avg confidence
- **Actual vs Predicted** — confusion matrix + class distribution
- **Explore Predictions** — filterable table with color-coded predictions, CSV download
- **Feature Insights** — feature importance + correlation matrix charts

**Run locally:**
```bash
pip install -r requirements.txt
python telegram_bot.py
```

**Deploy 24/7:**
```bash
nohup python telegram_bot.py &
```

→ Full guide: [Deployment_Guide.md](Deployment_Guide.md)

---

## 11. Output Files

| Status | File | Description |
|--------|------|-------------|
| ✅ | `df1_silver.parquet` | Silver layer — standardized raw data + imputation |
| ✅ | `df2_gold.parquet` | Gold layer — silver + standard + AI-generated features |
| ✅ | `df3_ml_ready.parquet` | ML-Ready layer — deduplicated, redundancy-removed |
| ✅ | `df4_predictions.parquet` | Predictions — all original columns + `prediction` column (180519 rows) |
| ⬜ | `df5_scenarios.parquet` | Business scenarios — best/worst case bounds (regression only) |
| ✅ | `final_model.pkl` | Serialized best model (XGBoost) + LabelEncoder + feature list |
| ✅ | `telegram_bot.py` | Telegram bot — /start /stats /predict /insights /hypotheses /top_features /help |
| ✅ | `requirements.txt` | Python dependencies for the Telegram bot |
| ✅ | `analysis_notebook.ipynb` | Full pipeline story — renders on GitHub |
| ✅ | `Quality_Report.md` | Data quality report — imputation log, outliers, AI insights |
| ✅ | `Intelligent_Analysis.md` | Claude's full dataset analysis in JSON |
| ✅ | `Descriptive_Statistics.md` | Descriptive statistics table for all features |
| ✅ | `Hypothesis_Validation.md` | 10 business hypotheses — 2 TRUE / 7 FALSE / 1 INCONCLUSIVE |
| ✅ | `Model_Metrics.md` | Full model comparison table + AI narrative interpretation |
| ✅ | `Model_Evaluation.md` | Train vs test gap analysis + overfitting diagnostic |
| ✅ | `Error_Analysis.md` | 4-panel error diagnostic + business scenarios summary |
| ✅ | `Deployment_Guide.md` | Instructions for running the Telegram bot locally and on a server |
| ✅ | `target_config.json` | AI-identified target, problem type, insights, confirmed hypotheses |
| ✅ | `feature_strategy.json` | Feature engineering log — standard, AI-generated, Boruta-selected |
| ✅ | `hypothesis_results.json` | Full hypothesis results with verdicts and business insights |
| ✅ | `README.md` | This file |


---

## 12. How to Reproduce

### Prerequisites
```bash
# 1. Clone the repo
git clone https://github.com/bttisrael/ecommerce-ds-agent.git
cd ecommerce-ds-agent

# 2. Create .env
echo "KAGGLE_USERNAME=your_username"   >> .env
echo "KAGGLE_KEY=your_kaggle_key"      >> .env
echo "ANTHROPIC_API_KEY=sk-ant-..."    >> .env

# 3. (Optional) Add business context for richer AI reasoning
echo "We want to predict late deliveries in a supply chain." > business_context.txt

# 4. Install dependencies
pip install crewai kagglehub pandas pyarrow python-dotenv optuna anthropic \
            scikit-learn matplotlib seaborn tabulate numpy xgboost lightgbm \
            python-telegram-bot anthropic nbformat scipy boruta
```

### Run the pipeline
```bash
python auto_data_scientist_v7.py
```

### Run only the Telegram bot (after pipeline completes)
```bash
python telegram_bot.py
```

### Open the notebook
```bash
jupyter notebook analysis_notebook.ipynb
```

### Configuration knobs (`CONFIG` dict)
| Key | Default | Effect |
|-----|---------|--------|
| `test_size` | `0.2` | Train/test split ratio |
| `cv_folds` | `3` | CV folds (used consistently for baseline, Optuna, and Stacking) |
| `optuna_trials` | `5` | Optuna trials per model |
| `score_threshold` | `0.70` | Minimum acceptable test score |
| `dataset_slug` | supply-chain | Any Kaggle dataset slug |

---

## 13. Agent Architecture Reference

| # | Agent | Tool | Max Iter | Retry | Intelligence inside |
|---|-------|------|----------|-------|---------------------|
| 1 | Ingestor | `download_and_save_silver` | 3 | 1 | Multi-encoding CSV fallback |
| 2 | Analyst | `analyze_data_with_ai` | 8 | 3 | Claude: target ID + code gen + self-healing |
| 3 | Feature Engineer | `generate_features_with_ai_strategy` | 6 | 2 | Claude: custom feature code + Boruta |
| 4 | EDA Analyst | `generate_eda_and_ml_ready` | 4 | 1 | 6 charts + Cramér's V + row-index key (_src_idx) |
| 5 | Hypothesis Validator | `validate_hypotheses` | 6 | 2 | Claude: generate + test + verdict × 10 |
| 6 | ML Scientist | `train_and_save_model` | 8 | 2 | CV + Optuna + Stacking + Claude narrative |
| 7 | Deployer | `deploy_telegram_bot` | 6 | 2 | Claude: full Telegram bot code |
| 8 | Notebook Writer | `generate_analysis_notebook` | 4 | 1 | Claude: exec summary + conclusion |

### Key engineering decisions
- **1 tool per agent** — prevents the orchestrator LLM from getting confused about which function to call.
- **Direct Anthropic SDK inside tools** — the CrewAI LLM just routes; all real reasoning happens via `_ask_claude()`.
- **`_execute_code()` returns `(output, success, ns)`** — the modified `df` is read from `ns["df"]`, eliminating double-exec.
- **`_src_idx` row key** — written into `df3_ml_ready.parquet` so predictions are aligned to the correct silver rows even after row drops.
- **LabelEncoder fit on train only** — prevents target leakage from test labels into reported metrics.
- **Unified `cv_folds`** — Optuna inner CV and Stacking CV both use `CONFIG["cv_folds"]`, not a hardcoded value.

---

## 14. Limitations & Next Steps

## Limitations & Next Steps

- **Optuna search was severely underpowered**: 3 trials is insufficient to meaningfully explore XGBoost's hyperparameter space (learning rate, max_depth, subsample, etc.); re-run with ≥100 trials and a proper TPE sampler to verify the current parameters aren't leaving performance on the table or causing overfitting.

- **97.45% accuracy likely masks class imbalance issues**: Without calibration curves or a precision-recall breakdown per class, it is unknown whether the model is simply predicting the majority class well — evaluate with F1, MCC, and AUC-PR before trusting this number for production decisions.

- **No model explainability layer**: SHAP values were not generated, meaning there is no way to validate that Boruta's 23 selected features make *business sense* or to detect spurious correlations (e.g., a data-leakage proxy feature); add SHAP summary and dependence plots as a mandatory pre-deployment step.

- **Prediction confidence is uncalibrated**: XGBoost probability outputs are known to be poorly calibrated out of the box; apply Platt scaling or isotonic regression and validate with reliability diagrams, especially if downstream decisions are threshold-sensitive.

- **No experiment tracking creates reproducibility risk**: Hyperparameters, Boruta random seeds, train/test splits, and metric history are not versioned; integrate MLflow or Weights & Biases before any further iteration to ensure experiments are auditable and comparable.

- **Generalization to production data is unvalidated**: Evaluate temporal stability (if delivery data has a time component, use a time-based train/test split), monitor for feature drift on live data using PSI scores, and define a re

---

*Auto Data Scientist v7.1 · CrewAI + Claude 4.6 Sonnet + Optuna · [MIT License](LICENSE)*
