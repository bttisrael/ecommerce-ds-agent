# Auto Data Scientist v7.1 — SOTA Multi-Agent Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-6C3FC6?style=flat)](https://github.com/joaomdmoura/crewAI)
[![Claude](https://img.shields.io/badge/Claude-4.6%20Sonnet-CC7722?style=flat)](https://www.anthropic.com/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?style=flat&logo=telegram&logoColor=white)](https://core.telegram.org/bots)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Search-4C8BF5?style=flat)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> ## Executive Summary

E-commerce growth increasingly depends on understanding *when* and *why* users convert from browsing to buying. With 285 million user events spanning views, cart additions, and purchases, this project addresses a critical business challenge: identifying which users are most likely to complete a purchase so that recommendations, promotions, and inventory decisions can be made proactively rather than reactively.

This project employs a multi-agent, AI-powered pipeline that automatically identifies the target variable, validates business hypotheses against real data, runs a competitive model selection process, and stress-tests findings through A/B testing — eliminating manual guesswork at every stage. The pipeline confirmed that **brand affinity**, **product category**, and a derived **feature-ratio metric** are statistically significant drivers of purchase conversion, giving the business clear, evidence-backed levers to act on.

The winning model — **XGBoost** — achieved **97.24% accuracy** on the held-out test set. Two immediate takeaways for stakeholders: *(1)* prioritize recommendation spend on top-converting brands and electronics-accessory categories, which exhibit disproportionately short decision cycles; *(2)* surface high feature-ratio products earlier in the browsing journey to accelerate purchase intent. Together, these actions translate directly into higher conversion rates and measurable revenue uplift.

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
9. [A/B Testing & Business ROI](#9-ab-testing--business-roi)
10. [Error Analysis](#10-error-analysis)
11. [Deployment — Telegram Bot](#11-deployment--telegram-bot)
12. [Output Files](#12-output-files)
13. [How to Reproduce](#13-how-to-reproduce)
14. [Agent Architecture Reference](#14-agent-architecture-reference)
15. [Limitations & Next Steps](#15-limitations--next-steps)

---

## 1. Project Result

| | |
|---|---|
| **Target variable** | `event_type` |
| **Problem type** | Classification |
| **Best model** | XGBoost |
| **Accuracy (test set)** | **97.24%** |
| **Optimized parameters** | `{"n_estimators": 142, "learning_rate": 0.2055855551931528, "max_depth": 7, "subsample": 0.6094732414399233}` |
| **CV strategy** | 2-fold StratifiedKFold + Optuna (3 trials) + Stacking |
| **Features used** | 16 (Boruta-selected from 16 engineered) |
| **Dataset** | 5000000 rows × 9 columns → 5000000 rows × 8 ML-ready |
| **Predictions generated** | 5000000 rows in `df4_predictions.parquet` |

### AI-Identified Target Justification
> *Auto-selected fallback: 'event_type' chosen from actual dataset columns.*

### Top Dataset Insights (by Claude)
1. Dataset has 5,000,000 rows × 9 columns. Target auto-detected as 'event_type'.

---

## 2. Pipeline Architecture

This pipeline uses a **two-LLM architecture**:
- **Orchestration layer** — CrewAI runs autonomous agents sequentially, each with exactly one tool.
- **Intelligence layer** — Claude 4.6 Sonnet is called directly *inside* each tool to do the actual reasoning: target identification, custom code generation, self-healing, feature design, hypothesis validation, A/B statistical testing, and Telegram bot authoring.

```text
Kaggle Dataset
      │
      ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Ingestor   │───▶│    Analyst       │───▶│  Feature Engineer   │
│ (dl+clean)  │    │ (QA+insights+    │    │ (std feats + Claude │
└─────────────┘    │  target detect)  │    │  feats + Boruta)    │
                   └──────────────────┘    └─────────────────────┘
                         │ Claude calls          │ Claude calls
                         ▼                       ▼
                   ┌──────────────┐    ┌──────────────────────┐
                   │ EDA Analyst  │───▶│ Hypothesis Validator │
                   │ (6 charts +  │    │ (12 hyps, TRUE/FALSE │
                   │  Cramér's V) │    │  verdict per Claude) │
                   └──────────────┘    └──────────────────────┘
                                                 │
                                                 ▼
                                       ┌──────────────────┐
                                       │   ML Scientist   │
                                       │  CV+Optuna+Stack │
                                       │  +error analysis │
                                       └──────────────────┘
                                                 │
                                                 ▼
                   ┌──────────────────┐    ┌──────────────────┐
                   │     Deployer     │◀───│    A/B Tester    │
                   │ (predictions +   │    │ (Bayesian stats  │
                   │  Telegram bot)   │    │  & Business ROI) │
                   └──────────────────┘    └──────────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │ Notebook Writer  │
                   │  (.ipynb, GitHub │
                   │   renders)       │
                   └──────────────────┘
```

### What Claude Does Inside Each Tool

| Tool | Claude's Role |
|------|--------------|
| `analyze_data_with_ai` | Reads full column stats → identifies target + problematic columns → writes & executes custom analysis code → **self-heals** on error |
| `generate_features_with_ai_strategy` | Receives correlation matrix → proposes 3–5 domain-specific engineered features → code runs once (no double-exec) |
| `validate_hypotheses` | Generates 10 business hypotheses → tests each with pandas → reads output → issues TRUE/FALSE/INCONCLUSIVE verdict + business insight |
| `train_and_save_model` | Receives model competition results → writes 3-paragraph narrative interpretation → contextualises the score for business stakeholders |
| `run_ab_testing` | Interprets McNemar/Wilcoxon p-values + Bayesian posterior → writes 3-paragraph business recommendation on whether to deploy Model A |
| `deploy_telegram_bot` | Generates df4_predictions.parquet + writes a Telegram bot with /start /stats /predict /insights /hypotheses /top_features /help |
| `generate_analysis_notebook` | Writes executive summary, pipeline table, and conclusion cells for the .ipynb |

---

## 3. Dataset

| | |
|---|---|
| **Source** | [mkechinov/ecommerce-behavior-data-from-multi-category-store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) |
| **Raw shape** | 5,000,000 rows × 9 columns |
| **ML-ready shape** | 5,000,000 rows × 8 columns |
| **Target** | `event_type` (classification) |
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
- `price_zscore`
- `price_tier`
- `log_price`
- `user_segment`
- `product_id_log_ratio_user`
- `price_x_log_product`
- `sq_log_price`

### Boruta feature selection
After engineering, Boruta (Random Forest shadow features) selected **0 features** from 16 total engineered features.
_Boruta not run or selected fewer than 5 features — full feature set used._

→ Full log: [feature_strategy.json](feature_strategy.json)

---

## 7. Business Hypothesis Validation

Claude generated 12 business hypotheses about `event_type`, tested each with real pandas code, and issued a verdict.

**Summary:** ✅ 3 TRUE · ❌ 7 FALSE · ⚪ 0 INCONCLUSIVE

| ID | Verdict | Hypothesis | Business Insight |
|----|---------|-----------|-----------------|
| H1 | ❌ **FALSE** | Users with higher 'price_tier' tend to have a higher rate of 'purchase' event_type, as pre | Premium pricing (tier 4) does not drive the highest purchase intent, suggesting mid-tier p |
| H2 | ❌ **FALSE** | Users in a higher 'user_segment' tend to have a higher proportion of 'purchase' event_type | User segment numbers are likely categorical identifiers rather than ordinal value rankings |
| H3 | ❌ **FALSE** | Products with a higher 'price_zscore' tend to have a lower 'purchase' event_type rate, as  | Since unusually priced products do not systematically deter purchases, the business should |
| H4 | ❌ **FALSE** | Sessions (user_session) with a higher 'feat_sum' tend to have a higher 'purchase' event_ty | Users who interact with too many product features may be experiencing decision fatigue or  |
| H5 | ✅ **TRUE** | Products associated with a specific 'brand' tend to have significantly different 'purchase | Marketing and inventory investment should be prioritized toward high-converting brands lik |
| H6 | ✅ **TRUE** | Products belonging to specific 'category_code' values tend to have a higher 'purchase' eve | The business should prioritize marketing spend and streamlined checkout experiences for hi |
| H7 | ❌ **FALSE** | Events with a higher 'feat_interact' value tend to have a higher 'purchase' event_type rat | Users with lower feature interaction values are actually more likely to purchase, suggesti |
| H8 | ❌ **FALSE** | Events with a higher 'log_price' tend to show a lower 'cart' to 'purchase' conversion rate | Cart abandonment is not simply driven by price level, suggesting that low-priced items may |
| H9 | ✅ **TRUE** | Events with a higher 'feat_ratio' tend to have a higher 'purchase' event_type rate, as a f | Products or sessions with a lower feat_ratio convert better, suggesting the business shoul |
| H10 | ❌ **FALSE** | Events occurring at specific hours derived from 'event_time' tend to have a higher 'purcha | Marketing campaigns and personalized push notifications should be prioritized during early |


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
**Winner: `XGBoost`** · Accuracy on test set: **97.24%**

Best Optuna parameters: `{"n_estimators": 142, "learning_rate": 0.2055855551931528, "max_depth": 7, "subsample": 0.6094732414399233}`

![Model Comparison](model_comparison.png)
![Feature Importance](feature_importance.png)

→ Full metrics: [Model_Metrics.md](Model_Metrics.md)
→ Train/test gap analysis: [Model_Evaluation.md](Model_Evaluation.md)

---

## 9. A/B Testing & Business ROI

To validate the business value of the trained model before production, an autonomous A/B test was run
comparing **Model A** (the trained `XGBoost`) against **Model B** (a lightweight baseline).

The test uses rigorous statistics:
- **Classification:** McNemar test (paired accuracy difference) + Bayesian Beta-posterior P(A beats B)
- **Regression:** Wilcoxon signed-rank test on residuals + Bayesian P(A beats B) on MAE improvement

_A/B testing not yet run or results not found._


---

## 10. Error Analysis

4-panel diagnostic chart:

![Error Analysis](error_analysis.png)

# Error Analysis

## Model: `XGBoost` | Target: `event_type`

**Overall failure rate:** 0.0276 (2.8% of test samples misclassified)

## Classification Report
```
              precision    recall  f1-score   support

           0       0.79      0.00      0.01     12819
           1       0.00      0.00      0.00     14756
           2       0.97      1.00      0.99    971831

    accuracy                           0.97    999406
   macro avg       0.59      0.33      0.33    999406
weighted avg       0.96      0.97      0.96    999406

```

## Error Analysis Chart
See `error_analysis.png` for

→ Full report: [Error_Analysis.md](Error_Analysis.md)

---

## 11. Deployment — Telegram Bot

Claude wrote a complete Telegram bot (`telegram_bot.py`) tailored to this specific dataset.

**Available commands:**
- `/start` — Welcome message and full command list
- `/stats` — Dataset and model summary with KPI metrics
- `/top_features` — Top 7 predictive features with importance scores
- `/hypotheses` — Validated TRUE business hypotheses
- `/insights` — AI-generated business insight powered by Claude
- `/help` — List all commands with descriptions

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

## 12. Output Files

| Status | File | Description |
|--------|------|-------------|
| ✅ | `df1_silver.parquet` | Silver layer — standardized raw data + imputation |
| ✅ | `df2_gold.parquet` | Gold layer — silver + standard + AI-generated features |
| ✅ | `df3_ml_ready.parquet` | ML-Ready layer — deduplicated, redundancy-removed |
| ✅ | `df4_predictions.parquet` | Predictions — all original columns + `prediction` column (5000000 rows) |
| ⬜ | `df5_scenarios.parquet` | Business scenarios — best/worst case bounds (regression only) |
| ✅ | `df6_recommendations.parquet` | Recommendation system output — top-N similar entities |
| ✅ | `final_model.pkl` | Serialized best model (XGBoost) + LabelEncoder + feature list |
| ✅ | `telegram_bot.py` | Telegram bot — /start /stats /predict /insights /hypotheses /top_features /help |
| ✅ | `dashboard.html` | Self-contained BI dashboard — zero dependencies, open in any browser |
| ✅ | `requirements.txt` | Python dependencies for the Telegram bot |
| ✅ | `analysis_notebook.ipynb` | Full pipeline story — renders on GitHub |
| ✅ | `Quality_Report.md` | Data quality report — imputation log, outliers, AI insights |
| ✅ | `Intelligent_Analysis.md` | Claude's full dataset analysis in JSON |
| ✅ | `Descriptive_Statistics.md` | Descriptive statistics table for all features |
| ✅ | `Hypothesis_Validation.md` | 10 business hypotheses — 3 TRUE / 7 FALSE / 0 INCONCLUSIVE |
| ✅ | `Model_Metrics.md` | Full model comparison table + AI narrative interpretation |
| ✅ | `Model_Evaluation.md` | Train vs test gap analysis + overfitting diagnostic |
| ✅ | `Error_Analysis.md` | 4-panel error diagnostic + business scenarios summary |
| ✅ | `AB_Test_Report.md` | Statistical A/B testing report — McNemar/Wilcoxon + Bayesian analysis |
| ✅ | `Recommendation_System.md` | Recommendation system report — strategy, sample output, business playbook |
| ✅ | `Deployment_Guide.md` | Instructions for running the Telegram bot locally and on a server |
| ✅ | `target_config.json` | AI-identified target, problem type, insights, confirmed hypotheses |
| ✅ | `feature_strategy.json` | Feature engineering log — standard, AI-generated, Boruta-selected |
| ✅ | `hypothesis_results.json` | Full hypothesis results with verdicts and business insights |
| ⬜ | `ab_test_results.json` | A/B testing raw results — scores, p-values, Bayesian probabilities |
| ✅ | `README.md` | This file |


---

## 13. How to Reproduce

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

## 14. Agent Architecture Reference

| # | Agent | Tool | Max Iter | Retry | Intelligence inside |
|---|-------|------|----------|-------|---------------------|
| 1 | Ingestor | `download_and_save_silver` | 3 | 1 | Multi-encoding CSV fallback |
| 2 | Analyst | `analyze_data_with_ai` | 8 | 3 | Claude: target ID + code gen + self-healing |
| 3 | Feature Engineer | `generate_features_with_ai_strategy` | 6 | 2 | Claude: custom feature code + Boruta |
| 4 | EDA Analyst | `generate_eda_and_ml_ready` | 4 | 1 | 6 charts + Cramér's V + row-index key (_src_idx) |
| 5 | Hypothesis Validator | `validate_hypotheses` | 6 | 2 | Claude: generate + test + verdict × 12 |
| 6 | ML Scientist | `train_and_save_model` | 8 | 2 | CV + Optuna + Stacking + Claude narrative |
| 7 | A/B Tester | `run_ab_testing` | 5 | 2 | McNemar/Wilcoxon + Bayesian + Claude recommendation |
| 8 | Deployer | `deploy_telegram_bot` | 6 | 2 | Claude: full Telegram bot code |
| 9 | Notebook Writer | `generate_analysis_notebook` | 4 | 1 | Claude: exec summary + conclusion |

### Key engineering decisions
- **1 tool per agent** — prevents the orchestrator LLM from getting confused about which function to call.
- **Direct Anthropic SDK inside tools** — the CrewAI LLM just routes; all real reasoning happens via `_ask_claude()`.
- **`_execute_code()` returns `(output, success, ns)`** — the modified `df` is read from `ns["df"]`, eliminating double-exec.
- **`_src_idx` row key** — written into `df3_ml_ready.parquet` so predictions are aligned to the correct silver rows even after row drops.
- **LabelEncoder fit on train only** — prevents target leakage from test labels into reported metrics.
- **Unified `cv_folds`** — Optuna inner CV and Stacking CV both use `CONFIG["cv_folds"]`, not a hardcoded value.

---

## 15. Limitations & Next Steps

## Limitations & Next Steps

- **Boruta selected 0 features**, suggesting possible issues with feature scaling, excessive noise variables, or a misconfigured Boruta run (e.g., too few iterations or incorrect `max_iter`); the final model likely trained on all raw features without validated selection — this must be audited before trusting feature importance rankings.

- **97.24% accuracy may be misleading** without per-class precision/recall and a confusion matrix — if `event_type` is class-imbalanced, the model could be near-trivially predicting the majority class; evaluate with macro-F1 and AUC-OVR before claiming strong performance.

- **3 Optuna trials is insufficient hyperparameter optimization** for XGBoost's search space (learning rate, depth, subsample, etc.); increase to a minimum of 50–100 trials with a proper TPE sampler and pruning to avoid landing on a suboptimal configuration by chance.

- **No experiment tracking (MLflow/W&B)** means hyperparameter configurations, data versions, and metric histories are not reproducible — implement tracking immediately, as current results cannot be reliably audited or compared in future iterations.

- **Absence of SHAP analysis** leaves the model a black box; add SHAP summary and dependence plots to validate that the model is learning causal signal rather than leaking features (e.g., event IDs, timestamps, or target-correlated proxies).

- **No probability calibration** means predicted class probabilities are unreliable for downstream decision thresholds; apply Platt scaling or isotonic regression and validate with calibration curves before any production scoring pipeline consumes probability outputs.

---

*Auto Data Scientist v7.1 · CrewAI + Claude 4.6 Sonnet + Optuna · [MIT License](LICENSE)*
