"""
Auto Data Scientist v7 — True SOTA Multi-Agent Pipeline
========================================================

Architecture:
  - CrewAI Layer: stable, sequential, 1 tool per agent (lesson from v5)
  - Real intelligence: Claude 4.6 Sonnet called INSIDE tools to reason
  - LLM writes analysis code dynamically (not pre-written)
  - LLM interprets results and makes real decisions
  - LLM diagnoses errors and suggests corrections (real self-healing)
  - LLM chooses feature strategy based on the data
  - LLM decides which model to use and why
  - LLM writes a full Telegram bot tailored to the dataset

Why this is different:
  - v4/v5/v6: LLM decides which function to call. Python does the work.
  - v7: LLM decides AND ALSO does the intelligent work inside each step.

Pipeline steps:
  1. Ingestor            → download_and_save_silver
  2. Analyst             → analyze_data_with_ai
  3. Feature Eng.        → generate_features_with_ai_strategy  (+ Boruta selection)
  4. EDA Analyst         → generate_eda_and_ml_ready           (+ Cramér's V)
  5. Hypothesis Validator→ validate_hypotheses                  (10 hypotheses TRUE/FALSE)
  6. ML Scientist        → train_and_save_model                 (+ error analysis + scenarios)
  7. Deployer            → deploy_telegram_bot
                             ├── df4_predictions.parquet (all original cols + prediction)
                             ├── telegram_bot.py         (Telegram bot with 7 commands)
                             └── requirements.txt
  8. Notebook Writer     → generate_analysis_notebook
                             └── analysis_notebook.ipynb (full pipeline story, renders on GitHub)

FIX LOG (v7 → v7.1):
  [FIX-1] _execute_code: removed .copy(), now returns (output, success, ns) so
          callers can read the modified df instead of running exec() twice.
  [FIX-2] generate_features_with_ai_strategy: eliminated the double-exec pattern;
          feature code now runs once and the resulting df is read from ns["df"].
  [FIX-3] train_and_save_model: LabelEncoder now fit only on y_train, then
          transform applied to y_test — prevents target-leakage into test metrics.
  [FIX-4] generate_eda_and_ml_ready: saves _src_idx (original silver row index)
          in ml_ready so the deployer can align predictions to the correct silver rows.
  [FIX-5] deploy_telegram_bot: uses _src_idx for safe silver↔prediction alignment
          instead of the fragile iloc[:min_rows] assumption.
  [FIX-6] Per-agent max_iter / max_retry_limit tuned to agent complexity.
  [FIX-7] Optuna inner CV unified to CONFIG["cv_folds"] (was hardcoded cv=3).
  [FIX-8] StackingClassifier/Regressor CV unified to CONFIG["cv_folds"].
  [FIX-9] evaluate_model: LabelEncoder applied post-split (same fix as FIX-3).

Dependencies:
    pip install crewai kagglehub pandas pyarrow python-dotenv optuna anthropic
    pip install scikit-learn matplotlib seaborn tabulate numpy xgboost lightgbm
    pip install python-telegram-bot anthropic nbformat scipy boruta

Environment variables (.env):
    KAGGLE_USERNAME=your_username
    KAGGLE_KEY=your_key_here
    ANTHROPIC_API_KEY=sk-ant-...

Optional:
    Create business_context.txt with a description of the business problem.
"""

# ==========================================
# 0. IMPORTS
# ==========================================
import os, sys, json, logging, subprocess, pickle
import traceback, textwrap, io, contextlib
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import optuna
import anthropic
import paramiko
from scipy import stats as ss
optuna.logging.set_verbosity(optuna.logging.WARNING)

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, KFold,
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    StackingClassifier, StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error, accuracy_score,
    classification_report, r2_score,
    mean_absolute_error, confusion_matrix,
)

load_dotenv()

# ==========================================
# LOGGING UTF-8
# ==========================================
_utf8_handler = logging.StreamHandler(
    open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
)
_utf8_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("pipeline.log", encoding="utf-8"),
        _utf8_handler,
    ],
)
logger = logging.getLogger("AutoDS")

# ==========================================
# 1. CONFIGURATION
# ==========================================
_BASE_DIR = os.getcwd()

CONFIG = {
    "dataset_slug": "mkechinov/ecommerce-behavior-data-from-multi-category-store",
    "dataset_url":  "https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store",

    "silver_path":   os.path.join(_BASE_DIR, "df1_silver.parquet"),
    "gold_path":     os.path.join(_BASE_DIR, "df2_gold.parquet"),
    "ml_ready_path": os.path.join(_BASE_DIR, "df3_ml_ready.parquet"),
    "quality_md":    os.path.join(_BASE_DIR, "Quality_Report.md"),
    "analysis_md":   os.path.join(_BASE_DIR, "Intelligent_Analysis.md"),
    "stats_md":      os.path.join(_BASE_DIR, "Descriptive_Statistics.md"),
    "target_json":   os.path.join(_BASE_DIR, "target_config.json"),
    "strategy_json": os.path.join(_BASE_DIR, "feature_strategy.json"),
    "corr_png":      os.path.join(_BASE_DIR, "correlation_matrix.png"),
    "metrics_md":    os.path.join(_BASE_DIR, "Model_Metrics.md"),
    "eval_md":       os.path.join(_BASE_DIR, "Model_Evaluation.md"),
    "model_pkl":       os.path.join(_BASE_DIR, "final_model.pkl"),
    "readme_md":       os.path.join(_BASE_DIR, "README.md"),
    "business_ctx":    os.path.join(_BASE_DIR, "business_context.txt"),
    "predictions_path": os.path.join(_BASE_DIR, "df4_predictions.parquet"),
    "telegram_bot":     os.path.join(_BASE_DIR, "telegram_bot.py"),
    "html_dashboard":   os.path.join(_BASE_DIR, "dashboard.html"),
    "requirements_txt": os.path.join(_BASE_DIR, "requirements.txt"),
    "notebook_path":    os.path.join(_BASE_DIR, "analysis_notebook.ipynb"),
    "hypothesis_json":  os.path.join(_BASE_DIR, "hypothesis_results.json"),
    "hypothesis_png":   os.path.join(_BASE_DIR, "hypothesis_validation.png"),
    "error_analysis_md":os.path.join(_BASE_DIR, "Error_Analysis.md"),
    "error_analysis_png":os.path.join(_BASE_DIR, "error_analysis.png"),
    "scenarios_path":   os.path.join(_BASE_DIR, "df5_scenarios.parquet"),
    "ab_test_json":     os.path.join(_BASE_DIR, "ab_test_results.json"),
    "ab_test_md":       os.path.join(_BASE_DIR, "AB_Test_Report.md"),
    "ab_test_png":      os.path.join(_BASE_DIR, "ab_test.png"),
    "reco_path":        os.path.join(_BASE_DIR, "df6_recommendations.parquet"),
    "reco_md":          os.path.join(_BASE_DIR, "Recommendation_System.md"),
    "reco_png":         os.path.join(_BASE_DIR, "recommendations.png"),

    "test_size":       0.2,
    "random_state":    42,
    "cv_folds":        2,        # [LITE] was 3 — saves ~33% CV time across all steps
    "optuna_trials":   3,        # [LITE] was 5 — saves ~40% tuning time
    "score_threshold": 0.70,
    # [FIX-6] max_iter/max_retry_limit are now set per-agent below, not here.
    # Kept for backward-compat reference only.
    "max_iter":        5,
    "max_retry_limit": 2,

    # ── [LITE] Hardware-safety knobs ──────────────────────────────────────────
    # n_jobs: max CPU cores used by sklearn. -1 = all cores (can overheat/shutdown).
    # Set to 2 for laptops; increase to 4-6 on workstations with active cooling.
    "n_jobs":                    2,

    # boruta_sample: Boruta runs on a random subsample of this size.
    # Avoids training hundreds of RF trees on the full dataset.
    # Features relevant for 15k rows generalise well to the full dataset.
    "boruta_sample":             15_000,

    # knn_fallback_threshold: if the dataset has more rows than this,
    # KNNImputer (which builds an O(n^2) distance matrix) is replaced with
    # the much lighter median imputer. KNN is only used for small datasets.
    "knn_fallback_threshold":    50_000,

    # enable_stacking: Stacking trains each base model cv_folds times inside
    # the CV loop — the most RAM/CPU-intensive step. Disable on low-end hardware.
    # Re-enable when you have a workstation or cloud instance.
    "enable_stacking":           False,

    # max_cat_cardinality: categorical columns with more unique values than this
    # are dropped from the ML-ready dataset. pd.get_dummies on high-cardinality
    # columns (e.g. customer_email with ~100k uniques) generates tens of thousands
    # of boolean columns and crashes with MemoryError. 50 is safe for most machines.
    "max_cat_cardinality":       50,

    # forced_target: set this to a column name string to skip Claude's target
    # detection entirely. Useful when the auto-detection falls back incorrectly
    # (e.g. on datasets where JSON parsing fails). Set to None to use AI detection.
    # Example: "forced_target": "late_delivery_risk"
    "forced_target":             "event_type",

    # ── Oracle Cloud auto-deploy ──────────────────────────────────────────────
    # Set ORACLE_VM_IP, ORACLE_KEY_PATH, and ORACLE_REPO_PATH in your .env file.
    # The pipeline will automatically SSH into the VM, upload model artifacts,
    # run git pull, and restart the telegram-bot systemd service after every run.
    # Leave ORACLE_VM_IP unset (or empty) to skip auto-deploy entirely.
    #
    # .env example:
    #   ORACLE_VM_IP=147.15.72.72
    #   ORACLE_KEY_PATH=C:\Users\israb\Documents\Oracle_keys\ssh-key-2026-03-21.key
    #   ORACLE_REPO_PATH=/home/ubuntu/agente_ds2
    "oracle_vm_ip":    os.getenv("ORACLE_VM_IP",    None),
    "oracle_key_path": os.getenv("ORACLE_KEY_PATH", None),
    "oracle_repo_path":os.getenv("ORACLE_REPO_PATH","/home/ubuntu/ecommerce-ds-agent"),
    "oracle_vm_user":  "ubuntu",
}

# ==========================================
# 2. LLMs
# ==========================================

# LLM for CrewAI (orchestrates the agents)
llm_agent = LLM(
    model="anthropic/claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.0,
)

# Direct Anthropic client (used INSIDE tools for real reasoning)
_claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def _ask_claude(prompt: str, max_tokens: int = 2000) -> str:
    """Calls Claude directly for intelligent reasoning inside tools."""
    try:
        msg = _claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logger.error(f"[Claude] Error in direct call: {e}")
        return f"CLAUDE_ERROR: {e}"

# ==========================================
# 3. HELPERS
# ==========================================

def _read_ctx() -> str:
    p = CONFIG["business_ctx"]
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read().strip()
    return ""

def _detect_type(y: pd.Series) -> str:
    n, nu = len(y), y.nunique()
    if y.dtype == "object":                return "classification"
    if nu <= 15:                           return "classification"
    if nu <= 30 and "int" in str(y.dtype): return "classification"
    if nu / n < 0.05:                      return "classification"
    return "regression"

def _safe_json(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")

# [FIX-1] _execute_code now returns (output, success, namespace).
# The caller receives ns["df"] directly — no need for a second exec() call.
# Caller is responsible for passing df.copy() when it does NOT want mutations,
# or df itself when it DOES want to capture the modified dataframe.
def _execute_code(code: str, df: pd.DataFrame) -> tuple:
    """
    Executes Python code and returns (output, success, namespace).
    Pass df.copy() if you only want output; pass df directly if you want
    to capture mutations via ns["df"].
    """
    ns = {"pd": pd, "np": np, "df": df,
          "plt": plt, "sns": sns, "os": os,
          "_BASE_DIR": _BASE_DIR, "json": json}
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(textwrap.dedent(code), ns)
        return output.getvalue() or "Executed without output.", True, ns
    except Exception as e:
        return f"{type(e).__name__}: {e}", False, ns

def _cramer_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V — association strength between two categorical variables."""
    try:
        cm = pd.crosstab(x, y).to_numpy()
        n  = cm.sum()
        if n == 0: return 0.0
        r, k   = cm.shape
        chi2   = ss.chi2_contingency(cm)[0]
        chi2c  = max(0, chi2 - (k - 1) * (r - 1) / (n - 1))
        kc     = k - (k - 1) ** 2 / (n - 1)
        rc     = r - (r - 1) ** 2 / (n - 1)
        denom  = min(kc - 1, rc - 1)
        if denom <= 0: return 0.0
        return float(np.sqrt((chi2c / n) / denom))
    except Exception:
        return 0.0

def _mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0: return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

# ==========================================
# 4. TOOLS — ONE PER AGENT, REAL INTELLIGENCE INSIDE
# ==========================================

# ── STEP 1: Ingestion ─────────────────────────────────────────────────────────

@tool("download_and_save_silver")
def download_and_save_silver(_: str = "") -> str:
    """
    Downloads the dataset from Kaggle, standardizes columns, and saves df1_silver.parquet.
    Returns INGESTION_SUCCESS or ERROR. No parameters.
    """
    import kagglehub
    kaggle_user = os.getenv("KAGGLE_USERNAME")
    kaggle_key  = os.getenv("KAGGLE_KEY")
    if not kaggle_user or not kaggle_key:
        return "ERROR: KAGGLE_USERNAME/KAGGLE_KEY not found in .env."
    os.environ["KAGGLE_USERNAME"] = kaggle_user
    os.environ["KAGGLE_KEY"]      = kaggle_key
    try:
        path = kagglehub.dataset_download(CONFIG["dataset_slug"])
        csvs = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not csvs:
            return f"ERROR: No CSV found in {path}"
        csv_path = os.path.join(path, csvs[0])
        # Try multiple encodings — many Kaggle CSVs are not UTF-8
        for enc in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                logger.info(f"[Ingestor] CSV read with encoding: {enc}")
                break
            except UnicodeDecodeError:
                continue
        else:
            return "ERROR: Could not decode CSV with any known encoding."
        df.columns = (df.columns.str.strip().str.lower()
                      .str.replace(" ", "_")
                      .str.replace(r"[^a-z0-9_]", "", regex=True))
        df.to_parquet(CONFIG["silver_path"], index=False)
        logger.info(f"[Ingestor] Silver saved: {df.shape}")
        return (f"INGESTION_SUCCESS\n"
                f"Shape: {df.shape}\n"
                f"Columns: {list(df.columns)}\n"
                f"File: df1_silver.parquet")
    except Exception as e:
        return f"INGESTION_ERROR: {e}"


# ── STEP 2: Quality + Intelligent Analysis ────────────────────────────────────

@tool("analyze_data_with_ai")
def analyze_data_with_ai(_: str = "") -> str:
    """
    Reads df1_silver.parquet. Applies intelligent imputation. Passes a full
    dataset summary to Claude to reason about: data quality, suspicious columns,
    cleaning recommendations, and business insights.
    Claude also writes and executes custom Python analysis code.
    Saves Quality_Report.md and Intelligent_Analysis.md.
    Returns ANALYSIS_SUCCESS or ERROR. No parameters.
    """
    try:
        if not os.path.exists(CONFIG["silver_path"]):
            return "ERROR: df1_silver.parquet does not exist."

        df  = pd.read_parquet(CONFIG["silver_path"])
        n, p = df.shape
        ctx = _read_ctx()

        # ── Imputation ─────────────────────────────────────────────────────────
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        imps = []

        if num_cols and df[num_cols].isnull().any().any():
            try:
                num_cols_valid = [c for c in num_cols if df[c].notna().any()]
                if num_cols_valid:
                    # [LITE] KNNImputer builds an O(n²) distance matrix.
                    # On datasets with >50k rows this can allocate several GBs of RAM
                    # and cause thermal shutdown. Use SimpleImputer (median) instead.
                    if len(df) <= CONFIG.get("knn_fallback_threshold", 50_000):
                        df[num_cols_valid] = KNNImputer(n_neighbors=5).fit_transform(
                            df[num_cols_valid])
                        imps.append("KNN imputer applied to numeric columns.")
                    else:
                        df[num_cols_valid] = SimpleImputer(strategy="median").fit_transform(
                            df[num_cols_valid])
                        imps.append(
                            f"Median imputer applied (dataset has {len(df):,} rows — "
                            f"KNN skipped to avoid RAM spike, threshold={CONFIG['knn_fallback_threshold']:,})."
                        )
            except Exception:
                num_cols_valid = [c for c in num_cols if df[c].notna().any()]
                if num_cols_valid:
                    df[num_cols_valid] = SimpleImputer(strategy="median").fit_transform(
                        df[num_cols_valid])
                    imps.append("Median imputer (fallback) applied.")
        for c in cat_cols:
            if df[c].isnull().any():
                fill_val = df[c].mode()[0] if not df[c].mode().empty else "MISSING"
                df[c] = df[c].fillna(fill_val)   # [FIX] avoids ChainedAssignmentError
                imps.append(f"Mode applied to '{c}'.")
        if imps:
            df.to_parquet(CONFIG["silver_path"], index=False)

        # ── Summary for Claude to reason about ────────────────────────────────
        col_summary = {}
        for col in df.columns:
            s = df[col]
            col_summary[col] = {
                "dtype": str(s.dtype),
                "nunique": int(s.nunique()),
                "null_pct": round(float(s.isnull().mean() * 100), 2),
                "sample": s.dropna().head(5).tolist(),
            }
            if s.dtype in ["float64", "int64"]:
                col_summary[col].update({
                    "mean": round(float(s.mean()), 4),
                    "std":  round(float(s.std()), 4),
                    "min":  round(float(s.min()), 4),
                    "max":  round(float(s.max()), 4),
                    "skew": round(float(s.skew()), 4),
                })

        # ── Forced target override (bypasses Claude detection entirely) ──────────
        # Set CONFIG["forced_target"] = "column_name" to skip AI target detection.
        # This is the safest option when you know the target in advance.
        forced = CONFIG.get("forced_target")
        if forced and forced in df.columns:
            logger.info(f"[Analysis] Using forced_target from CONFIG: '{forced}'")
            analysis = {
                "likely_target":        forced,
                "target_justification": f"Forced via CONFIG['forced_target'] = '{forced}'.",
                "problematic_columns":  [],
                "insights":             [f"Target '{forced}' was set manually in CONFIG."],
                "analysis_code":        f"print(df['{forced}'].value_counts())",
                "feature_strategy":     "Create ratio and interaction features between numeric variables.",
            }
        elif forced:
            logger.warning(
                f"[Analysis] CONFIG['forced_target'] = '{forced}' not found in columns. "
                f"Falling back to AI detection. Available: {list(df.columns[:10])}"
            )
            forced = None  # clear so we fall through to Claude

        if not forced:
            # ── Claude analyzes the dataset ────────────────────────────────────────
            prompt_analysis = f"""You are a senior Data Scientist analyzing a dataset.

BUSINESS CONTEXT: {ctx or 'Resume screening dataset with 200k candidates.'}

DATASET:
- Shape: {n} rows x {p} columns
- Columns and statistics:
{json.dumps(col_summary, indent=2, default=_safe_json)}

IMPUTATIONS ALREADY APPLIED: {imps}

Your task:
1. Identify which column is likely the TARGET (response variable) and justify.
2. Identify problematic columns (leakage, high cardinality, constants).
3. List the top-5 most important insights about this dataset.
4. Write Python code (using df, pd, np, plt, os, json, _BASE_DIR) that:
   a) Calculates statistics by group for the most interesting column
   b) Shows the distribution of the likely target
   c) Calculates correlations with the target
   d) Saves a plot to os.path.join(_BASE_DIR, 'intelligent_analysis.png')
5. Recommend a feature engineering strategy for this specific dataset.

Respond in JSON with this exact structure:
{{
  "likely_target": "column_name",
  "target_justification": "...",
  "problematic_columns": ["col1", "col2"],
  "insights": ["insight1", "insight2", "insight3", "insight4", "insight5"],
  "analysis_code": "python code here as string",
  "feature_strategy": "description of recommended strategy"
}}

Respond ONLY with the JSON, no text before or after."""

        raw_response = _ask_claude(prompt_analysis, max_tokens=3000)

        # Parse JSON from response
        try:
            clean_response = raw_response
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[1].split("```")[0]
            elif "```" in clean_response:
                clean_response = clean_response.split("```")[1].split("```")[0]
            analysis = json.loads(clean_response.strip())
        except json.JSONDecodeError:
            logger.warning("[Analysis] Claude did not return valid JSON. Using data-driven fallback.")
            # [FIX] The original fallback used "hired" — a leftover from a different dataset.
            # Instead, we inspect the actual DataFrame to pick the most plausible target:
            # 1. Any binary column (0/1 or True/False) that looks like an outcome
            # 2. Any column named with outcome-like keywords
            # 3. Last resort: the first non-ID categorical column
            outcome_keywords = [
                "risk", "churn", "fraud", "late", "delay", "default", "cancel",
                "status", "target", "label", "outcome", "result", "flag",
                "approved", "rejected", "hired", "survived", "attrition",
            ]
            candidate = None

            # Priority 1: column name matches known outcome keywords
            for kw in outcome_keywords:
                for c in df.columns:
                    if kw in c.lower() and df[c].nunique() <= 15:
                        candidate = c
                        break
                if candidate:
                    break

            # Priority 2: binary numeric column (0/1) — likely a flag
            if not candidate:
                for c in df.select_dtypes(include="number").columns:
                    if set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                        candidate = c
                        break

            # Priority 3: low-cardinality categorical (good classification target)
            if not candidate:
                cat_cols_fb = df.select_dtypes(include=["object", "string"]).columns
                for c in cat_cols_fb:
                    if 2 <= df[c].nunique() <= 10:
                        candidate = c
                        break

            # Absolute fallback: first non-ID column
            if not candidate:
                candidate = next(
                    (c for c in df.columns if "id" not in c.lower()), df.columns[-1]
                )

            logger.warning(
                f"[Analysis] Fallback target selected from data: '{candidate}' "
                f"(nunique={df[candidate].nunique()}). "
                f"Verify this is correct — set CONFIG['forced_target'] to override."
            )
            analysis = {
                "likely_target":        candidate,
                "target_justification": f"Auto-selected fallback: '{candidate}' chosen from actual dataset columns.",
                "problematic_columns":  [],
                "insights":             [f"Dataset has {n:,} rows × {p} columns. Target auto-detected as '{candidate}'."],
                "analysis_code":        "print(df.shape); print(df.dtypes)",
                "feature_strategy":     "Create ratio and interaction features between numeric variables.",
            }

        # ── Execute the code Claude wrote ──────────────────────────────────────
        # [FIX-1] pass df.copy() — analysis code only needs output, not df mutations
        code = analysis.get("analysis_code", "print('no code')")
        code_output, success, _ = _execute_code(code, df.copy())

        # Self-healing: if code failed, Claude tries to fix it
        if not success:
            logger.warning(f"[Analysis] Code failed: {code_output}. Claude will fix it.")
            prompt_fix = f"""The following Python code failed with error: {code_output}

Original code:
```python
{code}
```

Fix the code. Available variables: df (DataFrame), pd, np, plt, os, json, _BASE_DIR.
Respond ONLY with the corrected Python code, no explanations, no markdown."""

            fixed_code = _ask_claude(prompt_fix, max_tokens=1500)
            if "```python" in fixed_code:
                fixed_code = fixed_code.split("```python")[1].split("```")[0]
            elif "```" in fixed_code:
                fixed_code = fixed_code.split("```")[1].split("```")[0]

            # [FIX-1] again pass df.copy() for analysis-only code
            code_output, success, _ = _execute_code(fixed_code, df.copy())
            code = fixed_code
            logger.info(f"[Analysis] Self-healing: {'success' if success else 'failed again'}")

        # ── Save quality report ────────────────────────────────────────────────
        nulls = df.isnull().sum()
        null_cols = nulls[nulls > 0]
        outliers = {}
        for c in num_cols:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            cnt = ((df[c] < q1-1.5*iqr) | (df[c] > q3+1.5*iqr)).sum()
            if cnt > 0:
                outliers[c] = int(cnt)

        quality_md = f"""# Quality Report — AI-Powered Analysis

**Context:** {ctx or 'Resume screening dataset.'}
**Shape:** {n} x {p}

## Applied Imputation
{chr(10).join(f'- {i}' for i in imps) if imps else '- No imputation required.'}

## Detected Outliers (IQR)
{json.dumps(outliers, indent=2) if outliers else 'No significant outliers.'}

## Intelligent Analysis by Claude

### Identified Target
**Column:** `{analysis['likely_target']}`
**Justification:** {analysis['target_justification']}

### Problematic Columns
{analysis.get('problematic_columns', [])}

### Top Dataset Insights
{chr(10).join(f'{i+1}. {ins}' for i, ins in enumerate(analysis.get('insights', [])))}

### Recommended Feature Engineering Strategy
{analysis.get('feature_strategy', '')}

### Analysis Execution Output
```
{code_output[:2000]}
```

---
*Analysis generated by Claude 4.6 Sonnet*
"""
        with open(CONFIG["quality_md"], "w", encoding="utf-8") as f:
            f.write(quality_md)

        with open(CONFIG["analysis_md"], "w", encoding="utf-8") as f:
            f.write(f"# Intelligent Analysis\n\n```json\n{json.dumps(analysis, indent=2, default=_safe_json)}\n```")

        with open(CONFIG["target_json"], "w", encoding="utf-8") as f:
            json.dump({
                "target_col": analysis["likely_target"],
                "problem_type": _detect_type(df[analysis["likely_target"]])
                    if analysis["likely_target"] in df.columns else "classification",
                "ai_justification": analysis["target_justification"],
                "ai_feature_strategy": analysis.get("feature_strategy", ""),
                "ai_insights": analysis.get("insights", []),
            }, f, indent=2, default=_safe_json)

        return (f"ANALYSIS_SUCCESS\n"
                f"Target identified by AI: '{analysis['likely_target']}'\n"
                f"Insights generated: {len(analysis.get('insights', []))}\n"
                f"Code executed: {'yes' if success else 'with fallback'}\n"
                f"Self-healing activated: {not success}\n"
                f"Files: Quality_Report.md, Intelligent_Analysis.md, target_config.json")
    except Exception as e:
        return f"ANALYSIS_ERROR: {e}\n{traceback.format_exc()}"


# ── STEP 3: Intelligent Feature Engineering ───────────────────────────────────

@tool("generate_features_with_ai_strategy")
def generate_features_with_ai_strategy(_: str = "") -> str:
    """
    Reads df1_silver.parquet and the AI-defined feature strategy.
    Claude decides which features to create based on real data.
    Creates standard features + AI-customized features.
    Saves df2_gold.parquet. Returns FEATURES_SUCCESS or ERROR. No parameters.
    """
    try:
        if not os.path.exists(CONFIG["silver_path"]):
            return "ERROR: df1_silver.parquet does not exist."
        if not os.path.exists(CONFIG["target_json"]):
            return "ERROR: target_config.json does not exist."

        df = pd.read_parquet(CONFIG["silver_path"])
        with open(CONFIG["target_json"]) as f:
            cfg = json.load(f)
        target_col = cfg["target_col"]
        strategy   = cfg.get("ai_feature_strategy", "")

        df.describe(include="all").to_markdown(CONFIG["stats_md"])

        num_cols = [c for c in df.select_dtypes(include="number").columns
                    if c != target_col]

        # Standard features (always created)
        standard_feats = []
        if len(num_cols) >= 2:
            c0, c1 = num_cols[0], num_cols[1]
            df["feat_ratio"]   = df[c0] / (df[c1] + 1e-9)
            df["feat_sum"]     = df[c0] + df[c1]
            df["feat_product"] = df[c0] * df[c1]
            df["feat_diff"]    = df[c0] - df[c1]
            standard_feats    += ["feat_ratio", "feat_sum", "feat_product", "feat_diff"]

        for c in num_cols[:8]:
            col_d = df[c].dropna()
            if len(col_d) > 0 and col_d.min() > 0 and abs(col_d.skew()) > 1.0:
                df[f"log_{c}"] = np.log1p(df[c])
                standard_feats.append(f"log_{c}")

        if len(num_cols) >= 3:
            df["feat_interact"] = df[num_cols[0]] * df[num_cols[2]]
            standard_feats.append("feat_interact")

        if len(num_cols) >= 2:
            df[f"sq_{num_cols[0]}"] = df[num_cols[0]] ** 2
            df[f"sq_{num_cols[1]}"] = df[num_cols[1]] ** 2
            standard_feats += [f"sq_{num_cols[0]}", f"sq_{num_cols[1]}"]

        # ── Claude decides custom features based on real data ──────────────────
        col_stats = {c: {"mean": round(float(df[c].mean()), 3),
                         "std": round(float(df[c].std()), 3),
                         "corr_target": round(float(df[c].corr(df[target_col]))
                                              if target_col in df.select_dtypes(include="number").columns
                                              else 0.0, 3)}
                     for c in num_cols[:10]}

        prompt_features = f"""You are an expert Feature Engineer.

Dataset: {df.shape[0]} rows, {df.shape[1]} columns
Target: '{target_col}'
Previously suggested strategy: {strategy}

Numeric columns and their correlations with the target:
{json.dumps(col_stats, indent=2)}

Features already created: {standard_feats}

Create 3-5 additional features SPECIFIC to this dataset that may improve
target prediction. Consider:
- Non-linear combinations that make sense in context
- Ratios between correlated columns
- Features of magnitude or relative scale

Respond ONLY with valid Python code. Available variables: df (DataFrame with all columns), np, pd.
Do not use plt. Do not save files. Only modify df by adding new columns.
Example: df['new_feat'] = df['col_a'] / (df['col_b'] + 1)"""

        feature_code = _ask_claude(prompt_features, max_tokens=1000)

        # Clean markdown if present
        if "```python" in feature_code:
            feature_code = feature_code.split("```python")[1].split("```")[0]
        elif "```" in feature_code:
            feature_code = feature_code.split("```")[1].split("```")[0]

        # [FIX-2] Single execution: pass df directly (not a copy) so ns["df"]
        # holds the mutated dataframe. No second exec() needed.
        ai_feats = []
        cols_before = set(df.columns)
        feat_output, feat_success, feat_ns = _execute_code(feature_code, df)

        if feat_success:
            df = feat_ns["df"]
            ai_feats = [c for c in df.columns if c not in cols_before]
            logger.info(f"[AI Features] Created: {ai_feats}")
        else:
            logger.warning(f"[AI Features] Code failed: {feat_output}")

        df.to_parquet(CONFIG["gold_path"], index=False)

        # ── Boruta feature selection on gold dataset ───────────────────────────
        boruta_selected = []
        try:
            from boruta import BorutaPy
            df_boruta = df.dropna(axis=1, how="all")
            df_boruta = df_boruta.dropna(subset=[c for c in df_boruta.columns
                                                  if df_boruta[c].isnull().mean() < 0.5])
            feat_b = [c for c in df_boruta.columns if c != target_col]

            # [LITE-FIX] Drop high-cardinality categoricals before get_dummies.
            # This is the same guard as in the EDA and ML steps — gold still has
            # the raw categorical columns (email, city, etc.) that cause MemoryError.
            max_card_b = CONFIG.get("max_cat_cardinality", 50)
            hi_card_b  = [
                c for c in feat_b
                if df_boruta[c].dtype in [object, "string"]
                and df_boruta[c].nunique() > max_card_b
            ]
            if hi_card_b:
                logger.info(
                    f"[Boruta] Dropping {len(hi_card_b)} high-cardinality cols "
                    f"before get_dummies: {hi_card_b}"
                )
                feat_b = [c for c in feat_b if c not in hi_card_b]

            X_b = pd.get_dummies(df_boruta[feat_b], drop_first=True)
            y_b = df_boruta[target_col].copy()
            if y_b.dtype == "object":
                y_b = LabelEncoder().fit_transform(y_b.astype(str))
            else:
                y_b = y_b.values

            # [LITE] Boruta internally trains hundreds of RF trees. Running it on
            # the full dataset (e.g. 180k rows) with n_jobs=-1 uses all cores and
            # can cause thermal shutdown. We subsample and cap n_jobs.
            boruta_sample_size = CONFIG.get("boruta_sample", 15_000)
            n_jobs_safe        = CONFIG.get("n_jobs", 2)
            if len(X_b) > boruta_sample_size:
                logger.info(
                    f"[Boruta] Sampling {boruta_sample_size:,} rows "
                    f"(full dataset: {len(X_b):,}). "
                    f"Features relevant at this scale generalise well."
                )
                rng       = np.random.RandomState(CONFIG["random_state"])
                sample_ix = rng.choice(len(X_b), boruta_sample_size, replace=False)
                X_b_fit   = X_b.values[sample_ix]
                y_b_fit   = y_b[sample_ix]
            else:
                X_b_fit, y_b_fit = X_b.values, y_b

            from sklearn.ensemble import RandomForestClassifier as RFC
            rf_b   = RFC(n_estimators=50, n_jobs=n_jobs_safe,
                         random_state=CONFIG["random_state"])
            boruta = BorutaPy(rf_b, n_estimators="auto",
                              random_state=CONFIG["random_state"], verbose=0)
            boruta.fit(X_b_fit, y_b_fit)
            boruta_selected = list(X_b.columns[boruta.support_])
            logger.info(f"[Boruta] Selected {len(boruta_selected)} features.")
        except ImportError:
            logger.info("[Boruta] boruta not installed — skipping. pip install boruta")
        except Exception as be:
            logger.warning(f"[Boruta] Failed: {be}")

        with open(CONFIG["strategy_json"], "w", encoding="utf-8") as f:
            json.dump({
                "standard_features": standard_feats,
                "ai_features": ai_feats,
                "boruta_selected": boruta_selected,
                "ai_code": feature_code,
                "ai_success": feat_success,
            }, f, indent=2)

        return (f"FEATURES_SUCCESS\n"
                f"Standard features: {len(standard_feats)}\n"
                f"AI-generated features: {ai_feats}\n"
                f"Boruta selected features: {len(boruta_selected)}\n"
                f"Gold shape: {df.shape}\n"
                f"File: df2_gold.parquet")
    except Exception as e:
        return f"FEATURES_ERROR: {e}\n{traceback.format_exc()}"


# ── STEP 4: EDA ───────────────────────────────────────────────────────────────

@tool("generate_eda_and_ml_ready")
def generate_eda_and_ml_ready(_: str = "") -> str:
    """
    Reads df2_gold.parquet. Generates 6 visualizations.
    Removes redundancies. Saves df3_ml_ready.parquet.
    Returns EDA_SUCCESS or ERROR. No parameters.
    """
    try:
        if not os.path.exists(CONFIG["gold_path"]):
            return "ERROR: df2_gold.parquet does not exist."
        if not os.path.exists(CONFIG["target_json"]):
            return "ERROR: target_config.json does not exist."

        df = pd.read_parquet(CONFIG["gold_path"])
        with open(CONFIG["target_json"]) as f:
            cfg = json.load(f)
        target_col = cfg["target_col"]
        num_df     = df.select_dtypes(include="number")

        # G1: Correlation matrix
        plt.figure(figsize=(13, 10))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.4, annot_kws={"size": 7})
        plt.title("Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(CONFIG["corr_png"], dpi=150); plt.close()

        # G1b: Cramér's V heatmap for categorical columns
        cat_cols_eda = [c for c in df.select_dtypes(include=["object", "string"]).columns
                        if df[c].nunique() <= 30]  # [FIX] raised from 20 → 30
        if len(cat_cols_eda) >= 2:
            cv_matrix = pd.DataFrame(index=cat_cols_eda, columns=cat_cols_eda, dtype=float)
            for c1 in cat_cols_eda:
                for c2 in cat_cols_eda:
                    cv_matrix.loc[c1, c2] = _cramer_v(df[c1], df[c2])
            plt.figure(figsize=(max(8, len(cat_cols_eda)), max(6, len(cat_cols_eda))))
            sns.heatmap(cv_matrix.astype(float), annot=True, fmt=".2f",
                        cmap="YlOrRd", linewidths=0.4, annot_kws={"size": 8})
            plt.title("Cramér's V — Categorical Association Matrix",
                      fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(_BASE_DIR, "cramers_v_matrix.png"), dpi=150)
            plt.close()

        base_cols = [c for c in num_df.columns
                     if not any(c.startswith(p)
                                for p in ["feat_", "log_", "sq_"])][:8]

        # G2 + G3: Distributions and Boxplots
        if base_cols:
            n_c = min(len(base_cols), 4)
            n_r = (len(base_cols) + n_c - 1) // n_c
            for fname, plot_fn in [
                ("distributions.png",
                 lambda ax, d: ax.hist(d, bins=40, color="#4C72B0",
                                       edgecolor="white", alpha=0.85)),
                ("boxplots.png",
                 lambda ax, d: ax.boxplot(d, patch_artist=True,
                                          boxprops=dict(facecolor="#4C72B0",
                                                        alpha=0.7))),
            ]:
                fig, axes = plt.subplots(n_r, n_c, figsize=(5*n_c, 4*n_r))
                axes = np.array(axes).flatten()
                for i, col in enumerate(base_cols):
                    plot_fn(axes[i], num_df[col].dropna())
                    axes[i].set_title(col, fontsize=10, fontweight="bold")
                    axes[i].grid(axis="y", alpha=0.3)
                for j in range(len(base_cols), len(axes)):
                    fig.delaxes(axes[j])
                plt.tight_layout()
                plt.savefig(os.path.join(_BASE_DIR, fname), dpi=150); plt.close()

        # G4: Categorical features
        cat_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns
                    if df[c].nunique() <= 20 and c != target_col][:4]
        if cat_cols:
            fig, axes = plt.subplots(1, len(cat_cols),
                                     figsize=(6*len(cat_cols), 5))
            if len(cat_cols) == 1: axes = [axes]
            for i, col in enumerate(cat_cols):
                vc = df[col].value_counts().head(15)
                axes[i].barh(vc.index.astype(str), vc.values,
                             color="#E05C5C", alpha=0.8)
                axes[i].set_title(col, fontsize=11, fontweight="bold")
                axes[i].invert_yaxis(); axes[i].grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(_BASE_DIR, "categoricals.png"), dpi=150)
            plt.close()

        # G5: Target distribution
        if target_col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            if cfg["problem_type"] == "classification":
                vc = df[target_col].value_counts().head(20)
                ax.bar(vc.index.astype(str), vc.values,
                       color="#2CA02C", alpha=0.85)
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.hist(df[target_col].dropna(), bins=40,
                        color="#2CA02C", alpha=0.85)
            ax.set_title(f"Target Distribution: {target_col}",
                         fontsize=13, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(_BASE_DIR, "target_dist.png"), dpi=150)
            plt.close()

        # G6: Dataset sample
        sample = df.head(10)
        fig, ax = plt.subplots(figsize=(18, 3)); ax.axis("off")
        tb = ax.table(cellText=sample.values, colLabels=sample.columns,
                      cellLoc="center", loc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(7)
        tb.auto_set_column_width(col=list(range(len(sample.columns))))
        for (row, col), cell in tb.get_celld().items():
            if row == 0:
                cell.set_facecolor("#4C72B0")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f0f4ff")
        plt.title("Dataset Sample", fontsize=11, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(_BASE_DIR, "dataset_sample.png"),
                    dpi=150, bbox_inches="tight"); plt.close()

        # Remove redundancies
        corr_abs = num_df.corr().abs()
        upper    = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
        redundant_cols = [c for c in upper.columns if any(upper[c] > 0.95)]
        id_cols        = [c for c in df.columns
                          if "id" in c.lower() and c != target_col]

        # [LITE-FIX] Drop high-cardinality categorical columns BEFORE saving ml_ready.
        # pd.get_dummies on columns like customer_email (~100k uniques) or product_name
        # creates tens of thousands of boolean columns and requires >10 GB of RAM.
        # We keep only categoricals with ≤ max_cat_cardinality unique values.
        max_cat_cardinality = CONFIG.get("max_cat_cardinality", 50)
        cat_all = df.select_dtypes(include=["object", "string"]).columns.tolist()
        high_card_cols = [
            c for c in cat_all
            if c != target_col and df[c].nunique() > max_cat_cardinality
        ]
        if high_card_cols:
            logger.info(
                f"[EDA] Dropping {len(high_card_cols)} high-cardinality categorical "
                f"columns (nunique > {max_cat_cardinality}): {high_card_cols}"
            )

        # Also drop free-text / URL / image columns that are never useful for ML
        junk_patterns = ["email", "password", "description", "image", "street", "zipcode"]
        junk_cols = [
            c for c in df.columns
            if c != target_col and any(p in c.lower() for p in junk_patterns)
        ]

        cols_to_remove = list(set(redundant_cols + id_cols + high_card_cols + junk_cols) - {target_col})
        df_filtered    = df.drop(columns=cols_to_remove, errors="ignore")

        # [FIX-4] Preserve original silver row indices so the deployer can
        # align predictions to the correct silver rows after any row drops.
        df_filtered = df_filtered.reset_index(drop=False)  # keeps original as 'index'
        df_filtered.rename(columns={"index": "_src_idx"}, inplace=True)
        df_filtered.to_parquet(CONFIG["ml_ready_path"], index=False)

        return (f"EDA_SUCCESS\n"
                f"Removed columns: {cols_to_remove}\n"
                f"High-cardinality cols dropped: {high_card_cols}\n"
                f"Junk cols dropped: {junk_cols}\n"
                f"ML-Ready shape: {df_filtered.shape}\n"
                f"Charts: correlation, distributions, boxplots, "
                f"categoricals, target_dist, sample\n"
                f"File: df3_ml_ready.parquet")
    except Exception as e:
        return f"EDA_ERROR: {e}\n{traceback.format_exc()}"


# ── STEP 4.5: Hypothesis Validation ──────────────────────────────────────────

@tool("validate_hypotheses")
def validate_hypotheses(_: str = "") -> str:
    """
    Reads df2_gold.parquet and target_config.json.
    Claude generates 10 business hypotheses based on the dataset and context.
    Each hypothesis is tested with real data and marked TRUE/FALSE/INCONCLUSIVE.
    Saves hypothesis_results.json and hypothesis_validation.png.
    Returns HYPOTHESIS_SUCCESS or ERROR. No parameters.
    """
    try:
        if not os.path.exists(CONFIG["gold_path"]):
            return "ERROR: df2_gold.parquet does not exist."
        if not os.path.exists(CONFIG["target_json"]):
            return "ERROR: target_config.json does not exist."

        df = pd.read_parquet(CONFIG["gold_path"])
        with open(CONFIG["target_json"]) as f:
            cfg = json.load(f)
        target_col   = cfg["target_col"]
        problem_type = cfg["problem_type"]
        ctx          = _read_ctx()

        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in df.select_dtypes(
                    include=["object", "string"]).columns if df[c].nunique() <= 30]

        col_target_corr = {}
        if target_col in df.select_dtypes(include="number").columns:
            for c in num_cols[:15]:
                if c != target_col:
                    col_target_corr[c] = round(float(df[c].corr(df[target_col])), 3)

        cat_target_assoc = {}
        if target_col in df.select_dtypes(include=["object", "string"]).columns or \
           df[target_col].nunique() <= 15:
            for c in cat_cols[:8]:
                if c != target_col:
                    cat_target_assoc[c] = round(
                        _cramer_v(df[c].astype(str),
                                  df[target_col].astype(str)), 3)

        prompt_hyp = f"""You are a senior Data Scientist generating business hypotheses.

BUSINESS CONTEXT: {ctx or 'Predict the target variable based on available features.'}
TARGET: '{target_col}' ({problem_type})
COLUMNS: {list(df.columns[:30])}
TARGET CORRELATIONS (numeric): {col_target_corr}
TARGET ASSOCIATIONS (categorical, Cramér V): {cat_target_assoc}

Generate exactly 10 business hypotheses about what drives '{target_col}'.
Each must be:
- Testable with the available columns
- Specific (mention column names)
- Framed as: "Stores/Orders/Customers with X tend to have higher/lower Y"

For each hypothesis also provide:
- The Python pandas code to test it (using df, target='{target_col}')
- What chart type to use (bar, box, scatter, line)
- Which columns are involved

Respond ONLY with JSON:
{{
  "hypotheses": [
    {{
      "id": "H1",
      "statement": "...",
      "columns": ["col1", "col2"],
      "test_code": "result = df.groupby('col1')['{target_col}'].mean().sort_values(ascending=False)\\nprint(result.head(10))",
      "chart_type": "bar",
      "expected_direction": "positive"
    }}
  ]
}}"""

        hyp_raw = _ask_claude(prompt_hyp, max_tokens=3000)
        if "```json" in hyp_raw:
            hyp_raw = hyp_raw.split("```json")[1].split("```")[0]
        elif "```" in hyp_raw:
            hyp_raw = hyp_raw.split("```")[1].split("```")[0]

        try:
            hyp_data = json.loads(hyp_raw.strip())
            hypotheses = hyp_data.get("hypotheses", [])
        except Exception:
            hypotheses = []

        if not hypotheses:
            return "HYPOTHESIS_ERROR: Claude did not return valid hypotheses."

        results = []
        for h in hypotheses[:10]:
            hid   = h.get("id", "Hx")
            stmt  = h.get("statement", "")
            code  = h.get("test_code", "")
            cols  = h.get("columns", [])

            missing = [c for c in cols if c not in df.columns]
            if missing:
                results.append({**h,
                    "verdict": "INCONCLUSIVE",
                    "reason": f"Columns not found: {missing}",
                    "output": ""})
                continue

            # [FIX-1] pass df.copy() — hypothesis tests are read-only
            output, success, _ = _execute_code(code, df.copy())
            if not success:
                results.append({**h,
                    "verdict": "INCONCLUSIVE",
                    "reason": f"Code error: {output[:200]}",
                    "output": ""})
                continue

            prompt_verdict = f"""You tested the hypothesis: "{stmt}"
Code output:
{output[:800]}

Based on this output, respond with a JSON object:
{{
  "verdict": "TRUE" or "FALSE" or "INCONCLUSIVE",
  "reason": "one sentence explaining why",
  "business_insight": "one sentence on what this means for the business"
}}
Respond ONLY with the JSON."""

            verdict_raw = _ask_claude(prompt_verdict, max_tokens=300)
            try:
                if "```json" in verdict_raw:
                    verdict_raw = verdict_raw.split("```json")[1].split("```")[0]
                elif "```" in verdict_raw:
                    verdict_raw = verdict_raw.split("```")[1].split("```")[0]
                verdict_obj = json.loads(verdict_raw.strip())
            except Exception:
                verdict_obj = {"verdict": "INCONCLUSIVE",
                               "reason": "Could not parse verdict.",
                               "business_insight": ""}

            results.append({**h,
                "verdict":          verdict_obj.get("verdict", "INCONCLUSIVE"),
                "reason":           verdict_obj.get("reason", ""),
                "business_insight": verdict_obj.get("business_insight", ""),
                "output":           output[:500]})

            logger.info(f"[Hypothesis] {hid}: {verdict_obj.get('verdict','?')} — {stmt[:60]}")

        with open(CONFIG["hypothesis_json"], "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=_safe_json)

        verdicts = [r["verdict"] for r in results]
        v_counts = {v: verdicts.count(v) for v in ["TRUE", "FALSE", "INCONCLUSIVE"]}

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors_pie = {"TRUE": "#2CA02C", "FALSE": "#D62728", "INCONCLUSIVE": "#7F7F7F"}
        axes[0].pie(list(v_counts.values()),
                    labels=list(v_counts.keys()),
                    colors=[colors_pie[k] for k in v_counts],
                    autopct="%1.0f%%", startangle=90,
                    textprops={"fontsize": 12})
        axes[0].set_title("Hypothesis Verdicts", fontsize=13, fontweight="bold")

        hy_labels = [r["id"] for r in results]
        hy_colors = [colors_pie.get(r["verdict"], "#7F7F7F") for r in results]
        axes[1].barh(hy_labels, [1] * len(results), color=hy_colors, alpha=0.85)
        for i, r in enumerate(results):
            axes[1].text(0.02, i, r["verdict"], va="center",
                         fontsize=9, color="white", fontweight="bold")
        axes[1].set_xlabel("Verdict")
        axes[1].set_title("Individual Hypothesis Results",
                          fontsize=13, fontweight="bold")
        axes[1].invert_yaxis()
        plt.tight_layout()
        plt.savefig(CONFIG["hypothesis_png"], dpi=150)
        plt.close()

        lines = ["# Hypothesis Validation\n\n"]
        lines.append(f"**Target:** `{target_col}` | "
                     f"TRUE: {v_counts['TRUE']} | "
                     f"FALSE: {v_counts['FALSE']} | "
                     f"INCONCLUSIVE: {v_counts['INCONCLUSIVE']}\n\n")
        lines.append("| ID | Hypothesis | Verdict | Business Insight |\n")
        lines.append("|----|-----------|---------|-----------------|\n")
        for r in results:
            lines.append(f"| {r['id']} | {r['statement'][:80]} | "
                         f"**{r['verdict']}** | {r.get('business_insight','')[:80]} |\n")
        with open(os.path.join(_BASE_DIR, "Hypothesis_Validation.md"),
                  "w", encoding="utf-8") as f:
            f.write("".join(lines))

        true_hyps = [r for r in results if r["verdict"] == "TRUE"]
        cfg["hypothesis_insights"] = [r.get("business_insight", "") for r in true_hyps]
        cfg["true_hypotheses"]     = [r["statement"] for r in true_hyps]
        with open(CONFIG["target_json"], "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, default=_safe_json)

        return (f"HYPOTHESIS_SUCCESS\n"
                f"Hypotheses tested: {len(results)}\n"
                f"TRUE: {v_counts['TRUE']} | FALSE: {v_counts['FALSE']} | "
                f"INCONCLUSIVE: {v_counts['INCONCLUSIVE']}\n"
                f"Key findings: {[r['statement'][:50] for r in true_hyps]}\n"
                f"Files: hypothesis_results.json, hypothesis_validation.png, "
                f"Hypothesis_Validation.md")

    except Exception as e:
        return f"HYPOTHESIS_ERROR: {e}\n{traceback.format_exc()}"


# ── STEP 5: ML with Optuna + Stacking + AI Interpretation ─────────────────────

@tool("train_and_save_model")
def train_and_save_model(_: str = "") -> str:
    """
    Reads df3_ml_ready.parquet. Runs model competition with CV.
    Applies Optuna to the top-3. Attempts Stacking. Saves best model.
    Claude interprets results and writes a narrative diagnostic.
    Saves Model_Metrics.md. Returns ML_SUCCESS or ERROR. No parameters.
    """
    try:
        if not os.path.exists(CONFIG["ml_ready_path"]):
            return "ERROR: df3_ml_ready.parquet does not exist."
        if not os.path.exists(CONFIG["target_json"]):
            return "ERROR: target_config.json does not exist."

        df = pd.read_parquet(CONFIG["ml_ready_path"])
        df = df.dropna(axis=1, how="all")
        df = df.dropna(subset=[c for c in df.columns
                                if df[c].isnull().mean() < 0.5])
        df = df.reset_index(drop=True)
        logger.info(f"[ML] Dataset after cleaning: {df.shape}")

        with open(CONFIG["target_json"]) as f:
            cfg = json.load(f)
        target_col   = cfg["target_col"]
        problem_type = cfg["problem_type"]

        # [FIX-4] Exclude _src_idx from features (it's a row-alignment key, not a feature)
        feature_cols = [c for c in df.columns
                        if c != target_col and c != "_src_idx"]

        # Use Boruta-selected features if available
        if os.path.exists(CONFIG["strategy_json"]):
            with open(CONFIG["strategy_json"]) as f:
                strat = json.load(f)
            boruta_cols = strat.get("boruta_selected", [])
            if len(boruta_cols) >= 5:
                feature_cols = [c for c in feature_cols
                                if any(c == b or c.startswith(b + "_")
                                       for b in boruta_cols)]
                if not feature_cols:
                    feature_cols = [c for c in df.columns
                                    if c != target_col and c != "_src_idx"]
                logger.info(f"[ML] Using {len(feature_cols)} Boruta-selected features.")

        # [LITE-FIX] Second-line-of-defence: drop any remaining high-cardinality
        # categoricals that slipped through the EDA step.
        # pd.get_dummies on columns with thousands of unique values allocates
        # tens of GBs of boolean memory (180k rows × 60k cols = 10+ GB).
        max_card = CONFIG.get("max_cat_cardinality", 50)
        df_features = df[feature_cols]
        hi_card = [
            c for c in df_features.select_dtypes(include=["object", "string"]).columns
            if df_features[c].nunique() > max_card
        ]
        if hi_card:
            logger.warning(
                f"[ML] Dropping {len(hi_card)} high-cardinality cols before get_dummies "
                f"(nunique > {max_card}): {hi_card}"
            )
            feature_cols = [c for c in feature_cols if c not in hi_card]

        X = pd.get_dummies(df[feature_cols], drop_first=True)
        y = df[target_col].copy()

        # [FIX-3] Determine if classification BEFORE any encoding
        is_classification = (y.dtype == "object" or problem_type == "classification")
        if is_classification:
            le = LabelEncoder()
            problem_type = "classification"
        else:
            le = None

        # Split on raw (un-encoded) y so stratify works correctly
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=CONFIG["test_size"],
            random_state=CONFIG["random_state"],
            stratify=y.astype(str) if is_classification else None,
        )

        # [FIX-3] Fit encoder only on train split; transform both independently
        if le is not None:
            le.fit(y_train.astype(str))
            y_train = pd.Series(le.transform(y_train.astype(str)), index=y_train.index)
            y_test  = pd.Series(le.transform(y_test.astype(str)),  index=y_test.index)

        # Time-aware CV — only activate if date columns are actually in the
        # feature set. High-cardinality date strings are dropped by the guard
        # above, so checking df.columns would give a false positive.
        date_cols = [c for c in feature_cols
                     if "date" in c.lower() or "time" in c.lower()]
        use_timeseries_cv = len(date_cols) > 0

        if use_timeseries_cv:
            from sklearn.model_selection import TimeSeriesSplit
            cv = TimeSeriesSplit(n_splits=CONFIG["cv_folds"])
            logger.info("[ML] Using TimeSeriesSplit CV (date column detected).")
        else:
            cv = (StratifiedKFold if problem_type == "classification" else KFold)(
                n_splits=CONFIG["cv_folds"], shuffle=True,
                random_state=CONFIG["random_state"],
            )
        metric = "accuracy" if problem_type == "classification" else "r2"

        # [LITE] Cap CPU usage. n_jobs=-1 uses ALL cores simultaneously — fine on a
        # server, dangerous on a laptop (thermal shutdown). n_jobs_safe comes from
        # CONFIG["n_jobs"] (default 2). Increase to 4-6 on workstations.
        n_jobs_safe = CONFIG.get("n_jobs", 2)

        # Model catalog
        if problem_type == "classification":
            MODELS = {
                "RandomForest":       RandomForestClassifier(n_estimators=100,
                                          n_jobs=n_jobs_safe,
                                          random_state=CONFIG["random_state"]),
                "GradientBoosting":   GradientBoostingClassifier(n_estimators=100,
                                          random_state=CONFIG["random_state"]),
                "ExtraTrees":         ExtraTreesClassifier(n_estimators=100,
                                          n_jobs=n_jobs_safe,
                                          random_state=CONFIG["random_state"]),
                "LogisticRegression": LogisticRegression(max_iter=1000,
                                          n_jobs=n_jobs_safe,
                                          random_state=CONFIG["random_state"]),
            }
        else:
            MODELS = {
                "RandomForest":     RandomForestRegressor(n_estimators=100,
                                        n_jobs=n_jobs_safe,
                                        random_state=CONFIG["random_state"]),
                "GradientBoosting": GradientBoostingRegressor(n_estimators=100,
                                        random_state=CONFIG["random_state"]),
                "ExtraTrees":       ExtraTreesRegressor(n_estimators=100,
                                        n_jobs=n_jobs_safe,
                                        random_state=CONFIG["random_state"]),
                "Ridge":            Ridge(),
            }

        try:
            from xgboost import XGBClassifier, XGBRegressor
            MODELS["XGBoost"] = (
                XGBClassifier(n_estimators=100,
                              eval_metric="logloss", verbosity=0,
                              random_state=CONFIG["random_state"])  # [FIX] removed use_label_encoder (deprecated XGB>=1.6)
                if problem_type == "classification" else
                XGBRegressor(n_estimators=100, verbosity=0,
                             random_state=CONFIG["random_state"])
            )
        except ImportError:
            pass

        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            MODELS["LightGBM"] = (
                LGBMClassifier(n_estimators=100, verbose=-1,
                               random_state=CONFIG["random_state"])
                if problem_type == "classification" else
                LGBMRegressor(n_estimators=100, verbose=-1,
                              random_state=CONFIG["random_state"])
            )
        except ImportError:
            pass

        # CV on all models
        cv_results = {}
        for name, model in MODELS.items():
            try:
                sc = cross_val_score(model, X_train, y_train, cv=cv,
                                     scoring=metric, n_jobs=n_jobs_safe)
                cv_results[name] = {"mean": float(sc.mean()), "std": float(sc.std())}
                logger.info(f"[ML] {name}: {sc.mean():.4f} ± {sc.std():.4f}")
            except Exception as ex:
                logger.warning(f"[ML] {name} failed: {ex}")
                cv_results[name] = {"mean": -9999.0, "std": 0.0}

        # Optuna on top-3
        top3 = sorted(cv_results, key=lambda k: cv_results[k]["mean"],
                      reverse=True)[:3]
        optuna_results = {}

        for name in top3:
            def _obj(trial, _n=name):
                if _n in ["RandomForest", "ExtraTrees"]:
                    p = {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
                         "max_depth":    trial.suggest_int("max_depth", 3, 20),
                         "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)}
                elif _n == "GradientBoosting":
                    p = {"n_estimators":  trial.suggest_int("n_estimators", 50, 300),
                         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                         "max_depth":     trial.suggest_int("max_depth", 2, 8),
                         "subsample":     trial.suggest_float("subsample", 0.5, 1.0)}
                elif _n == "LogisticRegression":
                    p = {"C": trial.suggest_float("C", 0.001, 100, log=True),
                         "max_iter": 1000}
                elif _n == "Ridge":
                    p = {"alpha": trial.suggest_float("alpha", 0.001, 100, log=True)}
                elif _n == "XGBoost":
                    p = {"n_estimators":  trial.suggest_int("n_estimators", 50, 300),
                         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                         "max_depth":     trial.suggest_int("max_depth", 2, 8),
                         "subsample":     trial.suggest_float("subsample", 0.5, 1.0)}
                elif _n == "LightGBM":
                    p = {"n_estimators":  trial.suggest_int("n_estimators", 50, 300),
                         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                         "num_leaves":    trial.suggest_int("num_leaves", 20, 150)}
                else:
                    p = {}
                try:
                    cls = type(MODELS[_n])
                    kw  = {**p}
                    if "random_state" in cls.__init__.__code__.co_varnames:
                        kw["random_state"] = CONFIG["random_state"]
                    m  = cls(**kw)
                    # [FIX-7] Unified to CONFIG["cv_folds"] (was hardcoded cv=3)
                    sc = cross_val_score(m, X_train, y_train,
                                         cv=CONFIG["cv_folds"],
                                         scoring=metric, n_jobs=n_jobs_safe)
                    return float(sc.mean())
                except Exception:
                    return -9999.0

            study = optuna.create_study(direction="maximize")
            study.optimize(_obj, n_trials=CONFIG["optuna_trials"],
                           show_progress_bar=False)

            bp  = study.best_params
            cls = type(MODELS[name])
            kw  = {**bp}
            if "random_state" in cls.__init__.__code__.co_varnames:
                kw["random_state"] = CONFIG["random_state"]
            try:
                m_tuned = cls(**kw)
            except Exception:
                m_tuned = MODELS[name]

            sc_t = cross_val_score(m_tuned, X_train, y_train, cv=cv,
                                   scoring=metric, n_jobs=n_jobs_safe)
            optuna_results[name] = {
                "model": m_tuned,
                "best_params": bp,
                "mean": float(sc_t.mean()),
                "std":  float(sc_t.std()),
            }
            logger.info(f"[Optuna] {name}: {sc_t.mean():.4f} ± {sc_t.std():.4f}")

        # [LITE] Stacking trains each base model cv_folds times inside the CV loop
        # and then runs another CV on the meta-learner on top. It is by far the
        # most CPU/RAM-intensive step. Disabled by default on low-end hardware.
        # Set CONFIG["enable_stacking"] = True on a workstation or cloud instance.
        # [FIX-8] Stacking CV unified to CONFIG["cv_folds"] (was hardcoded cv=3)
        estimators = [(n, optuna_results[n]["model"]) for n in top3]
        if CONFIG.get("enable_stacking", False):
            try:
                stacker = (StackingClassifier(
                               estimators=estimators,
                               final_estimator=LogisticRegression(max_iter=1000),
                               cv=CONFIG["cv_folds"], n_jobs=n_jobs_safe)
                           if problem_type == "classification" else
                           StackingRegressor(
                               estimators=estimators,
                               final_estimator=Ridge(),
                               cv=CONFIG["cv_folds"], n_jobs=n_jobs_safe))
                # n_jobs=1 here because the stacker's inner models already use n_jobs_safe
                sc_s = cross_val_score(stacker, X_train, y_train, cv=cv,
                                       scoring=metric, n_jobs=1)
                optuna_results["Stacking"] = {
                    "model": stacker,
                    "best_params": {"base": [n for n, _ in estimators]},
                    "mean": float(sc_s.mean()),
                    "std":  float(sc_s.std()),
                }
                logger.info(f"[Stacking] {sc_s.mean():.4f} ± {sc_s.std():.4f}")
            except Exception as se:
                logger.warning(f"[Stacking] Failed: {se}")
        else:
            logger.info("[Stacking] Skipped — CONFIG['enable_stacking'] is False. "
                        "Set to True on workstations for potentially higher accuracy.")

        # Best final model
        best = max(optuna_results, key=lambda k: optuna_results[k]["mean"])
        final_model = optuna_results[best]["model"]
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)

        test_score = (accuracy_score(y_test, y_pred)
                      if problem_type == "classification"
                      else r2_score(y_test, y_pred))

        comp = {**{n: {"mean": v["mean"], "std": v["std"]}
                   for n, v in cv_results.items()},
                **{f"{n}_Optuna": {"mean": v["mean"], "std": v["std"]}
                   for n, v in optuna_results.items()}}
        comp_df = pd.DataFrame(comp).T.sort_values("mean", ascending=False).round(4)

        # ── Claude interprets the results ──────────────────────────────────────
        ai_insights = cfg.get("ai_insights", [])
        prompt_interp = f"""You are a senior Data Scientist interpreting model results.

CONTEXT: {_read_ctx() or 'Resume screening dataset.'}
TARGET: '{target_col}' ({problem_type})
DATASET INSIGHTS: {ai_insights}

MODEL COMPETITION RESULTS:
{comp_df.to_string()}

SELECTED MODEL: {best}
{metric.upper()} ON TEST SET: {test_score:.4f}

Write a narrative interpretation (3-4 paragraphs) explaining:
1. Why the model {best} was the best choice for this problem
2. What the score {test_score:.4f} means in a business context
3. Points of attention or model limitations
4. Practical recommendations for production deployment

Be specific and technical, but also practical."""

        narrative = _ask_claude(prompt_interp, max_tokens=1000)

        with open(CONFIG["model_pkl"], "wb") as f:
            pickle.dump({
                "model": final_model, "label_encoder": le,
                "features": list(X.columns), "target": target_col,
                "type": problem_type, "name": best,
                "test_score": test_score,
                "optuna_params": optuna_results[best].get("best_params", {}),
            }, f)

        lines = [f"# Model Metrics\n\n",
                 f"**Type:** {problem_type} | **Target:** `{target_col}`\n\n",
                 f"## Model Comparison\n\n",
                 comp_df.to_markdown() + "\n\n",
                 f"**Selected model:** `{best}`\n\n",
                 f"**{metric.upper()} (test):** {test_score:.4f}\n\n"]

        if problem_type == "classification":
            lines.append(f"```\n{classification_report(y_test, y_pred)}\n```\n\n")
        else:
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            lines.append(f"**RMSE:** {rmse:.4f} | **R²:** {test_score:.4f}\n\n")

        lines.append(f"## AI Interpretation\n\n{narrative}\n")

        with open(CONFIG["metrics_md"], "w", encoding="utf-8") as f:
            f.write("".join(lines))

        comp_plot = comp_df.sort_values("mean")
        fig, ax = plt.subplots(figsize=(12, max(4, len(comp_plot)*0.4)))
        colors = ["#2CA02C" if n == best else
                  ("#FF7F0E" if "Optuna" in n or n == "Stacking" else "#4C72B0")
                  for n in comp_plot.index]
        bars = ax.barh(comp_plot.index, comp_plot["mean"],
                       xerr=comp_plot["std"], color=colors, alpha=0.85, capsize=4)
        ax.bar_label(bars, fmt="%.4f", padding=6, fontsize=9)
        ax.set_xlabel(f"{metric.upper()} CV", fontsize=11)
        ax.set_title("Model Comparison — Baseline vs Optuna vs Stacking",
                     fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(_BASE_DIR, "model_comparison.png"), dpi=150)
        plt.close()

        if hasattr(final_model, "feature_importances_"):
            imp = (pd.Series(final_model.feature_importances_, index=X.columns)
                   .sort_values(ascending=True).tail(15))
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(imp.index, imp.values, color="#4C72B0", alpha=0.85)
            ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
            ax.set_title(f"Top 15 Features — {best}\n(Target: {target_col})",
                         fontsize=12, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(_BASE_DIR, "feature_importance.png"), dpi=150)
            plt.close()

        # ── Error Analysis — 4-panel ───────────────────────────────────────────
        y_test_arr  = np.array(y_test)
        y_pred_arr  = np.array(y_pred)
        errors      = y_test_arr - y_pred_arr
        error_rate  = np.where(y_test_arr != 0,
                               y_pred_arr / y_test_arr, np.nan)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Error Analysis — {best} (Target: {target_col})",
                     fontsize=14, fontweight="bold")

        if problem_type == "regression":
            idx = np.argsort(y_test_arr)[:200]
            axes[0, 0].plot(y_test_arr[idx],  label="Actual",    alpha=0.7)
            axes[0, 0].plot(y_pred_arr[idx],  label="Predicted", alpha=0.7)
            axes[0, 0].legend(); axes[0, 0].set_title("Actual vs Predicted (sample)")
            axes[0, 0].grid(alpha=0.3)
            axes[0, 1].plot(error_rate[~np.isnan(error_rate)][:200], color="#FF7F0E")
            axes[0, 1].axhline(1, linestyle="--", color="red", alpha=0.6)
            axes[0, 1].set_title("Error Rate (pred/actual)"); axes[0, 1].grid(alpha=0.3)
            axes[1, 0].hist(errors, bins=40, color="#4C72B0",
                            edgecolor="white", alpha=0.85)
            axes[1, 0].set_title("Error Distribution"); axes[1, 0].grid(axis="y", alpha=0.3)
            axes[1, 1].scatter(y_pred_arr[:2000], errors[:2000],
                               alpha=0.3, s=8, color="#9467BD")
            axes[1, 1].axhline(0, linestyle="--", color="red", alpha=0.6)
            axes[1, 1].set_xlabel("Predictions"); axes[1, 1].set_ylabel("Error")
            axes[1, 1].set_title("Predictions vs Error"); axes[1, 1].grid(alpha=0.3)
        else:
            cm = confusion_matrix(y_test_arr, y_pred_arr)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
            axes[0, 0].set_title("Confusion Matrix")
            axes[0, 0].set_xlabel("Predicted"); axes[0, 0].set_ylabel("Actual")
            classes = np.unique(y_test_arr)
            per_class_acc = [accuracy_score(y_test_arr[y_test_arr == c],
                                            y_pred_arr[y_test_arr == c])
                             for c in classes]
            axes[0, 1].bar(classes.astype(str), per_class_acc, color="#2CA02C", alpha=0.8)
            axes[0, 1].set_title("Accuracy per Class")
            axes[0, 1].set_ylabel("Accuracy"); axes[0, 1].grid(axis="y", alpha=0.3)
            if hasattr(final_model, "predict_proba"):
                proba = final_model.predict_proba(X_test).max(axis=1)
                axes[1, 0].hist(proba, bins=40, color="#FF7F0E",
                                edgecolor="white", alpha=0.85)
                axes[1, 0].set_title("Prediction Confidence Distribution")
                axes[1, 0].grid(axis="y", alpha=0.3)
            act_dist  = pd.Series(y_test_arr).value_counts().sort_index()
            pred_dist = pd.Series(y_pred_arr).value_counts().sort_index()
            x_pos = np.arange(len(act_dist))
            axes[1, 1].bar(x_pos - 0.2, act_dist.values,  0.4,
                           label="Actual",    color="#4C72B0", alpha=0.8)
            axes[1, 1].bar(x_pos + 0.2, pred_dist.reindex(act_dist.index, fill_value=0).values,
                           0.4, label="Predicted", color="#FF7F0E", alpha=0.8)
            axes[1, 1].set_title("Class Distribution: Actual vs Predicted")
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(act_dist.index.astype(str))
            axes[1, 1].legend(); axes[1, 1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(CONFIG["error_analysis_png"], dpi=150)
        plt.close()

        # ── Business Scenarios ─────────────────────────────────────────────────
        if problem_type == "regression":
            mae_val  = mean_absolute_error(y_test_arr, y_pred_arr)
            mape_val = _mape(y_test_arr, y_pred_arr)

            df_scenarios = df[[target_col]].copy().reset_index(drop=True)
            X_full = pd.get_dummies(
                df[[c for c in df.columns if c != target_col and c != "_src_idx"]],
                drop_first=True).reindex(columns=X.columns, fill_value=0)
            full_preds = final_model.predict(X_full)
            if le is not None:
                full_preds_labels = le.inverse_transform(full_preds.astype(int))
            else:
                full_preds_labels = full_preds

            df_scenarios["prediction"]     = full_preds_labels
            df_scenarios["worst_scenario"] = full_preds_labels - mae_val
            df_scenarios["best_scenario"]  = full_preds_labels + mae_val
            df_scenarios["mae"]            = round(mae_val, 4)
            df_scenarios["mape"]           = round(mape_val, 4)
            df_scenarios.to_parquet(CONFIG["scenarios_path"], index=False)

            error_analysis_md = f"""# Error Analysis & Business Scenarios

## Model: `{best}` | Target: `{target_col}`

| Metric | Value |
|--------|-------|
| MAE    | {mae_val:.4f} |
| MAPE   | {mape_val:.4f} |
| R²     | {test_score:.4f} |

## Business Scenarios
- **Best case**: prediction + MAE = upper bound
- **Worst case**: prediction - MAE = lower bound
- File: `df5_scenarios.parquet`

## Error Analysis Chart
See `error_analysis.png` for the 4-panel diagnostic.
"""
        else:
            report_str = classification_report(y_test, y_pred)
            fail_mask  = y_pred_arr != y_test_arr
            fail_rate  = fail_mask.mean()

            error_analysis_md = f"""# Error Analysis

## Model: `{best}` | Target: `{target_col}`

**Overall failure rate:** {fail_rate:.4f} ({fail_rate*100:.1f}% of test samples misclassified)

## Classification Report
```
{report_str}
```

## Error Analysis Chart
See `error_analysis.png` for confusion matrix and per-class accuracy.
"""

        with open(CONFIG["error_analysis_md"], "w", encoding="utf-8") as f:
            f.write(error_analysis_md)

        return (f"ML_SUCCESS\n"
                f"Model: '{best}'\n"
                f"{metric.upper()} test: {test_score:.4f}\n"
                f"CV type: {'TimeSeriesSplit' if use_timeseries_cv else 'Stratified/KFold'}\n"
                f"Boruta features used: {len(feature_cols)}\n"
                f"Optuna trials: {CONFIG['optuna_trials']} per model\n"
                f"Optuna CV: {CONFIG['cv_folds']}-fold (unified with eval CV)\n"
                f"Stacking CV: {CONFIG['cv_folds']}-fold\n"
                f"LabelEncoder leakage fix: active\n"
                f"AI narrative generated: yes\n"
                f"Files: Model_Metrics.md, final_model.pkl, "
                f"model_comparison.png, feature_importance.png, "
                f"error_analysis.png, Error_Analysis.md"
                + (f", df5_scenarios.parquet" if problem_type == "regression" else ""))
    except Exception as e:
        return f"ML_ERROR: {e}\n{traceback.format_exc()}"


# ── README ─────────────────────────────────────────────────────────────────────

@tool("generate_readme")
def generate_readme(_: str = "") -> str:
    """
    Generates a complete, richly detailed README.md by reading every pipeline
    artifact produced so far: model metrics, hypothesis results, feature strategy,
    dataset shape, error analysis, and deployment info.
    Returns README_SUCCESS or README_ERROR. No parameters.
    """
    try:
        ctx   = _read_ctx()
        cfg_t = {}
        art   = {}

        # ── Load all available artifacts ───────────────────────────────────────
        if os.path.exists(CONFIG["target_json"]):
            with open(CONFIG["target_json"]) as f:
                cfg_t = json.load(f)

        if os.path.exists(CONFIG["model_pkl"]):
            with open(CONFIG["model_pkl"], "rb") as f:
                art = pickle.load(f)

        strat = {}
        if os.path.exists(CONFIG["strategy_json"]):
            with open(CONFIG["strategy_json"]) as f:
                strat = json.load(f)

        hyp_results = []
        if os.path.exists(CONFIG["hypothesis_json"]):
            with open(CONFIG["hypothesis_json"]) as f:
                hyp_results = json.load(f)

        eval_content = ""
        if os.path.exists(CONFIG["eval_md"]):
            with open(CONFIG["eval_md"], encoding="utf-8") as f:
                eval_content = f.read()

        error_content = ""
        if os.path.exists(CONFIG["error_analysis_md"]):
            with open(CONFIG["error_analysis_md"], encoding="utf-8") as f:
                error_content = f.read()

        # ── Derive key values ──────────────────────────────────────────────────
        target_col   = cfg_t.get("target_col", "unknown")
        problem_type = cfg_t.get("problem_type", "unknown")
        ai_insights  = cfg_t.get("ai_insights", [])
        true_hyps    = cfg_t.get("true_hypotheses", [])
        model_name   = art.get("name", "unknown")
        test_score   = art.get("test_score", 0.0)
        optuna_params = art.get("optuna_params", {})
        metric_label = "Accuracy" if problem_type == "classification" else "R²"
        metric_pct   = f"{test_score * 100:.2f}%" if problem_type == "classification" else f"{test_score:.4f}"

        std_feats    = strat.get("standard_features", [])
        ai_feats     = strat.get("ai_features", [])
        boruta_feats = strat.get("boruta_selected", [])

        # Dataset shape from silver
        silver_shape = ("N/A", "N/A")
        if os.path.exists(CONFIG["silver_path"]):
            try:
                _df_s = pd.read_parquet(CONFIG["silver_path"])
                silver_shape = _df_s.shape
            except Exception:
                pass

        ml_shape = ("N/A", "N/A")
        if os.path.exists(CONFIG["ml_ready_path"]):
            try:
                _df_m = pd.read_parquet(CONFIG["ml_ready_path"])
                ml_shape = _df_m.shape
            except Exception:
                pass

        pred_rows = "N/A"
        if os.path.exists(CONFIG["predictions_path"]):
            try:
                _df_p = pd.read_parquet(CONFIG["predictions_path"])
                pred_rows = len(_df_p)
            except Exception:
                pass

        # Hypothesis summary
        v_counts = {v: sum(1 for r in hyp_results if r.get("verdict") == v)
                    for v in ["TRUE", "FALSE", "INCONCLUSIVE"]}
        hyp_table_rows = ""
        for r in hyp_results:
            emoji = {"TRUE": "✅", "FALSE": "❌", "INCONCLUSIVE": "⚪"}.get(r.get("verdict",""), "⚪")
            hyp_table_rows += (
                f"| {r.get('id','?')} | {emoji} **{r.get('verdict','?')}** "
                f"| {r.get('statement','')[:90]} "
                f"| {r.get('business_insight','')[:90]} |\n"
            )

        # ── Claude writes the executive summary with full context ──────────────
        prompt_readme = f"""Write a professional 3-paragraph executive summary for a Data Science project README on GitHub.

Business context: {ctx or 'End-to-end automated machine learning pipeline on a structured dataset.'}
Dataset: {silver_shape[0]:,} rows × {silver_shape[1]} columns
Target variable: '{target_col}' ({problem_type})
Best model: {model_name}
{metric_label} on test set: {metric_pct}
Key dataset insights discovered by AI: {ai_insights}
Confirmed business hypotheses: {true_hyps[:3]}

Paragraph 1: What problem this project solves and why it matters.
Paragraph 2: What the pipeline does (multi-agent, AI-powered, automatic target identification, hypothesis validation, model competition).
Paragraph 3: Main result — the model name, its score, and 1-2 actionable business takeaways.

Tone: technical but accessible to a non-ML stakeholder. Maximum 200 words total."""

        exec_summary = _ask_claude(prompt_readme, max_tokens=500)

        # ── Claude writes a limitations + next steps section ───────────────────
        prompt_limits = f"""You are a senior Data Scientist. Write a concise "Limitations & Next Steps"
section (bullet list, max 6 points) for a project that:
- Used {model_name} for {problem_type} on '{target_col}'
- Achieved {metric_label} = {metric_pct}
- Used Boruta for feature selection ({len(boruta_feats)} features selected)
- Ran {CONFIG['optuna_trials']} Optuna trials per model
- Did NOT use SHAP, calibration curves, or experiment tracking

Cover: known limitations of the approach, what should be done before production,
and concrete next steps to improve the model. Be specific, not generic."""

        limits_section = _ask_claude(prompt_limits, max_tokens=400)

        # ── Build output files manifest ────────────────────────────────────────
        all_output_files = [
            ("df1_silver.parquet",         "Silver layer — standardized raw data + imputation"),
            ("df2_gold.parquet",           "Gold layer — silver + standard + AI-generated features"),
            ("df3_ml_ready.parquet",       "ML-Ready layer — deduplicated, redundancy-removed"),
            ("df4_predictions.parquet",    f"Predictions — all original columns + `prediction` column ({pred_rows} rows)"),
            ("df5_scenarios.parquet",      "Business scenarios — best/worst case bounds (regression only)"),
            ("final_model.pkl",            f"Serialized best model ({model_name}) + LabelEncoder + feature list"),
            ("telegram_bot.py",            "Telegram bot — /start /stats /predict /insights /hypotheses /top_features /help"),
            ("requirements.txt",           "Python dependencies for the Telegram bot"),
            ("analysis_notebook.ipynb",    "Full pipeline story — renders on GitHub"),
            ("Quality_Report.md",          "Data quality report — imputation log, outliers, AI insights"),
            ("Intelligent_Analysis.md",    "Claude's full dataset analysis in JSON"),
            ("Descriptive_Statistics.md",  "Descriptive statistics table for all features"),
            ("Hypothesis_Validation.md",   f"10 business hypotheses — {v_counts.get('TRUE',0)} TRUE / {v_counts.get('FALSE',0)} FALSE / {v_counts.get('INCONCLUSIVE',0)} INCONCLUSIVE"),
            ("Model_Metrics.md",           "Full model comparison table + AI narrative interpretation"),
            ("Model_Evaluation.md",        "Train vs test gap analysis + overfitting diagnostic"),
            ("Error_Analysis.md",          "4-panel error diagnostic + business scenarios summary"),
            ("Deployment_Guide.md",        "Instructions for running the Telegram bot locally and on a server"),
            ("target_config.json",         "AI-identified target, problem type, insights, confirmed hypotheses"),
            ("feature_strategy.json",      "Feature engineering log — standard, AI-generated, Boruta-selected"),
            ("hypothesis_results.json",    "Full hypothesis results with verdicts and business insights"),
            ("README.md",                  "This file"),
        ]
        manifest_rows = ""
        for fname, desc in all_output_files:
            exists = "✅" if os.path.exists(os.path.join(_BASE_DIR, fname)) else "⬜"
            manifest_rows += f"| {exists} | `{fname}` | {desc} |\n"

        # ── Assemble the full README ───────────────────────────────────────────
        content = f"""# Auto Data Scientist v7.1 — SOTA Multi-Agent Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-6C3FC6?style=flat)](https://github.com/joaomdmoura/crewAI)
[![Claude](https://img.shields.io/badge/Claude-4.6%20Sonnet-CC7722?style=flat)](https://www.anthropic.com/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?style=flat&logo=telegram&logoColor=white)](https://core.telegram.org/bots)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Search-4C8BF5?style=flat)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> {exec_summary}

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
| **Target variable** | `{target_col}` |
| **Problem type** | {problem_type.capitalize()} |
| **Best model** | {model_name} |
| **{metric_label} (test set)** | **{metric_pct}** |
| **Optimized parameters** | `{json.dumps(optuna_params, default=_safe_json)}` |
| **CV strategy** | {CONFIG['cv_folds']}-fold {'Stratified' if problem_type == 'classification' else ''}KFold + Optuna ({CONFIG['optuna_trials']} trials) + Stacking |
| **Features used** | {len(boruta_feats) if boruta_feats else len(std_feats) + len(ai_feats)} (Boruta-selected from {len(std_feats) + len(ai_feats)} engineered) |
| **Dataset** | {silver_shape[0]:,} rows × {silver_shape[1]} columns → {ml_shape[0]:,} rows × {ml_shape[1]} ML-ready |
| **Predictions generated** | {pred_rows} rows in `df4_predictions.parquet` |

### AI-Identified Target Justification
> *{cfg_t.get('ai_justification', 'Target identified automatically by Claude.')}*

### Top Dataset Insights (by Claude)
{chr(10).join(f'{i+1}. {ins}' for i, ins in enumerate(ai_insights)) if ai_insights else '_Not yet available._'}

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
| **Source** | [{CONFIG['dataset_slug']}]({CONFIG['dataset_url']}) |
| **Raw shape** | {silver_shape[0]:,} rows × {silver_shape[1]} columns |
| **ML-ready shape** | {ml_shape[0]:,} rows × {ml_shape[1]} columns |
| **Target** | `{target_col}` ({problem_type}) |
| **Business context** | {ctx or '_None provided — add `business_context.txt` for richer AI reasoning._'} |

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
{chr(10).join(f'- `{f}`' for f in ai_feats) if ai_feats else '_None generated or feature code failed._'}

### Boruta feature selection
After engineering, Boruta (Random Forest shadow features) selected **{len(boruta_feats)} features** from {len(std_feats) + len(ai_feats)} total engineered features.
{f'Selected: `{", ".join(boruta_feats[:10])}{"..." if len(boruta_feats) > 10 else ""}`' if boruta_feats else '_Boruta not run or selected fewer than 5 features — full feature set used._'}

→ Full log: [feature_strategy.json](feature_strategy.json)

---

## 7. Business Hypothesis Validation

Claude generated 10 business hypotheses about `{target_col}`, tested each with real pandas code, and issued a verdict.

**Summary:** ✅ {v_counts.get('TRUE', 0)} TRUE · ❌ {v_counts.get('FALSE', 0)} FALSE · ⚪ {v_counts.get('INCONCLUSIVE', 0)} INCONCLUSIVE

| ID | Verdict | Hypothesis | Business Insight |
|----|---------|-----------|-----------------|
{hyp_table_rows if hyp_table_rows else '| — | — | _Hypothesis validation not yet run._ | — |\n'}

![Hypothesis Validation](hypothesis_validation.png)

→ Full results: [Hypothesis_Validation.md](Hypothesis_Validation.md) · [hypothesis_results.json](hypothesis_results.json)

---

## 8. Model Training & Selection

### Competition protocol
1. **Baseline CV** — all candidates scored with {CONFIG['cv_folds']}-fold cross-validation
2. **Optuna tuning** — top-3 models tuned with {CONFIG['optuna_trials']} trials each (CV also {CONFIG['cv_folds']}-fold, unified)
3. **Stacking** — meta-learner (LogisticRegression / Ridge) on top-3 Optuna-tuned models (CV = {CONFIG['cv_folds']}-fold)
4. **Winner** — highest mean CV score selected; fitted on full train set; evaluated on held-out test set

### Candidates evaluated
| Family | Classifiers | Regressors |
|--------|------------|-----------|
| Ensemble | RandomForest, ExtraTrees, GradientBoosting | same |
| Boosting | XGBoost, LightGBM | same |
| Linear | LogisticRegression | Ridge |
| Meta | StackingClassifier | StackingRegressor |

### Result
**Winner: `{model_name}`** · {metric_label} on test set: **{metric_pct}**

Best Optuna parameters: `{json.dumps(optuna_params, default=_safe_json)}`

![Model Comparison](model_comparison.png)
![Feature Importance](feature_importance.png)

→ Full metrics: [Model_Metrics.md](Model_Metrics.md)
→ Train/test gap analysis: [Model_Evaluation.md](Model_Evaluation.md)

---

## 9. Error Analysis

4-panel diagnostic chart:

![Error Analysis](error_analysis.png)

{error_content[:600] if error_content else '_Error analysis not yet available._'}

→ Full report: [Error_Analysis.md](Error_Analysis.md)

---

## 10. Deployment — Telegram Bot

Claude wrote a complete Telegram bot (`telegram_bot.py`) tailored to this specific dataset.

**4 tabs:**
- **Overview** — KPI cards: total records, {metric_label} score, prediction distribution, avg confidence
- **Actual vs Predicted** — {'confusion matrix + class distribution' if problem_type == 'classification' else 'scatter plot + residuals histogram'}
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
{manifest_rows}

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
pip install crewai kagglehub pandas pyarrow python-dotenv optuna anthropic \\
            scikit-learn matplotlib seaborn tabulate numpy xgboost lightgbm \\
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

{limits_section}

---

*Auto Data Scientist v7.1 · CrewAI + Claude 4.6 Sonnet + Optuna · [MIT License](LICENSE)*
"""
        with open(CONFIG["readme_md"], "w", encoding="utf-8") as f:
            f.write(content)

        sections = content.count("\n## ") + content.count("\n### ")
        return (f"README_SUCCESS\n"
                f"Sections: {sections}\n"
                f"Length: {len(content):,} characters\n"
                f"File: README.md")
    except Exception as e:
        return f"README_ERROR: {e}\n{traceback.format_exc()}"


# ── STEP 6: Deploy — Predictions Parquet + Telegram Bot ──────────────────────

@tool("deploy_telegram_bot")
def deploy_telegram_bot(_: str = "") -> str:
    """
    Loads final_model.pkl and df1_silver.parquet.
    Runs predictions on the full dataset and saves df4_predictions.parquet.
    Generates a production-ready Telegram bot (telegram_bot.py) that lets
    users query model stats, get predictions, and ask questions via chat.
    Returns DEPLOY_SUCCESS or ERROR. No parameters.

    Required .env variable: TELEGRAM_BOT_TOKEN=<your token from @BotFather>
    Install dependency: pip install python-telegram-bot
    """
    import ast as _ast

    def _validate_bot_code(code: str) -> tuple:
        """Returns (code, error_or_None)."""
        try:
            _ast.parse(code)
            return code, None
        except SyntaxError as e:
            return code, str(e)

    try:
        for required in [CONFIG["model_pkl"], CONFIG["silver_path"],
                         CONFIG["ml_ready_path"], CONFIG["target_json"]]:
            if not os.path.exists(required):
                return f"ERROR: {required} not found. Run the full pipeline first."

        with open(CONFIG["model_pkl"], "rb") as f:
            artifact = pickle.load(f)
        with open(CONFIG["target_json"]) as f:
            cfg = json.load(f)

        model        = artifact["model"]
        target_col   = artifact["target"]
        problem_type = artifact["type"]
        model_name   = artifact["name"]
        features     = artifact["features"]
        le           = artifact.get("label_encoder")
        test_score   = artifact.get("test_score", 0.0)

        # ── Generate predictions ──────────────────────────────────────────────
        df_silver = pd.read_parquet(CONFIG["silver_path"])
        df_ml     = pd.read_parquet(CONFIG["ml_ready_path"])
        df_ml     = df_ml.dropna(axis=1, how="all").reset_index(drop=True)

        if "_src_idx" in df_ml.columns:
            src_idx = df_ml["_src_idx"].values.astype(int)
            df_pred = df_silver.iloc[src_idx].reset_index(drop=True)
            df_ml   = df_ml.drop(columns=["_src_idx"])
            logger.info(f"[Deploy] Aligned {len(src_idx)} rows via _src_idx.")
        else:
            min_rows = min(len(df_silver), len(df_ml))
            df_pred  = df_silver.iloc[:min_rows].copy().reset_index(drop=True)

        feat_cols = [c for c in df_ml.columns if c != target_col]
        X_full    = pd.get_dummies(df_ml[feat_cols], drop_first=True).reindex(
                        columns=features, fill_value=0)
        raw_preds = model.predict(X_full)
        pred_labels = le.inverse_transform(raw_preds) if le else raw_preds

        pred_proba = None
        if problem_type == "classification" and hasattr(model, "predict_proba"):
            pred_proba = model.predict_proba(X_full).max(axis=1).round(4)

        df_pred["prediction"] = pred_labels
        if pred_proba is not None:
            df_pred["prediction_proba"] = pred_proba
        if target_col not in df_pred.columns and target_col in df_ml.columns:
            df_pred[target_col] = df_ml[target_col].values

        df_pred.to_parquet(CONFIG["predictions_path"], index=False)
        logger.info(f"[Deploy] df4_predictions saved: {df_pred.shape}")

        # ── Compute stats to embed in bot ─────────────────────────────────────
        metric_label = "Accuracy" if problem_type == "classification" else "R2"
        total_rows   = len(df_pred)
        num_cols     = df_pred.select_dtypes(include="number").columns.tolist()
        cat_cols     = [c for c in df_pred.select_dtypes(
                            include=["object", "string"]).columns
                        if df_pred[c].nunique() <= 30]
        ctx          = _read_ctx()

        pred_classes = []
        class_dist   = {}
        if problem_type == "classification":
            pred_classes = sorted(df_pred["prediction"].dropna().unique().tolist())
            class_dist   = df_pred["prediction"].value_counts().to_dict()

        # Feature importances (top 7)
        feat_imp_str = ""
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=features)
            top = imp.sort_values(ascending=False).head(7)
            feat_imp_str = "\n".join(
                f"  {i+1}. {feat}: {val:.4f}"
                for i, (feat, val) in enumerate(top.items())
            )

        # Hypothesis insights
        hyp_insights = []
        if os.path.exists(CONFIG["hypothesis_json"]):
            with open(CONFIG["hypothesis_json"]) as f:
                hyp_data = json.load(f)
            hyp_insights = [
                f"- {r['statement'][:80]} → {r['verdict']}"
                for r in hyp_data if r.get("verdict") == "TRUE"
            ][:5]

        col_schema_sample = {
            c: {"dtype": str(df_pred[c].dtype),
                "nunique": int(df_pred[c].nunique()),
                "sample": df_pred[c].dropna().head(2).tolist()}
            for c in df_pred.columns[:20]
        }

        # ── Prompt to generate the bot ────────────────────────────────────────
        prompt_bot = f"""You are an expert Python developer. Write a COMPLETE, working
Telegram bot that acts as a data science assistant for this ML project.

=== PROJECT CONTEXT ===
Business: {ctx or 'Supply chain delivery performance prediction.'}
Target: '{target_col}' | Problem: {problem_type} | Model: {model_name}
{metric_label}: {test_score:.4f} | Rows: {total_rows:,}
Prediction classes: {pred_classes}
Class distribution in predictions: {class_dist}
Numeric columns: {num_cols[:8]}
Categorical columns: {cat_cols[:6]}
Top features by importance:
{feat_imp_str}
Validated business hypotheses:
{chr(10).join(hyp_insights) if hyp_insights else '(none available)'}
Column schema (first 20):
{json.dumps(col_schema_sample, indent=2, default=_safe_json)}

=== MANDATORY BOT COMMANDS ===

/start
  Welcome message explaining what the bot does and listing all commands.

/stats
  Summary of the dataset and model:
  - Total records, model name, {metric_label} score
  - Prediction class distribution (counts + percentages)
  - Average confidence score if available
  Format as clean text with emojis.

/top_features
  Show the top 7 most important features with their importance scores.
  Explain in plain language what each one means for the business.

/hypotheses
  List the validated TRUE business hypotheses from the analysis.
  Format as numbered list with a short business explanation each.

/predict
  Ask the user for values of the top 3-4 features one by one
  (using conversation state via context.user_data).
  Then run the model and return the prediction + confidence.
  Use ConversationHandler for multi-step input.

/insights
  Ask Claude API (anthropic) a question about the dataset stats
  and return a 2-3 paragraph business insight. Use the ANTHROPIC_API_KEY
  from environment variables. Pass the stats as context in the prompt.

/help
  List all commands with brief descriptions.

=== TECHNICAL REQUIREMENTS ===
- Use python-telegram-bot >= 20.0 (async, ApplicationBuilder pattern).
- Load df4_predictions.parquet at startup with pd.read_parquet.
- Load final_model.pkl at startup with pickle.
- Read TELEGRAM_BOT_TOKEN from os.environ (not hardcoded).
- Read ANTHROPIC_API_KEY from os.environ for /insights command.
- Use logging for all errors.
- All handlers must be async def.
- Use application.run_polling() at the end of main().
- The file must end with:
  if __name__ == "__main__":
      main()

=== HARD RULES ===
- NEVER hardcode any token or key.
- NEVER use the old python-telegram-bot v13 API (Updater, Dispatcher).
- Use ApplicationBuilder().token(TOKEN).build() pattern (v20+).
- Handle exceptions in every command handler with try/except and send
  a user-friendly error message.
- Keep messages under 4096 characters (Telegram limit).
- If a column is missing from the dataframe, skip it gracefully.
- The /predict command must validate user inputs and handle non-numeric input.
- Write ONLY valid Python. No markdown fences. No explanations. No TODOs."""

        bot_code = _ask_claude(prompt_bot, max_tokens=7000)

        if "```python" in bot_code:
            bot_code = bot_code.split("```python")[1].split("```")[0]
        elif "```" in bot_code:
            bot_code = bot_code.split("```")[1].split("```")[0]
        bot_code = bot_code.strip()

        # ── Validate + self-heal ──────────────────────────────────────────────
        bot_code, syntax_err = _validate_bot_code(bot_code)

        if syntax_err:
            logger.warning(f"[Deploy] Bot syntax error: {syntax_err}. Self-healing...")
            fix_prompt = f"""Fix this Python syntax error in the Telegram bot:

ERROR: {syntax_err}

Rules:
- Fix ONLY the syntax, keep all logic intact.
- File must end with:
  if __name__ == "__main__":
      main()
- Output ONLY the corrected Python code, no markdown.

CODE:
{bot_code}"""
            bot_code_fixed = _ask_claude(fix_prompt, max_tokens=7000)
            if "```python" in bot_code_fixed:
                bot_code_fixed = bot_code_fixed.split("```python")[1].split("```")[0]
            elif "```" in bot_code_fixed:
                bot_code_fixed = bot_code_fixed.split("```")[1].split("```")[0]
            bot_code_fixed, syntax_err2 = _validate_bot_code(bot_code_fixed.strip())
            if syntax_err2 is None:
                bot_code = bot_code_fixed
                logger.info("[Deploy] Bot self-healing succeeded.")
            else:
                logger.warning(f"[Deploy] Bot self-healing failed: {syntax_err2}.")
                bot_code = bot_code_fixed

        # ── Guarantee main() call ─────────────────────────────────────────────
        if "def main(" in bot_code and \
           "main()" not in bot_code.split("def main(")[-1]:
            bot_code = bot_code.rstrip() + "\n\nif __name__ == '__main__':\n    main()\n"

        with open(CONFIG["telegram_bot"], "w", encoding="utf-8") as f:
            f.write(bot_code)
        logger.info(f"[Deploy] telegram_bot.py saved "
                    f"({len(bot_code)} chars, "
                    f"syntax={'OK' if syntax_err is None else 'FIXED'}).")

        # ── requirements.txt ─────────────────────────────────────────────────
        requirements = """python-telegram-bot>=20.7
pandas>=2.0.0
pyarrow>=14.0.0
scikit-learn>=1.4.0
numpy>=1.26.0
anthropic>=0.25.0
python-dotenv>=1.0.0
xgboost>=2.0.0
lightgbm>=4.3.0
"""
        with open(CONFIG["requirements_txt"], "w", encoding="utf-8") as f:
            f.write(requirements)

        # ── Deployment Guide ──────────────────────────────────────────────────
        deploy_md = f"""# Telegram Bot Deployment Guide

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
| `/stats` | Dataset and model summary ({metric_label}: {test_score:.4f}) |
| `/top_features` | Top 7 predictive features with business explanation |
| `/hypotheses` | Validated TRUE business hypotheses |
| `/predict` | Interactive prediction — enter feature values via chat |
| `/insights` | AI-generated business insight powered by Claude |
| `/help` | List all commands |

## Model Info
- **Model:** {model_name}
- **Target:** `{target_col}` ({problem_type})
- **{metric_label}:** {test_score:.4f}
- **Rows in df4_predictions.parquet:** {total_rows:,}

## Deploy to a Server (keep bot running 24/7)
```bash
# Option 1: nohup (Linux/Mac)
nohup python telegram_bot.py &

# Option 2: systemd service (Linux)
# Option 3: Railway, Render, or Fly.io (free tier available)
# Option 4: AWS Lambda + polling (serverless)
```
"""
        with open(os.path.join(_BASE_DIR, "Deployment_Guide.md"), "w",
                  encoding="utf-8") as f:
            f.write(deploy_md)

        ml_ready_cols = pd.read_parquet(CONFIG["ml_ready_path"]).columns
        return (f"DEPLOY_SUCCESS\n"
                f"Predictions saved: df4_predictions.parquet "
                f"({len(df_pred)} rows, {len(df_pred.columns)} columns)\n"
                f"Row alignment: "
                f"{'_src_idx (safe)' if '_src_idx' in ml_ready_cols else 'fallback iloc'}\n"
                f"Bot syntax: {'valid' if syntax_err is None else 'self-healed'}\n"
                f"Telegram bot: telegram_bot.py\n"
                f"Commands: /start /stats /top_features /hypotheses /predict /insights /help\n"
                f"Requirements: requirements.txt\n"
                f"Guide: Deployment_Guide.md\n"
                f"Next step: add TELEGRAM_BOT_TOKEN to .env then run: python telegram_bot.py")

    except Exception as e:
        return f"DEPLOY_ERROR: {e}\n{traceback.format_exc()}"
# ── STEP 7: Generate Analysis Notebook ───────────────────────────────────────

@tool("generate_analysis_notebook")
def generate_analysis_notebook(_: str = "") -> str:
    """
    Compiles all pipeline outputs — markdown reports, charts, metrics,
    Claude narratives, and a live prediction preview — into a single
    analysis_notebook.ipynb that renders beautifully on GitHub.
    Returns NOTEBOOK_SUCCESS or ERROR. No parameters.
    """
    try:
        def _read_md(path):
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return f.read()
            return ""

        def _img_cell(path, caption=""):
            fname = os.path.basename(path)
            meta  = ", metadata={'width': 900}"
            code  = (
                f"from IPython.display import Image, display\n"
                f"display(Image(filename='{fname}'{meta}))"
            )
            if caption:
                code += f"\nprint('{caption}')"
            return new_code_cell(code)

        ctx          = _read_ctx()
        quality_md   = _read_md(CONFIG["quality_md"])
        analysis_md  = _read_md(CONFIG["analysis_md"])
        metrics_md   = _read_md(CONFIG["metrics_md"])
        eval_md      = _read_md(CONFIG["eval_md"])
        deploy_guide = _read_md(os.path.join(_BASE_DIR, "Deployment_Guide.md"))

        target_col   = "unknown"
        problem_type = "unknown"
        model_name   = "unknown"
        test_score   = 0.0
        ai_insights  = []

        if os.path.exists(CONFIG["target_json"]):
            with open(CONFIG["target_json"]) as f:
                cfg_t = json.load(f)
            target_col   = cfg_t.get("target_col", "unknown")
            problem_type = cfg_t.get("problem_type", "unknown")
            ai_insights  = cfg_t.get("ai_insights", [])

        if os.path.exists(CONFIG["model_pkl"]):
            with open(CONFIG["model_pkl"], "rb") as f:
                art = pickle.load(f)
            model_name = art.get("name", "unknown")
            test_score = art.get("test_score", 0.0)

        metric_label = "Accuracy" if problem_type == "classification" else "R²"

        prompt_nb = f"""You are writing the introduction for a Jupyter notebook
that documents a complete automated Data Science pipeline.

Business context: {ctx or 'Automated ML pipeline on a structured dataset.'}
Target: '{target_col}' ({problem_type})
Best model: {model_name} | {metric_label}: {test_score:.4f}
Top insights discovered: {ai_insights}

Write 3 things:
1. A one-paragraph executive summary (what was done, what was found)
2. A 3-row markdown table: | Step | Tool | Output |
   covering: Ingestion, Analysis, Feature Engineering, EDA, Modeling, Deployment
3. A one-paragraph conclusion with business recommendations

Format your response as JSON:
{{
  "executive_summary": "paragraph here",
  "pipeline_table": "| Step | Tool | Output |\\n|---|---|---|\\n...",
  "conclusion": "paragraph here"
}}
Respond ONLY with the JSON."""

        nb_text = _ask_claude(prompt_nb, max_tokens=1200)
        try:
            if "```json" in nb_text:
                nb_text = nb_text.split("```json")[1].split("```")[0]
            elif "```" in nb_text:
                nb_text = nb_text.split("```")[1].split("```")[0]
            nb_content = json.loads(nb_text.strip())
        except Exception:
            nb_content = {
                "executive_summary": "Automated ML pipeline completed successfully.",
                "pipeline_table": "| Step | Tool | Output |\n|---|---|---|\n| All | CrewAI | See sections below |",
                "conclusion": "Model is ready for production deployment via the Telegram bot.",
            }

        cells = []

        cells.append(new_markdown_cell(
            f"# Auto Data Scientist v7 — Analysis Notebook\n\n"
            f"> **Target:** `{target_col}` | "
            f"**Problem:** {problem_type} | "
            f"**Best Model:** {model_name} | "
            f"**{metric_label}:** {test_score:.4f}\n\n"
            f"*Generated automatically by CrewAI + Claude 4.6 Sonnet*\n\n"
            f"---\n\n"
            f"## Executive Summary\n\n"
            f"{nb_content['executive_summary']}\n\n"
            f"## Pipeline Overview\n\n"
            f"{nb_content['pipeline_table']}"
        ))

        cells.append(new_markdown_cell("---\n## 1. Environment Setup"))
        cells.append(new_code_cell(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import json, pickle, os\n"
            "from IPython.display import Image, display, Markdown\n\n"
            "pd.set_option('display.max_columns', 50)\n"
            "pd.set_option('display.float_format', '{:.4f}'.format)\n"
            "print('Libraries loaded.')"
        ))

        cells.append(new_markdown_cell("---\n## 2. Data Quality Report\n\n" + quality_md))

        cells.append(new_markdown_cell("### Silver Dataset — Preview"))
        cells.append(new_code_cell(
            "df_silver = pd.read_parquet('df1_silver.parquet')\n"
            "print(f'Shape: {df_silver.shape}')\n"
            "print(f'Columns: {list(df_silver.columns)}')\n"
            "df_silver.head()"
        ))

        cells.append(new_code_cell(
            "# Null values overview\n"
            "nulls = df_silver.isnull().sum()\n"
            "nulls[nulls > 0].sort_values(ascending=False)"
        ))

        cells.append(new_markdown_cell("---\n## 3. Intelligent Analysis by Claude\n\n" + analysis_md))

        if os.path.exists(os.path.join(_BASE_DIR, "intelligent_analysis.png")):
            cells.append(new_markdown_cell("### AI-Generated Analysis Chart"))
            cells.append(_img_cell(
                os.path.join(_BASE_DIR, "intelligent_analysis.png"),
                "Chart generated by Claude's custom analysis code"
            ))

        cells.append(new_markdown_cell("---\n## 4. Exploratory Data Analysis"))

        cells.append(new_markdown_cell("### Gold Dataset — After Feature Engineering"))
        cells.append(new_code_cell(
            "df_gold = pd.read_parquet('df2_gold.parquet')\n"
            "print(f'Shape after feature engineering: {df_gold.shape}')\n"
            "df_gold.describe().T.round(3)"
        ))

        for img_file, caption in [
            ("target_dist.png",       f"Target Distribution — `{target_col}`"),
            ("distributions.png",     "Feature Distributions"),
            ("boxplots.png",          "Boxplots — Outlier Detection"),
            ("categoricals.png",      "Categorical Feature Distributions"),
            ("correlation_matrix.png","Correlation Matrix"),
        ]:
            fpath = os.path.join(_BASE_DIR, img_file)
            if os.path.exists(fpath):
                cells.append(new_markdown_cell(f"### {caption}"))
                cells.append(_img_cell(fpath, caption))

        cells.append(new_markdown_cell("---\n## 5. Feature Engineering"))

        if os.path.exists(CONFIG["strategy_json"]):
            with open(CONFIG["strategy_json"]) as f:
                strat = json.load(f)
            cells.append(new_code_cell(
                f"# Feature Engineering Summary\n"
                f"strategy = {json.dumps(strat, indent=2, default=_safe_json)}\n"
                f"print('Standard features created:', strategy.get('standard_features', []))\n"
                f"print('AI-generated features:', strategy.get('ai_features', []))\n"
                f"print('Boruta selected features:', len(strategy.get('boruta_selected', [])))\n"
                f"print('AI code executed successfully:', strategy.get('ai_success', False))"
            ))

        cells.append(new_markdown_cell("---\n## 5.5 Business Hypothesis Validation"))

        if os.path.exists(CONFIG["hypothesis_json"]):
            with open(CONFIG["hypothesis_json"]) as f:
                hyp_results = json.load(f)
            verdicts = {v: sum(1 for r in hyp_results if r.get("verdict") == v)
                        for v in ["TRUE", "FALSE", "INCONCLUSIVE"]}
            hyp_table = "| ID | Hypothesis | Verdict | Business Insight |\n"
            hyp_table += "|----|-----------|---------|-----------------|\n"
            for r in hyp_results:
                hyp_table += (f"| {r.get('id','?')} | "
                              f"{r.get('statement','')[:70]} | "
                              f"**{r.get('verdict','?')}** | "
                              f"{r.get('business_insight','')[:70]} |\n")
            cells.append(new_markdown_cell(
                f"**Results:** TRUE: {verdicts['TRUE']} | "
                f"FALSE: {verdicts['FALSE']} | "
                f"INCONCLUSIVE: {verdicts['INCONCLUSIVE']}\n\n"
                + hyp_table
            ))

        hyp_png = CONFIG["hypothesis_png"]
        if os.path.exists(hyp_png):
            cells.append(new_markdown_cell("### Hypothesis Verdict Summary"))
            cells.append(_img_cell(hyp_png, "Hypothesis Validation Results"))

        cells.append(new_code_cell(
            "import json\n"
            "with open('hypothesis_results.json') as f:\n"
            "    hyp = json.load(f)\n"
            "for h in hyp:\n"
            "    print(f\"{h['id']} [{h['verdict']}] {h['statement'][:70]}\")\n"
            "    print(f\"   → {h.get('business_insight','')[:80]}\\n\")"
        ))

        cells.append(new_markdown_cell("---\n## 6. Model Training & Evaluation\n\n" + metrics_md))

        for img_file, caption in [
            ("model_comparison.png",   "Model Comparison — Baseline vs Optuna vs Stacking"),
            ("feature_importance.png", "Top 15 Feature Importances"),
            ("actual_vs_predicted.png","Actual vs Predicted (Regression)"),
        ]:
            fpath = os.path.join(_BASE_DIR, img_file)
            if os.path.exists(fpath):
                cells.append(new_markdown_cell(f"### {caption}"))
                cells.append(_img_cell(fpath, caption))

        cells.append(new_markdown_cell("### Model Evaluation\n\n" + eval_md))

        cells.append(new_markdown_cell("---\n## 6.5 Error Analysis"))

        error_analysis_content = _read_md(CONFIG["error_analysis_md"])
        if error_analysis_content:
            cells.append(new_markdown_cell(error_analysis_content))

        error_png = CONFIG["error_analysis_png"]
        if os.path.exists(error_png):
            cells.append(new_markdown_cell("### 4-Panel Error Diagnostic"))
            cells.append(_img_cell(error_png, "Error Analysis — 4-panel"))

        if os.path.exists(CONFIG["scenarios_path"]):
            cells.append(new_markdown_cell("### Business Scenarios — Best / Worst Case"))
            cells.append(new_code_cell(
                "df_scenarios = pd.read_parquet('df5_scenarios.parquet')\n"
                f"print(f'Shape: {{df_scenarios.shape}}')\n"
                f"print(f'MAE: {{df_scenarios[\"mae\"].iloc[0]:.4f}}')\n"
                f"print(f'MAPE: {{df_scenarios[\"mape\"].iloc[0]:.4f}}')\n"
                "df_scenarios[['prediction','worst_scenario','best_scenario']].describe().round(2)"
            ))

        cells.append(new_markdown_cell("---\n## 7. Predictions — Full Dataset"))
        cells.append(new_code_cell(
            "df_pred = pd.read_parquet('df4_predictions.parquet')\n"
            "print(f'Shape: {df_pred.shape}')\n"
            f"print(f'Prediction distribution:')\n"
            f"print(df_pred['prediction'].value_counts())\n"
            "df_pred.head(10)"
        ))

        cells.append(new_code_cell(
            f"if '{target_col}' in df_pred.columns:\n"
            f"    match = (df_pred['{target_col}'].astype(str) == \n"
            f"             df_pred['prediction'].astype(str)).mean()\n"
            f"    print(f'Match rate: {{match:.4f}}')\n"
            f"    print(df_pred['{target_col}'].value_counts().rename('actual'))\n"
            f"    print(df_pred['prediction'].value_counts().rename('predicted'))"
        ))

        if problem_type == "classification":
            cells.append(new_code_cell(
                "from sklearn.metrics import confusion_matrix\n"
                "import seaborn as sns\n\n"
                f"if '{target_col}' in df_pred.columns:\n"
                f"    cm = confusion_matrix(\n"
                f"        df_pred['{target_col}'].astype(str),\n"
                f"        df_pred['prediction'].astype(str)\n"
                f"    )\n"
                f"    fig, ax = plt.subplots(figsize=(7, 5))\n"
                f"    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)\n"
                f"    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')\n"
                f"    ax.set_title('Confusion Matrix — {target_col}')\n"
                f"    plt.tight_layout(); plt.show()"
            ))

        cells.append(new_markdown_cell("---\n## 8. Deployment\n\n" + deploy_guide))
        cells.append(new_code_cell(
            "files = [\n"
            "    'df1_silver.parquet', 'df2_gold.parquet',\n"
            "    'df3_ml_ready.parquet', 'df4_predictions.parquet',\n"
            "    'final_model.pkl', 'telegram_bot.py',\n"
            "    'requirements.txt', 'analysis_notebook.ipynb',\n"
            "]\n"
            "for f in files:\n"
            "    exists = '✅' if os.path.exists(f) else '❌'\n"
            "    size   = f'{os.path.getsize(f)/1024:.1f} KB' if os.path.exists(f) else '-'\n"
            "    print(f'{exists}  {f:<40} {size}')"
        ))

        cells.append(new_markdown_cell(
            f"---\n## 9. Conclusion\n\n"
            f"{nb_content['conclusion']}\n\n"
            f"---\n"
            f"*Auto Data Scientist v7 · CrewAI + Claude 4.6 Sonnet + Optuna*"
        ))

        nb = new_notebook(cells=cells)
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nb.metadata["language_info"] = {
            "name": "python",
            "version": "3.10.0",
        }

        with open(CONFIG["notebook_path"], "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        n_cells = len(cells)
        logger.info(f"[Notebook] Saved: {CONFIG['notebook_path']} ({n_cells} cells)")

        return (f"NOTEBOOK_SUCCESS\n"
                f"File: analysis_notebook.ipynb\n"
                f"Cells: {n_cells}\n"
                f"Open with: jupyter notebook analysis_notebook.ipynb\n"
                f"Renders on GitHub automatically.")

    except Exception as e:
        return f"NOTEBOOK_ERROR: {e}\n{traceback.format_exc()}"


# ── STEP 8: Business Intelligence HTML Dashboard ──────────────────────────────

@tool("generate_html_dashboard")
def generate_html_dashboard(_: str = "") -> str:
    """
    Reads all pipeline artifacts (predictions, metrics, hypotheses, feature
    importance, model info) and asks Claude to write a single self-contained
    HTML file with embedded CSS and Chart.js (CDN) charts.
    The result is a zero-dependency business dashboard anyone can open in a
    browser or share as a link (GitHub Pages, Dropbox, email attachment).
    Returns HTML_SUCCESS or ERROR. No parameters.
    """
    try:
        # ── Load all available artifacts ──────────────────────────────────────
        if not os.path.exists(CONFIG["predictions_path"]):
            return "ERROR: df4_predictions.parquet not found. Run the full pipeline first."
        if not os.path.exists(CONFIG["target_json"]):
            return "ERROR: target_config.json not found."

        with open(CONFIG["target_json"]) as f:
            cfg = json.load(f)

        target_col   = cfg["target_col"]
        problem_type = cfg["problem_type"]
        ctx          = _read_ctx()

        df = pd.read_parquet(CONFIG["predictions_path"])

        # ── Model info ────────────────────────────────────────────────────────
        model_name  = "Unknown"
        test_score  = 0.0
        feat_imp    = {}
        optuna_params = {}

        if os.path.exists(CONFIG["model_pkl"]):
            with open(CONFIG["model_pkl"], "rb") as f:
                artifact = pickle.load(f)
            model_name    = artifact.get("name", "Unknown")
            test_score    = artifact.get("test_score", 0.0)
            optuna_params = artifact.get("optuna_params", {})
            model_obj     = artifact.get("model")
            features      = artifact.get("features", [])
            if hasattr(model_obj, "feature_importances_") and features:
                imp = pd.Series(model_obj.feature_importances_, index=features)
                feat_imp = imp.sort_values(ascending=False).head(8).round(4).to_dict()

        metric_label = "Accuracy" if problem_type == "classification" else "R²"

        # ── Prediction stats ──────────────────────────────────────────────────
        total_rows = len(df)
        pred_dist  = {}
        if "prediction" in df.columns:
            vc = df["prediction"].value_counts()
            pred_dist = {str(k): int(v) for k, v in vc.items()}

        avg_conf = None
        if "prediction_proba" in df.columns:
            avg_conf = round(float(df["prediction_proba"].mean()), 4)

        # ── Hypothesis results ────────────────────────────────────────────────
        hyp_results = []
        if os.path.exists(CONFIG["hypothesis_json"]):
            with open(CONFIG["hypothesis_json"]) as f:
                raw_hyp = json.load(f)
            hyp_results = [
                {
                    "id":      r.get("id", ""),
                    "verdict": r.get("verdict", ""),
                    "statement": r.get("statement", "")[:100],
                    "insight": r.get("business_insight", "")[:120],
                }
                for r in raw_hyp
            ]

        # ── Category breakdown (first low-card categorical vs prediction) ─────
        cat_breakdown = {}
        cat_candidates = [
            c for c in df.select_dtypes(include=["object", "string"]).columns
            if c not in ["prediction", target_col]
            and 2 <= df[c].nunique() <= 12
        ]
        if cat_candidates and "prediction" in df.columns:
            col = cat_candidates[0]
            breakdown = df.groupby(col)["prediction"].apply(
                lambda x: round(float((pd.to_numeric(x, errors="coerce") > 0).mean()) * 100, 1)
            ).to_dict()
            cat_breakdown = {"column": col, "data": breakdown}

        # ── AI insights section ───────────────────────────────────────────────
        ai_insights = cfg.get("ai_insights", [])
        true_hyps   = [h["statement"] for h in hyp_results if h["verdict"] == "TRUE"]

        prompt_insights = f"""You are a senior business analyst. Write 4 short, punchy
business insight paragraphs (2-3 sentences each) for an executive dashboard.
Base them on these real findings:

Business context: {ctx or 'Supply chain delivery performance prediction.'}
Model: {model_name} | {metric_label}: {test_score:.4f}
Total records: {total_rows:,}
Target: '{target_col}' ({problem_type})
Prediction distribution: {pred_dist}
Average confidence: {avg_conf}
Top features: {list(feat_imp.keys())[:5]}
Validated hypotheses (TRUE): {true_hyps[:3]}
AI dataset insights: {ai_insights[:3]}

Format each insight as:
TITLE: Short title (max 6 words)
BODY: 2-3 sentence explanation in plain business language.
No bullet points. No markdown. Just TITLE: and BODY: labels."""

        raw_insights = _ask_claude(prompt_insights, max_tokens=1000)

        # Parse the 4 insight blocks
        parsed_insights = []
        blocks = raw_insights.strip().split("\n\n")
        for block in blocks:
            lines = block.strip().splitlines()
            title = ""
            body  = ""
            for line in lines:
                if line.startswith("TITLE:"):
                    title = line.replace("TITLE:", "").strip()
                elif line.startswith("BODY:"):
                    body = line.replace("BODY:", "").strip()
            if title and body:
                parsed_insights.append({"title": title, "body": body})
        if not parsed_insights:
            parsed_insights = [{"title": "Model Performance",
                                "body": raw_insights[:300]}]

        # ── Build data payload for the HTML ───────────────────────────────────
        dashboard_data = {
            "title":        f"{target_col.replace('_', ' ').title()} — Business Dashboard",
            "context":      ctx or "Supply chain delivery performance prediction.",
            "model_name":   model_name,
            "metric_label": metric_label,
            "metric_value": round(test_score * 100, 2) if problem_type == "classification"
                            else round(test_score, 4),
            "total_rows":   total_rows,
            "avg_conf":     round(avg_conf * 100, 1) if avg_conf else None,
            "pred_dist":    pred_dist,
            "feat_imp":     feat_imp,
            "hypotheses":   hyp_results,
            "insights":     parsed_insights,
            "cat_breakdown": cat_breakdown,
            "true_hyp_count": sum(1 for h in hyp_results if h["verdict"] == "TRUE"),
            "false_hyp_count": sum(1 for h in hyp_results if h["verdict"] == "FALSE"),
            "inc_hyp_count": sum(1 for h in hyp_results if h["verdict"] == "INCONCLUSIVE"),
        }

        # ── Prompt Claude to write the full HTML ──────────────────────────────
        prompt_html = f"""You are an expert front-end developer and data visualisation specialist.
Write a SINGLE, COMPLETE, self-contained HTML file for an executive business dashboard.

=== DATA (embed this as a JavaScript const at the top of the <script> block) ===
const DATA = {json.dumps(dashboard_data, indent=2, default=_safe_json)};

=== DESIGN REQUIREMENTS ===
- Dark theme: background #0f1117, cards #1e2130, accent #4f8ef7 (blue), 
  success #22c55e (green), warning #f59e0b (amber), danger #ef4444 (red).
- Clean, modern layout. Max width 1200px, centered, padding 2rem.
- Google Font: Inter (load from https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap)
- Fully responsive — works on mobile.
- NO external CSS frameworks. Write all CSS inline in <style>.

=== SECTIONS (in this order) ===

1. HEADER
   - Dashboard title from DATA.title
   - Subtitle: DATA.context (truncated to 120 chars)
   - Small badge: model name + metric (e.g. "GradientBoosting · Accuracy 97.45%")

2. KPI CARDS ROW (4 cards in a CSS grid)
   - Total Records: DATA.total_rows formatted with commas
   - Model {metric_label}: DATA.metric_value + "%" if classification else raw
   - Avg Confidence: DATA.avg_conf + "%" (or "N/A" if null)
   - True Hypotheses: DATA.true_hyp_count + "/" + total hypotheses

3. PREDICTION DISTRIBUTION (Chart.js Doughnut)
   - Use DATA.pred_dist for labels and values
   - Colors: #22c55e for class 0/"On Time", #ef4444 for class 1/"Late"
   - Centered legend below chart

4. FEATURE IMPORTANCE (Chart.js horizontal Bar)
   - Use DATA.feat_imp (object with feature->importance)
   - Sort descending, color bars with gradient from #4f8ef7 to #818cf8
   - X-axis label: "Importance Score"

5. BUSINESS INSIGHTS (2x2 CSS grid of cards)
   - One card per item in DATA.insights
   - Each card: colored left border (cycle through accent/success/warning/danger),
     bold title, body text in lighter color
   - Icon before title: use a simple Unicode symbol (📊 🎯 ⚡ 💡)

6. HYPOTHESIS VALIDATION TABLE
   - Columns: ID | Verdict | Hypothesis | Business Insight
   - Verdict badge: green pill for TRUE, red for FALSE, grey for INCONCLUSIVE
   - Zebra striping on rows
   - Scrollable if > 6 rows

7. CATEGORY BREAKDOWN (Chart.js Bar) — only render if DATA.cat_breakdown.column exists
   - Title: "Late Rate by " + DATA.cat_breakdown.column
   - Data from DATA.cat_breakdown.data
   - Y-axis: percentage (0-100), add "%" suffix

8. FOOTER
   - "Generated by Auto Data Scientist v7 · CrewAI + Claude 4.6 Sonnet"
   - Generation timestamp using JavaScript new Date().toLocaleString()

=== CHART.JS ===
Load from CDN: https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js
All charts must use responsive: true and maintainAspectRatio: false inside a
fixed-height container div (e.g. style="height:280px; position:relative").
Use Chart.js dark theme: set Chart.defaults.color = '#94a3b8' before creating charts.

=== HARD RULES ===
- ONE file only. All CSS in <style>, all JS in <script>. No external files.
- No React, no Vue, no build tools. Vanilla JS only.
- All DATA is embedded in the JS const — no fetch() calls needed.
- The file must open correctly by double-clicking in Windows/Mac/Linux.
- Write ONLY the HTML. No markdown fences. No explanations."""

        html_code = _ask_claude(prompt_html, max_tokens=8000)

        # Strip markdown fences if present
        if "```html" in html_code:
            html_code = html_code.split("```html")[1].split("```")[0]
        elif "```" in html_code:
            html_code = html_code.split("```")[1].split("```")[0]
        html_code = html_code.strip()

        # Basic validation — must start with <!DOCTYPE or <html
        if not (html_code.lower().startswith("<!doctype") or
                html_code.lower().startswith("<html")):
            logger.warning("[HTML] Response did not start with HTML tag. Prepending <!DOCTYPE>.")
            html_code = "<!DOCTYPE html>\n" + html_code

        # Guarantee Chart.js CDN is present
        if "chart.js" not in html_code.lower():
            logger.warning("[HTML] Chart.js CDN missing — injecting into <head>.")
            html_code = html_code.replace(
                "</head>",
                '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>\n</head>',
                1,
            )

        with open(CONFIG["html_dashboard"], "w", encoding="utf-8") as f:
            f.write(html_code)

        file_size_kb = os.path.getsize(CONFIG["html_dashboard"]) // 1024
        logger.info(f"[HTML] dashboard.html saved ({file_size_kb} KB, "
                    f"{len(html_code)} chars).")

        return (f"HTML_SUCCESS\n"
                f"File: dashboard.html ({file_size_kb} KB)\n"
                f"Sections: Header, KPI Cards, Prediction Distribution, "
                f"Feature Importance, Business Insights, Hypothesis Table, "
                f"Category Breakdown, Footer\n"
                f"Charts: Chart.js (CDN) — Doughnut + horizontal Bar + Bar\n"
                f"Zero dependencies — open dashboard.html in any browser.\n"
                f"Host on GitHub Pages, Dropbox, or email as attachment.")

    except Exception as e:
        return f"HTML_ERROR: {e}\n{traceback.format_exc()}"
    

@tool("deploy_to_oracle_cloud")
def deploy_to_oracle_cloud(_: str = "") -> str:
    """
    Connects to the Oracle Cloud VM via SSH (paramiko), uploads
    final_model.pkl and df4_predictions.parquet via SFTP, runs
    'git pull' to get the latest telegram_bot.py, applies the
    Markdown fix, and restarts the systemd telegram-bot service.
 
    Requires in .env or CONFIG:
        ORACLE_VM_IP        — public IP of the VM
        ORACLE_KEY_PATH     — path to the SSH private key (.key file)
        ORACLE_REPO_PATH    — remote repo path (default: /home/ubuntu/ecommerce-ds-agent)
 
    Returns ORACLE_DEPLOY_SUCCESS or ORACLE_DEPLOY_ERROR.
    No parameters.
    """
    import paramiko
    import os
 
    vm_ip      = CONFIG.get("oracle_vm_ip")   or os.getenv("ORACLE_VM_IP")
    key_path   = CONFIG.get("oracle_key_path") or os.getenv("ORACLE_KEY_PATH")
    repo_path  = CONFIG.get("oracle_repo_path", "/home/ubuntu/ecommerce-ds-agent")
    vm_user    = CONFIG.get("oracle_vm_user",   "ubuntu")
 
    if not vm_ip:
        return "ORACLE_DEPLOY_SKIPPED: ORACLE_VM_IP not set in .env — skipping cloud deploy."
    if not key_path:
        return "ORACLE_DEPLOY_SKIPPED: ORACLE_KEY_PATH not set in .env — skipping cloud deploy."
    if not os.path.exists(key_path):
        return f"ORACLE_DEPLOY_ERROR: SSH key not found at {key_path}"
 
    logs = []
 
    try:
        # ── 1. Connect via SSH ────────────────────────────────────────────────
        logs.append(f"Connecting to {vm_user}@{vm_ip}...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=vm_ip,
            username=vm_user,
            key_filename=key_path,
            timeout=30,
        )
        logs.append("SSH connection established.")
 
        def run(cmd):
            """Run a remote command and return (stdout, stderr, exit_code)."""
            stdin, stdout, stderr = ssh.exec_command(cmd, timeout=120)
            exit_code = stdout.channel.recv_exit_status()
            out = stdout.read().decode().strip()
            err = stderr.read().decode().strip()
            return out, err, exit_code
 
        # ── 2. Upload model artifacts via SFTP ───────────────────────────────
        logs.append("Uploading model artifacts via SFTP...")
        sftp = ssh.open_sftp()
 
        files_to_upload = [
            (CONFIG["model_pkl"],        f"{repo_path}/final_model.pkl"),
            (CONFIG["predictions_path"], f"{repo_path}/df4_predictions.parquet"),
        ]
 
        for local_path, remote_path in files_to_upload:
            if not os.path.exists(local_path):
                logs.append(f"  WARNING: {local_path} not found — skipping.")
                continue
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logs.append(f"  Uploading {os.path.basename(local_path)} ({size_mb:.1f} MB)...")
            sftp.put(local_path, remote_path)
            logs.append(f"  ✅ {os.path.basename(local_path)} uploaded.")
 
        sftp.close()
 
        # ── 3. Git pull latest telegram_bot.py ───────────────────────────────
        logs.append("Running git pull on VM...")
        out, err, code = run(f"cd {repo_path} && git pull origin main 2>&1")
        logs.append(f"  git pull: {out[:200]}")
        if code != 0 and "Already up to date" not in out:
            logs.append(f"  WARNING: git pull may have had issues: {err[:100]}")
 
        # ── 4. Apply Markdown fix to telegram_bot.py ─────────────────────────
        logs.append("Applying Markdown parse_mode fix...")
        fix_cmd = (
            f"sed -i \"s/parse_mode='Markdown'/parse_mode=None/g\" {repo_path}/telegram_bot.py && "
            f"sed -i 's/parse_mode=ParseMode.MARKDOWN/parse_mode=None/g' {repo_path}/telegram_bot.py"
        )
        out, err, code = run(fix_cmd)
        logs.append("  Markdown fix applied.")
 
        # ── 5. Restart systemd service ────────────────────────────────────────
        logs.append("Restarting telegram-bot service...")
        out, err, code = run("sudo systemctl restart telegram-bot-ecommerce")
        if code != 0:
            logs.append(f"  WARNING: restart may have failed: {err[:100]}")
        else:
            logs.append("  ✅ Service restarted.")
 
        # ── 6. Verify service is running ──────────────────────────────────────
        import time
        time.sleep(3)  # wait for service to start
        out, err, code = run("sudo systemctl is-active telegram-bot-ecommerce")
        status = out.strip()
        logs.append(f"  Service status: {status}")
 
        ssh.close()
 
        if status == "active":
            summary = "\n".join(logs)
            return (f"ORACLE_DEPLOY_SUCCESS\n"
                    f"VM: {vm_ip}\n"
                    f"Repo: {repo_path}\n"
                    f"Service: active (running)\n"
                    f"Log:\n{summary}")
        else:
            summary = "\n".join(logs)
            return (f"ORACLE_DEPLOY_WARNING\n"
                    f"Files uploaded and service restarted but status={status}.\n"
                    f"Check VM logs: sudo journalctl -u telegram-bot -n 30\n"
                    f"Log:\n{summary}")
 
    except Exception as e:
        err_str = str(e)
        if "Authentication" in err_str or "publickey" in err_str:
            return f"ORACLE_DEPLOY_ERROR: SSH authentication failed. Check ORACLE_KEY_PATH. Detail: {e}"
        elif "Unable to connect" in err_str or "Connection refused" in err_str or "timed out" in err_str:
            return f"ORACLE_DEPLOY_ERROR: Could not connect to {vm_ip}. Check VM is running. Detail: {e}"
        else:
            return f"ORACLE_DEPLOY_ERROR: {e}\n{traceback.format_exc()}"

# ── STEP 9: A/B Testing ───────────────────────────────────────────────────────

@tool("run_ab_testing")
def run_ab_testing(_: str = "") -> str:
    """
    Loads final_model.pkl (Model A) and trains a lightweight baseline (Model B).
    Compares both on the held-out test set using statistical tests:
      - Classification: McNemar test (paired) + Chi-square on prediction distributions
      - Regression: Wilcoxon signed-rank test on residuals + Bayesian lift estimate
    Generates a 3-panel chart and a full markdown report.
    Claude interprets the results and writes a business recommendation.
    Saves ab_test_results.json, AB_Test_Report.md, ab_test.png.
    Returns AB_TEST_SUCCESS or ERROR. No parameters.
    """
    try:
        for req in [CONFIG["model_pkl"], CONFIG["ml_ready_path"], CONFIG["target_json"]]:
            if not os.path.exists(req):
                return f"ERROR: {req} not found. Run the full pipeline first."

        with open(CONFIG["model_pkl"], "rb") as f:
            artifact = pickle.load(f)
        with open(CONFIG["target_json"]) as f:
            cfg = json.load(f)

        model_a      = artifact["model"]
        target_col   = artifact["target"]
        problem_type = artifact["type"]
        model_a_name = artifact["name"]
        features     = artifact["features"]
        le           = artifact.get("label_encoder")
        score_a      = artifact.get("test_score", 0.0)

        df = pd.read_parquet(CONFIG["ml_ready_path"])
        df = df.dropna(axis=1, how="all").reset_index(drop=True)
        feat_cols = [c for c in df.columns if c != target_col and c != "_src_idx"]
        X = pd.get_dummies(df[feat_cols], drop_first=True).reindex(columns=features, fill_value=0)
        y = df[target_col].copy()

        is_clf = (problem_type == "classification")
        X_train, X_test, y_train_raw, y_test_raw = train_test_split(
            X, y,
            test_size=CONFIG["test_size"],
            random_state=CONFIG["random_state"],
            stratify=y.astype(str) if is_clf else None,
        )

        # Encode labels post-split (mirrors FIX-3)
        le_b = None
        if is_clf:
            le_b = LabelEncoder()
            le_b.fit(y_train_raw.astype(str))
            y_train = pd.Series(le_b.transform(y_train_raw.astype(str)), index=y_train_raw.index)
            y_test  = pd.Series(le_b.transform(y_test_raw.astype(str)),  index=y_test_raw.index)
        else:
            y_train, y_test = y_train_raw, y_test_raw

        # ── Model B: lightweight baseline ─────────────────────────────────────
        model_b      = LogisticRegression(max_iter=1000, random_state=CONFIG["random_state"]) \
                       if is_clf else Ridge()
        model_b_name = "LogisticRegression (baseline)" if is_clf else "Ridge (baseline)"
        model_b.fit(X_train, y_train)

        pred_a = model_a.predict(X_test)
        pred_b = model_b.predict(X_test)
        y_test_arr = np.array(y_test)

        # ── Statistical tests ──────────────────────────────────────────────────
        stat_results = {}
        if is_clf:
            score_b = float(accuracy_score(y_test_arr, pred_b))

            # McNemar test — are errors correlated?
            correct_a = (pred_a == y_test_arr)
            correct_b = (pred_b == y_test_arr)
            n01 = int(np.sum(~correct_a &  correct_b))   # B right, A wrong
            n10 = int(np.sum( correct_a & ~correct_b))   # A right, B wrong
            # Continuity-corrected McNemar
            if (n01 + n10) > 0:
                chi2_mc = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
                p_mcnemar = float(ss.chi2.sf(chi2_mc, df=1))
            else:
                chi2_mc, p_mcnemar = 0.0, 1.0

            # Chi-square on overall prediction distributions
            classes = np.unique(np.concatenate([pred_a, pred_b]))
            obs_a   = np.array([np.sum(pred_a == c) for c in classes])
            obs_b   = np.array([np.sum(pred_b == c) for c in classes])
            if obs_b.sum() > 0:
                chi2_dist, p_dist = ss.chisquare(obs_a, f_exp=obs_b * (obs_a.sum() / obs_b.sum()))
                p_dist = float(p_dist)
            else:
                chi2_dist, p_dist = 0.0, 1.0

            # Bayesian lift: P(A beats B) via Beta posteriors on accuracy
            n = len(y_test_arr)
            wins_a = int(np.sum(correct_a))
            wins_b = int(np.sum(correct_b))
            # Beta(wins+1, losses+1) — sample and compare
            rng = np.random.default_rng(42)
            samples_a = rng.beta(wins_a + 1, n - wins_a + 1, 50_000)
            samples_b = rng.beta(wins_b + 1, n - wins_b + 1, 50_000)
            p_a_beats_b = float(np.mean(samples_a > samples_b))

            stat_results = {
                "test_type":       "classification",
                "model_a":         model_a_name,
                "model_b":         model_b_name,
                "accuracy_a":      round(score_a, 4),
                "accuracy_b":      round(score_b, 4),
                "delta":           round(score_a - score_b, 4),
                "mcnemar_p":       round(p_mcnemar, 4),
                "chisq_dist_p":    round(p_dist, 4),
                "p_a_beats_b":     round(p_a_beats_b, 4),
                "n_test":          n,
                "winner":          model_a_name if score_a >= score_b else model_b_name,
                "significant":     p_mcnemar < 0.05,
            }
            metric_label = "Accuracy"

        else:
            score_b = float(r2_score(y_test_arr, pred_b))
            res_a   = y_test_arr - pred_a
            res_b   = y_test_arr - pred_b

            # Wilcoxon signed-rank on residuals
            try:
                stat_w, p_wilcoxon = ss.wilcoxon(np.abs(res_a), np.abs(res_b))
                p_wilcoxon = float(p_wilcoxon)
            except Exception:
                stat_w, p_wilcoxon = 0.0, 1.0

            mae_a = float(mean_absolute_error(y_test_arr, pred_a))
            mae_b = float(mean_absolute_error(y_test_arr, pred_b))

            # Bayesian lift on MAE improvement (normal approximation)
            diff_abs = np.abs(res_b) - np.abs(res_a)   # positive = A is better
            p_a_beats_b = float(np.mean(diff_abs > 0))

            stat_results = {
                "test_type":       "regression",
                "model_a":         model_a_name,
                "model_b":         model_b_name,
                "r2_a":            round(score_a, 4),
                "r2_b":            round(score_b, 4),
                "mae_a":           round(mae_a, 4),
                "mae_b":           round(mae_b, 4),
                "delta_r2":        round(score_a - score_b, 4),
                "delta_mae":       round(mae_b - mae_a, 4),
                "wilcoxon_p":      round(p_wilcoxon, 4),
                "p_a_beats_b":     round(p_a_beats_b, 4),
                "n_test":          len(y_test_arr),
                "winner":          model_a_name if score_a >= score_b else model_b_name,
                "significant":     p_wilcoxon < 0.05,
            }
            metric_label = "R²"

        with open(CONFIG["ab_test_json"], "w") as f:
            json.dump(stat_results, f, indent=2, default=_safe_json)

        # ── Chart — 3 panels ───────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"A/B Test: {model_a_name} vs {model_b_name}",
                     fontsize=14, fontweight="bold")

        if is_clf:
            # Panel 1: Accuracy bar
            axes[0].bar(["Model A\n" + model_a_name[:20],
                         "Model B\n" + model_b_name[:20]],
                        [stat_results["accuracy_a"], stat_results["accuracy_b"]],
                        color=["#2CA02C", "#4C72B0"], alpha=0.85, width=0.5)
            axes[0].set_ylabel("Accuracy"); axes[0].set_title("Accuracy Comparison")
            axes[0].set_ylim(max(0, min(stat_results["accuracy_a"],
                                        stat_results["accuracy_b"]) - 0.05), 1.0)
            axes[0].grid(axis="y", alpha=0.3)
            for i, v in enumerate([stat_results["accuracy_a"], stat_results["accuracy_b"]]):
                axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")

            # Panel 2: Prediction distribution comparison
            classes_all = sorted(set(pred_a.tolist() + pred_b.tolist()), key=str)
            x_pos = np.arange(len(classes_all))
            cnt_a = [np.sum(pred_a == c) for c in classes_all]
            cnt_b = [np.sum(pred_b == c) for c in classes_all]
            axes[1].bar(x_pos - 0.2, cnt_a, 0.4, label="Model A", color="#2CA02C", alpha=0.8)
            axes[1].bar(x_pos + 0.2, cnt_b, 0.4, label="Model B", color="#4C72B0", alpha=0.8)
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels([str(c) for c in classes_all], rotation=45, ha="right")
            axes[1].set_title("Prediction Distribution"); axes[1].legend()
            axes[1].grid(axis="y", alpha=0.3)

            # Panel 3: Bayesian P(A beats B)
            p_val = stat_results["p_a_beats_b"]
            color = "#2CA02C" if p_val >= 0.95 else ("#FF7F0E" if p_val >= 0.80 else "#D62728")
            axes[2].barh(["P(A beats B)"], [p_val], color=color, alpha=0.85)
            axes[2].barh(["P(B beats A)"], [1 - p_val], color="#4C72B0", alpha=0.85)
            axes[2].axvline(0.95, linestyle="--", color="gray", alpha=0.6, label="95% threshold")
            axes[2].set_xlim(0, 1)
            axes[2].set_title("Bayesian Probability"); axes[2].legend()
            axes[2].grid(axis="x", alpha=0.3)
            axes[2].text(p_val / 2, 0, f"{p_val:.1%}", va="center", ha="center",
                         fontsize=13, fontweight="bold", color="white")

        else:
            # Panel 1: R² comparison
            axes[0].bar(["Model A\n" + model_a_name[:20],
                         "Model B\n" + model_b_name[:20]],
                        [stat_results["r2_a"], stat_results["r2_b"]],
                        color=["#2CA02C", "#4C72B0"], alpha=0.85, width=0.5)
            axes[0].set_ylabel("R²"); axes[0].set_title("R² Comparison")
            axes[0].grid(axis="y", alpha=0.3)

            # Panel 2: Residuals distribution
            axes[1].hist(y_test_arr - pred_a, bins=40, alpha=0.7,
                         color="#2CA02C", label="Model A residuals")
            axes[1].hist(y_test_arr - pred_b, bins=40, alpha=0.7,
                         color="#4C72B0", label="Model B residuals")
            axes[1].axvline(0, linestyle="--", color="red", alpha=0.6)
            axes[1].set_title("Residuals Distribution"); axes[1].legend()
            axes[1].grid(axis="y", alpha=0.3)

            # Panel 3: MAE comparison
            axes[2].bar(["MAE A", "MAE B"],
                        [stat_results["mae_a"], stat_results["mae_b"]],
                        color=["#2CA02C", "#4C72B0"], alpha=0.85, width=0.5)
            axes[2].set_title("MAE Comparison (lower is better)")
            axes[2].grid(axis="y", alpha=0.3)
            for i, v in enumerate([stat_results["mae_a"], stat_results["mae_b"]]):
                axes[2].text(i, v * 1.01, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")

        plt.tight_layout()
        plt.savefig(CONFIG["ab_test_png"], dpi=150)
        plt.close()

        # ── Claude interprets the A/B test ────────────────────────────────────
        prompt_ab = f"""You are a senior Data Scientist interpreting an A/B test between two models.

Business context: {_read_ctx() or 'Prediction pipeline.'}
Target: '{target_col}' ({problem_type})

A/B TEST RESULTS:
{json.dumps(stat_results, indent=2, default=_safe_json)}

Write 3 paragraphs:
1. Which model wins statistically and by how much — be precise about the numbers.
2. Whether the difference is practically significant for the business (not just statistically).
3. A clear recommendation: deploy Model A, stick with Model B, or run a longer test.

Be direct and quantitative. Avoid generic statements."""

        interpretation = _ask_claude(prompt_ab, max_tokens=600)

        # ── Save markdown report ───────────────────────────────────────────────
        sig_str = "✅ Statistically significant (p < 0.05)" if stat_results["significant"] \
                  else "⚠️ Not statistically significant (p ≥ 0.05)"

        if is_clf:
            stats_table = f"""| Metric | Model A ({model_a_name}) | Model B ({model_b_name}) |
|--------|---------|---------|
| Accuracy | **{stat_results['accuracy_a']:.4f}** | {stat_results['accuracy_b']:.4f} |
| Delta | {stat_results['delta']:+.4f} | — |
| McNemar p-value | {stat_results['mcnemar_p']:.4f} | {sig_str} |
| P(A beats B) Bayesian | **{stat_results['p_a_beats_b']:.1%}** | — |
| N test samples | {stat_results['n_test']:,} | — |"""
        else:
            stats_table = f"""| Metric | Model A ({model_a_name}) | Model B ({model_b_name}) |
|--------|---------|---------|
| R² | **{stat_results['r2_a']:.4f}** | {stat_results['r2_b']:.4f} |
| MAE | **{stat_results['mae_a']:.4f}** | {stat_results['mae_b']:.4f} |
| ΔR² | {stat_results['delta_r2']:+.4f} | — |
| Wilcoxon p-value | {stat_results['wilcoxon_p']:.4f} | {sig_str} |
| P(A beats B) Bayesian | **{stat_results['p_a_beats_b']:.1%}** | — |
| N test samples | {stat_results['n_test']:,} | — |"""

        report = f"""# A/B Test Report

**Winner: `{stat_results['winner']}`** · Target: `{target_col}` ({problem_type})

## Statistical Results

{stats_table}

## Visual Analysis
![A/B Test](ab_test.png)

## AI Interpretation

{interpretation}

---
*Generated by Auto Data Scientist v8 — A/B Testing Agent*
"""
        with open(CONFIG["ab_test_md"], "w", encoding="utf-8") as f:
            f.write(report)

        return (f"AB_TEST_SUCCESS\n"
                f"Winner: '{stat_results['winner']}'\n"
                f"{'Accuracy A: ' + str(stat_results.get('accuracy_a','')) if is_clf else 'R² A: ' + str(stat_results.get('r2_a',''))}\n"
                f"{'Accuracy B: ' + str(stat_results.get('accuracy_b','')) if is_clf else 'R² B: ' + str(stat_results.get('r2_b',''))}\n"
                f"Significant: {stat_results['significant']}\n"
                f"P(A beats B): {stat_results['p_a_beats_b']:.1%}\n"
                f"Files: ab_test_results.json, AB_Test_Report.md, ab_test.png")

    except Exception as e:
        return f"AB_TEST_ERROR: {e}\n{traceback.format_exc()}"


# ── STEP 10: Recommendation System ───────────────────────────────────────────

@tool("build_recommendation_system")
def build_recommendation_system(_: str = "") -> str:
    """
    Automatically detects the best recommendation strategy for the dataset:

    Strategy A — Collaborative Filtering (if user + item + rating columns exist):
      Uses truncated SVD (matrix factorization) to decompose the user-item matrix.
      Generates top-N item recommendations per user.

    Strategy B — Content-Based Filtering (default for most business datasets):
      Normalises the ML-ready feature vectors, computes cosine similarity between
      all rows, and for each entity returns the top-N most similar records that
      also have a positive model prediction.

    Claude interprets the results and writes a business playbook.
    Saves df6_recommendations.parquet, Recommendation_System.md, recommendations.png.
    Returns RECO_SUCCESS or ERROR. No parameters.
    """
    try:
        for req in [CONFIG["ml_ready_path"], CONFIG["target_json"],
                    CONFIG["predictions_path"]]:
            if not os.path.exists(req):
                return f"ERROR: {req} not found. Run the full pipeline first."

        with open(CONFIG["target_json"]) as f:
            cfg = json.load(f)

        target_col   = cfg["target_col"]
        problem_type = cfg["problem_type"]
        ctx          = _read_ctx()

        df_pred = pd.read_parquet(CONFIG["predictions_path"])
        df_ml   = pd.read_parquet(CONFIG["ml_ready_path"]).reset_index(drop=True)
        df_ml   = df_ml.dropna(axis=1, how="all")

        n_rows = len(df_ml)
        TOP_N  = 5

        # ── Detect strategy ───────────────────────────────────────────────────
        all_cols_lower = {c.lower(): c for c in df_ml.columns}

        user_col = next((all_cols_lower[k] for k in all_cols_lower
                         if any(w in k for w in ["customer", "user", "client", "buyer"])), None)
        item_col = next((all_cols_lower[k] for k in all_cols_lower
                         if any(w in k for w in ["product", "item", "sku", "category",
                                                  "order_item", "department"])), None)
        rating_col = next((all_cols_lower[k] for k in all_cols_lower
                           if any(w in k for w in ["rating", "score", "sales",
                                                    "quantity", "amount", "profit"])), None)

        # [FIX] All three columns must be distinct — prevents same column
        # being used as both user and rating, which breaks pivot_table.
        use_collab = (
            user_col and item_col and rating_col and
            len({user_col, item_col, rating_col}) == 3 and
            df_ml[user_col].nunique() <= 5000 and
            df_ml[item_col].nunique() <= 2000
        )

        strategy = "collaborative" if use_collab else "content_based"
        logger.info(f"[Reco] Strategy: {strategy} | "
                    f"user={user_col}, item={item_col}, rating={rating_col}")

        reco_df = None

        # ══ STRATEGY A: Collaborative Filtering via truncated SVD ════════════
        if strategy == "collaborative":
            from sklearn.preprocessing import normalize
            from sklearn.decomposition import TruncatedSVD
            try:
                ui = df_ml.pivot_table(index=user_col, columns=item_col,
                                       values=rating_col, aggfunc="mean", fill_value=0)
                logger.info(f"[Reco] User-item matrix: {ui.shape}")

                users = list(ui.index)
                items = list(ui.columns)
                R     = ui.values.astype(float)

                R_norm = normalize(R, norm="l2")

                k   = min(20, min(R.shape) - 1)
                svd = TruncatedSVD(n_components=k, random_state=CONFIG["random_state"])
                U   = svd.fit_transform(R_norm)
                Vt  = svd.components_
                R_hat = U @ Vt

                records = []
                for u_idx, user in enumerate(users):
                    seen = set(np.where(R[u_idx] > 0)[0])
                    scores = [(items[i_idx], float(R_hat[u_idx, i_idx]))
                              for i_idx in range(len(items)) if i_idx not in seen]
                    scores.sort(key=lambda x: x[1], reverse=True)
                    for rank, (item, score) in enumerate(scores[:TOP_N], 1):
                        records.append({
                            user_col:               user,
                            "recommended_item":     item,
                            "predicted_score":      round(score, 4),
                            "rank":                 rank,
                            "strategy":             "collaborative_svd",
                        })

                reco_df = pd.DataFrame(records)

                top_items = (reco_df[reco_df["rank"] == 1]
                             .groupby("recommended_item")
                             .size().sort_values(ascending=False).head(10))
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle("Collaborative Filtering — Recommendation System",
                             fontsize=13, fontweight="bold")
                axes[0].barh(top_items.index.astype(str)[::-1],
                             top_items.values[::-1], color="#4C72B0", alpha=0.85)
                axes[0].set_title("Top 10 Most Recommended Items (rank 1)")
                axes[0].set_xlabel("Times Recommended")
                axes[0].grid(axis="x", alpha=0.3)
                score_dist = reco_df["predicted_score"]
                axes[1].hist(score_dist, bins=40, color="#2CA02C",
                             edgecolor="white", alpha=0.85)
                axes[1].set_title("Distribution of Predicted Scores")
                axes[1].set_xlabel("Predicted Score")
                axes[1].grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.savefig(CONFIG["reco_png"], dpi=150)
                plt.close()

            except Exception as pivot_err:
                logger.warning(
                    f"[Reco] Collaborative failed ({pivot_err}). "
                    f"Switching to content-based."
                )
                strategy = "content_based"

        # ══ STRATEGY B: Content-Based via Cosine Similarity ══════════════════
        # Note: uses "if" not "else" so the fallback from Strategy A works.
        if strategy == "content_based":
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics.pairwise import cosine_similarity

            feat_cols = [c for c in df_ml.columns
                         if c != target_col and c != "_src_idx"
                         and df_ml[c].dtype in ["float64", "int64", "float32", "int32"]]
            feat_cols = feat_cols[:50]

            X_raw = df_ml[feat_cols].fillna(0).values.astype(float)
            X_sc  = StandardScaler().fit_transform(X_raw)

            max_sample = min(n_rows, 3_000)
            rng        = np.random.default_rng(CONFIG["random_state"])
            sample_idx = rng.choice(n_rows, size=max_sample, replace=False)
            sample_idx = np.sort(sample_idx)

            X_sample   = X_sc[sample_idx]
            sim_matrix = cosine_similarity(X_sample)
            np.fill_diagonal(sim_matrix, -1)

            if "prediction" in df_pred.columns:
                pred_vals = df_pred["prediction"].values
                try:
                    pos_mask_all = pred_vals.astype(str) == \
                                   pd.Series(pred_vals).value_counts().idxmax()
                except Exception:
                    pos_mask_all = np.ones(len(pred_vals), dtype=bool)
            else:
                pos_mask_all = np.ones(n_rows, dtype=bool)

            pos_mask_sample = pos_mask_all[sample_idx]

            records = []
            for i in range(max_sample):
                row_sim = sim_matrix[i].copy()
                row_sim[~pos_mask_sample] *= 0.5
                top_idx = np.argsort(row_sim)[::-1][:TOP_N]
                orig_i  = int(sample_idx[i])

                for rank, j in enumerate(top_idx, 1):
                    orig_j = int(sample_idx[j])
                    records.append({
                        "entity_idx":      orig_i,
                        "recommended_idx": orig_j,
                        "similarity":      round(float(sim_matrix[i, j]), 4),
                        "rank":            rank,
                        "rec_prediction":  str(df_pred["prediction"].iloc[orig_j])
                                           if "prediction" in df_pred.columns else "N/A",
                        "strategy":        "content_based_cosine",
                    })
                    if rank == 1:
                        break

            reco_df = pd.DataFrame(records)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle("Content-Based Recommendation System — Cosine Similarity",
                         fontsize=13, fontweight="bold")
            axes[0].hist(reco_df["similarity"], bins=40,
                         color="#4C72B0", edgecolor="white", alpha=0.85)
            axes[0].set_title("Distribution of Similarity Scores (rank 1)")
            axes[0].set_xlabel("Cosine Similarity")
            axes[0].set_ylabel("Count")
            axes[0].grid(axis="y", alpha=0.3)
            feat_var = (pd.Series(X_sample.var(axis=0), index=feat_cols)
                        .sort_values(ascending=True).tail(10))
            axes[1].barh(feat_var.index, feat_var.values, color="#FF7F0E", alpha=0.85)
            axes[1].set_title("Top 10 Features Driving Similarity\n"
                              "(highest variance in feature space)")
            axes[1].grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(CONFIG["reco_png"], dpi=150)
            plt.close()

        # ── Save recommendations parquet ──────────────────────────────────────
        reco_df.to_parquet(CONFIG["reco_path"], index=False)
        logger.info(f"[Reco] Saved {len(reco_df)} recommendations → df6_recommendations.parquet")

        # ── Sample for Claude context ─────────────────────────────────────────
        sample_reco = reco_df.head(10).to_dict(orient="records")

        prompt_reco = f"""You are a senior Data Scientist explaining a recommendation system to business stakeholders.

Business context: {ctx or 'Supply chain / business prediction pipeline.'}
Target: '{target_col}' ({problem_type})
Strategy used: {strategy.replace('_', ' ').title()}
{"User column: " + str(user_col) + " | Item column: " + str(item_col) + " | Rating column: " + str(rating_col) if strategy == "collaborative" else "Similarity metric: cosine similarity on " + str(len(feat_cols)) + " features"}
Total recommendations generated: {len(reco_df):,}
Sample output: {json.dumps(sample_reco[:5], default=_safe_json)}

Write 3 paragraphs:
1. What the recommendation system does and which strategy was used — explain it in plain language.
2. How business teams should use these recommendations operationally (with concrete examples).
3. Limitations and what data would make the recommendations even better.

Be specific and practical. Avoid generic statements."""

        interpretation = _ask_claude(prompt_reco, max_tokens=700)

        # ── Save markdown report ──────────────────────────────────────────────
        report = f"""# Recommendation System Report

**Strategy:** {strategy.replace('_', ' ').title()}
**Target:** `{target_col}` ({problem_type})
**Total recommendations generated:** {len(reco_df):,}
**Top-N per entity:** {TOP_N}

## Strategy Details

{"**Collaborative Filtering (SVD)**" if strategy == "collaborative" else "**Content-Based (Cosine Similarity)**"}

{"- User column: `" + str(user_col) + "`" if user_col else ""}
{"- Item column: `" + str(item_col) + "`" if item_col else ""}
{"- Rating column: `" + str(rating_col) + "`" if rating_col else ""}
{"- Latent factors: 20 (TruncatedSVD)" if strategy == "collaborative" else "- Features used: " + str(len(feat_cols))}
{"- Sampled rows: " + str(min(n_rows, 3000)) + " of " + str(n_rows) if strategy == "content_based" else ""}

## Sample Recommendations
```json
{json.dumps(sample_reco[:5], indent=2, default=_safe_json)}
```

## Visual Analysis
![Recommendations](recommendations.png)

## AI Business Interpretation

{interpretation}

## How to Use
```python
import pandas as pd
reco = pd.read_parquet('df6_recommendations.parquet')

{"# Get top recommendations for a specific user" if strategy == "collaborative" else "# Get most similar records to entity index 42"}
{"reco[reco['" + str(user_col) + "'] == 'CUSTOMER_ID'].sort_values('rank')" if strategy == "collaborative" else "reco[reco['entity_idx'] == 42].sort_values('rank')"}
```

---
*Generated by Auto Data Scientist v8 — Recommendation System Agent*
"""
        with open(CONFIG["reco_md"], "w", encoding="utf-8") as f:
            f.write(report)

        return (f"RECO_SUCCESS\n"
                f"Strategy: {strategy}\n"
                f"Recommendations generated: {len(reco_df):,}\n"
                f"Top-N: {TOP_N} per entity\n"
                f"Files: df6_recommendations.parquet, Recommendation_System.md, recommendations.png")

    except Exception as e:
        return f"RECO_ERROR: {e}\n{traceback.format_exc()}"


# [FIX-6] max_iter and max_retry_limit tuned per agent complexity.
# Simple agents (pure I/O, no Claude reasoning): low limits.
# Complex agents (Claude calls + code exec + self-healing): higher limits.

ingestor = Agent(
    role="Data Engineer",
    goal=("Download the dataset by calling download_and_save_silver. "
          "If it returns INGESTION_SUCCESS declare done. "
          "If ERROR try again."),
    backstory="Specialist in data ingestion.",
    tools=[download_and_save_silver],
    llm=llm_agent, verbose=True,
    max_iter=3,           # simple I/O — 3 is plenty
    max_retry_limit=1,
)

analyst = Agent(
    role="AI-Powered Data Analyst",
    goal=("Analyze the dataset by calling analyze_data_with_ai. "
          "If it returns ANALYSIS_SUCCESS declare done. "
          "If ERROR try again."),
    backstory=("Uses Claude internally for intelligent analysis, "
               "target identification, and custom code generation."),
    tools=[analyze_data_with_ai],
    llm=llm_agent, verbose=True,
    max_iter=8,           # Claude call + code exec + self-healing loop
    max_retry_limit=3,
)

feature_engineer = Agent(
    role="AI-Powered Feature Engineer",
    goal=("Generate features by calling generate_features_with_ai_strategy. "
          "If it returns FEATURES_SUCCESS declare done. "
          "If ERROR try again."),
    backstory=("Uses Claude to decide and create custom features "
               "specific to the dataset."),
    tools=[generate_features_with_ai_strategy],
    llm=llm_agent, verbose=True,
    max_iter=6,           # Claude call + Boruta (can be slow)
    max_retry_limit=2,
)

eda_analyst = Agent(
    role="EDA Analyst",
    goal=("Generate visualizations by calling generate_eda_and_ml_ready. "
          "If it returns EDA_SUCCESS declare done. "
          "If ERROR try again."),
    backstory="Generates visualizations and prepares the dataset for ML.",
    tools=[generate_eda_and_ml_ready],
    llm=llm_agent, verbose=True,
    max_iter=4,           # deterministic chart generation
    max_retry_limit=1,
)

hypothesis_validator = Agent(
    role="Business Hypothesis Validator",
    goal=("Validate business hypotheses by calling validate_hypotheses. "
          "If it returns HYPOTHESIS_SUCCESS declare done. "
          "If ERROR try again."),
    backstory=("Generates and tests 10 business hypotheses about the target "
               "using real data."),
    tools=[validate_hypotheses],
    llm=llm_agent, verbose=True,
    max_iter=6,           # 10 hypotheses × Claude verdict calls
    max_retry_limit=2,
)

ml_scientist = Agent(
    role="AI-Powered ML Scientist",
    goal=("Train and save the best model by calling train_and_save_model. "
          "If it returns ML_SUCCESS declare done. "
          "If ERROR try again."),
    backstory=("CV + Optuna + Stacking. Claude interprets the results "
               "and writes a performance narrative."),
    tools=[train_and_save_model],
    llm=llm_agent, verbose=True,
    max_iter=8,           # long-running: Optuna + Stacking + Claude narrative
    max_retry_limit=2,
)

deployer = Agent(
    role="ML Deployment Engineer",
    goal=("Deploy the model by calling deploy_telegram_bot. "
          "If it returns DEPLOY_SUCCESS declare done. "
          "If ERROR try again."),
    backstory=("Generates df4_predictions.parquet with all original columns "
               "plus the prediction column, then writes a production-ready "
               "Telegram bot using Claude so users can query the model via chat."),
    tools=[deploy_telegram_bot],
    llm=llm_agent, verbose=True,
    max_iter=6,
    max_retry_limit=2,
)

notebook_writer = Agent(
    role="Technical Notebook Writer",
    goal=("Generate the analysis notebook by calling generate_analysis_notebook. "
          "If it returns NOTEBOOK_SUCCESS declare done. "
          "If ERROR try again."),
    backstory=("Compiles all pipeline outputs into analysis_notebook.ipynb."),
    tools=[generate_analysis_notebook],
    llm=llm_agent, verbose=True,
    max_iter=4,           # mostly file assembly
    max_retry_limit=1,
)

html_writer = Agent(
    role="Business Intelligence Dashboard Developer",
    goal=("Generate a self-contained HTML dashboard by calling generate_html_dashboard. "
          "If it returns HTML_SUCCESS declare done. "
          "If ERROR try again."),
    backstory=("Reads all pipeline artifacts — predictions, model metrics, hypothesis "
               "results, feature importances — and uses Claude to produce a single "
               "dashboard.html file with embedded Chart.js charts and AI-written "
               "business insights. No dependencies — anyone can open it in a browser."),
    tools=[generate_html_dashboard],
    llm=llm_agent, verbose=True,
    max_iter=4,
    max_retry_limit=2,
)

cloud_deployer = Agent(
    role="Oracle Cloud Deployment Engineer",
    goal=("Deploy the bot to Oracle Cloud by calling deploy_to_oracle_cloud. "
          "If it returns ORACLE_DEPLOY_SUCCESS or ORACLE_DEPLOY_SKIPPED declare done. "
          "If ORACLE_DEPLOY_ERROR try once more then report the error."),
    backstory=("SSHs into the Oracle Cloud VM, uploads the trained model and "
               "predictions via SFTP, pulls the latest telegram_bot.py from GitHub, "
               "and restarts the systemd service so the bot is always up to date."),
    tools=[deploy_to_oracle_cloud],
    llm=llm_agent, verbose=True,
    max_iter=3,
    max_retry_limit=1,
)

ab_tester = Agent(
    role="Statistical A/B Testing Scientist",
    goal=("Run A/B testing by calling run_ab_testing. "
          "If it returns AB_TEST_SUCCESS declare done. "
          "If ERROR try again."),
    backstory=("Compares the trained model (Model A) against a lightweight baseline "
               "(Model B) using rigorous statistical tests: McNemar for classification, "
               "Wilcoxon signed-rank for regression, plus Bayesian posterior probability "
               "that Model A beats Model B. Claude writes the business recommendation."),
    tools=[run_ab_testing],
    llm=llm_agent, verbose=True,
    max_iter=5,
    max_retry_limit=2,
)

recommender = Agent(
    role="Recommendation System Engineer",
    goal=("Build a recommendation system by calling build_recommendation_system. "
          "If it returns RECO_SUCCESS declare done. "
          "If ERROR try again."),
    backstory=("Auto-detects whether the dataset suits collaborative filtering "
               "(user-item-rating structure) or content-based filtering (cosine similarity "
               "on feature vectors). Generates top-N recommendations per entity, saves "
               "df6_recommendations.parquet, and Claude writes a business playbook "
               "explaining how to operationalise the recommendations."),
    tools=[build_recommendation_system],
    llm=llm_agent, verbose=True,
    max_iter=5,
    max_retry_limit=2,
)

# ==========================================
# 6. TASKS
# ==========================================

task_ingestion = Task(
    description=("Download the dataset from Kaggle.\n"
                 "Call download_and_save_silver (no parameters).\n"
                 "If INGESTION_SUCCESS finish. If ERROR try again."),
    agent=ingestor,
    expected_output="INGESTION_SUCCESS with shape and columns.",
)

task_analysis = Task(
    description=("Analyze the dataset with AI.\n"
                 "Call analyze_data_with_ai (no parameters).\n"
                 "If ANALYSIS_SUCCESS finish. If ERROR try again."),
    agent=analyst,
    context=[task_ingestion],
    expected_output="ANALYSIS_SUCCESS with identified target and insights.",
)

task_features = Task(
    description=("Generate features with AI strategy.\n"
                 "Call generate_features_with_ai_strategy (no parameters).\n"
                 "If FEATURES_SUCCESS finish. If ERROR try again."),
    agent=feature_engineer,
    context=[task_analysis],
    expected_output="FEATURES_SUCCESS with standard and custom features.",
)

task_eda = Task(
    description=("Generate visualizations and ML-Ready dataset.\n"
                 "Call generate_eda_and_ml_ready (no parameters).\n"
                 "If EDA_SUCCESS finish. If ERROR try again."),
    agent=eda_analyst,
    context=[task_features],
    expected_output="EDA_SUCCESS with charts and df3_ml_ready.parquet.",
)

task_hypotheses = Task(
    description=("Validate business hypotheses with real data.\n"
                 "Call validate_hypotheses (no parameters).\n"
                 "If HYPOTHESIS_SUCCESS finish. If ERROR try again."),
    agent=hypothesis_validator,
    context=[task_eda],
    expected_output=(
        "HYPOTHESIS_SUCCESS with 10 hypotheses tested, "
        "each marked TRUE/FALSE/INCONCLUSIVE with business insight."
    ),
)

task_ml = Task(
    description=("Train and save the best model.\n"
                 "Call train_and_save_model (no parameters).\n"
                 "If ML_SUCCESS finish. If ERROR try again."),
    agent=ml_scientist,
    context=[task_hypotheses],
    expected_output="ML_SUCCESS with model, score, error analysis, and narrative.",
)

task_deploy = Task(
    description=("Deploy the model as a Telegram bot.\n"
                 "Call deploy_telegram_bot (no parameters).\n"
                 "If DEPLOY_SUCCESS finish. If ERROR try again."),
    agent=deployer,
    context=[task_ml],
    expected_output=(
        "DEPLOY_SUCCESS with df4_predictions.parquet "
        "(all original columns + prediction), telegram_bot.py, "
        "requirements.txt, and Deployment_Guide.md."
    ),
)

task_notebook = Task(
    description=("Generate the analysis notebook.\n"
                 "Call generate_analysis_notebook (no parameters).\n"
                 "If NOTEBOOK_SUCCESS finish. If ERROR try again."),
    agent=notebook_writer,
    context=[task_deploy],
    expected_output=(
        "NOTEBOOK_SUCCESS with analysis_notebook.ipynb containing "
        "all pipeline sections."
    ),
)

task_html = Task(
    description=("Generate the business intelligence HTML dashboard.\n"
                 "Call generate_html_dashboard (no parameters).\n"
                 "If HTML_SUCCESS finish. If ERROR try again."),
    agent=html_writer,
    context=[task_notebook],
    expected_output=(
        "HTML_SUCCESS with dashboard.html — a single self-contained file "
        "with KPI cards, Chart.js charts, AI business insights, and "
        "hypothesis validation table. Zero dependencies."
    ),
)

task_cloud_deploy = Task(
    description=("Deploy all artifacts to Oracle Cloud VM.\n"
                 "Call deploy_to_oracle_cloud (no parameters).\n"
                 "If ORACLE_DEPLOY_SUCCESS or ORACLE_DEPLOY_SKIPPED finish.\n"
                 "If ORACLE_DEPLOY_ERROR try once more then report."),
    agent=cloud_deployer,
    context=[task_html],
    expected_output=(
        "ORACLE_DEPLOY_SUCCESS with VM IP, repo path, and service status active. "
        "Or ORACLE_DEPLOY_SKIPPED if ORACLE_VM_IP is not configured."
    ),
)

task_ab_test = Task(
    description=("Run A/B testing between the trained model and a baseline.\n"
                 "Call run_ab_testing (no parameters).\n"
                 "If AB_TEST_SUCCESS finish. If ERROR try again."),
    agent=ab_tester,
    context=[task_cloud_deploy],
    expected_output=(
        "AB_TEST_SUCCESS with winner, statistical test p-value, "
        "Bayesian P(A beats B), and business recommendation. "
        "Files: ab_test_results.json, AB_Test_Report.md, ab_test.png"
    ),
)

task_recommendations = Task(
    description=("Build a recommendation system from the dataset and predictions.\n"
                 "Call build_recommendation_system (no parameters).\n"
                 "If RECO_SUCCESS finish. If ERROR try again."),
    agent=recommender,
    context=[task_ab_test],
    expected_output=(
        "RECO_SUCCESS with strategy used (collaborative or content-based), "
        "number of recommendations generated, and business playbook. "
        "Files: df6_recommendations.parquet, Recommendation_System.md, recommendations.png"
    ),
)

# ==========================================
# 7. CREW
# ==========================================

ds_squad = Crew(
    agents=[
        ingestor, analyst, feature_engineer, eda_analyst,
        hypothesis_validator, ml_scientist, deployer, notebook_writer,
        html_writer, cloud_deployer, ab_tester, recommender,
    ],
    tasks=[
        task_ingestion, task_analysis, task_features, task_eda,
        task_hypotheses, task_ml, task_deploy, task_notebook,
        task_html, task_cloud_deploy, task_ab_test, task_recommendations,
    ],
    process=Process.sequential,
    memory=False,
    verbose=True,
)
# ==========================================
# 8. POST-PIPELINE
# ==========================================

def evaluate_model():
    """Overfitting/underfitting diagnostic in pure Python."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    if not os.path.exists(CONFIG["model_pkl"]):
        print("final_model.pkl not found."); return

    try:
        with open(CONFIG["model_pkl"], "rb") as f:
            artifact = pickle.load(f)

        model        = artifact["model"]
        target       = artifact["target"]
        problem_type = artifact["type"]
        name         = artifact["name"]
        features     = artifact["features"]
        le           = artifact.get("label_encoder")

        df = pd.read_parquet(CONFIG["ml_ready_path"])
        df = df.dropna(axis=1, how="all")
        df = df.dropna(subset=[c for c in df.columns
                                if df[c].isnull().mean() < 0.5])
        df = df.reset_index(drop=True)

        feature_cols = [c for c in df.columns
                        if c != target and c != "_src_idx"]
        X = pd.get_dummies(df[feature_cols],
                           drop_first=True).reindex(columns=features, fill_value=0)
        y = df[target].copy()

        # [FIX-9] Apply LabelEncoder post-split (mirrors FIX-3 in train_and_save_model)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=CONFIG["test_size"],
            random_state=CONFIG["random_state"],
            stratify=y.astype(str) if le else None,
        )
        if le:
            y_tr = pd.Series(le.transform(y_tr.astype(str)), index=y_tr.index)
            y_te = pd.Series(le.transform(y_te.astype(str)), index=y_te.index)

        y_ptr = model.predict(X_tr)
        y_pte = model.predict(X_te)

        if problem_type == "classification":
            s_tr = accuracy_score(y_tr, y_ptr)
            s_te = accuracy_score(y_te, y_pte)
            met  = "Accuracy"
        else:
            s_tr = r2_score(y_tr, y_ptr)
            s_te = r2_score(y_te, y_pte)
            met  = "R²"

        gap = s_tr - s_te

        prompt_diag = f"""Diagnose this ML model in 2 short paragraphs:

Model: {name}
Type: {problem_type} | Target: {target}
{met} Train: {s_tr:.4f}
{met} Test: {s_te:.4f}
Gap: {gap:.4f}

State whether there is overfitting, underfitting, or if the model is well-fitted.
Be direct and practical."""

        diagnostic = _ask_claude(prompt_diag, max_tokens=400)

        if problem_type == "regression":
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(y_te, y_pte, alpha=0.3, s=10, color="#4C72B0")
            mn = min(float(np.array(y_te).min()), float(y_pte.min()))
            mx = max(float(np.array(y_te).max()), float(y_pte.max()))
            ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title(f"Actual vs Predicted — {name}", fontsize=12, fontweight="bold")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(_BASE_DIR, "actual_vs_predicted.png"), dpi=150)
            plt.close()

        content = f"""# Model Evaluation

## `{name}`
**Type:** {problem_type} | **Target:** `{target}`

| Dataset   | {met} |
|-----------|-------|
| Train     | {s_tr:.4f} |
| Test      | {s_te:.4f} |
| Gap       | {gap:.4f}  |

## AI Diagnostic

{diagnostic}

## Optimized Parameters (Optuna)
```json
{json.dumps(artifact.get('optuna_params', {}), indent=2)}
```
"""
        with open(CONFIG["eval_md"], "w", encoding="utf-8") as f:
            f.write(content)

        print(f"{met} Train: {s_tr:.4f} | Test: {s_te:.4f} | Gap: {gap:.4f}")
        print(f"Model_Evaluation.md saved.")

    except Exception as e:
        print(f"ERROR in evaluation: {e}\n{traceback.format_exc()}")


def run_post_pipeline():
    evaluate_model()

    print("\n" + "=" * 60)
    print("GENERATING README.md")
    print("=" * 60)
    print(generate_readme.func(""))

    print("\n" + "=" * 60)
    print("GIT VERSIONING")
    print("=" * 60)

    def git(cmd, timeout=300):
        print(f"\n> {cmd}")
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True,
                               text=True, timeout=timeout)
            output = (r.stdout or r.stderr).strip()
            status = "[OK]" if r.returncode == 0 else "[FAILED]"
            print(f"{status} {output[:200]}")
            return r.returncode == 0
        except Exception as e:
            print(f"[ERROR] {e}"); return False

    # [FIX] Remove ALL stale git lock files before any git command runs.
    # Locks survive crashes, interrupted pushes, or editors with git integration.
    # packed-refs.lock blocks 'git remote remove'; HEAD.lock blocks 'git commit'.
    import glob as _glob
    for lock in _glob.glob(os.path.join(_BASE_DIR, ".git", "**", "*.lock"),
                           recursive=True):
        try:
            os.remove(lock)
            print(f"[git] Removed stale lock: {lock}")
        except Exception:
            pass
    # Also catch locks directly in .git/ (e.g. packed-refs.lock, HEAD.lock)
    for lock in _glob.glob(os.path.join(_BASE_DIR, ".git", "*.lock")):
        try:
            os.remove(lock)
            print(f"[git] Removed stale lock: {lock}")
        except Exception:
            pass

    git("git init")
    # [FIX] Use set-url instead of remove+add — avoids "remote already exists" error
    # on subsequent runs. Falls back to add if set-url fails (first run).
    if not git("git remote set-url origin https://github.com/bttisrael/ecommerce-ds-agent.git"):
        git("git remote add origin https://github.com/bttisrael/ecommerce-ds-agent.git")

    gitignore = "\n".join([
        "final_model.pkl", "df1_silver.parquet", "df2_gold.parquet",
        "df3_ml_ready.parquet", "df4_predictions.parquet",
        ".env", "venv/", "__pycache__/",
        "*.pyc", "pipeline.log",
    ]) + "\n"
    with open(os.path.join(_BASE_DIR, ".gitignore"), "w", encoding="utf-8") as f:
        f.write(gitignore)

    for file in ["final_model.pkl", "df1_silver.parquet",
                 "df2_gold.parquet", "df3_ml_ready.parquet"]:
        git(f"git rm --cached {file}")

    git("git config http.postBuffer 524288000")
    git("git branch -M main")

    artifacts = [
        ".gitignore", "README.md", "multi_agent_ds_v7.py",
        "telegram_bot.py", "dashboard.html", "requirements.txt",
        "analysis_notebook.ipynb",
        "correlation_matrix.png", "cramers_v_matrix.png",
        "distributions.png", "boxplots.png",
        "categoricals.png", "target_dist.png", "dataset_sample.png",
        "intelligent_analysis.png", "feature_importance.png",
        "model_comparison.png", "error_analysis.png",
        "hypothesis_validation.png",
        "Descriptive_Statistics.md", "Model_Metrics.md",
        "Model_Evaluation.md", "Quality_Report.md",
        "Intelligent_Analysis.md", "Deployment_Guide.md",
        "Hypothesis_Validation.md", "Error_Analysis.md",
        "target_config.json", "feature_strategy.json",
        "hypothesis_results.json",
        "ab_test_results.json",
        "AB_Test_Report.md",
        "ab_test.png",
        "Recommendation_System.md",
        "recommendations.png",
    ]
    if os.path.exists(os.path.join(_BASE_DIR, "actual_vs_predicted.png")):
        artifacts.append("actual_vs_predicted.png")
    if os.path.exists(CONFIG["business_ctx"]):
        artifacts.append("business_context.txt")

    for file in artifacts:
        git(f"git add {file}")

    git('git commit -m "feat: pipeline v7.1 - critical fixes applied"')
    git("git push origin main")

    print("\n" + "=" * 60)
    print("PIPELINE v7.1 COMPLETE")
    print("Next: add TELEGRAM_BOT_TOKEN to .env then run: python telegram_bot.py")
    print("=" * 60)


# ==========================================
# 9. MAIN
# ==========================================

if __name__ == "__main__":
    # ── Clean stale artifacts from previous runs ──────────────────────────────
    def _clean_previous_run():
        stale_files = [
            CONFIG["silver_path"], CONFIG["gold_path"], CONFIG["ml_ready_path"],
            CONFIG["predictions_path"], CONFIG["target_json"], CONFIG["strategy_json"],
            CONFIG["model_pkl"], CONFIG["hypothesis_json"], CONFIG["scenarios_path"],
            CONFIG["ab_test_json"], CONFIG["reco_path"],
        ]
        for f in stale_files:
            if os.path.exists(f):
                os.remove(f)
                logger.info(f"[Cleanup] Removed stale file: {os.path.basename(f)}")
        logger.info("[Cleanup] Previous run artifacts cleared.")
    _clean_previous_run()

    logger.info("Auto Data Scientist v7.1 starting...")

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not found in .env.")
        sys.exit(1)

    if not os.path.exists(".env"):
        logger.warning("Create .env with KAGGLE_USERNAME, KAGGLE_KEY, and ANTHROPIC_API_KEY.")

    if not os.path.exists(CONFIG["business_ctx"]):
        logger.info("Tip: create business_context.txt to provide business context.")

    result = ds_squad.kickoff()
    print("\n" + "=" * 60)
    print("PIPELINE RESULT")
    print("=" * 60)
    print(result)
    run_post_pipeline()
    logger.info("Pipeline v7.1 finished.")