# A/B Test Report

**Winner:** `XGBoost`

```json
{
  "model_a": "XGBoost",
  "model_b": "LogisticRegression",
  "accuracy_a": 0.9609,
  "accuracy_b": 0.02,
  "delta": 0.9409,
  "mcnemar_p": 0.0,
  "p_a_beats_b": 1.0,
  "n_test": 100000,
  "winner": "XGBoost",
  "significant": true
}
```

## Interpretation

## A/B Test Interpretation: XGBoost vs. Logistic Regression

**1. Model Performance & Winner**
XGBoost is the clear and decisive winner of this A/B test, achieving an accuracy of **96.09%** compared to Logistic Regression's remarkably poor **2.0%** — a staggering delta of **94.09 percentage points**. The statistical evidence is overwhelming: the McNemar p-value rounds to **0.0** and the Bayesian probability of XGBoost beating Logistic Regression is **1.0 (100%)**, tested across a robust sample of **100,000 observations**. This is not a close race — it is a complete and total dominance by one model over the other.

**2. Business Significance**
A 2% accuracy rate from Logistic Regression is worse than random chance for most multi-class problems and barely better than random for binary classification, suggesting the model is **fundamentally broken** — likely due to misconfiguration, a preprocessing failure, missing feature scaling, or a data pipeline error rather than a genuine model limitation. Deploying or continuing to use Logistic Regression in this state would carry **serious business risk**, potentially leading to near-total misclassification, flawed decisions, financial loss, or customer harm depending on the use case.

**3. Recommendation**
**Deploy XGBoost immediately** and retire the Logistic Regression model from this pipeline. However, before closing the books, it is strongly recommended to **investigate why Logistic Regression failed so catastrophically** — since a properly tuned logistic regression should perform far better than 2%. Diagnosing the root cause (e.g., unscaled features, label encoding errors, class imbalance mishandling) will strengthen your overall ML pipeline hygiene and prevent similar failures in future experiments.
