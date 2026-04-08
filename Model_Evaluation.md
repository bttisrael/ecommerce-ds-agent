# Model Evaluation

## `XGBoost`
**Type:** classification | **Target:** `event_type`

| Dataset   | Accuracy |
|-----------|-------|
| Train     | 0.9724 |
| Test      | 0.9724 |
| Gap       | 0.0000  |

## AI Diagnostic

## Diagnosis: Well-Fitted Model

The model shows nearly identical train and test accuracy (97.24% both), with a gap of exactly 0.0000, which strongly indicates the model generalizes well to unseen data. There is **no overfitting or underfitting**. XGBoost's built-in regularization (L1/L2) and ensemble structure are working effectively here, preventing the model from memorizing the training data while still capturing meaningful patterns.

## Practical Considerations

Despite the clean metrics, a zero gap can occasionally signal **data leakage** (e.g., target-correlated features accidentally included) or an overly simple problem — worth a quick sanity check on feature sources. Also verify that the test set was properly held out and that class distribution is balanced, since high accuracy on imbalanced targets can be misleading. If those checks pass, the model is production-ready as-is, with no immediate need for regularization tuning or architectural changes.

## Optimized Parameters (Optuna)
```json
{
  "n_estimators": 142,
  "learning_rate": 0.2055855551931528,
  "max_depth": 7,
  "subsample": 0.6094732414399233
}
```
