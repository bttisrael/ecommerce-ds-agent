# Model Evaluation

## `XGBoost`
**Type:** classification | **Target:** `event_type`

| Dataset   | Accuracy |
|-----------|-------|
| Train     | 0.9702 |
| Test      | 0.9702 |
| Gap       | 0.0000  |

## AI Diagnostic

## Diagnosis: Well-Fitted Model

The model shows near-identical train and test accuracy (97.02% both), with a gap of exactly 0.00, which strongly indicates the model is **well-fitted**. XGBoost is handling the classification task effectively, generalizing well to unseen data without memorizing training patterns. This is the ideal scenario.

## Practical Considerations

Despite the clean numbers, the **perfect 0.0000 gap deserves a second look** — it can occasionally signal data leakage, an overly representative train/test split, or a dataset with low variance. Verify that the split was done correctly (random shuffle, no target-correlated features leaking in) and check performance on class-level metrics (precision, recall, F1 per `event_type`), especially if classes are imbalanced. If everything checks out, the model is production-ready.

## Optimized Parameters (Optuna)
```json
{
  "n_estimators": 205,
  "learning_rate": 0.01005509835164253,
  "max_depth": 3,
  "subsample": 0.9972228926059423
}
```
