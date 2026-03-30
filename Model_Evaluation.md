# Model Evaluation

## `XGBoost`
**Type:** classification | **Target:** `late_delivery_risk`

| Dataset   | Accuracy |
|-----------|-------|
| Train     | 0.9758 |
| Test      | 0.9745 |
| Gap       | 0.0013  |

## AI Diagnostic

## Diagnosis: Well-Fitted Model

The model shows **no signs of overfitting or underfitting**. The train/test accuracy gap of just 0.0013 (0.13%) is negligible, indicating the model generalizes extremely well to unseen data. Both scores sitting above 97% confirm the model learned meaningful patterns from the data rather than memorizing noise.

## Practical Recommendation

The model is **production-ready** as-is. The only flags worth investigating are unrelated to fit quality: verify that the **97.45% test accuracy holds across balanced classes** (check precision/recall/F1 if late deliveries are a minority class, since high accuracy can be misleading in imbalanced datasets). Also confirm the test set was kept strictly separate during training to rule out any data leakage inflating these numbers.

## Optimized Parameters (Optuna)
```json
{
  "n_estimators": 63,
  "learning_rate": 0.023386742875571638,
  "max_depth": 7,
  "subsample": 0.5830823325796838
}
```
