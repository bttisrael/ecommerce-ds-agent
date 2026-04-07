# Error Analysis

## Model: `XGBoost` | Target: `event_type`

**Overall failure rate:** 0.0275 (2.8% of test samples misclassified)

## Classification Report
```
              precision    recall  f1-score   support

           0       0.83      0.00      0.01     12819
           1       0.00      0.00      0.00     14755
           2       0.97      1.00      0.99    972426

    accuracy                           0.97   1000000
   macro avg       0.60      0.33      0.33   1000000
weighted avg       0.96      0.97      0.96   1000000

```

## Error Analysis Chart
See `error_analysis.png` for confusion matrix and per-class accuracy.
