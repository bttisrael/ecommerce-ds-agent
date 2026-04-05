# Error Analysis

## Model: `XGBoost` | Target: `event_type`

**Overall failure rate:** 0.0298 (3.0% of test samples misclassified)

## Classification Report
```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00      5493
           1       0.00      0.00      0.00      6412
           2       0.97      1.00      0.98    388095

    accuracy                           0.97    400000
   macro avg       0.32      0.33      0.33    400000
weighted avg       0.94      0.97      0.96    400000

```

## Error Analysis Chart
See `error_analysis.png` for confusion matrix and per-class accuracy.
