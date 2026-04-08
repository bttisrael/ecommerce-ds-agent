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
See `error_analysis.png` for confusion matrix and per-class accuracy.
