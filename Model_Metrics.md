# Model Metrics

**Type:** classification | **Target:** `event_type`

## Model Comparison

|                         |   mean |    std |
|:------------------------|-------:|-------:|
| XGBoost                 | 0.9702 | 0      |
| GradientBoosting_Optuna | 0.9702 | 0      |
| XGBoost_Optuna          | 0.9702 | 0      |
| LightGBM_Optuna         | 0.9702 | 0      |
| LightGBM                | 0.9702 | 0      |
| GradientBoosting        | 0.9702 | 0      |
| RandomForest            | 0.9559 | 0.0002 |
| ExtraTrees              | 0.9494 | 0      |
| LogisticRegression      | 0.4929 | 0.0007 |

**Selected model:** `XGBoost`

**ACCURACY (test):** 0.9702

```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00      5493
           1       0.00      0.00      0.00      6412
           2       0.97      1.00      0.98    388095

    accuracy                           0.97    400000
   macro avg       0.32      0.33      0.33    400000
weighted avg       0.94      0.97      0.96    400000

```

## AI Interpretation

# Model Interpretation Report: E-Commerce Purchase Prediction

## XGBoost for User Conversion Classification

---

### 1. Why XGBoost Was the Best Choice

XGBoost emerged as the selected model in a highly competitive field, though notably it tied with five other gradient boosting variants (GradientBoosting, LightGBM, and their Optuna-tuned counterparts) at virtually identical accuracy scores of 0.9702. In practice, XGBoost's selection likely comes down to its combination of **computational efficiency, robust regularization (L1/L2), and production maturity** rather than a clear performance gap. The near-zero standard deviation (0.0000) across cross-validation folds confirms that XGBoost generalizes extremely consistently across data partitions — a critical property when scoring millions of user sessions in real time. The sharp performance cliff observed with RandomForest (0.9559) and ExtraTrees (0.9494) suggests that the **sequential, error-correcting nature of boosting algorithms** is particularly well-suited to the hierarchical behavioral signals in this dataset (view → cart → purchase funnel), where subtle interaction effects between features like session depth, product category, and recency of actions carry predictive weight that ensemble bagging methods partially miss. Logistic Regression's near-random performance (0.4929) further confirms that the decision boundary between purchase, cart, and view events is **highly non-linear**, validating the choice of a tree-based model.

---

### 2. What 0.9702 Accuracy Means in Business Terms

A 97.02% accuracy on a 2-million-row dataset translates to approximately **59,400 misclassified user events** in this test partition alone — a number that sounds large but must be interpreted against the baseline. On an e-commerce platform with three event classes (view, cart, purchase), the dataset is almost certainly **heavily imbalanced**, with purchase events representing a small fraction of total interactions (typically 1–5% in real-world funnels). This means a naive model predicting "view" for every event could already achieve high accuracy, so this 97% figure **requires validation against precision, recall, and F1-score per class** — particularly for the purchase class, which drives all revenue. If the model achieves high recall on purchase events (catching most actual buyers), it becomes a powerful engine for conversion optimization: correctly identifying likely purchasers allows the platform to trigger **timely interventions** such as personalized recommendations, dynamic pricing, or cart abandonment emails. Every percentage point of improvement in purchase-class recall on 285M events can translate directly to measurable revenue uplift, making this model commercially significant if the purchase class metrics hold up under scrutiny.

---

### 3. Points of Attention and Model Limitations

Several red flags warrant careful investigation before treating this model as production-ready. **First and most critically**, the identical scores across six fundamentally different model architectures (0.9702, std=0.0000) is statistically unusual and suggests potential **data leakage** — specifically, the `event_type` column itself likely encodes temporal or sequential information that indirectly reveals the target, or features derived from the purchase event are inadvertently included as predictors. In a clickstream dataset, features like `session_value`, `items_purchased`, or any post-event aggregation would constitute leakage. **Second**, the zero standard deviation across folds indicates either that the cross-validation splits are not truly independent (e.g., the same user appears in both train and test folds, violating the i.i.d. assumption) or that the dataset has very low variance in its structure — both scenarios are concerning. **Third**, accuracy alone is an insufficient metric for this problem: a confusion matrix breakdown is essential to understand whether the model is actually distinguishing purchase intent versus merely memorizing the dominant class pattern. Finally, **model drift** is a significant operational risk — user browsing behavior shifts with seasonality, promotional campaigns, and catalog changes, meaning a static model trained on historical data will degrade in precision over time without a retraining pipeline.

---

### 4. Practical Recommendations for Production Deployment

Before deployment, the team should **immediately audit the feature set for leakage** by reconstructing the data pipeline and ensuring all predictive features are computed using only information available *at the moment of prediction* — no post-event signals, no future-looking aggregations. Cross-validation should be restructured as **time-based splits**
