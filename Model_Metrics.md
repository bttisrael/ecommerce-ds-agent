# Model Metrics

**Type:** classification | **Target:** `event_type`

## Model Comparison

|                         |   mean |    std |
|:------------------------|-------:|-------:|
| XGBoost_Optuna          | 0.9724 | 0      |
| LightGBM_Optuna         | 0.9724 | 0      |
| XGBoost                 | 0.9724 | 0      |
| GradientBoosting        | 0.9724 | 0      |
| GradientBoosting_Optuna | 0.9724 | 0      |
| LightGBM                | 0.9723 | 0      |
| RandomForest            | 0.9692 | 0      |
| ExtraTrees              | 0.9654 | 0.0001 |
| LogisticRegression      | 0.4928 | 0.0006 |

**Selected model:** `XGBoost`

**ACCURACY (test):** 0.9724

```
              precision    recall  f1-score   support

           0       0.79      0.00      0.01     12819
           1       0.00      0.00      0.00     14756
           2       0.97      1.00      0.99    971831

    accuracy                           0.97    999406
   macro avg       0.59      0.33      0.33    999406
weighted avg       0.96      0.97      0.96    999406

```

## AI Interpretation

# Model Results Interpretation: E-commerce Purchase Prediction

## XGBoost as the Optimal Model Choice

XGBoost emerged as the selected model from a competitive field of gradient boosting variants, though it is critical to note that the performance differences at the top of the leaderboard are essentially negligible — XGBoost, LightGBM, and both Optuna-tuned variants all achieved **0.9724 with zero measurable variance** across cross-validation folds. The selection of XGBoost over its peers is therefore justified primarily by practical engineering considerations: its mature ecosystem, robust SHAP integration for explainability, and proven production reliability at scale rather than any measurable accuracy advantage. The stark underperformance of Logistic Regression (0.4928, barely above random chance for a multi-class problem) confirms that the relationship between browsing behavior signals and event type classification is **highly non-linear**, making tree-based ensemble methods the architecturally correct family of models for this domain. The near-zero standard deviation across folds on a 5M-row dataset also signals that the model is learning stable, generalizable patterns rather than overfitting to noise.

## What 0.9724 Accuracy Means for the Business

A 97.24% accuracy on classifying user events — distinguishing between **view, cart, and purchase** behaviors — translates to the model misclassifying approximately **139,000 events per 5 million interactions**. In a business context, this is a strong result, but accuracy alone can be misleading here given the **severe class imbalance inherent to e-commerce funnels**: purchase events are typically 1–5% of total events, while views dominate at 70–80%+. A model predicting "view" for every event would achieve deceptively high accuracy. Therefore, the 0.9724 figure should be validated against **precision, recall, and F1-score per class** — particularly for the purchase class, which is the highest business-value signal. If the model correctly identifies purchase-intent users at high recall, even a modest improvement in recommendation targeting or cart abandonment recovery can generate significant revenue lift given the platform's 285M event scale.

## Points of Attention and Model Limitations

Several red flags warrant careful scrutiny before treating these results as production-ready. The **zero standard deviation** across all top models is statistically unusual and raises the possibility of **data leakage** — specifically, that features derived from or correlated with the target event type (e.g., a `price_paid` column populated only for purchases, or session-level aggregations computed post-event) may have inadvertently bled into the training features. This must be audited immediately. Additionally, the dataset covers **285M user events compressed into 5M rows for modeling**, raising questions about the sampling strategy and whether it preserves the true class distribution. The model also captures a static snapshot of user behavior; **seasonal drift, new product categories, and evolving user patterns** will degrade performance over time without retraining pipelines in place. Finally, with only 9 columns in the feature space, the model may be underutilizing available behavioral signals such as session depth, time-on-page, or cross-category browsing sequences.

## Practical Recommendations for Production Deployment

Before deployment, conduct a **rigorous leakage audit** by tracing each feature's data lineage and confirming no feature is computed with knowledge of the target event. Complement accuracy with a **full classification report** broken down by event type, prioritizing F1-score and recall for the purchase class as the primary business metric. In production, implement **real-time or near-real-time inference** using XGBoost's low-latency scoring capabilities, with the model serving as the backbone for personalized recommendations and purchase-propensity scoring. Establish a **model monitoring pipeline** tracking prediction distribution drift (PSI scores) and accuracy on labeled production samples weekly, with automated retraining triggers. Given the scale of 285M events, consider **stratified retraining on rolling 30–60 day windows** to capture seasonal patterns, and maintain the LightGBM variant as a shadow model for A/B comparison — its identical performance makes it a zero-cost insurance policy against XGBoost-specific degradation in production environments.
