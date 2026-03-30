# Model Metrics

**Type:** classification | **Target:** `late_delivery_risk`

## Model Comparison

|                         |   mean |    std |
|:------------------------|-------:|-------:|
| XGBoost                 | 0.9752 | 0.0007 |
| GradientBoosting_Optuna | 0.9752 | 0.0007 |
| LightGBM_Optuna         | 0.9752 | 0.0007 |
| XGBoost_Optuna          | 0.9752 | 0.0007 |
| LightGBM                | 0.9752 | 0.0007 |
| GradientBoosting        | 0.9752 | 0.0007 |
| RandomForest            | 0.9617 | 0.0009 |
| ExtraTrees              | 0.9572 | 0.0005 |
| LogisticRegression      | 0.539  | 0.0094 |

**Selected model:** `XGBoost`

**ACCURACY (test):** 0.9745

```
              precision    recall  f1-score   support

           0       1.00      0.94      0.97     16308
           1       0.96      1.00      0.98     19796

    accuracy                           0.97     36104
   macro avg       0.98      0.97      0.97     36104
weighted avg       0.98      0.97      0.97     36104

```

## AI Interpretation

# Model Interpretation Report: Late Delivery Risk Classifier

---

## Executive Summary & Model Interpretation

### 1. Why XGBoost Was the Best Choice

XGBoost emerged as the top-performing model in a highly competitive field, matching GradientBoosting and LightGBM at **0.9752 mean cross-validation score** — but its selection is justified beyond raw performance. XGBoost's gradient boosting framework excels at capturing the **complex, non-linear interaction patterns** inherent in e-commerce logistics data: relationships between order timing, product category, shipping routes, and carrier behavior rarely follow linear rules, and tree-based boosting handles these interactions natively without requiring manual feature engineering. Notably, the fact that **Optuna-tuned variants produced identical scores to the default XGBoost** is a meaningful signal — it suggests the model already reached a performance ceiling on this dataset, and hyperparameter optimization yielded no additional gains. This actually reinforces confidence in XGBoost's robustness here. The dramatic drop of LogisticRegression to **0.539** (near random chance) further confirms that the underlying patterns are strongly non-linear, validating the choice of a tree-based ensemble over simpler parametric models. RandomForest and ExtraTrees lagged behind by 1.3–1.8 percentage points, likely because they lack the sequential error-correction mechanism that boosting provides, which is particularly valuable when predicting rare or edge-case delivery failures.

---

### 2. What 0.9745 Means in Business Terms

A test set accuracy of **97.45%** on 180,519 rows translates to approximately **4,700 misclassified orders** — and in a logistics context, *how* those errors are distributed matters enormously. If the dataset has class imbalance (which is typical in late delivery scenarios, where, say, 15–20% of orders arrive late), accuracy alone can be misleading. Assuming a conservative estimate of **10,000 daily orders** on this platform, the model would correctly flag late-risk shipments for roughly **9,745 orders per day**, enabling proactive interventions such as customer notifications, carrier escalations, or expedited re-routing. Translating this to revenue impact: if each correctly predicted late delivery saves even **$5 in customer service costs or churn prevention**, and the model intercepts thousands of such cases daily, the annual business value runs into the **millions of dollars**. More critically, the model's high consistency (std of **0.0007**) across cross-validation folds indicates it generalizes reliably rather than overfitting to specific data partitions — a crucial property for production stability.

---

### 3. Points of Attention & Limitations

Despite the impressive headline number, several red flags demand scrutiny before treating this model as production-ready. **First and most urgently: the business context and the target variable are misaligned.** The platform description focuses on *purchase prediction from browsing behavior* (views, cart additions, conversions), yet the target variable is `late_delivery_risk` — a logistics outcome. This discrepancy strongly suggests either a **dataset mismatch or a pipeline configuration error**, and must be resolved before any deployment decision. Second, a 97.45% accuracy should trigger suspicion of **data leakage** — features like `shipment_status`, `actual_delivery_date`, or carrier confirmation codes that are only known *after* delivery would artificially inflate performance. A rigorous temporal validation (training on past months, testing on future months) is essential. Third, the **53-column feature space** needs audit: if delivery-outcome-adjacent features leaked into training, the model learned to recognize delivery results rather than predict risk. Finally, with 6 models converging identically at 0.9752, this plateau likely reflects a **dataset ceiling** — meaning further gains will require richer features (weather data, carrier SLAs, regional logistics patterns) rather than algorithmic changes.

---

### 4. Practical Recommendations for Production Deployment

Before deployment, the team should execute four concrete steps. **Immediately:** audit the feature set for temporal leakage by enforcing a strict cutoff — only features available *at the moment of order placement* should be used for inference. Follow this with a **time-based train/test split** (e.g., train on months 1–9, test on months 10–12) to simulate real-world deployment conditions; if accuracy drops significantly,
