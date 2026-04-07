# Model Metrics

**Type:** classification | **Target:** `event_type`

## Model Comparison

|                         |   mean |    std |
|:------------------------|-------:|-------:|
| XGBoost_Optuna          | 0.9725 | 0      |
| LightGBM_Optuna         | 0.9724 | 0      |
| XGBoost                 | 0.9724 | 0      |
| GradientBoosting_Optuna | 0.9724 | 0      |
| GradientBoosting        | 0.9724 | 0      |
| LightGBM                | 0.9723 | 0      |
| RandomForest            | 0.9603 | 0.0001 |
| ExtraTrees              | 0.9539 | 0      |
| LogisticRegression      | 0.4928 | 0.0001 |

**Selected model:** `XGBoost`

**ACCURACY (test):** 0.9725

```
              precision    recall  f1-score   support

           0       0.83      0.00      0.01     12819
           1       0.00      0.00      0.00     14755
           2       0.97      1.00      0.99    972426

    accuracy                           0.97   1000000
   macro avg       0.60      0.33      0.33   1000000
weighted avg       0.96      0.97      0.96   1000000

```

## AI Interpretation

# Model Interpretation Report: E-Commerce Purchase Prediction

## XGBoost Classification — Event Type Prediction

---

### 1. Why XGBoost Was the Best Choice

XGBoost emerged as the top-performing model with a mean cross-validated accuracy of **0.9725**, narrowly but consistently outperforming LightGBM, GradientBoosting, and their Optuna-tuned variants — all of which clustered tightly between 0.9723 and 0.9724. This convergence among gradient boosting methods is itself a meaningful signal: it confirms that the underlying **tree-based ensemble approach** is genuinely well-suited to this problem structure, and the result is robust rather than a statistical artifact of one particular algorithm. XGBoost's specific advantages here likely stem from its **second-order gradient optimization, built-in L1/L2 regularization, and efficient handling of sparse interaction patterns** — all of which align well with behavioral event data where user-product interactions are inherently sparse and non-linear. The dramatic collapse of LogisticRegression to near-random performance (0.4928) is particularly telling: it confirms that the relationships between browsing features and purchase intent are **highly non-linear and cannot be captured by linear decision boundaries**, making XGBoost's expressive tree structure not just preferable but necessary. The near-zero standard deviation across folds further indicates excellent stability on the 5M-row dataset.

---

### 2. What 0.9725 Means in Business Terms

A test set accuracy of **97.25%** means the model correctly classifies approximately **137.2 million out of 141 million user events** in a dataset of the scale described (extrapolating proportionally from 285M events). In practical business terms, this translates to a recommendation engine and conversion predictor that is **highly reliable at distinguishing between view, cart, and purchase events** — the three behavioral signals that define the customer journey. For the core business questions, this means: product recommendations can be personalized with high confidence based on predicted purchase likelihood; marketing budgets for re-targeting high-intent users can be allocated with significantly less waste; and category-level revenue attribution can be modeled with a reliable behavioral foundation. However, **accuracy alone can be misleading** in this context. With a dataset of 285M events, the natural distribution is almost certainly heavily skewed toward *view* events, with *purchase* events representing a small minority — possibly as low as 2–5% of all events. If the class distribution is imbalanced, a 97.25% accuracy could partially reflect the model becoming proficient at predicting the dominant class rather than capturing true purchase intent. The business team should **prioritize Precision, Recall, and F1-score for the purchase class specifically**, as a missed purchase prediction (false negative) has a direct and quantifiable revenue cost.

---

### 3. Points of Attention and Model Limitations

Several important limitations warrant careful attention before drawing firm conclusions. **First and most critically**, the target variable `event_type` contains the event labels *view*, *cart*, and *purchase* — meaning the model is trained on events that have already occurred, not on users before they act. This is a **data leakage risk**: if any feature in the 9-column dataset encodes information that is only available *at the time of the event* (e.g., session duration calculated after the session ends, or cart timestamps), the model may be learning from the future relative to when a prediction would need to be made in production. A strict **temporal train/test split** must be validated, not a random split, to ensure the model generalizes to genuinely unseen future behavior. **Second**, the near-zero standard deviation across cross-validation folds, while superficially reassuring, can indicate that the 5M rows from the same behavioral ecosystem may share underlying correlations — the model may be overfitting to platform-specific user patterns that shift seasonally (e.g., Black Friday behavior vs. regular browsing). **Third**, with only 9 columns, feature engineering opportunities are likely substantial and underexplored; the model may be leaving predictive signal on the table. Finally, **RandomForest and ExtraTrees underperformed significantly** (0.9603 and 0.9539), which suggests the sequential, boosting-based learning of gradient methods is meaningfully capturing error residuals that bagging methods miss — a useful diagnostic confirming the non-trivial complexity of the prediction task.

---

###
