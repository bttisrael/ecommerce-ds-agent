# Model Metrics

**Type:** classification | **Target:** `event_type`

## Model Comparison

|                         |   mean |    std |
|:------------------------|-------:|-------:|
| XGBoost_Optuna          | 0.9609 | 0      |
| LightGBM                | 0.9609 | 0      |
| GradientBoosting_Optuna | 0.9609 | 0      |
| XGBoost                 | 0.9609 | 0      |
| LightGBM_Optuna         | 0.9609 | 0      |
| GradientBoosting        | 0.9609 | 0      |
| RandomForest            | 0.9582 | 0      |
| ExtraTrees              | 0.955  | 0.0001 |
| LogisticRegression      | 0.4919 | 0.0028 |

**Selected model:** `XGBoost`

**ACCURACY (test):** 0.9609

```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00      2610
           1       0.00      0.00      0.00      1299
           2       0.96      1.00      0.98     96091

    accuracy                           0.96    100000
   macro avg       0.32      0.33      0.33    100000
weighted avg       0.92      0.96      0.94    100000

```

## AI Interpretation

# ML Results Interpretation

## Why XGBoost Won (And What the Score Actually Means)

XGBoost was selected as the winning model with a test accuracy of 96.09%, tied with LightGBM and GradientBoosting variants. The selection of XGBoost over its equally-performing peers likely came down to a tiebreaker such as training speed, memory efficiency, or convention — practically speaking, all three gradient boosting frameworks are functionally equivalent here. The more revealing story is the dramatic performance cliff between ensemble tree methods (~95-96%) and Logistic Regression (~49%). This gap tells us the relationship between browsing behavior and event type is **highly non-linear**, with complex interaction effects that tree-based models capture naturally through recursive feature splitting. The near-zero standard deviation across all folds also indicates the model is extremely stable across data partitions, which is a positive signal for deployment reliability.

## What 96% Accuracy Really Means for the Business

Here is where you need to pump the brakes on excitement. That 96% accuracy is **almost entirely explained by the class imbalance in your data**, not genuine predictive power. Your dataset is composed of approximately 96.1% views, 2.6% carts, and 1.3% purchases. A completely naive model that predicts "view" for every single user event would achieve roughly **96% accuracy without learning anything at all**. This is textbook class imbalance masking, and it means the headline metric is essentially meaningless for your core business questions. What the business actually cares about — identifying the 1.3% of events that result in a purchase, or the 2.6% that reach the cart — is precisely what this accuracy score tells you nothing about. You need to immediately pull **precision, recall, F1-score, and AUC-ROC broken down by class**, with particular focus on the minority purchase and cart classes.

## Critical Limitations to Address Before Trusting This Model

Several red flags deserve serious attention before acting on these results. First, as described above, the **target leakage / imbalance problem** means the model may have learned to nearly always predict "view" and still score well. Second, the fact that the target variable is `event_type` itself raises a conceptual concern — you are classifying what an event *is*, not predicting what a user *will do next*, which is the actual business goal stated in your brief. If `event_type` is recorded at the time of the event, the model may be trivially learning to re-label known events rather than forecasting future behavior, which would constitute **data leakage** from the feature construction process. Third, the perfectly identical scores across XGBoost, LightGBM, and GradientBoosting (0.9609 with 0.0000 std) is statistically suspicious and warrants investigation — this could indicate the models are all converging on the same dominant-class prediction strategy rather than genuinely learning.

## Deployment Recommendations

Before any production deployment, the following steps are strongly advised:

- **Reframe the modeling problem**: Shift the target to a binary `will_purchase` flag, or build a sequential session model that predicts the *next* event given prior events in a browsing session
- **Rebalance the training data**: Apply SMOTE, class weighting (`scale_pos_weight` in XGBoost), or strategic undersampling to force the model to learn purchase and cart signals
- **Replace accuracy with business-aligned metrics**: Optimize for **Recall on purchase class** (catch as many real buyers as possible) and monitor **Precision** to avoid over-recommending; set an explicit operating threshold based on revenue impact per recommendation
- **Audit features for leakage**: Confirm that no feature in the 9-column dataset is derived from or correlated with the event outcome after the fact
- **A/B test incrementally**: Deploy the recommendation engine to a small user segment first, measuring actual conversion lift against a baseline recommender rather than relying solely on offline accuracy metrics

The infrastructure around XGBoost is sound and the model framework is the right choice — the work remaining is in problem formulation and evaluation design, not in the algorithm selection itself.
