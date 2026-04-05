# Hypothesis Validation

**Target:** `event_type` | TRUE: 2 | FALSE: 5 | INCONCLUSIVE: 3

| ID | Hypothesis | Verdict | Business Insight |
|----|-----------|---------|-----------------|
| H1 | Higher-priced products are less likely to result in a purchase event compared to | **INCONCLUSIVE** | Price alone is not a reliable predictor of purchase conversion, suggesting other |
| H2 | Users who interact with more products within a single session (higher feat_sum)  | **FALSE** | Users with extremely high product interactions may be browsing or comparing with |
| H3 | Certain product categories (category_code) drive disproportionately higher purch | **INCONCLUSIVE** | Non-electronics categories such as fans, belts, and diapers drive the highest co |
| H4 | Specific brands are associated with higher purchase conversion rates, suggesting | **TRUE** | Marketing efforts should prioritize high-converting brands like uralsport and la |
| H5 | The time of day embedded in event_time influences event_type distribution, with  | **FALSE** | Marketing and promotional efforts should be targeted at early morning hours (4-8 |
| H6 | A higher category_price_ratio (product price relative to category average) is ne | **FALSE** | Price relative to category average does not straightforwardly deter purchases, s |
| H7 | Users with higher user_price_interact values (a proxy for total spend potential) | **INCONCLUSIVE** | Since high-value users do not consistently convert at higher rates, targeted re- |
| H8 | Users with more events per session (higher feat_ratio) are more likely to purcha | **FALSE** | Users who trigger fewer events per session are actually more likely to convert,  |
| H9 | The log-transformed price (log_price) shows a stronger linear relationship with  | **TRUE** | Although both correlations are extremely weak in absolute terms, the stronger lo |
| H10 | The interaction feature feat_interact captures a combined signal from user and p | **FALSE** | The interaction feature does not reliably distinguish purchase intent from brows |
