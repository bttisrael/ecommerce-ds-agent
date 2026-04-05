# Hypothesis Validation

**Target:** `event_type` | TRUE: 1 | FALSE: 8 | INCONCLUSIVE: 1

| ID | Hypothesis | Verdict | Business Insight |
|----|-----------|---------|-----------------|
| H1 | Users with higher 'price' products in their sessions tend to have lower purchase | **FALSE** | Price alone does not deter purchases, and mid-to-high priced products may attrac |
| H2 | Users with higher 'feat_ratio' (engineered feature) tend to have higher purchase | **FALSE** | Users with lower feat_ratio values are actually stronger purchase intent signals |
| H3 | Users with higher 'feat_sum' values tend to have higher purchase event_type rate | **FALSE** | Mid-range engaged users (Q2) are the most likely to purchase, suggesting that ve |
| H4 | Sessions/Events with extreme 'price_zscore_abs' (far from mean price) tend to ha | **FALSE** | Unusually priced items do not systematically deter purchases, suggesting custome |
| H5 | Events associated with specific 'brand' values tend to have significantly differ | **INCONCLUSIVE** | Oral-b and tyrex show the highest conversion rates among displayed brands, sugge |
| H6 | Events from specific 'category_code' values tend to have higher purchase event_t | **TRUE** | The business should prioritize marketing spend and inventory optimization for hi |
| H7 | Users with higher 'user_price_ratio' tend to have higher purchase event_type rat | **FALSE** | Price-to-user-budget alignment does not appear to be a meaningful driver of purc |
| H8 | Events with higher 'log_price' values tend to have lower purchase event_type rat | **FALSE** | Price alone does not linearly drive conversion behavior, suggesting other factor |
| H9 | Events with higher 'feat_interact' values tend to have higher purchase event_typ | **FALSE** | Lower feat_interact values are actually stronger predictors of purchase intent,  |
| H10 | Users with higher 'price_per_product_magnitude' tend to have lower purchase even | **FALSE** | Outlier pricing relative to product magnitude does not discourage purchases, sug |
