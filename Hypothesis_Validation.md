# Hypothesis Validation

**Target:** `event_type` | TRUE: 1 | FALSE: 9 | INCONCLUSIVE: 0

| ID | Hypothesis | Verdict | Business Insight |
|----|-----------|---------|-----------------|
| H1 | Users with a higher 'price_zscore' (premium-priced products) tend to have a lowe | **TRUE** | Extremely high-priced products are almost exclusively browsed and rarely purchas |
| H2 | Products with a higher 'feat_ratio' (engineered feature capturing relative produ | **FALSE** | Products with lower feat_ratio values are actually more likely to convert to pur |
| H3 | Users with a lower 'user_price_ratio' (user's interaction price relative to a ba | **FALSE** | Higher-spending, premium-oriented users are actually the most valuable converter |
| H4 | Products associated with specific 'brand' values tend to have significantly diff | **FALSE** | Since purchase conversion rates are remarkably consistent across brands regardle |
| H5 | Products in certain 'category_code' segments (e.g., electronics vs. apparel) ten | **FALSE** | Stationery and electronics drive the highest purchase conversion rates, suggesti |
| H6 | Products with a higher 'category_deviation' (price deviation from category avera | **FALSE** | Products priced near the category average convert best, but premium-priced items |
| H7 | Users with higher 'feat_interact' values (interaction feature combining product  | **FALSE** | The feat_interact feature appears to capture disengagement or browse-without-buy |
| H8 | Events occurring in certain 'price_bucket_x_log_product' segments tend to have s | **FALSE** | The interaction feature between price bucket and log product popularity does not |
| H9 | Products with lower 'price_quantile' values (i.e., positioned in the cheapest ti | **FALSE** | Since mid-priced products drive the highest conversion rates, the business shoul |
| H10 | User sessions ('user_session') with higher 'feat_sum' values (aggregate session- | **FALSE** | Sessions with moderate product interaction signals convert better than those wit |
