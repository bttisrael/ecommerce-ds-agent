# Hypothesis Validation

**Target:** `event_type` | TRUE: 3 | FALSE: 7 | INCONCLUSIVE: 0

| ID | Hypothesis | Verdict | Business Insight |
|----|-----------|---------|-----------------|
| H1 | Users with higher 'price_tier' tend to have a higher rate of 'purchase' event_ty | **FALSE** | Premium pricing (tier 4) does not drive the highest purchase intent, suggesting  |
| H2 | Users in a higher 'user_segment' tend to have a higher proportion of 'purchase'  | **FALSE** | User segment numbers are likely categorical identifiers rather than ordinal valu |
| H3 | Products with a higher 'price_zscore' tend to have a lower 'purchase' event_type | **FALSE** | Since unusually priced products do not systematically deter purchases, the busin |
| H4 | Sessions (user_session) with a higher 'feat_sum' tend to have a higher 'purchase | **FALSE** | Users who interact with too many product features may be experiencing decision f |
| H5 | Products associated with a specific 'brand' tend to have significantly different | **TRUE** | Marketing and inventory investment should be prioritized toward high-converting  |
| H6 | Products belonging to specific 'category_code' values tend to have a higher 'pur | **TRUE** | The business should prioritize marketing spend and streamlined checkout experien |
| H7 | Events with a higher 'feat_interact' value tend to have a higher 'purchase' even | **FALSE** | Users with lower feature interaction values are actually more likely to purchase |
| H8 | Events with a higher 'log_price' tend to show a lower 'cart' to 'purchase' conve | **FALSE** | Cart abandonment is not simply driven by price level, suggesting that low-priced |
| H9 | Events with a higher 'feat_ratio' tend to have a higher 'purchase' event_type ra | **TRUE** | Products or sessions with a lower feat_ratio convert better, suggesting the busi |
| H10 | Events occurring at specific hours derived from 'event_time' tend to have a high | **FALSE** | Marketing campaigns and personalized push notifications should be prioritized du |
