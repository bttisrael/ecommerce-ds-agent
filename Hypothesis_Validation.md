# Hypothesis Validation

**Target:** `late_delivery_risk` | TRUE: 2 | FALSE: 7 | INCONCLUSIVE: 1

| ID | Hypothesis | Verdict | Business Insight |
|----|-----------|---------|-----------------|
| H1 | Orders where days_for_shipping_real exceeds days_for_shipment_scheduled tend to  | **TRUE** | Orders that take longer than their scheduled shipping window are highly likely t |
| H2 | Orders with a higher days_for_shipping_real tend to have higher late_delivery_ri | **FALSE** | The business should focus monitoring efforts on orders expected to ship in the 4 |
| H3 | Orders with a lower days_for_shipment_scheduled tend to have higher late_deliver | **FALSE** | Shipments scheduled with very tight windows (around 0.8-1.6 days) pose the great |
| H4 | Orders shipped to certain markets (e.g., LATAM or Africa) tend to have higher la | **FALSE** | Since late delivery risk is uniformly high (~55%) across all markets, the root c |
| H5 | Orders belonging to certain order types (e.g., 'DEBIT' or 'TRANSFER') tend to ha | **FALSE** | Since TRANSFER orders actually have the lowest late delivery risk, payment type  |
| H6 | Orders from certain department_names (e.g., large/heavy item departments like 'O | **FALSE** | Late delivery risk is fairly uniform across all departments (ranging only from ~ |
| H7 | Orders placed by customers in certain customer_segments (e.g., 'Consumer' vs 'Co | **FALSE** | Since late delivery risk is uniformly high (~55%) across all customer segments,  |
| H8 | Orders shipped to distant or remote order_countries tend to have higher late_del | **INCONCLUSIVE** | The business should investigate logistics partnerships and fulfillment strategie |
| H9 | Orders with lower benefit_per_order tend to have higher late_delivery_risk, as l | **FALSE** | Fulfillment and delivery delays appear to be driven by operational or logistical |
| H10 | Orders placed in certain category_names (e.g., bulky or high-volume categories)  | **TRUE** | The business should prioritize targeted logistics improvements and buffer invent |
