Finance Gold Zone Schema Design

1. Goal

Thiết kế Gold model phục vụ analytics/BI và hệ thống phát hiện gian lận. Phương pháp: Fact-Dimension + OBT + Feature Store.
Yêu cầu: Bronze và Silver layers phải sử dụng kiến trúc lakehouse (VD: Delta Lake) để tối ưu chi phí. SLA mục tiêu: Gold table freshness <= 30 phút, Feature freshness <= 5-60 phút.

2. Dimension Tables
- dim_customer (customer_key, customer_id, risk_segment...).
- dim_account (account_key, account_id, credit_limit...)
- dim_merchant (merchant_key, merchant_id, category_code...)
- dim_date (date_key, calendar_date...).

3. Fact Tables
- fact_transaction (Grain: 1 per transaction. Keys: account_key, merchant_key, date_key. Measures: transaction_amount, fee_amount).
- fact_auth_attempt (Grain: 1 per login/auth. Keys: account_key. Measures: is_success)

4. OBT Table
- obt_transaction_fraud_view (Grain: 1 per transaction. Denormalized cho BI và Fraud Analysts. Chứa: transaction_id, account_id, merchant_category, amount, is_flagged_fraud).

5. Feature Store
Lưu ML features tại phân vùng Gold.
1. feat_account_90d (f_account_total_tx_90d, f_account_avg_tx_value_90d).
2. feat_stream_60m (f_stream_tx_velocity_60m, f_stream_login_failures_30m).
3. feat_account_unified (Join offline + streaming).

7. Data Pipeline Design & SLA
- Bronze: append-only.
- Silver: incremental processing + deduplication.
- Gold: incremental merge/upsert.
- Feature: merge by (account_id, event_timestamp).