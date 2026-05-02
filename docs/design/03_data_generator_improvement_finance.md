Finance Data Generator Improvement: Feature Drift & Labels

1. Objective

Mở rộng generator với tính năng mô phỏng feature drift để kiểm thử hệ thống cảnh báo của Feature Store và tạo bảng nhãn (Labels) cho ML.

2. Drift Scenarios

Scenario A: Simple - Transaction Frequency Drift (Mô phỏng Carding Attack).
- Thay đổi: Tần suất giao dịch tăng đột biến từ 1.2 lên 5.5 giao dịch/tài khoản/ngày sau `drift_start_date`.
- Feature bị ảnh hưởng: `f_account_total_tx_90d`.
- Cảnh báo: Kích hoạt khi PSI > 0.1.

Scenario B: Simple - Average Transaction Value Drift (Mô phỏng High-ticket Fraud).
- Thay đổi: Giá trị trung bình tăng từ $45 lên $180.
- Feature bị ảnh hưởng: `f_account_avg_tx_value_90d`.

6. Gold Layer Monitoring Tables
- Table 1: agg_feature_health_daily (Cảnh báo khi PSI > 0.15).
- Table 3: ml_transaction_label (Chứa `event_timestamp`, `created_ts`, `label = is_fraudulent` (1 or 0)).
- Table 4: ml_fraud_detection_training (Join giữa bảng label và feature tables).