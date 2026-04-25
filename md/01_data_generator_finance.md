Finance Data Generator Sample Solution

1. Domain Overview

Dự án này mô phỏng một hệ thống giao dịch thẻ tín dụng của một ngân hàng hoặc tổ chức tài chính. Trình tạo dữ liệu (generator) sẽ tạo ra:
  - Dữ liệu lịch sử/offline (định dạng Parquet).
  - Dữ liệu sự kiện streaming theo thời gian thực (định dạng JSON).
Mục tiêu là hỗ trợ quá trình ingestion, transformation và feature engineering, đồng thời cố ý đưa vào các thử thách thực tế về chất lượng dữ liệu để kiểm thử hệ thống.

2. Offline Dataset Design

2.1 Offline Tables
| Table | Grain | Key Columns |
| ------ | ------ | ------ |
| customers | one per customer | customer_id, signup_ts, country, credit_segment, KYC_status |
| accounts | one per account | account_id, customer_id, account_type, credit_limit, created_ts |
| merchants | one per merchant | merchant_id, category_code (MCC), country, risk_tier |
| transactions | one per transaction | transaction_id, account_id, merchant_id, transaction_timestamp, amount, currency, status |

2.2 Offline Data Problems
Bắt buộc (Compulsory):
  - Skew: 85% giao dịch diễn ra tại các thành phố lớn, 80% merchant thuộc nhóm bán lẻ/siêu thị.
  - High cardinality: customer_id, merchant_id, transaction_id gần như là unique.
  - Schema evolution: Các partition cũ (60% timeline) bị thiếu trường `device_id` và `ip_address`.
Tự chọn: Tỷ lệ duplicate 2% trong bảng transactions.

3. Streaming Dataset Design
3.1 Event Stream Schema
Một Kafka topic hợp nhất với trường `event_type`. Các cột chính:
  - event_id, event_type (login_attempt | balance_inquiry | transaction_auth | pin_change)
  - event_timestamp, created_ts.
  - account_id, session_id, device_type, location_ip
  - transaction_id (nullable), amount (nullable)

3.2 Streaming Data Problems
Bắt buộc:
  - Bursts: Baseline 100 events/min → tăng vọt lên 3000 events/min trong các khung giờ cao điểm (ví dụ: Black Friday hoặc ngày nhận lương).
  - Late arrivals: 12% sự kiện có `created_ts` trễ hơn `event_timestamp`.

4. Feature Engineering
Offline (stable, rolling 90 ngày):
  - f_account_total_tx_90d - tổng số giao dịch
  - f_account_avg_tx_value_90d - giá trị giao dịch trung bình
  - f_account_distinct_merchants_90d - độ đa dạng merchant
Streaming (rolling 30-60 phút):
  - f_stream_login_failures_30m - số lần đăng nhập lỗi
  - f_stream_tx_velocity_60m - tần suất giao dịch trong 1 giờ