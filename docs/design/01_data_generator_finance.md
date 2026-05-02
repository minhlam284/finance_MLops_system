# Finance Data Generator Sample Solution

## 1. Domain Overview

This project simulates a credit card transaction system for a bank or financial institution. The generator produces:

- Offline historical/reference data (Parquet)
- Streaming real-time events (JSON)

The goal is to support downstream ingestion, transformation, and feature engineering while intentionally injecting realistic data quality and processing challenges.

---

## 2. Offline Dataset Design

### 2.1 Offline Tables

| Table | Grain | Key Columns |
|-------|-------|------------|
| customers | one per customer | customer_id, signup_ts, country, credit_segment, kyc_status |
| accounts | one per account | account_id, customer_id, account_type, credit_limit, created_ts |
| merchants | one per merchant | merchant_id, category_code (MCC), country, risk_tier |
| transactions | one per transaction | transaction_id, account_id, merchant_id, transaction_timestamp, amount, currency, status |
| transaction_details | one per line | detail_id, transaction_id, merchant_id, quantity, unit_amount, fee_amount |

### 2.2 Offline Data Problems

**Compulsory:**
- **Skew**: 85% of transactions occur in major cities, 80% of merchants belong to the retail/supermarket category.
- **High cardinality**: customer_id, merchant_id, transaction_id are mostly unique.
- **Schema evolution**: old partitions (60% of timeline) missing `device_id` and `ip_address`.

**Optional chosen:** 2% duplicate rate in transactions (same account_id, merchant_id, amount, transaction_timestamp repeated).

**Output:** Parquet partitioned by transaction_date, settlement_date.

---

## 3. Streaming Dataset Design

### 3.1 Event Stream Schema

Single unified Kafka/streaming topic with `event_type` field.

Key columns:
- `event_id`, `event_type` (login_attempt|balance_inquiry|transaction_auth|pin_change|fraud_alert)
- `event_timestamp`, `created_ts` (event time vs row creation time)
- `account_id`, `session_id`, `device_type`, `source` (app|web|atm|pos)
- `transaction_id` (nullable), `merchant_id` (nullable), `amount` (nullable), `location_ip` (nullable)

### 3.2 Streaming Data Problems

**Compulsory:**
- **Bursts**: 100 events/min baseline → 3000 events/min in 20-min windows during peak periods (e.g., Black Friday or salary payment days).
- **Late arrivals**: 12% of events have a later `created_ts` than `event_timestamp`.

**Optional chosen:** 1.5% duplicate events (same event_id, immediate or 1-3 minute delay).

**Output:** JSON or Avro.

---

## 4. Feature Engineering

Compute from account transaction and event data:

**Offline (stable, 90-day windows):**
- `f_account_total_tx_90d` - transaction count
- `f_account_avg_tx_value_90d` - average transaction value
- `f_account_distinct_merchants_90d` - merchant diversity
- `f_account_payment_fail_rate_90d` - failed transaction ratio

**Streaming (rolling windows):**
- `f_stream_login_failures_30m` - failed login attempts
- `f_stream_tx_velocity_60m` - transaction frequency per hour
- `f_stream_high_amount_flag_30m` - spike in high-value transactions
- `f_stream_burst_activity_flag` - burst period traffic indicator

Merge offline + streaming for unified feature table keyed by account_id, refreshed every 15 minutes.

---

## 5. Generator Configuration

```yaml
n_customers: 120000
n_accounts: 135000
n_merchants: 45000
days_history: 180
skew_ratio_city: 0.85
skew_ratio_merchant_category: 0.80
duplicate_rate_offline: 0.02
schema_change_date: "2025-07-01"
base_events_per_min: 100
burst_multiplier: 30
burst_windows: ["08:00-08:20", "12:00-12:20"]
late_arrival_rate: 0.12
late_delay_min_max: [5, 45]
duplicate_rate_stream: 0.015
random_seed: 42
```

---

## 6. Deliverables

1. **Generator code** with configurable parameters.
2. **Data outputs**: Parquet (offline), JSON (streaming).
3. **Quality report**:
   - Skew distribution (city/merchant category %)
   - Cardinality: approx_count_distinct by ID
   - Schema evolution: nulls in old partitions
   - Duplicate rate before/after dedup
   - Streaming burst/late/duplicate rates
4. **Write-up**: explain optional problem choice and feature design.

---

## 7. Implementation Tips

- Use deterministic seeds for reproducibility.
- Define dedup keys: transaction_id/account_id (offline), event_id + created_ts (streaming).