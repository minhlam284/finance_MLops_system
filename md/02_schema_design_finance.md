# Finance Gold Zone Schema Design

## 1. Goal

Business-ready Gold model for analytics/BI and fraud detection systems, plus implemented data pipelines with lineage visibility.

**Approach:** Fact-Dimension + OBT + Optional aggregates/features.

**Coursework requirement:** design and implement data pipelines end-to-end, and capture lineage for key datasets (for example, using DataHub or an equivalent metadata/lineage tool).

**Storage requirement (cost-focused):** Bronze and Silver layers must be stored using lakehouse architecture/storage (for example, Delta Lake tables on object storage) to reduce storage and processing cost.

**Schema:** `gold_finance` with naming: `dim_`, `fact_`, `obt_`, `feat_` prefix.

**Naming note:** for upstream layers, you can name Bronze tables with `raw_` prefix and Silver tables with `stg_` prefix.

**Input data profile (required before design)**
- Source from 01: list input datasets/events and key columns used for Bronze/Silver/Gold modeling.
- Data volume: estimated rows/day, historical backfill range, and expected table sizes.
- Data velocity: arrival/update frequency (batch interval or streaming rate).
- Data characteristics: key identifiers, timestamp columns, null/duplicate patterns, schema evolution risks.
- Known data issues from 01 generation: missing fields, duplicates, late-arriving records, or outliers.

Students must state this input profile first, then design schema and pipelines.

**Assumptions**
- Business objective: provide reliable, query-efficient Gold datasets for BI, fraud analytics, and downstream ML use.
- Decision usage: Gold tables and features are used for analytics/reporting, fraud detection, and ML training/scoring support.
- Service level expectation: Gold and feature data must meet agreed freshness and reliability targets.
- Explainability expectation: out of scope for the current phase.
- Risk and governance expectation: out of scope for the current phase.

**SLA targets (example)**
- Gold table freshness: <= 30 minutes for incremental loads.
- Feature freshness: <= 5-60 minutes depending on feature type.
- Pipeline run success rate: >= 99% scheduled-run success per week.

Given coursework compute constraints, you may adjust SLA targets and report the final achieved values after implementation.

---

## 2. Dimension Tables

| Dimension | Grain | Key Columns |
|-----------|-------|------------|
| dim_customer | one per customer | customer_key (SK), customer_id (BK), signup_ts, country, credit_segment, kyc_status |
| dim_account | one per account | account_key (SK), account_id (BK), account_type, credit_limit, created_ts |
| dim_merchant | one per merchant | merchant_key (SK), merchant_id (BK), category_code (MCC), country, risk_tier |
| dim_date | one per date | date_key (yyyymmdd), calendar_date, day_of_week, month, year, is_weekend |
| dim_transaction_status | one per status | transaction_status_key (SK), transaction_status (name) |

**Notes:**
- Use SCD2 (valid_from_ts, valid_to_ts, is_current) if attributes change over time (e.g., credit_limit changes on dim_account, risk_tier changes on dim_merchant).
- SK = surrogate key (data warehouse-generated), BK = business key (natural identifier).

---

## 3. Fact Tables

### 3.1 fact_transaction
**Grain:** one per transaction. **Keys:** account_key, merchant_key, transaction_date_key, transaction_status_key.  
**Measures:** transaction_amount, fee_amount, is_transaction_success (0/1), is_transaction_failed (0/1).  
**Note:** Handles schema evolution (old partitions missing `device_id` and `ip_address`; treat as nullable with default NULL).

### 3.2 fact_transaction_detail
**Grain:** one per transaction line item. **Keys:** account_key, merchant_key, transaction_date_key.  
**Measures:** quantity, unit_amount, fee_amount, line_net_amount.  
**Note:** Apply deduplication before load (2% duplicate rate from source, dedup key: account_id + merchant_id + amount + transaction_timestamp).

### 3.3 fact_auth_attempt
**Grain:** one per login/auth event. **Keys:** account_key, event_date_key.  
**Measures:** is_success (0/1), is_failed (0/1), is_fraud_flagged (0/1).  
**Note:** Source from streaming events (login_attempt, transaction_auth, pin_change).

---

## 4. OBT Table

### 4.1 obt_transaction_fraud_view
**Grain:** one per transaction.  
**Purpose:** Denormalized table for BI queries and Fraud Analysts.  
**Columns:** transaction_id, account_id, customer_id, transaction_timestamp, merchant_id, category_code (MCC), country, credit_segment, transaction_amount, currency, transaction_status, is_fraud_flagged, device_id (nullable), ip_address (nullable) (+ needed fact/dimension columns).

---

## 5. Refresh & Data Quality

**Refresh SLAs:**
- Dimensions: daily (or real-time if attributes change, e.g., account credit_limit, merchant risk_tier)
- Facts: incremental append/merge every 15-30 minutes
- OBT: merge by transaction_id every 15-30 minutes

*Note:* SLA (Service Level Agreement) is basically an agreed target for service quality, such as freshness, latency, availability, and reliability.

**Quality checks:**
- Uniqueness: transaction_id, detail_id, event_id per fact table
- Referential: facts link to dimensions (account_key, merchant_key, date_key must exist)
- Total match check: sum(line_net_amount) in fact_transaction_detail should reconcile with transaction_amount in fact_transaction
- Duplicate check: monitor transactions before and after dedup
- Null check: required keys/measures (account_key, merchant_key, transaction_amount) must stay filled; device_id/ip_address allowed NULL for old partitions

---

## 6. Feature Store

Keep ML features in Gold:

Each feature row should include `event_timestamp` for point-in-time joins and `created_ts` for dedup.

**Feature tables:**
1. `feat_account_90d` (grain: account_id, event_timestamp)
   - f_account_total_tx_90d, f_account_avg_tx_value_90d, f_account_distinct_merchants_90d, f_account_payment_fail_rate_90d
2. `feat_stream_60m` (grain: account_id, event_timestamp)
   - f_stream_login_failures_30m, f_stream_tx_velocity_60m, f_stream_high_amount_flag_30m, f_stream_burst_activity_flag
3. `feat_account_unified` (grain: account_id, event_timestamp)
   - Join offline + streaming for fraud detection training/scoring

**Point-in-time correctness:** Do not use feature data later than the label/reference timestamp. This is especially critical for fraud models to avoid label leakage.

**Dedup note:** use `created_ts` to keep the latest row when multiple rows share the same entity key and `event_timestamp`.

**Refresh:** 15-60 min (feat_account_90d), 1-5 min (feat_stream_60m), 5-15 min (feat_account_unified).

---

## 7. Data Pipeline Design and Implementation Scope

**Requirement (for students):** In 02, you must both design and implement all data pipelines, including feature pipelines.

Pipeline groups to cover:
1. Bronze ingestion pipelines
   - load raw source events/tables (customers, accounts, merchants, transactions, streaming auth events) into Bronze with schema checks and ingest metadata
   - store Bronze tables in lakehouse storage format
2. Silver transformation pipelines
   - clean, deduplicate, and standardize records for downstream modeling
   - store Silver tables in lakehouse storage format
3. Gold modeling pipelines
   - build/update dimensions, facts, and OBT tables in `gold_finance`
4. Feature pipelines (required)
   - build/update `feat_account_90d`, `feat_stream_60m`, and `feat_account_unified`
   - publish freshness and data-quality checks for feature tables

### 7.1 Pipeline SLA Targets (example)

- Bronze ingest freshness: <= 10 minutes from source arrival.
- Silver table freshness: <= 30 minutes.
- Gold fact/OBT freshness: <= 30 minutes.
- Feature freshness:
   - `feat_account_90d`: <= 60 minutes
   - `feat_stream_60m`: <= 5 minutes
   - `feat_account_unified`: <= 15 minutes
- Pipeline availability target: >= 99% successful scheduled runs per week.

Given coursework compute constraints, students may tune SLA values, but must report the final achieved targets.

### 7.2 Pipeline Update Strategy (required)

- Bronze: append-only ingestion with ingest metadata (`ingest_ts`, `source_offset`/`batch_id`).
- Silver: incremental processing with deduplication by business key + event time (dedup key for transactions: transaction_id; for events: event_id + created_ts).
- Gold dimensions/facts/OBT: incremental merge/upsert using stable keys.
- Feature tables: incremental recomputation by rolling window + merge by (`account_id`, `event_timestamp`) with latest `created_ts` retained.
- Backfill policy: for coursework, use no backfill by default; if needed, limit re-runs to at most the last 1 day with idempotent writes.
- Late-arriving data policy: reprocess affected windows and reconcile downstream tables.

### 7.3 Pipeline Controls and Monitoring (required)

- Quality gates per run: schema checks, uniqueness checks, null checks, and referential checks.
- Freshness checks: alert when SLA thresholds are exceeded.
- Volume checks: alert on abnormal drops/spikes versus baseline (especially important for burst detection in streaming).
- Run metadata: store run_id, start/end time, status, input/output row counts, and error summary.
- Recovery controls: retry with backoff, dead-letter/quarantine for bad records, and rerun procedure.
- Lineage tracking: publish dataset and job lineage for Bronze -> Silver -> Gold -> Feature tables (for example via DataHub).
- Lineage evidence: include at least one lineage view/screenshot or exported lineage summary for core tables.

---

## 8. Warehouse Optimization

Students must state what warehouse optimizations were applied and why.

- Storage/layout: partitioning strategy and clustering/sorting strategy for large tables.
- Access path optimization: indexing (or warehouse equivalent) for common filters/joins.
- Query optimization: materialized views/summary tables where justified.

**NOTE (for students): Suggested write-up format**
- Workload: which query/job was slow (for example, daily fraud dashboard query on `obt_transaction_fraud_view`).
- Bottleneck: what caused the issue (for example, full table scan, expensive join, skewed partition).
- Optimization applied: what you changed (for example, index on `transaction_timestamp`, partition by `transaction_date_key`, clustering by `account_id`).
- Result: before/after metrics (runtime, scanned bytes, cost, or resource usage).
- Trade-off: one downside or maintenance cost of the optimization.

**Example (brief):**
- Workload: daily fraud analyst query filtering by date range and merchant category.
- Optimization: partition `fact_transaction` by `transaction_date_key` and add index on `category_code`.
- Result: runtime improved from 45s to 12s; scanned data reduced by ~70%.
- Trade-off: slightly higher write cost during incremental loads.

**Scope boundary with 04:** 04 reuses these implemented data pipelines and covers CI/CD for ML pipelines and inference services.

---

## 9. Deliverables

**Submission format (required):**
- Submit one Markdown file (`.md`) as your final 02 design + implementation document.
- The file should follow the same section structure and content coverage as this example file.

1. Goal setup: define the Gold-zone objective, modeling approach, naming conventions, required input data profile from 01 (volume, velocity, key attributes, and known data issues), plus assumptions and SLA targets before design.
2. Dimension design: define grain, keys, and SCD strategy for dimension tables.
3. Fact design: define grain, keys, measures, and handling for schema evolution/dedup.
4. OBT design: define purpose, grain, and core denormalized columns.
5. Refresh and data quality plan: define freshness SLAs and required validation checks.
6. Feature store design: define feature tables, point-in-time correctness, dedup policy, and refresh targets.
7. Data pipeline plan: design and implement Bronze, Silver, Gold, and feature pipelines, including lakehouse storage for Bronze/Silver, schedules/dependencies, SLA targets, update strategy (incremental/merge, limited backfill up to 1 day if needed, late-data handling), operational controls (monitoring/alerting, run metadata, retry/recovery, rerun procedures), and lineage tracking (for example DataHub).
8. Warehouse optimization plan: document indexing/partitioning/clustering (or warehouse equivalents), maintenance operations, measured impact, and follow the Section 8 write-up format.