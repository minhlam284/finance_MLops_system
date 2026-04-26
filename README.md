# Finance MLOps System – Lakehouse Pipeline

End-to-end Medallion Architecture pipeline with PySpark, Delta Lake, and Apache Airflow.

## Architecture

```
data_generator/
    ↓  (Parquet + JSONL)
data/bronze/          ← Append-only raw ingestion (Delta)
    ↓  (clean + dedup)
data/silver/          ← Cleaned & conformed (Delta Merge)
    ↓  (dim/fact modelling)
data/gold/            ← Star schema + OBT + Feature Store (Delta Merge)
```

## Project Structure

```
finance_MLops_system/
├── data_generator/          # Raw data generator (Parquet + JSONL)
├── jobs/
│   ├── utils/spark.py       # SparkSession factory (Delta Lake configured)
│   ├── bronze/
│   │   ├── ingest_offline.py
│   │   └── ingest_streaming.py
│   ├── silver/
│   │   ├── process_dimensions.py
│   │   └── process_facts.py
│   ├── gold/
│   │   ├── build_dimensions.py
│   │   ├── build_facts.py
│   │   └── build_obt.py
│   ├── features/
│   │   └── build_features.py
│   └── verify_pipeline.py
├── dags/
│   └── finance_lakehouse_dag.py   # Airflow DAG
├── data/                          # Delta Lake storage (created at runtime)
│   ├── bronze/
│   ├── silver/
│   └── gold/
└── output/                        # Raw generated data (created at runtime)
    ├── offline/
    └── streaming/
```

## Setup

```bash
# 1. Install dependencies
pip install -e .

# 2. Verify JAVA_HOME is set (required for PySpark)
java -version
```

## Running the Pipeline Manually

```bash
# Step 0: Generate source data
python main.py

# Step 1: Bronze ingestion
python -m jobs.bronze.ingest_offline
python -m jobs.bronze.ingest_streaming

# Step 2: Silver processing
python -m jobs.silver.process_dimensions
python -m jobs.silver.process_facts

# Step 3: Gold dimensions (order matters!)
python -m jobs.gold.build_dimensions

# Step 4: Gold facts
python -m jobs.gold.build_facts

# Step 5: Gold OBT
python -m jobs.gold.build_obt

# Step 6: Feature Store
python -m jobs.features.build_features

# Step 7: Verify
python -m jobs.verify_pipeline
```

## Running with Airflow

```bash
# Set AIRFLOW_HOME to project root (optional)
export AIRFLOW_HOME=$(pwd)
export PYTHONPATH=$(pwd)

# Initialise Airflow DB (first time only)
airflow db init

# Start the scheduler + webserver
airflow scheduler &
airflow webserver --port 8080

# Trigger the DAG manually
airflow dags trigger finance_lakehouse_pipeline
```

## SLA Targets

| Layer | Freshness Target |
|-------|-----------------|
| Gold tables | ≤ 30 minutes |
| Feature Store (offline 90d) | ≤ 60 minutes |
| Feature Store (streaming 60m) | ≤ 5 minutes |

## Gold Schema Summary

| Table | Grain | Key Columns |
|-------|-------|-------------|
| `dim_customer` | 1/customer | `customer_key`, `customer_id` |
| `dim_account` | 1/account | `account_key`, `account_id` |
| `dim_merchant` | 1/merchant | `merchant_key`, `merchant_id` |
| `dim_date` | 1/calendar day | `date_key`, `calendar_date` |
| `fact_transaction` | 1/transaction | `transaction_id`, `account_key`, `merchant_key`, `date_key` |
| `fact_auth_attempt` | 1/auth event | `event_id`, `account_key` |
| `obt_transaction_fraud_view` | 1/transaction | All dims denormalized, `is_flagged_fraud` |
| `feat_account_90d` | 1/account | Offline rolling 90-day aggregates |
| `feat_stream_60m` | 1/account | Near-RT 60-min velocity features |
| `feat_account_unified` | 1/account | Offline + streaming feature vector |