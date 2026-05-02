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

## KServe Online Inference (Fraud Model)

This repository now supports a KServe-style custom predictor for online fraud
inference while keeping MLflow local tracking for training.

### 1) Train and promote model

Run the ML DAG or jobs below so `fraud_detection_model` has a `Production` stage:

```bash
python -m jobs.ml.train_job
python -m jobs.ml.evaluate_and_register_job
```

### 2) Export serving artifacts

```bash
python -m jobs.ml.export_model_for_serving_job
```

Artifacts are written to:

```text
output/serving/fraud_detection_model/<version>/
```

Optional remote sync (MinIO/S3) via env vars:
`MODEL_ARTIFACT_BUCKET`, `MODEL_ARTIFACT_PREFIX`, `AWS_ENDPOINT_URL`,
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.

### 3) Local smoke test for predictor

```bash
uvicorn jobs.kserve.predictor:app --host 0.0.0.0 --port 8080
```

Then call:

```bash
curl -X POST "http://localhost:8080/v1/models/fraud-detection:predict" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "f_account_total_tx_90d": 120.0,
        "f_account_avg_tx_value_90d": 42.5,
        "f_account_max_tx_value_90d": 420.0,
        "f_account_declined_ratio_90d": 0.10,
        "f_account_foreign_tx_ratio_90d": 0.05,
        "f_stream_tx_velocity_60m": 3.0,
        "f_stream_unique_devices_60m": 1.0,
        "f_stream_login_failures_30m": 0.0
      }
    ]
  }'
```

Expected response fields per prediction:
`fraud_score`, `is_blocked`, `model_version`.

### 4) KServe manifest smoke test

```bash
kubectl apply -k k8s/kserve/overlays/dev
kubectl get inferenceservice -n ml-serving fraud-detection
```

Update image tag and storage env vars in `k8s/kserve/overlays/dev` before deploy.

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