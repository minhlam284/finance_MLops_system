# Finance ML Pipeline – Walkthrough

## What Was Built

Full implementation of `docs/design/04.1_ml_design_finance.md` – a production-grade fraud detection ML system on top of the existing Finance Lakehouse.

---

## New Files

| File | Purpose |
|---|---|
| `src/finance_mlops/pipelines/training/train_job.py` | XGBoost training + MLflow logging |
| `src/finance_mlops/pipelines/training/evaluate_and_register_job.py` | Model evaluation + registry promotion |
| `src/finance_mlops/pipelines/inference/batch_streaming/streaming_scoring_job.py` | Spark Structured Streaming scoring |
| `orchestration/airflow/dags/finance_ml_training_dag.py` | Airflow retraining DAG (weekly) |
| `.github/workflows/ml_pipeline.yml` | CI/CD: lint → unit tests → smoke train |

**Dependencies added to `pyproject.toml`:** `mlflow>=2.12`, `xgboost>=2.0`, `scikit-learn>=1.4`, `numpy>=1.26`  
**Installed versions:** mlflow 3.11.1, xgboost 3.2.0, sklearn 1.8.0, numpy 2.4.4

---

## Data Flow

```
Gold/ml_fraud_detection_training
          │
    [train_job.py]
          │  XGBoost (scale_pos_weight handles fraud imbalance)
          │  Time-based split (last 20% = validation)
          │  Metrics: Precision, Recall, PR-AUC
          ▼
    MLflow experiment: "fraud_detection"
    MLflow registry:   "fraud_detection_model"
          │
    [evaluate_and_register_job.py]
          │  PR-AUC >= 0.30 → promote to Production stage
          ▼
    artifacts/reports/ml_eval_report.json

Bronze/events (`artifacts/lakehouse/bronze/events`) (Delta streaming source)
          │  filter event_type = 'transaction_auth'
          │  LEFT JOIN Gold/feat_account_unified
    [streaming_scoring_job.py]
          │  XGBoost UDF (loaded from MLflow Production)
          │  fraud_score ∈ [0, 1]
          │  is_blocked = True if fraud_score >= 0.60
          ▼
    Gold/fraud_scores (`artifacts/lakehouse/gold/fraud_scores`)
```

---

## How to Run

### Step 1 – Ensure the full lakehouse pipeline has run
```bash
# Run bronze → silver → gold → features → ml_tables first
python -m finance_mlops.pipelines.lakehouse.gold.build_ml_tables
```

### Step 2 – Train the model
```bash
python -m finance_mlops.pipelines.training.train_job
# Output: artifacts/mlruns/ directory created with XGBoost experiment
```

### Step 3 – Evaluate & register
```bash
python -m finance_mlops.pipelines.training.evaluate_and_register_job
# Output: artifacts/reports/ml_eval_report.json
#         Model promoted to Production if PR-AUC >= 0.30
```

### Step 4 – Start real-time scoring stream
```bash
# Runs indefinitely (Ctrl-C to stop)
python -m finance_mlops.pipelines.inference.batch_streaming.streaming_scoring_job
# Output: artifacts/lakehouse/gold/fraud_scores/ (Delta table, streaming appended)
```

### Airflow retraining (weekly)
The `finance_ml_training_dag` DAG runs every Sunday at midnight and chains:
`build_ml_tables → train_model → check_prauc_gate → evaluate_and_register`

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `scale_pos_weight` on XGBoost | Fraud is rare; this compensates imbalance without oversampling |
| Time-based split (not random) | Prevents data leakage from future events into training |
| PR-AUC as primary metric | More informative than ROC-AUC for heavily imbalanced fraud data |
| `ShortCircuitOperator` in DAG | Automatically skips registry promotion if model quality degrades |
| Delta streaming source | Unified with existing bronze/events table; no Kafka needed |
| `timeout_secs` param on streaming job | Makes the job Airflow-compatible (micro-batch runs) |
