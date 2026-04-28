"""
dags/finance_ml_training_dag.py
---------------------------------
Apache Airflow DAG – Finance ML Training Pipeline.

Orchestrates the batch retraining workflow for the fraud detection model:

  [Build ML Tables]  ← updates training dataset from Gold layer
        ↓
  [Train Model]      ← XGBoost training + MLflow experiment logging
        ↓
  [Evaluate & Register]  ← promote best model to Production in registry

Schedule  : Weekly (Sunday midnight) by default.
             Can be triggered manually via Airflow UI or when data drift
             is detected (PR-AUC drop below baseline).

Requirements:
  - AIRFLOW_HOME configured.
  - All jobs/* modules on PYTHONPATH.
  - JAVA_HOME set for PySpark.
  - MLflow local tracking server in `mlruns/` (created automatically).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago

# ── Default args ──────────────────────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner":            "finance-mlops",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}


# ── Helper: project root ───────────────────────────────────────────────────────
def _project_root() -> str:
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="finance_ml_training_pipeline",
    description="Fraud Detection ML: build tables → train → evaluate → register",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 0 * * 0",   # Weekly, Sunday at midnight
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["finance", "ml", "fraud", "xgboost", "mlflow"],
) as dag:

    # ── Task 0: Build / refresh ML training tables ─────────────────────────────
    def _build_ml_tables(**ctx):
        """
        Re-build Gold ML tables:
          - agg_feature_health_daily
          - ml_transaction_label
          - ml_fraud_detection_training
        """
        import sys
        sys.path.insert(0, _project_root())
        from jobs.gold.build_ml_tables import run
        from jobs.utils.spark import get_spark

        spark = get_spark("gold_build_ml_tables")
        try:
            paths = run(spark)
            for name, path in paths.items():
                print(f"  ✓ {name} → {path}")
        finally:
            spark.stop()

    task_build_ml_tables = PythonOperator(
        task_id="build_ml_tables",
        python_callable=_build_ml_tables,
        execution_timeout=timedelta(minutes=30),
    )

    # ── Task 1: Train model ────────────────────────────────────────────────────
    def _train_model(**ctx):
        """Train XGBoost and log to MLflow. Push run_id + gate result to XCom."""
        import sys
        sys.path.insert(0, _project_root())
        from jobs.ml.train_job import run
        from jobs.utils.spark import get_spark

        spark = get_spark("ml_train_job")
        try:
            result = run(spark)
        finally:
            spark.stop()

        # Push to XCom for downstream tasks
        ctx["ti"].xcom_push(key="run_id",           value=result["run_id"])
        ctx["ti"].xcom_push(key="passed_threshold",  value=result["passed_threshold"])
        ctx["ti"].xcom_push(key="val_pr_auc",        value=result["metrics"]["val_pr_auc"])

        print(f"  run_id          : {result['run_id']}")
        print(f"  val_pr_auc      : {result['metrics']['val_pr_auc']:.4f}")
        print(f"  passed_threshold: {result['passed_threshold']}")

    task_train = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        execution_timeout=timedelta(minutes=30),
    )

    # ── Task 2: Gate – only continue if training passed ────────────────────────
    def _check_gate(**ctx) -> bool:
        """ShortCircuit: skip registration if model did not pass the PR-AUC gate."""
        passed = ctx["ti"].xcom_pull(task_ids="train_model", key="passed_threshold")
        pr_auc = ctx["ti"].xcom_pull(task_ids="train_model", key="val_pr_auc")
        print(f"  Gate check — val_pr_auc={pr_auc:.4f} | passed={passed}")
        return bool(passed)

    task_gate = ShortCircuitOperator(
        task_id="check_prauc_gate",
        python_callable=_check_gate,
    )

    # ── Task 3: Evaluate & promote to registry ─────────────────────────────────
    def _evaluate_and_register(**ctx):
        """Evaluate the latest run and promote the model to Production."""
        import sys
        sys.path.insert(0, _project_root())
        from jobs.ml.evaluate_and_register_job import run

        result = run()
        promoted = result["promoted_to_production"]
        status   = "PROMOTED to Production" if promoted else "NOT promoted"
        print(f"  run_id  : {result['run_id']}")
        print(f"  PR-AUC  : {result['metrics'].get('val_pr_auc', 'N/A')}")
        print(f"  Status  : {status}")

    task_evaluate = PythonOperator(
        task_id="evaluate_and_register",
        python_callable=_evaluate_and_register,
        execution_timeout=timedelta(minutes=10),
    )

    # ── Dependency graph ───────────────────────────────────────────────────────
    #
    #   build_ml_tables
    #         │
    #   train_model
    #         │
    #   check_prauc_gate  ──(short-circuit if FAIL)──►  [end]
    #         │ (PASS)
    #   evaluate_and_register

    task_build_ml_tables >> task_train >> task_gate >> task_evaluate
