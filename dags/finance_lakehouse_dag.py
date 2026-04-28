"""
dags/finance_lakehouse_dag.py
------------------------------
Apache Airflow DAG – Finance Lakehouse Pipeline.

Orchestrates the full Medallion Architecture pipeline:

  [Data Generation]
       ↓
  [Bronze Offline]  [Bronze Streaming]
       ↓                   ↓
  [Silver Dimensions]  [Silver Facts]
       ↓                   ↓
  [Gold Dimensions]
       ↓
  [Gold Facts]
       ↓
  [Gold OBT]
       ↓
  [Feature Store]

Schedule:
  - Main pipeline (Bronze → Gold): every 30 minutes (SLA: Gold freshness ≤ 30 min)
  - Feature store: every 5 minutes sub-task within (SLA: Feature freshness ≤ 60 min)

Requirements:
  - Set AIRFLOW_HOME and ensure all jobs/* modules are on PYTHONPATH.
  - JAVA_HOME must be set for PySpark to work.
"""
from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ── Default DAG arguments ─────────────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner":            "finance-mlops",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=3),
}

# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="finance_lakehouse_pipeline",
    description="Medallion Architecture: Bronze → Silver → Gold → Features",
    default_args=DEFAULT_ARGS,
    schedule_interval="*/30 * * * *",   # every 30 minutes
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["finance", "lakehouse", "delta", "pyspark"],
) as dag:

    # ── Step 0: Generate raw data (optional, for dev/testing) ─────────────────
    def _generate_data(**ctx):
        """Re-generate source data if not yet present."""
        import os
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, project_root)

        offline_path   = os.path.join(project_root, "output", "offline", "transactions.parquet")
        streaming_path = os.path.join(project_root, "output", "streaming", "events_stream.jsonl")

        if not os.path.exists(offline_path) or not os.path.exists(streaming_path):
            from data_generator.main import run as gen_run
            gen_run()
        else:
            print("Source data already present – skipping generation.")

    task_generate = PythonOperator(
        task_id="generate_source_data",
        python_callable=_generate_data,
    )

    # ── Step 1a: Bronze – Offline ─────────────────────────────────────────────
    def _bronze_offline(**ctx):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from jobs.bronze.ingest_offline import run
        from jobs.utils.spark import get_spark
        spark = get_spark("bronze_offline_ingest")
        try:
            paths = run(spark)
            print("Bronze offline tables:", paths)
        finally:
            spark.stop()

    task_bronze_offline = PythonOperator(
        task_id="bronze_ingest_offline",
        python_callable=_bronze_offline,
        execution_timeout=timedelta(minutes=15),
    )

    # ── Step 1b: Bronze – Streaming ───────────────────────────────────────────
    def _bronze_streaming(**ctx):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from jobs.bronze.ingest_streaming import run
        from jobs.utils.spark import get_spark
        spark = get_spark("bronze_streaming_ingest")
        try:
            path = run(spark)
            print("Bronze streaming events →", path)
        finally:
            spark.stop()

    task_bronze_streaming = PythonOperator(
        task_id="bronze_ingest_streaming",
        python_callable=_bronze_streaming,
        execution_timeout=timedelta(minutes=10),
    )

    # ── Step 2a: Silver – Dimensions ──────────────────────────────────────────
    def _silver_dims(**ctx):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from jobs.silver.process_dimensions import run
        from jobs.utils.spark import get_spark
        spark = get_spark("silver_process_dimensions")
        try:
            paths = run(spark)
            print("Silver dimensions:", paths)
        finally:
            spark.stop()

    task_silver_dims = PythonOperator(
        task_id="silver_process_dimensions",
        python_callable=_silver_dims,
        execution_timeout=timedelta(minutes=15),
    )

    # ── Step 2b: Silver – Facts ───────────────────────────────────────────────
    def _silver_facts(**ctx):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from jobs.silver.process_facts import run
        from jobs.utils.spark import get_spark
        spark = get_spark("silver_process_facts")
        try:
            paths = run(spark)
            print("Silver facts:", paths)
        finally:
            spark.stop()

    task_silver_facts = PythonOperator(
        task_id="silver_process_facts",
        python_callable=_silver_facts,
        execution_timeout=timedelta(minutes=15),
    )

    # ── Step 3: Gold – Dimensions ─────────────────────────────────────────────
    def _gold_dims(**ctx):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from jobs.gold.build_dimensions import run
        from jobs.utils.spark import get_spark
        spark = get_spark("gold_build_dimensions")
        try:
            paths = run(spark)
            print("Gold dimensions:", paths)
        finally:
            spark.stop()

    task_gold_dims = PythonOperator(
        task_id="gold_build_dimensions",
        python_callable=_gold_dims,
        execution_timeout=timedelta(minutes=20),
    )

    # ── Step 4: Gold – Facts ──────────────────────────────────────────────────
    def _gold_facts(**ctx):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from jobs.gold.build_facts import run
        from jobs.utils.spark import get_spark
        spark = get_spark("gold_build_facts")
        try:
            paths = run(spark)
            print("Gold facts:", paths)
        finally:
            spark.stop()

    task_gold_facts = PythonOperator(
        task_id="gold_build_facts",
        python_callable=_gold_facts,
        execution_timeout=timedelta(minutes=20),
    )

    # ── Step 5: Gold – OBT ───────────────────────────────────────────────────
    def _gold_obt(**ctx):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from jobs.gold.build_obt import run
        from jobs.utils.spark import get_spark
        spark = get_spark("gold_build_obt")
        try:
            paths = run(spark)
            print("Gold OBT:", paths)
        finally:
            spark.stop()

    task_gold_obt = PythonOperator(
        task_id="gold_build_obt",
        python_callable=_gold_obt,
        execution_timeout=timedelta(minutes=15),
    )

    # ── Step 6: Feature Store ─────────────────────────────────────────────────
    def _build_features(**ctx):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from jobs.features.build_features import run
        from jobs.utils.spark import get_spark
        spark = get_spark("feature_store_build")
        try:
            paths = run(spark)
            print("Feature store:", paths)
        finally:
            spark.stop()

    task_features = PythonOperator(
        task_id="build_feature_store",
        python_callable=_build_features,
        execution_timeout=timedelta(minutes=15),
    )

    # ── Pipeline dependency graph ─────────────────────────────────────────────
    #
    #  generate_source_data
    #         ├── bronze_ingest_offline  ──► silver_process_dimensions ─┐
    #         └── bronze_ingest_streaming ► silver_process_facts        │
    #                                                                   ▼
    #                                                         gold_build_dimensions
    #                                                                   │
    #                                                         gold_build_facts
    #                                                                   │
    #                                                         gold_build_obt
    #                                                                   │
    #                                                         build_feature_store

    task_generate >> [task_bronze_offline, task_bronze_streaming]
    task_bronze_offline >> task_silver_dims
    task_bronze_streaming >> task_silver_facts
    [task_silver_dims, task_silver_facts] >> task_gold_dims
    task_gold_dims >> task_gold_facts
    task_gold_facts >> task_gold_obt
    task_gold_obt >> task_features
