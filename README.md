# Finance MLOps System

End-to-end fraud detection MLOps system with a Medallion lakehouse, ML training on MLflow, and online inference via KServe.

## Repository Layout

```text
finance_MLops_system/
├── src/finance_mlops/                # Application source package
├── orchestration/airflow/dags/       # Airflow DAGs
├── deploy/docker/                    # Container build specs
├── deploy/k8s/                       # KServe manifests and overlays
├── scripts/                          # Local run helpers
├── docs/                             # Architecture/design docs
├── tests/                            # Unit tests
└── artifacts/                        # Runtime data (gitignored)
```

## Runtime Artifact Layout

All generated data/model artifacts are stored under `artifacts/`:

```text
artifacts/source/{offline,streaming}
artifacts/lakehouse/{bronze,silver,gold}
artifacts/mlruns
artifacts/reports
artifacts/serving/fraud_detection_model/<version>
```

## Quickstart

```bash
pip install -e ".[dev]"
java -version

# Generate source
python -m finance_mlops.data_generator.main

# Lakehouse
python -m finance_mlops.pipelines.lakehouse.bronze.ingest_offline
python -m finance_mlops.pipelines.lakehouse.bronze.ingest_streaming
python -m finance_mlops.pipelines.lakehouse.silver.process_dimensions
python -m finance_mlops.pipelines.lakehouse.silver.process_facts
python -m finance_mlops.pipelines.lakehouse.gold.build_dimensions
python -m finance_mlops.pipelines.lakehouse.gold.build_facts
python -m finance_mlops.pipelines.lakehouse.gold.build_obt
python -m finance_mlops.pipelines.lakehouse.features.build_features
python -m finance_mlops.pipelines.verify_pipeline

# ML train/register/export
python -m finance_mlops.pipelines.lakehouse.gold.build_ml_tables
python -m finance_mlops.pipelines.training.train_job
python -m finance_mlops.pipelines.training.evaluate_and_register_job
python -m finance_mlops.pipelines.training.export_model_for_serving_job
```

## Airflow

```bash
export AIRFLOW_HOME=$(pwd)
airflow db init
airflow scheduler &
airflow webserver --port 8080
airflow dags trigger finance_lakehouse_pipeline
```

DAG files are in `orchestration/airflow/dags`.

## KServe

```bash
uvicorn finance_mlops.pipelines.inference.online_kserve.predictor:app --host 0.0.0.0 --port 8080
kubectl apply -k deploy/k8s/kserve/overlays/dev
kubectl get inferenceservice -n ml-serving fraud-detection
```

Container image is built from `deploy/docker/kserve-predictor.Dockerfile`.

## Documentation

- High-level architecture: `docs/high-level-design.md`
- Implementation walkthrough: `docs/walkthrough.md`
- Detailed design notes: `docs/design/`