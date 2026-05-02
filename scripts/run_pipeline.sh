#!/usr/bin/env bash
set -e

echo "=========================================="
echo "1. Cleaning up old data & models"
echo "=========================================="
# Clean runtime artifacts but keep parent directories
rm -rf artifacts/lakehouse/bronze/* artifacts/lakehouse/silver/* artifacts/lakehouse/gold/*
rm -rf artifacts/source/offline/* artifacts/source/streaming/*
rm -rf artifacts/mlruns/* artifacts/reports/* artifacts/serving/*
echo "✓ Old data cleared."

echo "=========================================="
echo "2. Generating fresh data"
echo "=========================================="
uv run python -m finance_mlops.data_generator.main

echo "=========================================="
echo "3. Bronze Layer (Ingestion)"
echo "=========================================="
uv run python -m finance_mlops.pipelines.lakehouse.bronze.ingest_offline
uv run python -m finance_mlops.pipelines.lakehouse.bronze.ingest_streaming

echo "=========================================="
echo "4. Silver Layer (Cleaning & Deduplication)"
echo "=========================================="
uv run python -m finance_mlops.pipelines.lakehouse.silver.process_dimensions
uv run python -m finance_mlops.pipelines.lakehouse.silver.process_facts

echo "=========================================="
echo "5. Gold Layer (Facts, Dimensions & ML Features)"
echo "=========================================="
uv run python -m finance_mlops.pipelines.lakehouse.gold.build_dimensions
uv run python -m finance_mlops.pipelines.lakehouse.gold.build_facts
uv run python -m finance_mlops.pipelines.lakehouse.features.build_features
uv run python -m finance_mlops.pipelines.lakehouse.gold.build_ml_tables

echo "=========================================="
echo "6. ML Model Training"
echo "=========================================="
uv run python -m finance_mlops.pipelines.training.train_job

echo "=========================================="
echo "🎉 End-to-end Pipeline Completed Successfully!"
echo "=========================================="
