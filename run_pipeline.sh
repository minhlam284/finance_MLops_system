#!/usr/bin/env bash
set -e

echo "=========================================="
echo "1. Cleaning up old data & models"
echo "=========================================="
# Clean data directories but keep the parent folders
rm -rf data/bronze/* data/silver/* data/gold/*
rm -rf output/offline/* output/streaming/*
rm -rf mlruns/*
echo "✓ Old data cleared."

echo "=========================================="
echo "2. Generating fresh data"
echo "=========================================="
uv run python -m data_generator.main

echo "=========================================="
echo "3. Bronze Layer (Ingestion)"
echo "=========================================="
uv run python -m jobs.bronze.ingest_offline
uv run python -m jobs.bronze.ingest_streaming

echo "=========================================="
echo "4. Silver Layer (Cleaning & Deduplication)"
echo "=========================================="
uv run python -m jobs.silver.process_dimensions
uv run python -m jobs.silver.process_facts

echo "=========================================="
echo "5. Gold Layer (Facts, Dimensions & ML Features)"
echo "=========================================="
uv run python -m jobs.gold.build_dimensions
uv run python -m jobs.gold.build_facts
uv run python -m jobs.features.build_features
uv run python -m jobs.gold.build_ml_tables

echo "=========================================="
echo "6. ML Model Training"
echo "=========================================="
uv run python -m jobs.ml.train_job

echo "=========================================="
echo "🎉 End-to-end Pipeline Completed Successfully!"
echo "=========================================="
