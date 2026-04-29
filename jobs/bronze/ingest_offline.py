"""
jobs/bronze/ingest_offline.py
------------------------------
Bronze layer – Offline Ingestion Job.

Reads raw Parquet files produced by `data_generator` and writes them
as Delta tables (append-only) into the Bronze zone.

Tables processed:
  - customers
  - accounts
  - merchants
  - transactions
  - transaction_details
  - transaction_status

Data-quality metadata columns added:
  - _ingestion_ts  : timestamp when the row was ingested
  - _source_file   : original Parquet filename (for lineage)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from jobs.utils.spark import get_spark

log = logging.getLogger(__name__)

# ── Path configuration ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFLINE_INPUT_DIR  = os.path.join(BASE_DIR, "output", "offline")
BRONZE_OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "bronze")

TABLES = [
    "customers",
    "accounts",
    "merchants",
    "transactions",
    "transaction_details",   # fact: line items with 2% duplicate rate from source
    "transaction_status",    # dim: static lookup (approved / declined / pending)
]


# ── Core helpers ──────────────────────────────────────────────────────────────

def _add_metadata(df: DataFrame, source_name: str) -> DataFrame:
    """Attach ingestion metadata columns to a DataFrame."""
    now_ts = F.lit(datetime.now(tz=timezone.utc).isoformat()).cast("timestamp")
    return (
        df
        .withColumn("_ingestion_ts", now_ts)
        .withColumn("_source_file",  F.lit(f"{source_name}.parquet"))
        .withColumn("_layer",        F.lit("bronze"))
    )


def ingest_table(spark: SparkSession, table_name: str) -> str:
    """
    Read one Parquet file from the offline input dir, attach metadata,
    and append to the corresponding Bronze Delta table.

    Returns the output Delta path.
    """
    src_path = os.path.join(OFFLINE_INPUT_DIR, f"{table_name}.parquet")
    dst_path = os.path.join(BRONZE_OUTPUT_DIR, table_name)

    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"Source Parquet not found: {src_path}\n"
            "Run `python main.py` (data_generator) first."
        )

    log.info("[BRONZE] Reading %s from %s …", table_name, src_path)
    df = spark.read.parquet(src_path)

    df = _add_metadata(df, table_name)

    log.info("[BRONZE] Writing %s → %s  (%d rows, %d cols)",
             table_name, dst_path, df.count(), len(df.columns))

    (
        df.write
        .format("delta")
        .mode("append")                  # Bronze = append-only
        .option("mergeSchema", "true")   # tolerate schema evolution
        .save(dst_path)
    )

    return dst_path


def run(spark: SparkSession | None = None) -> dict[str, str]:
    """
    Entry point – ingest all offline tables into Bronze.

    Returns a mapping of {table_name: delta_path}.
    """
    standalone = spark is None
    if standalone:
        spark = get_spark("bronze_offline_ingest")

    try:
        paths: dict[str, str] = {}
        for table in TABLES:
            paths[table] = ingest_table(spark, table)
        log.info("[BRONZE] Offline ingestion complete. Tables: %s", list(paths.keys()))
        return paths
    finally:
        if standalone:
            spark.stop()


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
    results = run()
    for name, path in results.items():
        print(f"  ✓ {name:20s} → {path}")
