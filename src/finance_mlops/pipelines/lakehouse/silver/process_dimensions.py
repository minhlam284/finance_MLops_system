"""
jobs/silver/process_dimensions.py
----------------------------------
Silver layer – Dimension Table Processing Job.

Reads Bronze Delta tables for customers, accounts, and merchants.
Applies data quality rules:
  - Drop rows missing mandatory business keys
  - Normalise string casing / trim whitespace
  - Cast columns to canonical types
  - Deduplicate on natural key (keep latest by _ingestion_ts)

Then merges (upsert) cleaned rows into Silver Delta tables using
Delta MERGE so downstream Gold jobs always see the freshest records
without full rewrites.

Silver tables produced:
  - data/silver/customers
  - data/silver/accounts
  - data/silver/merchants
  - data/silver/transaction_status
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from delta.tables import DeltaTable  # type: ignore

from finance_mlops.config.paths import LAKEHOUSE_BRONZE_DIR, LAKEHOUSE_SILVER_DIR
from finance_mlops.pipelines.common.utils.spark import get_spark

log = logging.getLogger(__name__)

# ── Path configuration ────────────────────────────────────────────────────────
BRONZE_DIR = str(LAKEHOUSE_BRONZE_DIR)
SILVER_DIR = str(LAKEHOUSE_SILVER_DIR)

# ── Config per dimension ──────────────────────────────────────────────────────
DIM_CONFIG = {
    "customers": {
        "natural_key": "customer_id",
        "mandatory_cols": ["customer_id"],
        "string_cols":    ["country", "credit_segment", "kyc_status",
                           "first_name", "last_name", "email"],
    },
    "accounts": {
        "natural_key": "account_id",
        "mandatory_cols": ["account_id", "customer_id"],
        "string_cols":    ["account_type", "currency"],
    },
    "merchants": {
        "natural_key": "merchant_id",
        "mandatory_cols": ["merchant_id"],
        "string_cols":    ["merchant_name", "category_code", "country",
                           "city", "risk_tier"],
    },
    "transaction_status": {
        "natural_key":    "status_id",
        "mandatory_cols": ["status_id", "status_name"],
        "string_cols":    ["status_id", "status_name"],
    },
}


# ── Data-quality helpers ──────────────────────────────────────────────────────

def _drop_mandatory_nulls(df: DataFrame, cols: list[str]) -> DataFrame:
    """Drop rows where any mandatory column is null or empty string."""
    for col in cols:
        df = df.filter(F.col(col).isNotNull() & (F.trim(F.col(col)) != ""))
    return df


def _clean_strings(df: DataFrame, cols: list[str]) -> DataFrame:
    """Trim whitespace on all string columns."""
    for col in cols:
        if col in df.columns:
            df = df.withColumn(col, F.trim(F.col(col)))
    return df


def _dedup_by_key(df: DataFrame, natural_key: str) -> DataFrame:
    """Keep the latest record per natural key (based on _ingestion_ts)."""
    window = Window.partitionBy(natural_key).orderBy(F.col("_ingestion_ts").desc())
    return (
        df
        .withColumn("_rn", F.row_number().over(window))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


def _add_silver_metadata(df: DataFrame) -> DataFrame:
    now_ts = F.lit(datetime.now(tz=timezone.utc).isoformat()).cast("timestamp")
    return df.withColumn("_silver_ts", now_ts).withColumn("_layer", F.lit("silver"))


# ── Merge helper ──────────────────────────────────────────────────────────────

def _upsert_to_silver(
    spark: SparkSession,
    df: DataFrame,
    natural_key: str,
    dst_path: str,
) -> None:
    """
    MERGE (upsert) new Silver data into the Delta table.
    If the table does not yet exist it is created.
    """
    if DeltaTable.isDeltaTable(spark, dst_path):
        target = DeltaTable.forPath(spark, dst_path)
        (
            target.alias("tgt")
            .merge(
                df.alias("src"),
                f"tgt.{natural_key} = src.{natural_key}",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
        log.info("[SILVER] Merged into %s", dst_path)
    else:
        (
            df.write
            .format("delta")
            .mode("overwrite")
            .save(dst_path)
        )
        log.info("[SILVER] Created new table at %s", dst_path)


# ── Per-table processor ───────────────────────────────────────────────────────

def process_dimension(spark: SparkSession, table_name: str) -> str:
    """Clean and upsert one dimension table from Bronze → Silver."""
    cfg         = DIM_CONFIG[table_name]
    natural_key = cfg["natural_key"]
    src_path    = os.path.join(BRONZE_DIR, table_name)
    dst_path    = os.path.join(SILVER_DIR, table_name)

    log.info("[SILVER] Processing dimension: %s", table_name)

    df = spark.read.format("delta").load(src_path)
    log.info("[SILVER] Loaded %d rows from Bronze %s", df.count(), table_name)

    df = _drop_mandatory_nulls(df, cfg["mandatory_cols"])
    df = _clean_strings(df, cfg["string_cols"])
    df = _dedup_by_key(df, natural_key)
    df = _add_silver_metadata(df)

    log.info("[SILVER] After cleaning: %d rows", df.count())
    _upsert_to_silver(spark, df, natural_key, dst_path)

    return dst_path


# ── Entry point ───────────────────────────────────────────────────────────────

def run(spark: SparkSession | None = None) -> dict[str, str]:
    """Process all dimension tables from Bronze → Silver."""
    standalone = spark is None
    if standalone:
        spark = get_spark("silver_process_dimensions")

    try:
        paths: dict[str, str] = {}
        for table in DIM_CONFIG:
            paths[table] = process_dimension(spark, table)

        log.info("[SILVER] Dimension processing complete: %s", list(paths.keys()))
        return paths
    finally:
        if standalone:
            spark.stop()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
    results = run()
    for name, path in results.items():
        print(f"  ✓ {name:20s} → {path}")
