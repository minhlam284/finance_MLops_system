"""
jobs/silver/process_facts.py
-----------------------------
Silver layer – Fact Table Processing Job.

Reads Bronze Delta tables for transactions and events.
Applies:
  - Deduplication on natural key (transaction_id / event_id)
  - Timestamp casting & validation (drop rows with null timestamps)
  - Incremental write to Silver (append rows not yet in Silver)

Silver tables produced:
  - data/silver/transactions
  - data/silver/events
  - data/silver/transaction_details
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from delta.tables import DeltaTable  # type: ignore

from jobs.utils.spark import get_spark

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRONZE_DIR = os.path.join(BASE_DIR, "data", "bronze")
SILVER_DIR = os.path.join(BASE_DIR, "data", "silver")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dedup_by_key(df: DataFrame, key_col: str) -> DataFrame:
    """Keep one row per natural key; prefer the latest by _ingestion_ts."""
    window = Window.partitionBy(key_col).orderBy(F.col("_ingestion_ts").desc())
    return (
        df
        .withColumn("_rn", F.row_number().over(window))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


def _add_silver_metadata(df: DataFrame) -> DataFrame:
    now_ts = F.lit(datetime.now(tz=timezone.utc).isoformat()).cast("timestamp")
    return (
        df
        .withColumn("_silver_ts", now_ts)
        .withColumn("_layer",     F.lit("silver"))
    )


def _incremental_append(
    spark: SparkSession,
    new_df: DataFrame,
    key_col: str,
    dst_path: str,
) -> None:
    """
    Append only rows whose key is not already present in the Silver table.
    If the table doesn't exist yet, do a full write.
    """
    if DeltaTable.isDeltaTable(spark, dst_path):
        existing_keys = (
            spark.read.format("delta").load(dst_path)
            .select(key_col)
        )
        incremental = new_df.join(existing_keys, on=key_col, how="left_anti")
        row_count = incremental.count()
        if row_count > 0:
            (
                incremental.write
                .format("delta")
                .mode("append")
                .option("mergeSchema", "true")
                .save(dst_path)
            )
            log.info("[SILVER] Appended %d new rows to %s", row_count, dst_path)
        else:
            log.info("[SILVER] No new rows for %s", dst_path)
    else:
        new_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(dst_path)
        log.info("[SILVER] Created %s (%d rows)", dst_path, new_df.count())


# ── Transactions ──────────────────────────────────────────────────────────────

def process_transactions(spark: SparkSession) -> str:
    """Bronze → Silver for the transactions fact table."""
    src_path = os.path.join(BRONZE_DIR, "transactions")
    dst_path = os.path.join(SILVER_DIR, "transactions")

    log.info("[SILVER] Processing transactions …")
    df = spark.read.format("delta").load(src_path)

    # Cast & validate timestamps
    df = df.withColumn(
        "transaction_timestamp",
        F.to_timestamp("transaction_timestamp"),
    )
    df = df.filter(F.col("transaction_timestamp").isNotNull())
    df = df.filter(F.col("transaction_id").isNotNull())

    # Normalise amount
    df = df.withColumn("amount", F.col("amount").cast("double"))

    df = _dedup_by_key(df, "transaction_id")
    df = _add_silver_metadata(df)

    _incremental_append(spark, df, "transaction_id", dst_path)
    return dst_path


# ── Events ────────────────────────────────────────────────────────────────────

def process_events(spark: SparkSession) -> str:
    """Bronze → Silver for the streaming events fact table."""
    src_path = os.path.join(BRONZE_DIR, "events")
    dst_path = os.path.join(SILVER_DIR, "events")

    log.info("[SILVER] Processing events …")
    df = spark.read.format("delta").load(src_path)

    # Validate mandatory fields
    df = df.filter(
        F.col("event_id").isNotNull() &
        F.col("event_timestamp").isNotNull() &
        F.col("account_id").isNotNull()
    )

    df = _dedup_by_key(df, "event_id")
    df = _add_silver_metadata(df)

    _incremental_append(spark, df, "event_id", dst_path)
    return dst_path


# ── Transaction Details ─────────────────────────────────────────────────────────

def process_transaction_details(spark: SparkSession) -> str:
    """
    Bronze → Silver for the transaction_details fact table.

    Dedup key: detail_id  (removes the ~2% duplicates injected at the source
    data generator as per schema design Section 3.2).
    """
    src_path = os.path.join(BRONZE_DIR, "transaction_details")
    dst_path = os.path.join(SILVER_DIR, "transaction_details")

    log.info("[SILVER] Processing transaction_details …")
    df = spark.read.format("delta").load(src_path)

    # Validate mandatory fields
    df = df.filter(
        F.col("detail_id").isNotNull() &
        F.col("transaction_id").isNotNull()
    )

    # Cast numeric measures to canonical types
    df = df.withColumn("unit_amount", F.col("unit_amount").cast("double"))
    df = df.withColumn("fee_amount",  F.col("fee_amount").cast("double"))
    df = df.withColumn("quantity",    F.col("quantity").cast("int"))

    # Dedup by detail_id – keeps latest ingest; removes source duplicates
    df = _dedup_by_key(df, "detail_id")
    df = _add_silver_metadata(df)

    _incremental_append(spark, df, "detail_id", dst_path)
    return dst_path


# ── Entry point ───────────────────────────────────────────────────────────────

def run(spark: SparkSession | None = None) -> dict[str, str]:
    """Process all fact tables from Bronze → Silver."""
    standalone = spark is None
    if standalone:
        spark = get_spark("silver_process_facts")

    try:
        paths = {
            "transactions":         process_transactions(spark),
            "events":               process_events(spark),
            "transaction_details":  process_transaction_details(spark),
        }
        log.info("[SILVER] Fact processing complete.")
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
