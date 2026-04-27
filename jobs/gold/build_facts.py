"""
jobs/gold/build_facts.py
--------------------------
Gold layer – Fact Table Builder.

Reads Silver fact tables, resolves surrogate keys from Gold Dim tables,
and writes Gold-level Fact tables using Delta MERGE (upsert by natural key).

Fact tables produced:
  - data/gold/fact_transaction
      Grain : 1 row per transaction
      Keys  : account_key, merchant_key, date_key
      Measures: transaction_amount, fee_amount (estimated 1.5%)
      Flags   : is_declined, is_foreign_currency

  - data/gold/fact_auth_attempt
      Grain : 1 row per login/auth event
      Keys  : account_key
      Measures: is_success (1/0)
      Flags   : is_late_arrival, is_burst
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from delta.tables import DeltaTable  # type: ignore

from jobs.utils.spark import get_spark

log = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SILVER_DIR = os.path.join(BASE_DIR, "data", "silver")
GOLD_DIR   = os.path.join(BASE_DIR, "data", "gold")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gold_metadata(df: DataFrame) -> DataFrame:
    now_ts = F.lit(datetime.now(tz=timezone.utc).isoformat()).cast("timestamp")
    return df.withColumn("_gold_ts", now_ts).withColumn("_layer", F.lit("gold"))


def _upsert_gold(spark: SparkSession, df: DataFrame, merge_key: str, dst_path: str) -> None:
    """SCD Type 1 upsert."""
    if DeltaTable.isDeltaTable(spark, dst_path):
        target = DeltaTable.forPath(spark, dst_path)
        (
            target.alias("tgt")
            .merge(df.alias("src"), f"tgt.{merge_key} = src.{merge_key}")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
        log.info("[GOLD] Upserted → %s", dst_path)
    else:
        df.write.format("delta").mode("overwrite").save(dst_path)
        log.info("[GOLD] Created → %s", dst_path)


# ── fact_transaction ──────────────────────────────────────────────────────────

def build_fact_transaction(spark: SparkSession) -> str:
    dst_path = os.path.join(GOLD_DIR, "fact_transaction")
    log.info("[GOLD] Building fact_transaction …")

    silver_tx  = spark.read.format("delta").load(os.path.join(SILVER_DIR, "transactions"))
    dim_acct   = spark.read.format("delta").load(os.path.join(GOLD_DIR, "dim_account"))
    dim_merch  = spark.read.format("delta").load(os.path.join(GOLD_DIR, "dim_merchant"))
    dim_date   = spark.read.format("delta").load(os.path.join(GOLD_DIR, "dim_date"))

    # Resolve surrogate keys
    df = (
        silver_tx
        .join(dim_acct.select("account_id", "account_key"),   on="account_id",  how="left")
        .join(dim_merch.select("merchant_id", "merchant_key"), on="merchant_id", how="left")
    )

    # Derive date_key from transaction timestamp
    df = df.withColumn(
        "date_key",
        F.date_format(F.col("transaction_timestamp"), "yyyyMMdd").cast("int"),
    )

    # Measures & flags
    df = (
        df
        .withColumn("transaction_amount", F.col("amount").cast("double"))
        .withColumn("fee_amount", F.round(F.col("amount") * 0.015, 2))   # 1.5% fee estimate
        .withColumn("is_declined",
            (F.col("status") == "declined").cast("int"))
        .withColumn("is_foreign_currency",
            (F.col("currency") != "VND").cast("int"))
    )

    df = df.select(
        "transaction_id",
        "account_id",           # kept for ML join (label → feature tables)
        "account_key",
        "merchant_key",
        "date_key",
        "transaction_timestamp",
        "transaction_amount",
        "fee_amount",
        "currency",
        "status",
        "is_declined",
        "is_foreign_currency",
        "city",
        "device_id",
        "ip_address",
        F.col("is_fraudulent").cast("int"),  # fraud label propagated from generator
    )
    df = _gold_metadata(df)

    _upsert_gold(spark, df, "transaction_id", dst_path)
    return dst_path


# ── fact_auth_attempt ─────────────────────────────────────────────────────────

def build_fact_auth_attempt(spark: SparkSession) -> str:
    dst_path = os.path.join(GOLD_DIR, "fact_auth_attempt")
    log.info("[GOLD] Building fact_auth_attempt …")

    silver_events = spark.read.format("delta").load(os.path.join(SILVER_DIR, "events"))
    dim_acct      = spark.read.format("delta").load(os.path.join(GOLD_DIR, "dim_account"))

    # Filter only auth-relevant events
    auth_events = silver_events.filter(
        F.col("event_type").isin(["login_attempt", "transaction_auth", "pin_change"])
    )

    df = auth_events.join(
        dim_acct.select("account_id", "account_key"),
        on="account_id",
        how="left",
    )

    df = df.withColumn(
        "is_success",
        F.when(F.col("event_type") == "login_attempt", 1)
         .otherwise(0)
         .cast("int")
    )

    df = df.withColumn(
        "date_key",
        F.date_format(F.col("event_timestamp"), "yyyyMMdd").cast("int"),
    )

    df = df.select(
        "event_id",
        "account_key",
        "date_key",
        "event_type",
        "event_timestamp",
        "session_id",
        "device_type",
        "location_ip",
        "is_success",
        F.col("_is_burst").alias("is_burst"),
        F.col("_late_arrival").alias("is_late_arrival"),
    )
    df = _gold_metadata(df)

    _upsert_gold(spark, df, "event_id", dst_path)
    return dst_path


# ── Entry point ───────────────────────────────────────────────────────────────

def run(spark: SparkSession | None = None) -> dict[str, str]:
    """Build all Gold fact tables."""
    standalone = spark is None
    if standalone:
        spark = get_spark("gold_build_facts")

    try:
        paths = {
            "fact_transaction":   build_fact_transaction(spark),
            "fact_auth_attempt":  build_fact_auth_attempt(spark),
        }
        log.info("[GOLD] Fact build complete.")
        return paths
    finally:
        if standalone:
            spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
    results = run()
    for name, path in results.items():
        print(f"  ✓ {name:30s} → {path}")
