"""
jobs/gold/build_dimensions.py
-------------------------------
Gold layer – Dimension Table Builder.

Reads from Silver and produces the final Gold dimension tables using
surrogate keys (hash-based UUID5) and SCD Type 1 (overwrite on match).

Gold dimension tables produced:
  - data/gold/dim_customer
  - data/gold/dim_account
  - data/gold/dim_merchant
  - data/gold/dim_date        (static date spine)

Schema follows finance schema design:
  dim_customer : customer_key (surrogate), customer_id, risk_segment, ...
  dim_account  : account_key  (surrogate), account_id, credit_limit, ...
  dim_merchant : merchant_key (surrogate), merchant_id, category_code, ...
  dim_date     : date_key (INT YYYYMMDD), calendar_date, year, quarter, ...
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta, datetime, timezone

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from delta.tables import DeltaTable  # type: ignore

from jobs.utils.spark import get_spark

log = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SILVER_DIR = os.path.join(BASE_DIR, "data", "silver")
GOLD_DIR   = os.path.join(BASE_DIR, "data", "gold")


# ── Surrogate key helper ──────────────────────────────────────────────────────

def _add_surrogate_key(df: DataFrame, natural_key: str, sk_col: str) -> DataFrame:
    """
    Generate a deterministic surrogate key via SHA-256 hash of the natural key.
    Stored as a hex string – stable across re-runs.
    """
    return df.withColumn(sk_col, F.sha2(F.col(natural_key).cast("string"), 256))


def _gold_metadata(df: DataFrame) -> DataFrame:
    now_ts = F.lit(datetime.now(tz=timezone.utc).isoformat()).cast("timestamp")
    return df.withColumn("_gold_ts", now_ts).withColumn("_layer", F.lit("gold"))


def _upsert_gold(
    spark: SparkSession,
    df: DataFrame,
    merge_key: str,
    dst_path: str,
) -> None:
    """SCD Type 1 upsert into a Gold Delta table."""
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


# ── dim_customer ──────────────────────────────────────────────────────────────

def build_dim_customer(spark: SparkSession) -> str:
    dst_path = os.path.join(GOLD_DIR, "dim_customer")
    log.info("[GOLD] Building dim_customer …")

    df = spark.read.format("delta").load(os.path.join(SILVER_DIR, "customers"))

    df = _add_surrogate_key(df, "customer_id", "customer_key")
    df = df.select(
        "customer_key",
        "customer_id",
        F.col("credit_segment").alias("risk_segment"),
        "kyc_status",
        "country",
        "first_name",
        "last_name",
        "email",
        F.to_timestamp("signup_ts").alias("signup_ts"),
    )
    df = _gold_metadata(df)

    _upsert_gold(spark, df, "customer_key", dst_path)
    return dst_path


# ── dim_account ───────────────────────────────────────────────────────────────

def build_dim_account(spark: SparkSession) -> str:
    dst_path = os.path.join(GOLD_DIR, "dim_account")
    log.info("[GOLD] Building dim_account …")

    df = spark.read.format("delta").load(os.path.join(SILVER_DIR, "accounts"))
    cust = spark.read.format("delta").load(os.path.join(GOLD_DIR, "dim_customer"))

    df = _add_surrogate_key(df, "account_id", "account_key")

    # Enrich with customer surrogate key
    df = df.join(
        cust.select("customer_id", "customer_key"),
        on="customer_id",
        how="left",
    )

    df = df.select(
        "account_key",
        "account_id",
        "customer_key",
        "account_type",
        F.col("credit_limit").cast("double"),
        "currency",
        F.to_timestamp("created_ts").alias("created_ts"),
        "is_active",
    )
    df = _gold_metadata(df)

    _upsert_gold(spark, df, "account_key", dst_path)
    return dst_path


# ── dim_merchant ──────────────────────────────────────────────────────────────

def build_dim_merchant(spark: SparkSession) -> str:
    dst_path = os.path.join(GOLD_DIR, "dim_merchant")
    log.info("[GOLD] Building dim_merchant …")

    df = spark.read.format("delta").load(os.path.join(SILVER_DIR, "merchants"))

    df = _add_surrogate_key(df, "merchant_id", "merchant_key")
    df = df.select(
        "merchant_key",
        "merchant_id",
        "merchant_name",
        F.col("category_code").alias("category_code"),
        "country",
        "city",
        "risk_tier",
    )
    df = _gold_metadata(df)

    _upsert_gold(spark, df, "merchant_key", dst_path)
    return dst_path


# ── dim_date ──────────────────────────────────────────────────────────────────

def build_dim_date(spark: SparkSession, start: str = "2020-01-01", end: str = "2030-12-31") -> str:
    dst_path = os.path.join(GOLD_DIR, "dim_date")
    log.info("[GOLD] Building dim_date (%s → %s) …", start, end)

    # Generate date spine using Spark range
    start_dt = date.fromisoformat(start)
    end_dt   = date.fromisoformat(end)
    num_days = (end_dt - start_dt).days + 1

    df = spark.range(num_days).select(
        F.expr(f"date_add(to_date('{start}'), cast(id as int))").alias("calendar_date")
    )

    df = (
        df
        .withColumn("date_key",    F.date_format("calendar_date", "yyyyMMdd").cast("int"))
        .withColumn("year",         F.year("calendar_date"))
        .withColumn("quarter",      F.quarter("calendar_date"))
        .withColumn("month",        F.month("calendar_date"))
        .withColumn("day_of_month", F.dayofmonth("calendar_date"))
        .withColumn("day_of_week",  F.dayofweek("calendar_date"))
        .withColumn("week_of_year", F.weekofyear("calendar_date"))
        .withColumn("is_weekend",
            F.col("day_of_week").isin([1, 7])  # Sunday=1, Saturday=7 in Spark
        )
        .withColumn("_layer", F.lit("gold"))
    )

    if DeltaTable.isDeltaTable(spark, dst_path):
        log.info("[GOLD] dim_date already exists – skipping rebuild.")
    else:
        df.write.format("delta").mode("overwrite").save(dst_path)
        log.info("[GOLD] Created dim_date (%d rows)", df.count())

    return dst_path


# ── Entry point ───────────────────────────────────────────────────────────────

def run(spark: SparkSession | None = None) -> dict[str, str]:
    """Build all Gold dimension tables."""
    standalone = spark is None
    if standalone:
        spark = get_spark("gold_build_dimensions")

    try:
        paths = {}
        paths["dim_customer"] = build_dim_customer(spark)
        paths["dim_date"]     = build_dim_date(spark)
        paths["dim_account"]  = build_dim_account(spark)   # needs dim_customer first
        paths["dim_merchant"] = build_dim_merchant(spark)

        log.info("[GOLD] Dimension build complete.")
        return paths
    finally:
        if standalone:
            spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
    results = run()
    for name, path in results.items():
        print(f"  ✓ {name:30s} → {path}")
