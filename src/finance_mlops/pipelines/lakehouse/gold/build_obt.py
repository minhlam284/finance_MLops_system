"""
jobs/gold/build_obt.py
------------------------
Gold layer – One Big Table (OBT) Builder.

Produces a fully denormalized view of transactions enriched with all
dimension attributes. Designed for direct consumption by BI tools and
Fraud Analysts without any further joins.

Table produced:
  - data/gold/obt_transaction_fraud_view

Grain   : 1 row per transaction
Contains: transaction_id, account_id, merchant_category, amount,
          customer risk_segment, merchant risk_tier, city, timestamps,
          is_flagged_fraud (heuristic rule-based flag)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from delta.tables import DeltaTable  # type: ignore

from finance_mlops.config.paths import LAKEHOUSE_GOLD_DIR
from finance_mlops.pipelines.common.utils.spark import get_spark

log = logging.getLogger(__name__)

GOLD_DIR = str(LAKEHOUSE_GOLD_DIR)


def _gold_metadata(df: DataFrame) -> DataFrame:
    now_ts = F.lit(datetime.now(tz=timezone.utc).isoformat()).cast("timestamp")
    return df.withColumn("_gold_ts", now_ts).withColumn("_layer", F.lit("gold"))


def _flag_fraud(df: DataFrame) -> DataFrame:
    """
    Heuristic fraud flag – combines several risk signals:
      1. Merchant is high-risk tier
      2. Transaction amount > 4,000 (top ~1 % of value distribution)
      3. Foreign currency
      4. Account risk segment is 'standard' (highest-volume, often targeted)

    This is a simple rule-based proxy; a real ML score will replace it.
    """
    return df.withColumn(
        "is_flagged_fraud",
        (
            (F.col("merchant_risk_tier") == "high") |
            (F.col("transaction_amount") > 4_000) |
            (F.col("is_foreign_currency") == 1)
        ).cast("int")
    )


def build_obt(spark: SparkSession) -> str:
    dst_path = os.path.join(GOLD_DIR, "obt_transaction_fraud_view")
    log.info("[GOLD] Building obt_transaction_fraud_view …")

    fact_tx   = spark.read.format("delta").load(os.path.join(GOLD_DIR, "fact_transaction"))
    dim_acct  = spark.read.format("delta").load(os.path.join(GOLD_DIR, "dim_account"))
    dim_cust  = spark.read.format("delta").load(os.path.join(GOLD_DIR, "dim_customer"))
    dim_merch = spark.read.format("delta").load(os.path.join(GOLD_DIR, "dim_merchant"))
    dim_date  = spark.read.format("delta").load(os.path.join(GOLD_DIR, "dim_date"))

    df = (
        fact_tx
        # Enrich with account + customer
        .join(
            dim_acct.select("account_key", "account_type", "customer_key"),
            on="account_key", how="left",
        )
        .join(
            dim_cust.select(
                "customer_key", "customer_id",
                F.col("risk_segment").alias("customer_risk_segment"),
                "kyc_status", "country",
            ),
            on="customer_key", how="left",
        )
        # Enrich with merchant
        .join(
            dim_merch.select(
                "merchant_key", "merchant_id", "merchant_name",
                F.col("category_code").alias("merchant_category"),
                F.col("risk_tier").alias("merchant_risk_tier"),
            ),
            on="merchant_key", how="left",
        )
        # Enrich with date attributes
        .join(
            dim_date.select(
                "date_key", "year", "quarter", "month",
                "day_of_week", "is_weekend",
            ),
            on="date_key", how="left",
        )
    )

    # Apply fraud heuristic flag
    df = _flag_fraud(df)

    # Final column selection (wide denormalized table for Fraud Analysts)
    df = df.select(
        # Transaction identifiers
        "transaction_id",
        "account_id",
        "customer_id",
        "merchant_id",
        # Timestamps & date
        "transaction_timestamp",
        "date_key",
        "year",
        "quarter",
        "month",
        "day_of_week",
        "is_weekend",
        # Financial measures
        "transaction_amount",
        "fee_amount",
        "currency",
        "status",
        "is_declined",
        "is_foreign_currency",
        # Customer attributes
        "customer_risk_segment",
        "kyc_status",
        F.col("country").alias("customer_country"),
        # Merchant attributes
        "merchant_name",
        "merchant_category",
        "merchant_risk_tier",
        # Location & device
        "city",
        "device_id",
        "ip_address",
        # Fraud signal
        "is_flagged_fraud",
    )

    df = _gold_metadata(df)

    if DeltaTable.isDeltaTable(spark, dst_path):
        target = DeltaTable.forPath(spark, dst_path)
        (
            target.alias("tgt")
            .merge(df.alias("src"), "tgt.transaction_id = src.transaction_id")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
        log.info("[GOLD] Merged obt_transaction_fraud_view → %s", dst_path)
    else:
        df.write.format("delta").mode("overwrite").save(dst_path)
        log.info("[GOLD] Created obt_transaction_fraud_view → %s", dst_path)

    return dst_path


# ── Entry point ───────────────────────────────────────────────────────────────

def run(spark: SparkSession | None = None) -> dict[str, str]:
    """Build the OBT fraud view."""
    standalone = spark is None
    if standalone:
        spark = get_spark("gold_build_obt")

    try:
        path = build_obt(spark)
        return {"obt_transaction_fraud_view": path}
    finally:
        if standalone:
            spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
    results = run()
    for name, path in results.items():
        print(f"  ✓ {name:40s} → {path}")
