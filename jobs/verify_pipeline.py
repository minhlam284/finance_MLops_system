"""
jobs/verify_pipeline.py
------------------------
Quick sanity-check script to validate the entire pipeline output.

Checks:
  1. All expected Delta tables exist and are readable
  2. Row counts are non-zero
  3. Schema contains expected key columns
  4. OBT sample print
  5. Feature store sample print

Run after executing the full pipeline end-to-end.
"""
from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPECTED_TABLES = {
    # layer → {table_name: required_columns}
    "bronze": {
        "customers":    ["customer_id", "_ingestion_ts"],
        "accounts":     ["account_id",  "_ingestion_ts"],
        "merchants":    ["merchant_id", "_ingestion_ts"],
        "transactions": ["transaction_id", "_ingestion_ts"],
        "events":       ["event_id", "_ingestion_ts"],
    },
    "silver": {
        "customers":    ["customer_id",    "_silver_ts"],
        "accounts":     ["account_id",     "_silver_ts"],
        "merchants":    ["merchant_id",    "_silver_ts"],
        "transactions": ["transaction_id", "_silver_ts"],
        "events":       ["event_id",       "_silver_ts"],
    },
    "gold": {
        "dim_customer":               ["customer_key", "customer_id"],
        "dim_account":                ["account_key",  "account_id"],
        "dim_merchant":               ["merchant_key", "merchant_id"],
        "dim_date":                   ["date_key",     "calendar_date"],
        "fact_transaction":           ["transaction_id", "account_key", "merchant_key", "date_key", "is_fraudulent"],
        "fact_auth_attempt":          ["event_id",     "account_key"],
        "obt_transaction_fraud_view": ["transaction_id", "is_flagged_fraud"],
        "feat_account_90d":           ["account_id",   "f_account_total_tx_90d"],
        "feat_stream_60m":            ["account_id",   "f_stream_tx_velocity_60m"],
        "feat_account_unified":       ["account_id",   "f_account_total_tx_90d", "f_stream_tx_velocity_60m"],
        # ML monitoring & training tables
        "agg_feature_health_daily":      ["metric_date", "feature_name", "psi", "is_alert"],
        "ml_transaction_label":          ["transaction_id", "account_id", "event_timestamp", "label"],
        "ml_fraud_detection_training":   ["transaction_id", "account_id", "label", "f_account_total_tx_90d"],
    },
}


def verify(spark) -> bool:
    from pyspark.sql import SparkSession
    from delta.tables import DeltaTable

    all_ok = True
    results = []

    for layer, tables in EXPECTED_TABLES.items():
        for table_name, required_cols in tables.items():
            path = os.path.join(BASE_DIR, "data", layer, table_name)

            # 1. Table exists?
            if not DeltaTable.isDeltaTable(spark, path):
                results.append(("❌", layer, table_name, "Table does NOT exist"))
                all_ok = False
                continue

            # 2. Readable & non-empty?
            try:
                df = spark.read.format("delta").load(path)
                count = df.count()
            except Exception as e:
                results.append(("❌", layer, table_name, f"Read error: {e}"))
                all_ok = False
                continue

            if count == 0:
                results.append(("⚠️ ", layer, table_name, "EMPTY table (0 rows)"))
                all_ok = False
                continue

            # 3. Required columns present?
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                results.append(("❌", layer, table_name, f"Missing columns: {missing}"))
                all_ok = False
                continue

            results.append(("✅", layer, table_name, f"{count:,} rows | {len(df.columns)} cols"))

    # Print results table
    print("\n" + "=" * 72)
    print(f"{'':2} {'Layer':8} {'Table':35} {'Result'}")
    print("=" * 72)
    for icon, layer, table, msg in results:
        print(f"{icon} {layer:8} {table:35} {msg}")
    print("=" * 72)

    # 4. OBT sample
    obt_path = os.path.join(BASE_DIR, "data", "gold", "obt_transaction_fraud_view")
    if DeltaTable.isDeltaTable(spark, obt_path):
        print("\n── OBT Sample (fraud-flagged rows) ──────────────────────────────────")
        (
            spark.read.format("delta").load(obt_path)
            .filter("is_flagged_fraud = 1")
            .select("transaction_id", "customer_risk_segment", "merchant_category",
                    "transaction_amount", "is_foreign_currency", "merchant_risk_tier",
                    "is_flagged_fraud")
            .show(5, truncate=False)
        )

    # 5. Feature store sample
    feat_path = os.path.join(BASE_DIR, "data", "gold", "feat_account_unified")
    if DeltaTable.isDeltaTable(spark, feat_path):
        print("\n── Feature Store Sample (feat_account_unified) ──────────────────────")
        (
            spark.read.format("delta").load(feat_path)
            .select(
                "account_id",
                "f_account_total_tx_90d",
                "f_account_avg_tx_value_90d",
                "f_stream_tx_velocity_60m",
                "f_stream_login_failures_30m",
            )
            .show(5, truncate=False)
        )

    # 6. ML tables spot-check
    health_path = os.path.join(BASE_DIR, "data", "gold", "agg_feature_health_daily")
    if DeltaTable.isDeltaTable(spark, health_path):
        print("\n── Feature Health Alerts (PSI > 0.15) ───────────────────────────────")
        (
            spark.read.format("delta").load(health_path)
            .filter("is_alert = true")
            .orderBy("feature_name", "metric_date")
            .select("metric_date", "feature_name", "psi", "is_alert")
            .show(10, truncate=False)
        )

    training_path = os.path.join(BASE_DIR, "data", "gold", "ml_fraud_detection_training")
    if DeltaTable.isDeltaTable(spark, training_path):
        print("\n── ML Training Dataset Sample (fraud rows) ──────────────────────────")
        (
            spark.read.format("delta").load(training_path)
            .filter("label = 1")
            .select(
                "transaction_id", "account_id", "label",
                "f_account_total_tx_90d", "f_account_avg_tx_value_90d",
            )
            .show(5, truncate=False)
        )

    return all_ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s – %(message)s")

    sys.path.insert(0, BASE_DIR)
    from jobs.utils.spark import get_spark

    spark = get_spark("verify_pipeline")
    try:
        ok = verify(spark)
        sys.exit(0 if ok else 1)
    finally:
        spark.stop()
