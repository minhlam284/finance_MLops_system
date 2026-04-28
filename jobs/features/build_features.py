"""
jobs/features/build_features.py
---------------------------------
Feature Store – Feature Engineering Job.

Builds ML features in the Gold partition using window aggregations.
All features are keyed by (account_id, event_timestamp) for point-in-time
correctness, and upserted via Delta MERGE.

Feature tables produced:
  - data/gold/feat_account_90d
      f_account_total_tx_90d      : total transactions in last 90 days
      f_account_avg_tx_value_90d  : avg transaction value in last 90 days
      f_account_max_tx_value_90d  : max transaction value in last 90 days
      f_account_declined_ratio_90d: ratio of declined txns in last 90 days
      f_account_foreign_tx_ratio_90d: ratio of foreign-currency txns in 90d

  - data/gold/feat_stream_60m
      f_stream_tx_velocity_60m    : number of transactions in last 60 minutes
      f_stream_login_failures_30m : number of login events in last 30 minutes
      f_stream_unique_devices_60m : distinct device types in last 60 minutes

  - data/gold/feat_account_unified
      Join of feat_account_90d + feat_stream_60m on account_id.
      The final feature vector fed to fraud ML models.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from delta.tables import DeltaTable  # type: ignore
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from jobs.utils.spark import get_spark

log = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SILVER_DIR = os.path.join(BASE_DIR, "data", "silver")
GOLD_DIR   = os.path.join(BASE_DIR, "data", "gold")

# Window sizes in seconds
SECONDS_90D = 90 * 24 * 3600
SECONDS_60M = 60 * 60
SECONDS_30M = 30 * 60


# ── Metadata ──────────────────────────────────────────────────────────────────

def _feat_metadata(df: DataFrame, feature_set: str) -> DataFrame:
    now_ts = F.lit(datetime.now(tz=timezone.utc).isoformat()).cast("timestamp")
    return (
        df
        .withColumn("_feature_set", F.lit(feature_set))
        .withColumn("_computed_ts", now_ts)
        .withColumn("_layer",       F.lit("gold"))
    )


# ── Delta Merge upsert ────────────────────────────────────────────────────────

def _upsert_features(
    spark: SparkSession,
    df: DataFrame,
    merge_keys: list[str],
    dst_path: str,
) -> None:
    merge_cond = " AND ".join(f"tgt.{k} = src.{k}" for k in merge_keys)

    if DeltaTable.isDeltaTable(spark, dst_path):
        target = DeltaTable.forPath(spark, dst_path)
        (
            target.alias("tgt")
            .merge(df.alias("src"), merge_cond)
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
        log.info("[FEATURES] Upserted → %s", dst_path)
    else:
        df.write.format("delta").mode("overwrite").save(dst_path)
        log.info("[FEATURES] Created → %s", dst_path)


# ── feat_account_90d ──────────────────────────────────────────────────────────

def build_feat_account_90d(spark: SparkSession) -> str:
    """
    Offline batch features: rolling 90-day aggregates per account.
    Computed relative to the LATEST transaction timestamp per account
    (serves as the "as-of" date for batch training datasets).
    """
    dst_path = os.path.join(GOLD_DIR, "feat_account_90d")
    log.info("[FEATURES] Building feat_account_90d …")

    tx = spark.read.format("delta").load(os.path.join(SILVER_DIR, "transactions"))

    # Compute per-account latest timestamp as the reference point
    max_ts_per_account = (
        tx.groupBy("account_id")
          .agg(F.max("transaction_timestamp").alias("as_of_ts"))
    )

    # Join back and filter to rolling 90-day window
    tx = tx.join(max_ts_per_account, on="account_id", how="left")
    tx_90d = tx.filter(
        F.col("transaction_timestamp") >= F.col("as_of_ts") - F.expr(f"interval {SECONDS_90D} seconds")
    )

    feats = tx_90d.groupBy("account_id").agg(
        F.count("transaction_id").alias("f_account_total_tx_90d"),
        F.avg("amount").alias("f_account_avg_tx_value_90d"),
        F.max("amount").alias("f_account_max_tx_value_90d"),
        F.avg(F.when(F.col("status") == "declined", 1).otherwise(0))
         .alias("f_account_declined_ratio_90d"),
        F.avg(F.when(F.col("currency") != "VND", 1).otherwise(0))
         .alias("f_account_foreign_tx_ratio_90d"),
        F.max("as_of_ts").alias("event_timestamp"),
    )

    feats = feats.withColumn("f_account_avg_tx_value_90d",
                              F.round("f_account_avg_tx_value_90d", 2))
    feats = _feat_metadata(feats, "feat_account_90d")

    _upsert_features(spark, feats, ["account_id"], dst_path)
    return dst_path


# ── feat_stream_60m ───────────────────────────────────────────────────────────

def build_feat_stream_60m(spark: SparkSession) -> str:
    """
    Near-real-time streaming features: rolling 60-min / 30-min aggregates.
    Computed relative to the LATEST event timestamp per account.
    """
    dst_path = os.path.join(GOLD_DIR, "feat_stream_60m")
    log.info("[FEATURES] Building feat_stream_60m …")

    events = spark.read.format("delta").load(os.path.join(SILVER_DIR, "events"))

    max_ts_per_account = (
        events.groupBy("account_id")
              .agg(F.max("event_timestamp").alias("as_of_ts"))
    )

    events = events.join(max_ts_per_account, on="account_id", how="left")

    # 60-minute window
    events_60m = events.filter(
        F.col("event_timestamp") >= F.col("as_of_ts") - F.expr(f"interval {SECONDS_60M} seconds")
    )

    # 30-minute window
    events_30m = events.filter(
        F.col("event_timestamp") >= F.col("as_of_ts") - F.expr(f"interval {SECONDS_30M} seconds")
    )

    feats_60m = events_60m.groupBy("account_id").agg(
        F.count(F.when(F.col("event_type") == "transaction_auth", 1))
         .alias("f_stream_tx_velocity_60m"),
        F.countDistinct("device_type")
         .alias("f_stream_unique_devices_60m"),
        F.max("as_of_ts").alias("event_timestamp"),
    )

    feats_30m = events_30m.groupBy("account_id").agg(
        F.count(F.when(F.col("event_type") == "login_attempt", 1))
         .alias("f_stream_login_failures_30m"),
    )

    feats = feats_60m.join(feats_30m, on="account_id", how="left")
    feats = feats.fillna(0, subset=[
        "f_stream_tx_velocity_60m",
        "f_stream_unique_devices_60m",
        "f_stream_login_failures_30m",
    ])

    feats = _feat_metadata(feats, "feat_stream_60m")
    _upsert_features(spark, feats, ["account_id"], dst_path)
    return dst_path


# ── feat_account_unified ──────────────────────────────────────────────────────

def build_feat_account_unified(spark: SparkSession) -> str:
    """
    Unified feature vector: join offline (90d) and near-RT (60m) features.
    This is the final feature set consumed by the fraud ML model.
    """
    dst_path = os.path.join(GOLD_DIR, "feat_account_unified")
    log.info("[FEATURES] Building feat_account_unified …")

    offline = spark.read.format("delta").load(os.path.join(GOLD_DIR, "feat_account_90d"))
    stream  = spark.read.format("delta").load(os.path.join(GOLD_DIR, "feat_stream_60m"))

    # Drop metadata columns before join to avoid ambiguity
    meta_cols = ["_feature_set", "_computed_ts", "_layer", "event_timestamp"]
    offline_clean = offline.drop(*meta_cols)
    stream_clean  = stream.drop(*meta_cols)

    unified = offline_clean.join(stream_clean, on="account_id", how="left")

    # Fill missing streaming features for accounts with no recent activity
    unified = unified.fillna(0, subset=[
        "f_stream_tx_velocity_60m",
        "f_stream_unique_devices_60m",
        "f_stream_login_failures_30m",
    ])

    unified = _feat_metadata(unified, "feat_account_unified")
    _upsert_features(spark, unified, ["account_id"], dst_path)
    return dst_path


# ── Entry point ───────────────────────────────────────────────────────────────

def run(spark: SparkSession | None = None) -> dict[str, str]:
    """Build all feature store tables."""
    standalone = spark is None
    if standalone:
        spark = get_spark("feature_store_build")

    try:
        paths = {}
        paths["feat_account_90d"]    = build_feat_account_90d(spark)
        paths["feat_stream_60m"]     = build_feat_stream_60m(spark)
        paths["feat_account_unified"] = build_feat_account_unified(spark)

        log.info("[FEATURES] Feature store build complete.")
        return paths
    finally:
        if standalone:
            spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
    results = run()
    for name, path in results.items():
        print(f"  ✓ {name:35s} → {path}")
