"""
jobs/gold/build_ml_tables.py
------------------------------
Gold layer – ML Monitoring & Training Table Builder.

Builds three ML-focused tables:

1. agg_feature_health_daily
   ─────────────────────────
   Daily aggregate statistics for two feature proxies computed from
   `fact_transaction`:
     • daily_avg_tx_value  →  proxy for f_account_avg_tx_value_90d
     • daily_tx_count      →  proxy for f_account_total_tx_90d

   PSI (Population Stability Index) is calculated against a reference
   period (pre-drift: 2023-01-01 → 2024-10-31) using 10 equal-width bins.
   An `is_alert` boolean flag is set when PSI > FEATURE_HEALTH_PSI_ALERT (0.15).

   Schema:
     metric_date DATE, feature_name STRING, psi DOUBLE,
     is_alert BOOLEAN, _gold_ts TIMESTAMP, _layer STRING

2. ml_transaction_label
   ──────────────────────
   Point-in-time fraud labels for every transaction. Consumed by
   Feature Store retrieval (point-in-time join).

   Schema:
     transaction_id STRING, account_id STRING,
     event_timestamp TIMESTAMP,   # = transaction_timestamp
     created_ts TIMESTAMP,        # = _gold_ts (when label was written)
     label INT                    # = is_fraudulent (0 / 1)

3. ml_fraud_detection_training
   ─────────────────────────────
   Unified training dataset: label table LEFT-JOINed with the pre-built
   `feat_account_unified` feature vector on `account_id`.
   Intended for offline ML model training / evaluation.

   Schema:
     transaction_id, account_id, event_timestamp, created_ts, label,
     + all feature columns from feat_account_unified
"""
from __future__ import annotations

import logging
import os
import math
from datetime import datetime, timezone

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from delta.tables import DeltaTable  # type: ignore

from finance_mlops.config.paths import LAKEHOUSE_GOLD_DIR
from finance_mlops.pipelines.common.utils.spark import get_spark

log = logging.getLogger(__name__)

GOLD_DIR = str(LAKEHOUSE_GOLD_DIR)

# Reference period for PSI baseline (pre-drift)
REFERENCE_START = "2023-01-01"
REFERENCE_END   = "2024-10-31"

# PSI alert threshold (from config — replicated here for standalone use)
FEATURE_HEALTH_PSI_ALERT = 0.15

# Number of equal-width bins for PSI calculation
PSI_N_BINS = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gold_metadata(df: DataFrame) -> DataFrame:
    now_ts = F.lit(datetime.now(tz=timezone.utc).isoformat()).cast("timestamp")
    return df.withColumn("_gold_ts", now_ts).withColumn("_layer", F.lit("gold"))


def _overwrite_delta(df: DataFrame, dst_path: str, spark: SparkSession) -> None:
    """Full overwrite for monitoring/ML tables (no MERGE needed)."""
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(dst_path)
    log.info("[ML] Written → %s  (%s rows)", dst_path, df.count())


def _upsert_delta(
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
        log.info("[ML] Upserted → %s", dst_path)
    else:
        df.write.format("delta").mode("overwrite").save(dst_path)
        log.info("[ML] Created → %s", dst_path)


# ── PSI utilities ─────────────────────────────────────────────────────────────

def _compute_psi_pandas(reference_values, current_values, n_bins: int = PSI_N_BINS) -> float:
    """
    Compute Population Stability Index between reference and current distributions.

    PSI = Σ (p_current - p_ref) * ln(p_current / p_ref)

    Interpretation:
      PSI < 0.10  → No significant shift
      PSI 0.10–0.25 → Moderate shift (alert at 0.1 for individual features)
      PSI > 0.25  → Major shift
    """
    import numpy as np

    ref  = np.array(reference_values, dtype=float)
    curr = np.array(current_values,   dtype=float)

    # Determine bin edges from both populations combined
    combined = np.concatenate([ref, curr])
    min_val, max_val = combined.min(), combined.max()
    if min_val == max_val:
        return 0.0

    edges = np.linspace(min_val, max_val, n_bins + 1)

    def _freq(arr):
        counts, _ = np.histogram(arr, bins=edges)
        # Laplace smoothing to avoid division by zero / log(0)
        counts = counts + 1e-6
        return counts / counts.sum()

    p_ref  = _freq(ref)
    p_curr = _freq(curr)

    psi = float(np.sum((p_curr - p_ref) * np.log(p_curr / p_ref)))
    return round(max(psi, 0.0), 6)  # clamp negative rounding artifacts


# ── Table 1: agg_feature_health_daily ────────────────────────────────────────

def build_agg_feature_health_daily(spark: SparkSession) -> str:
    """
    Compute daily PSI for two feature proxies and write agg_feature_health_daily.

    Feature proxies (computed from fact_transaction):
      - feature: 'f_account_avg_tx_value_90d'  metric: daily mean(transaction_amount)
      - feature: 'f_account_total_tx_90d'       metric: daily count(transaction_id)
    """
    dst_path = os.path.join(GOLD_DIR, "agg_feature_health_daily")
    log.info("[ML] Building agg_feature_health_daily …")

    fact_tx = spark.read.format("delta").load(os.path.join(GOLD_DIR, "fact_transaction"))

    # Derive metric_date
    fact_tx = fact_tx.withColumn(
        "metric_date", F.to_date("transaction_timestamp")
    )

    # Reference window (pre-drift)
    ref_df = fact_tx.filter(
        (F.col("metric_date") >= F.lit(REFERENCE_START)) &
        (F.col("metric_date") <= F.lit(REFERENCE_END))
    )

    # ── Daily aggregates across ALL dates ──
    daily_avg = (
        fact_tx
        .groupBy("metric_date")
        .agg(F.avg("transaction_amount").alias("metric_value"))
        .withColumn("feature_name", F.lit("f_account_avg_tx_value_90d"))
    )

    daily_cnt = (
        fact_tx
        .groupBy("metric_date")
        .agg(F.count("transaction_id").cast(DoubleType()).alias("metric_value"))
        .withColumn("feature_name", F.lit("f_account_total_tx_90d"))
    )

    # ── Reference distributions (collect once) ──
    ref_avg = (
        ref_df
        .groupBy("metric_date")
        .agg(F.avg("transaction_amount").alias("metric_value"))
        .select("metric_value")
        .dropna()
        .toPandas()["metric_value"].tolist()
    )

    ref_cnt = (
        ref_df
        .groupBy("metric_date")
        .agg(F.count("transaction_id").cast(DoubleType()).alias("metric_value"))
        .select("metric_value")
        .dropna()
        .toPandas()["metric_value"].tolist()
    )

    log.info("[ML] Reference distributions collected — avg(%d days), cnt(%d days)",
             len(ref_avg), len(ref_cnt))

    # ── Compute PSI per metric_date using Pandas UDF style ──
    # We collect daily metrics to driver and compute PSI month-by-month
    # (acceptable for this monitoring scale; replace with Spark UDF for very large data)
    def _add_psi_column(df_spark: DataFrame, ref_values: list, feature_name: str) -> DataFrame:
        """Add PSI relative to the reference distribution for each row date."""
        # Collect all daily values
        rows = df_spark.select("metric_date", "metric_value").dropna().toPandas()
        if rows.empty or not ref_values:
            return spark.createDataFrame([], df_spark.schema.add("psi", DoubleType()))

        # Compute PSI for each date: [ref] vs [single day value]
        # A single day's value isn't enough for distribution comparison,
        # so we use a rolling 30-day window worth of daily values vs reference.
        rows = rows.sort_values("metric_date").reset_index(drop=True)
        rows["psi"] = 0.0

        window = 30
        for i in range(len(rows)):
            start = max(0, i - window + 1)
            current_window = rows["metric_value"].iloc[start: i + 1].tolist()
            if len(current_window) >= 2:
                rows.at[i, "psi"] = _compute_psi_pandas(ref_values, current_window)

        # Attach feature_name
        rows["feature_name"] = feature_name

        result = spark.createDataFrame(rows[["metric_date", "feature_name", "metric_value", "psi"]])
        return result

    health_avg = _add_psi_column(daily_avg, ref_avg, "f_account_avg_tx_value_90d")
    health_cnt = _add_psi_column(daily_cnt, ref_cnt, "f_account_total_tx_90d")

    health = health_avg.unionByName(health_cnt)

    # Add alert flag and metadata
    health = (
        health
        .withColumn("is_alert", F.col("psi") > F.lit(FEATURE_HEALTH_PSI_ALERT))
        .drop("metric_value")   # raw metric not needed in monitoring table
    )
    health = _gold_metadata(health)

    _overwrite_delta(health, dst_path, spark)

    # Log alert summary
    n_alerts = health.filter("is_alert = true").count()
    log.info("[ML] agg_feature_health_daily complete — %d alert days (PSI > %.2f)",
             n_alerts, FEATURE_HEALTH_PSI_ALERT)
    return dst_path


# ── Table 2: ml_transaction_label ────────────────────────────────────────────

def build_ml_transaction_label(spark: SparkSession) -> str:
    """
    Extract point-in-time fraud labels from fact_transaction.

    Columns:
      transaction_id  – natural key
      account_id      – entity key for feature join
      event_timestamp – when the transaction occurred (point-in-time anchor)
      created_ts      – when the label was written to Gold (_gold_ts)
      label           – is_fraudulent (0 = legitimate, 1 = fraud)
    """
    dst_path = os.path.join(GOLD_DIR, "ml_transaction_label")
    log.info("[ML] Building ml_transaction_label …")

    fact_tx = spark.read.format("delta").load(os.path.join(GOLD_DIR, "fact_transaction"))

    labels = (
        fact_tx
        .select(
            "transaction_id",
            "account_id",
            F.col("transaction_timestamp").alias("event_timestamp"),
            F.col("_gold_ts").alias("created_ts"),
            F.col("is_fraudulent").alias("label"),
        )
        .dropDuplicates(["transaction_id"])
    )

    _upsert_delta(spark, labels, ["transaction_id"], dst_path)

    fraud_count = labels.filter("label = 1").count()
    total_count = labels.count()
    log.info(
        "[ML] ml_transaction_label complete — %d rows, %d fraud (%.1f%%)",
        total_count, fraud_count, 100 * fraud_count / max(1, total_count),
    )
    return dst_path


# ── Table 3: ml_fraud_detection_training ─────────────────────────────────────

def build_ml_fraud_detection_training(spark: SparkSession) -> str:
    """
    Build the final training dataset:
      ml_transaction_label (labels) LEFT JOIN feat_account_unified (features)
      on account_id.

    This provides a point-in-time accurate training table where each row
    contains:
      - The fraud label for a specific transaction
      - The 90-day and 60-min features computed for that account
    """
    dst_path = os.path.join(GOLD_DIR, "ml_fraud_detection_training")
    log.info("[ML] Building ml_fraud_detection_training …")

    labels  = spark.read.format("delta").load(os.path.join(GOLD_DIR, "ml_transaction_label"))
    unified = spark.read.format("delta").load(os.path.join(GOLD_DIR, "feat_account_unified"))

    # Drop feature metadata columns to keep the training set clean
    meta_cols = ["_feature_set", "_computed_ts", "_layer"]
    unified_clean = unified.drop(*meta_cols)

    training = labels.join(unified_clean, on="account_id", how="left")

    # Fill missing feature values (accounts with no feature history)
    feature_cols = [c for c in unified_clean.columns if c.startswith("f_")]
    training = training.fillna(0, subset=feature_cols)

    _upsert_delta(spark, training, ["transaction_id"], dst_path)

    total = training.count()
    fraud = training.filter("label = 1").count()
    log.info(
        "[ML] ml_fraud_detection_training complete — %d rows, "
        "%d fraud (%.1f%%), %d feature columns",
        total, fraud, 100 * fraud / max(1, total), len(feature_cols),
    )
    return dst_path


# ── Entry point ───────────────────────────────────────────────────────────────

def run(spark: SparkSession | None = None) -> dict[str, str]:
    """Build all ML monitoring and training tables."""
    standalone = spark is None
    if standalone:
        spark = get_spark("gold_build_ml_tables")

    try:
        paths = {}
        paths["agg_feature_health_daily"]      = build_agg_feature_health_daily(spark)
        paths["ml_transaction_label"]          = build_ml_transaction_label(spark)
        paths["ml_fraud_detection_training"]   = build_ml_fraud_detection_training(spark)

        log.info("[ML] All ML tables built successfully.")
        return paths
    finally:
        if standalone:
            spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
    results = run()
    for name, path in results.items():
        print(f"  ✓ {name:40s} → {path}")
