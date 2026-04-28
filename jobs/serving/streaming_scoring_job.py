"""
jobs/serving/streaming_scoring_job.py
---------------------------------------
Real-Time Fraud Scoring Pipeline (Spark Structured Streaming).

Architecture:
  bronze/events (Delta streaming source)
       │
       ▼  filter event_type = 'transaction_auth'
  enrichment: LEFT JOIN gold/feat_account_unified  (static batch lookup)
       │
       ▼  apply fraud_score_udf  (XGBoost Production model from MLflow)
  gold/fraud_scores  (Delta append sink)

Output schema for gold/fraud_scores:
  event_id        STRING    – original event identifier
  account_id      STRING    – entity key
  transaction_id  STRING    – linked transaction (nullable)
  event_timestamp TIMESTAMP – when the event occurred
  fraud_score     DOUBLE    – model probability of fraud [0, 1]
  is_blocked      BOOLEAN   – True if fraud_score >= BLOCK_THRESHOLD
  scored_at       TIMESTAMP – when scoring happened
  model_version   STRING    – MLflow model version used
  _layer          STRING    – 'gold'

SLA target: Online inference latency p95 < 200 ms.

Run modes:
  - Standalone  : python -m jobs.serving.streaming_scoring_job
                  (runs until manually stopped; uses Spark local[*])
  - Airflow     : call run(spark=spark, timeout_secs=300) from DAG task
  - Testing     : set max_files_per_trigger=1, timeout_secs=30
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import mlflow
import mlflow.xgboost
import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from jobs.utils.spark import get_spark

log = logging.getLogger(__name__)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRONZE_EVENTS_PATH   = os.path.join(BASE_DIR, "data", "bronze", "events")
GOLD_FEATURES_PATH   = os.path.join(BASE_DIR, "data", "gold", "feat_account_unified")
GOLD_SCORES_PATH     = os.path.join(BASE_DIR, "data", "gold", "fraud_scores")
CHECKPOINT_PATH      = os.path.join(BASE_DIR, "data", "_checkpoints", "fraud_scoring")
MLRUNS_DIR           = os.path.join(BASE_DIR, "mlruns")

MODEL_NAME       = "fraud_detection_model"
BLOCK_THRESHOLD  = 0.60   # fraud_score >= this → is_blocked = True

FEATURE_COLS = [
    "f_account_total_tx_90d",
    "f_account_avg_tx_value_90d",
    "f_account_max_tx_value_90d",
    "f_account_declined_ratio_90d",
    "f_account_foreign_tx_ratio_90d",
    "f_stream_tx_velocity_60m",
    "f_stream_unique_devices_60m",
    "f_stream_login_failures_30m",
]

# Default trigger: process new files as they arrive (micro-batch)
DEFAULT_TRIGGER_INTERVAL = "30 seconds"


# ── Model loader (cached at module level inside the UDF closure) ───────────────

def _load_production_model():
    """
    Load the Production XGBoost model from the local MLflow registry.
    Returns (model, version_str).
    """
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    client = mlflow.tracking.MlflowClient()

    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(
            f"No Production model found for '{MODEL_NAME}'. "
            "Run evaluate_and_register_job.py first to promote a model."
        )
    mv       = versions[0]
    model_uri = f"models:/{MODEL_NAME}/Production"
    model    = mlflow.xgboost.load_model(model_uri)
    log.info(
        "[SERVING] Loaded model '%s' version=%s (run_id=%s)",
        MODEL_NAME, mv.version, mv.run_id,
    )
    return model, str(mv.version)


# ── Spark UDF factory ─────────────────────────────────────────────────────────

def _make_score_udf(model, model_version: str):
    """
    Build a Pandas UDF that scores a batch of feature rows.

    Spark Pandas UDFs operate on pd.Series objects, so we receive
    one Series per feature column and return a Series of fraud scores.
    """
    import pandas as pd
    from pyspark.sql.functions import pandas_udf

    # Capture model reference in closure (broadcast-safe for local mode)
    _model = model
    _version = model_version

    @pandas_udf(DoubleType())
    def fraud_score_udf(*feature_series) -> pd.Series:
        """UDF: feature columns → fraud probability."""
        X = np.column_stack([s.fillna(0.0).values for s in feature_series]).astype("float32")
        probs = _model.predict_proba(X)[:, 1]
        return pd.Series(probs.astype(float))

    return fraud_score_udf, _version


# ── Stream processing ─────────────────────────────────────────────────────────

def _enrich_with_features(stream_df: DataFrame, features_df: DataFrame) -> DataFrame:
    """
    Filter for transaction events and left-join with feature vector.
    `features_df` is a static DataFrame (batch read).
    """
    # Keep only transaction auth events
    tx_events = stream_df.filter(F.col("event_type") == "transaction_auth")

    # Drop feature metadata cols to avoid conflicts
    meta_cols = ["_feature_set", "_computed_ts", "_layer", "event_timestamp"]
    features_clean = features_df.drop(*meta_cols)

    enriched = tx_events.join(features_clean, on="account_id", how="left")

    # Fill missing features with 0
    enriched = enriched.fillna(0.0, subset=FEATURE_COLS)
    return enriched


def _apply_scoring(df: DataFrame, score_udf, model_version: str) -> DataFrame:
    """Apply the fraud scoring UDF and add action columns."""
    now_ts = F.lit(datetime.now(tz=timezone.utc).isoformat()).cast("timestamp")

    scored = df.withColumn(
        "fraud_score",
        score_udf(*[F.col(c).cast(DoubleType()) for c in FEATURE_COLS]),
    )
    scored = (
        scored
        .withColumn("is_blocked",    F.col("fraud_score") >= F.lit(BLOCK_THRESHOLD))
        .withColumn("scored_at",     now_ts)
        .withColumn("model_version", F.lit(model_version))
        .withColumn("_layer",        F.lit("gold"))
        .select(
            "event_id",
            "account_id",
            "transaction_id",
            "event_timestamp",
            "fraud_score",
            "is_blocked",
            "scored_at",
            "model_version",
            "_layer",
        )
    )
    return scored


# ── Sink ──────────────────────────────────────────────────────────────────────

def _write_stream(scored_stream: DataFrame, trigger_interval: str):
    """Write streaming scores to Delta sink."""
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    query = (
        scored_stream.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT_PATH)
        .trigger(processingTime=trigger_interval)
        .start(GOLD_SCORES_PATH)
    )
    log.info("[SERVING] Streaming query started → %s", GOLD_SCORES_PATH)
    return query


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    spark: SparkSession | None = None,
    timeout_secs: int | None = None,
    trigger_interval: str = DEFAULT_TRIGGER_INTERVAL,
) -> None:
    """
    Start the real-time fraud scoring pipeline.

    Parameters
    ----------
    spark : SparkSession, optional
        Reuse an existing session (Airflow usage). Creates one if None.
    timeout_secs : int, optional
        If set, the streaming query will stop after this many seconds.
        Useful for testing or scheduled micro-batch runs via Airflow.
        If None, runs indefinitely (standalone mode).
    trigger_interval : str
        Spark Structured Streaming trigger interval.
    """
    standalone = spark is None
    if standalone:
        spark = get_spark("fraud_scoring_stream")

    try:
        # ── Load static assets ──────────────────────────────────────
        model, model_version = _load_production_model()
        score_udf, ver       = _make_score_udf(model, model_version)

        # Static feature table (refreshed each micro-batch startup)
        features_df = spark.read.format("delta").load(GOLD_FEATURES_PATH)

        # ── Streaming source ────────────────────────────────────────
        stream_df = (
            spark.readStream
            .format("delta")
            .option("maxFilesPerTrigger", 10)
            .load(BRONZE_EVENTS_PATH)
        )

        # ── Transformation ─────────────────────────────────────────
        enriched     = _enrich_with_features(stream_df, features_df)
        scored_stream = _apply_scoring(enriched, score_udf, ver)

        # ── Write to sink ──────────────────────────────────────────
        query = _write_stream(scored_stream, trigger_interval)

        if timeout_secs is not None:
            query.awaitTermination(timeout_secs)
            query.stop()
            log.info("[SERVING] Streaming query stopped after %d seconds.", timeout_secs)
        else:
            log.info("[SERVING] Running indefinitely. Ctrl-C to stop.")
            query.awaitTermination()

    finally:
        if standalone:
            spark.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    )
    run()
