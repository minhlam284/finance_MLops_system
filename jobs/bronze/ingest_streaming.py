"""
jobs/bronze/ingest_streaming.py
--------------------------------
Bronze layer – Streaming Events Ingestion Job (micro-batch).

Reads the JSONL events file produced by `data_generator`
(`output/streaming/events_stream.jsonl`) and appends it as a
Delta table into the Bronze zone.

This micro-batch approach is scheduled by Airflow on a short interval
(e.g. every 5 minutes) to satisfy the SLA of Feature freshness <= 60 min.

Schema (from streaming_generator.py):
  event_id, event_type, event_timestamp, created_ts,
  account_id, session_id, device_type, location_ip,
  transaction_id (nullable), amount (nullable),
  _burst_minute, _is_burst, _late_arrival
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, TimestampType, BooleanType, DoubleType, IntegerType,
)

from jobs.utils.spark import get_spark

log = logging.getLogger(__name__)

# ── Path configuration ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STREAMING_INPUT_PATH = os.path.join(BASE_DIR, "output", "streaming", "events_stream.jsonl")
BRONZE_EVENTS_PATH   = os.path.join(BASE_DIR, "data", "bronze", "events")

# ── Explicit schema avoids full-scan schema inference on large JSONL ──────────
EVENTS_SCHEMA = StructType([
    StructField("event_id",         StringType(),    nullable=False),
    StructField("event_type",       StringType(),    nullable=True),
    StructField("event_timestamp",  StringType(),    nullable=True),  # cast below
    StructField("created_ts",       StringType(),    nullable=True),
    StructField("account_id",       StringType(),    nullable=True),
    StructField("session_id",       StringType(),    nullable=True),
    StructField("device_type",      StringType(),    nullable=True),
    StructField("location_ip",      StringType(),    nullable=True),
    StructField("transaction_id",   StringType(),    nullable=True),
    StructField("amount",           DoubleType(),    nullable=True),
    StructField("_burst_minute",    IntegerType(),   nullable=True),
    StructField("_is_burst",        BooleanType(),   nullable=True),
    StructField("_late_arrival",    BooleanType(),   nullable=True),
])


def _cast_timestamps(df: DataFrame) -> DataFrame:
    """Cast ISO-string timestamp columns to proper TimestampType."""
    return (
        df
        .withColumn("event_timestamp", F.to_timestamp("event_timestamp"))
        .withColumn("created_ts",      F.to_timestamp("created_ts"))
    )


def _add_metadata(df: DataFrame) -> DataFrame:
    """Add Bronze-layer ingestion metadata."""
    now_ts = F.lit(datetime.now(tz=timezone.utc).isoformat()).cast("timestamp")
    return (
        df
        .withColumn("_ingestion_ts", now_ts)
        .withColumn("_source_file",  F.lit("events_stream.jsonl"))
        .withColumn("_layer",        F.lit("bronze"))
    )


def run(spark: SparkSession | None = None) -> str:
    """
    Entry point – ingest streaming events JSON into Bronze Delta table.

    Returns the output Delta path.
    """
    standalone = spark is None
    if standalone:
        spark = get_spark("bronze_streaming_ingest")

    try:
        if not os.path.exists(STREAMING_INPUT_PATH):
            raise FileNotFoundError(
                f"Events JSONL not found: {STREAMING_INPUT_PATH}\n"
                "Run `python main.py` (data_generator) first."
            )

        log.info("[BRONZE] Reading events stream from %s …", STREAMING_INPUT_PATH)

        df = (
            spark.read
            .schema(EVENTS_SCHEMA)
            .json(STREAMING_INPUT_PATH)
        )

        df = _cast_timestamps(df)
        df = _add_metadata(df)

        row_count = df.count()
        log.info("[BRONZE] Writing %d events → %s", row_count, BRONZE_EVENTS_PATH)

        (
            df.write
            .format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .partitionBy("event_type")       # partition for efficient Silver reads
            .save(BRONZE_EVENTS_PATH)
        )

        log.info("[BRONZE] Streaming ingestion complete.")
        return BRONZE_EVENTS_PATH

    finally:
        if standalone:
            spark.stop()


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
    path = run()
    print(f"  ✓ events → {path}")
