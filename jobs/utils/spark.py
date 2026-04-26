"""
jobs/utils/spark.py
-------------------
Centralised SparkSession factory with Delta Lake support.

Usage:
    from jobs.utils.spark import get_spark
    spark = get_spark("MyJobName")
"""
from __future__ import annotations

import os
import logging
from typing import Optional

log = logging.getLogger(__name__)

# Delta Lake / Spark version alignment
# delta-spark 3.x  requires Spark 3.5.x
DELTA_VERSION = "3.2.0"
SCALA_VERSION = "2.12"


def get_spark(app_name: str = "FinanceLakehouse", master: str = "local[*]"):
    """
    Build and return a SparkSession pre-configured for Delta Lake.

    Parameters
    ----------
    app_name : str
        Human-readable application name shown in the Spark UI.
    master : str
        Spark master URL; default is local[*] for development.

    Returns
    -------
    pyspark.sql.SparkSession
    """
    from pyspark.sql import SparkSession
    from delta import configure_spark_with_delta_pip  # type: ignore

    # Base builder
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        # ── Delta Lake extensions ──────────────────────────────────
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # ── Performance & local-mode tweaks ───────────────────────
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.driver.memory", "2g")
        # ── Logging verbosity ─────────────────────────────────────
        .config("spark.eventLog.enabled", "false")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    log.info("SparkSession '%s' created (master=%s).", app_name, master)
    return spark
