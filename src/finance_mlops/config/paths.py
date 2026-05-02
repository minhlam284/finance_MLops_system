"""Centralized filesystem paths for local development/runtime artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    env_root = os.getenv("FINANCE_MLOPS_HOME")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _project_root()
ARTIFACTS_ROOT = Path(os.getenv("FINANCE_MLOPS_ARTIFACTS", PROJECT_ROOT / "artifacts")).resolve()

SOURCE_DIR = ARTIFACTS_ROOT / "source"
SOURCE_OFFLINE_DIR = SOURCE_DIR / "offline"
SOURCE_STREAMING_DIR = SOURCE_DIR / "streaming"
SOURCE_STREAMING_FILE = SOURCE_STREAMING_DIR / "events_stream.jsonl"

LAKEHOUSE_DIR = ARTIFACTS_ROOT / "lakehouse"
LAKEHOUSE_BRONZE_DIR = LAKEHOUSE_DIR / "bronze"
LAKEHOUSE_SILVER_DIR = LAKEHOUSE_DIR / "silver"
LAKEHOUSE_GOLD_DIR = LAKEHOUSE_DIR / "gold"
LAKEHOUSE_CHECKPOINT_DIR = ARTIFACTS_ROOT / "checkpoints"

MLRUNS_DIR = ARTIFACTS_ROOT / "mlruns"
REPORTS_DIR = ARTIFACTS_ROOT / "reports"
SERVING_ROOT = ARTIFACTS_ROOT / "serving"
