"""
Export the current Production model from MLflow for online serving.

The job is designed for a local MLflow tracking backend (`file://mlruns`) and
creates a deterministic artifact bundle under:

  output/serving/<MODEL_NAME>/<version>/

Optional S3/MinIO sync is enabled via environment variables:
  - MODEL_ARTIFACT_BUCKET
  - MODEL_ARTIFACT_PREFIX (optional)
  - AWS_ENDPOINT_URL (optional, for MinIO)
  - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (optional)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import mlflow
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from jobs.ml.constants import BLOCK_THRESHOLD, FEATURE_COLS, MODEL_NAME

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MLRUNS_DIR = BASE_DIR / "mlruns"
OUTPUT_DIR = BASE_DIR / "output"
SERVING_ROOT = OUTPUT_DIR / "serving" / MODEL_NAME
EVAL_REPORT_PATH = OUTPUT_DIR / "ml_eval_report.json"


def _get_production_model(client: mlflow.tracking.MlflowClient) -> Any:
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(
            f"No Production model found for '{MODEL_NAME}'. "
            "Run evaluate_and_register_job.py first."
        )
    return versions[0]


def _load_eval_report() -> dict:
    if EVAL_REPORT_PATH.exists():
        return json.loads(EVAL_REPORT_PATH.read_text())
    return {}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _s3_client():
    endpoint = os.getenv("AWS_ENDPOINT_URL")
    kwargs = {
        "endpoint_url": endpoint,
        "config": Config(signature_version="s3v4"),
    }
    return boto3.client("s3", **kwargs)


def _sync_directory_to_s3(local_dir: Path, bucket: str, prefix: str) -> str:
    s3 = _s3_client()
    prefix = prefix.strip("/")
    base_prefix = f"{prefix}/{MODEL_NAME}/{local_dir.name}".strip("/")
    for file_path in local_dir.rglob("*"):
        if file_path.is_dir():
            continue
        rel = file_path.relative_to(local_dir).as_posix()
        key = f"{base_prefix}/{rel}".strip("/")
        s3.upload_file(str(file_path), bucket, key)
    return f"s3://{bucket}/{base_prefix}"


def run() -> dict:
    """
    Export Production model and metadata for KServe/custom online predictor.

    Returns
    -------
    dict with keys:
      model_name, model_version, run_id, local_export_dir, remote_uri
    """
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    client = mlflow.tracking.MlflowClient()
    production_mv = _get_production_model(client)

    model_version = str(production_mv.version)
    run_id = production_mv.run_id
    model_uri = f"models:/{MODEL_NAME}/Production"

    export_dir = SERVING_ROOT / model_version
    export_dir.mkdir(parents=True, exist_ok=True)

    model_target_dir = export_dir / "model"
    downloaded = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=str(model_target_dir),
    )
    log.info("[EXPORT] Downloaded model to %s", downloaded)

    report = _load_eval_report()
    metadata = {
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "run_id": run_id,
        "model_uri": model_uri,
        "block_threshold": BLOCK_THRESHOLD,
        "feature_count": len(FEATURE_COLS),
        "metrics": report.get("metrics", {}),
    }
    _write_json(export_dir / "feature_cols.json", {"feature_cols": FEATURE_COLS})
    _write_json(export_dir / "model_metadata.json", metadata)
    if report:
        _write_json(export_dir / "ml_eval_report.json", report)

    remote_uri = None
    bucket = os.getenv("MODEL_ARTIFACT_BUCKET")
    if bucket:
        prefix = os.getenv("MODEL_ARTIFACT_PREFIX", "").strip("/")
        try:
            remote_uri = _sync_directory_to_s3(export_dir, bucket, prefix)
            log.info("[EXPORT] Synced to %s", remote_uri)
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"Failed to sync artifacts to S3/MinIO: {exc}") from exc
    else:
        log.info("[EXPORT] MODEL_ARTIFACT_BUCKET is not set. Skip remote sync.")

    return {
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "run_id": run_id,
        "local_export_dir": str(export_dir),
        "remote_uri": remote_uri,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    result = run()
    print(json.dumps(result, indent=2))
