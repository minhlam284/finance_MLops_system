"""
Custom fraud model predictor for KServe (container-based predictor).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import boto3
import mlflow.xgboost
from fastapi import FastAPI, HTTPException

from jobs.ml.constants import MODEL_NAME
from jobs.ml.inference_contract import build_predictions, parse_instances_payload

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SERVING_ROOT = BASE_DIR / "output" / "serving" / MODEL_NAME
DEFAULT_LOCAL_CACHE = Path("/tmp/fraud-model")

app = FastAPI(title="fraud-detection-predictor", version="1.0.0")


def _latest_version_dir(root: Path) -> Path:
    if not root.exists():
        raise RuntimeError(f"Serving root not found: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError(f"No model version directories under {root}")
    return sorted(candidates, key=lambda p: int(p.name))[-1]


def _download_s3_prefix(s3_uri: str, target_dir: Path) -> Path:
    # format: s3://bucket/prefix...
    _, _, remainder = s3_uri.partition("s3://")
    bucket, _, prefix = remainder.partition("/")
    if not bucket:
        raise RuntimeError(f"Invalid s3 uri: {s3_uri}")

    endpoint = os.getenv("AWS_ENDPOINT_URL")
    s3 = boto3.client("s3", endpoint_url=endpoint)
    paginator = s3.get_paginator("list_objects_v2")
    target_dir.mkdir(parents=True, exist_ok=True)

    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            found = True
            rel = key[len(prefix) :].lstrip("/")
            local_file = target_dir / rel
            local_file.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_file))
    if not found:
        raise RuntimeError(f"No artifacts found at {s3_uri}")
    return target_dir


def _resolve_export_dir() -> Path:
    model_dir = os.getenv("MODEL_DIR")
    if model_dir:
        return Path(model_dir)

    model_artifact_uri = os.getenv("MODEL_ARTIFACT_URI")
    if model_artifact_uri and model_artifact_uri.startswith("s3://"):
        return _download_s3_prefix(model_artifact_uri, DEFAULT_LOCAL_CACHE)

    model_version = os.getenv("MODEL_VERSION")
    if model_version:
        return DEFAULT_SERVING_ROOT / model_version
    return _latest_version_dir(DEFAULT_SERVING_ROOT)


def _resolve_model_path(export_dir: Path) -> Path:
    model_path = export_dir / "model"
    if (model_path / "MLmodel").exists():
        return model_path

    # Fallback for nested download layouts.
    mlmodels = list(model_path.rglob("MLmodel"))
    if mlmodels:
        return mlmodels[0].parent
    raise RuntimeError(f"Could not locate MLmodel under {model_path}")


class FraudPredictor:
    def __init__(self) -> None:
        self.export_dir = _resolve_export_dir()
        self.model_path = _resolve_model_path(self.export_dir)
        self.model = mlflow.xgboost.load_model(str(self.model_path))
        self.model_version = self._load_version()
        log.info(
            "[PREDICTOR] Loaded model=%s version=%s from %s",
            MODEL_NAME,
            self.model_version,
            self.model_path,
        )

    def _load_version(self) -> str:
        metadata_path = self.export_dir / "model_metadata.json"
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text())
            return str(payload.get("model_version", "unknown"))
        return "unknown"

    def predict(self, payload: dict) -> dict:
        matrix = parse_instances_payload(payload)
        scores = self.model.predict_proba(matrix)[:, 1]
        return {"predictions": build_predictions(scores, self.model_version)}


_predictor: FraudPredictor | None = None


def _get_predictor() -> FraudPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FraudPredictor()
    return _predictor


@app.get("/healthz")
def healthz() -> dict:
    _get_predictor()
    return {"status": "ok"}


@app.get("/v1/models/fraud-detection")
def model_metadata() -> dict:
    predictor = _get_predictor()
    return {
        "name": MODEL_NAME,
        "version": predictor.model_version,
    }


@app.post("/v1/models/fraud-detection:predict")
def predict(payload: dict) -> dict:
    try:
        return _get_predictor().predict(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive server behavior
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

