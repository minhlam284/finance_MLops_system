"""
jobs/ml/evaluate_and_register_job.py
--------------------------------------
ML Evaluation & Model Registry Pipeline.

Loads the latest MLflow run from the `fraud_detection` experiment,
evaluates its PR-AUC against the SLA gate, and – if it passes –
promotes the corresponding registered model version to the
`Production` stage in the MLflow Model Registry.

Logic:
  1. Query the MLflow experiment for the latest completed run.
  2. Check val_pr_auc >= MIN_PRAUC_THRESHOLD.
  3. If passed → transition model version to Production.
  4. If any older version was Production → archive it to Archived.
  5. Write a short evaluation report to `output/ml_eval_report.json`.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from jobs.ml.constants import EXPERIMENT_NAME, MIN_PRAUC_THRESHOLD, MODEL_NAME

log = logging.getLogger(__name__)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REPORT_PATH = os.path.join(OUTPUT_DIR, "ml_eval_report.json")

LATENCY_SLA_MS      = 200   # p95 online-inference SLA (documented; not measured here)


# ── MLflow helpers ─────────────────────────────────────────────────────────────

def _get_client() -> MlflowClient:
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    return MlflowClient()


def _latest_run(client: MlflowClient) -> Any:
    """Return the most recent *FINISHED* run in the experiment."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found. Run train_job.py first."
        )
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No finished runs found. Run train_job.py first.")
    return runs[0]


def _latest_model_version(client: MlflowClient, run_id: str) -> Any:
    """Return the model version registered from a specific run_id."""
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for v in versions:
        if v.run_id == run_id:
            return v
    return None


# ── Promotion logic ────────────────────────────────────────────────────────────

def _promote_to_production(client: MlflowClient, version: str) -> None:
    """Archive any existing Production model, then promote `version`."""
    # Archive existing Production versions
    existing = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    for old in existing:
        if old.version != version:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=old.version,
                stage="Archived",
                archive_existing_versions=False,
            )
            log.info(
                "[EVAL] Archived old Production version %s (run: %s)",
                old.version, old.run_id,
            )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    log.info("[EVAL] Model version %s → Production", version)


# ── Report ─────────────────────────────────────────────────────────────────────

def _write_report(run_id: str, metrics: dict, passed: bool, promoted: bool) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = {
        "evaluated_at":         datetime.now(tz=timezone.utc).isoformat(),
        "run_id":               run_id,
        "experiment":           EXPERIMENT_NAME,
        "model_name":           MODEL_NAME,
        "metrics":              metrics,
        "prauc_threshold":      MIN_PRAUC_THRESHOLD,
        "latency_sla_ms":       LATENCY_SLA_MS,
        "passed_threshold":     passed,
        "promoted_to_production": promoted,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log.info("[EVAL] Report written → %s", REPORT_PATH)


# ── Entry point ────────────────────────────────────────────────────────────────

def run() -> dict:
    """
    Evaluate the latest training run and conditionally register it.

    Returns
    -------
    dict with keys: run_id, metrics, passed_threshold, promoted_to_production
    """
    client = _get_client()
    run    = _latest_run(client)
    run_id = run.info.run_id
    metrics = {k: v for k, v in run.data.metrics.items()}

    pr_auc  = metrics.get("val_pr_auc", 0.0)
    passed  = pr_auc >= MIN_PRAUC_THRESHOLD
    promoted = False
    model_version = None

    log.info(
        "[EVAL] Latest run: %s | val_pr_auc=%.4f | gate (%s): %s",
        run_id, pr_auc, f">={MIN_PRAUC_THRESHOLD}", "PASS" if passed else "FAIL",
    )

    if passed:
        model_version_obj = _latest_model_version(client, run_id)
        if model_version_obj is None:
            log.warning(
                "[EVAL] No registered model version found for run %s. "
                "Model logging may have failed in train_job.py.", run_id,
            )
        else:
            _promote_to_production(client, model_version_obj.version)
            promoted = True
            model_version = str(model_version_obj.version)
    else:
        log.warning(
            "[EVAL] Model did NOT pass the PR-AUC gate (%.4f < %.2f). "
            "Production model remains unchanged.",
            pr_auc, MIN_PRAUC_THRESHOLD,
        )

    _write_report(run_id, metrics, passed, promoted)

    return {
        "run_id":                  run_id,
        "metrics":                 metrics,
        "passed_threshold":        passed,
        "promoted_to_production":  promoted,
        "model_version":           model_version,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    )
    result = run()
    status = "✓ PROMOTED to Production" if result["promoted_to_production"] else "✗ NOT promoted"
    print(f"\n  run_id  : {result['run_id']}")
    print(f"  PR-AUC  : {result['metrics'].get('val_pr_auc', 'N/A'):.4f}")
    print(f"  Status  : {status}")
    print(f"  Report  : {REPORT_PATH}")
