"""
jobs/ml/train_job.py
---------------------
ML Training Pipeline – Fraud Detection (XGBoost).

Reads the pre-built `ml_fraud_detection_training` Gold table, performs a
time-based train/validation split, trains an XGBoost classifier with
class-imbalance compensation, and logs the model + metrics to MLflow.

Feature columns (from feat_account_unified):
  f_account_total_tx_90d, f_account_avg_tx_value_90d,
  f_account_max_tx_value_90d, f_account_declined_ratio_90d,
  f_account_foreign_tx_ratio_90d,
  f_stream_tx_velocity_60m, f_stream_unique_devices_60m,
  f_stream_login_failures_30m

Label column : label (0 = legit, 1 = fraud)
Metrics reported : Precision, Recall, PR-AUC  (imbalanced-class focus)
SLA              : Online inference latency p95 < 200 ms
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from jobs.ml.constants import (
    EXPERIMENT_NAME,
    FEATURE_COLS,
    LABEL_COL,
    MIN_PRAUC_THRESHOLD,
    MODEL_NAME,
    TIME_COL,
)
from jobs.utils.spark import get_spark

log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

# Validation set: last N% of rows sorted by time
VALIDATION_RATIO = 0.20

# XGBoost hyper-parameters (baseline)
XGBOOST_PARAMS: dict = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":      "aucpr",
    "random_state":     42,
    "n_jobs":           -1,
    "tree_method":      "hist",
}

# ── Data loading ───────────────────────────────────────────────────────────────

def _load_training_data(spark) -> pd.DataFrame:
    """Load ml_fraud_detection_training from Gold Delta table → Pandas."""
    path = os.path.join(GOLD_DIR, "ml_fraud_detection_training")
    log.info("[TRAIN] Reading training data from %s …", path)
    df = spark.read.format("delta").load(path)
    log.info("[TRAIN] Total rows: %d", df.count())
    return df.select([TIME_COL, LABEL_COL] + FEATURE_COLS).toPandas()


# ── Feature engineering ────────────────────────────────────────────────────────

def _prepare_features(df: pd.DataFrame):
    """Sort by time, impute nulls with 0, return X_train, X_val, y_train, y_val."""
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0).astype("float32")

    split_idx = int(len(df) * (1 - VALIDATION_RATIO))
    train_df  = df.iloc[:split_idx]
    val_df    = df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[LABEL_COL].values.astype(int)
    X_val   = val_df[FEATURE_COLS].values
    y_val   = val_df[LABEL_COL].values.astype(int)

    fraud_ratio = y_train.sum() / max(len(y_train), 1)
    log.info(
        "[TRAIN] Train: %d rows (%.2f%% fraud) | Val: %d rows",
        len(y_train), 100 * fraud_ratio, len(y_val),
    )
    return X_train, X_val, y_train, y_val


# ── Training ───────────────────────────────────────────────────────────────────

def _compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compensate class imbalance: negatives / positives."""
    n_pos = max(y_train.sum(), 1)
    n_neg = len(y_train) - n_pos
    spw   = n_neg / n_pos
    log.info("[TRAIN] scale_pos_weight = %.2f (neg=%d, pos=%d)", spw, n_neg, n_pos)
    return float(spw)


def _train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> tuple[XGBClassifier, dict]:
    """Train XGBoost and return (model, metrics_dict)."""
    spw    = _compute_scale_pos_weight(y_train)
    params = {**XGBOOST_PARAMS, "scale_pos_weight": spw}

    model  = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "val_precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "val_recall":    float(recall_score(y_val, y_pred, zero_division=0)),
        "val_pr_auc":    float(average_precision_score(y_val, y_prob)),
        "val_rows":      int(len(y_val)),
        "train_rows":    int(len(y_train)),
        "fraud_ratio":   float(y_train.sum() / max(len(y_train), 1)),
    }
    log.info(
        "[TRAIN] Metrics — PR-AUC=%.4f | Precision=%.4f | Recall=%.4f",
        metrics["val_pr_auc"], metrics["val_precision"], metrics["val_recall"],
    )
    return model, metrics


# ── MLflow logging ─────────────────────────────────────────────────────────────

def _log_to_mlflow(model: XGBClassifier, metrics: dict, params: dict) -> str:
    """Log model + metrics to MLflow; return the run_id."""
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"xgboost_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log feature column list as artifact
        feat_path = "/tmp/feature_cols.txt"
        with open(feat_path, "w") as f:
            f.write("\n".join(FEATURE_COLS))
        mlflow.log_artifact(feat_path, artifact_path="metadata")

        # Log model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=None,
        )
        run_id = run.info.run_id
        log.info("[TRAIN] MLflow run logged: %s", run_id)
        return run_id


# ── Entry point ────────────────────────────────────────────────────────────────

def run(spark=None) -> dict:
    """
    Full training pipeline.

    Returns
    -------
    dict with keys: run_id, metrics, passed_threshold
    """
    standalone = spark is None
    if standalone:
        spark = get_spark("ml_train_job")

    try:
        df_pd = _load_training_data(spark)

        if df_pd.empty:
            raise RuntimeError(
                "Training data is empty. Run the full lakehouse pipeline first "
                "(bronze → silver → gold → features → build_ml_tables)."
            )

        X_train, X_val, y_train, y_val = _prepare_features(df_pd)
        model, metrics = _train_xgboost(X_train, y_train, X_val, y_val)

        # Determine if model passes gate
        passed = metrics["val_pr_auc"] >= MIN_PRAUC_THRESHOLD
        log.info(
            "[TRAIN] Threshold gate (PR-AUC >= %.2f): %s",
            MIN_PRAUC_THRESHOLD, "PASS" if passed else "FAIL",
        )

        # Build combined params dict for MLflow
        params_to_log = {
            "model_type":        "XGBoostClassifier",
            "validation_ratio":  VALIDATION_RATIO,
            "min_prauc_threshold": MIN_PRAUC_THRESHOLD,
            **{k: v for k, v in XGBOOST_PARAMS.items() if k != "use_label_encoder"},
        }

        run_id = _log_to_mlflow(model, metrics, params_to_log)

        return {
            "run_id":           run_id,
            "metrics":          metrics,
            "passed_threshold": passed,
        }

    finally:
        if standalone:
            spark.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
    )
    result = run()
    print(f"\n  ✓ run_id          : {result['run_id']}")
    print(f"  ✓ PR-AUC          : {result['metrics']['val_pr_auc']:.4f}")
    print(f"  ✓ Precision       : {result['metrics']['val_precision']:.4f}")
    print(f"  ✓ Recall          : {result['metrics']['val_recall']:.4f}")
    print(f"  ✓ Passed threshold: {result['passed_threshold']}")
