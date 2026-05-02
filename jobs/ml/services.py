"""
jobs/ml/services.py
--------------------
Low-Level ML Design – 5 Core Service Classes (OOP).

Implements the class contracts defined in Section 4 of 04.1_ml_design_finance.md:
  - TrainingDataService   : data loading & validation
  - SplitService          : time-based train/val split
  - ModelService          : XGBoost training, evaluation, MLflow logging
  - ScoringService        : online / stream / batch scoring
  - MonitoringService     : metric publishing & drift alerts
"""
from __future__ import annotations

import json
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
    MIN_PRAUC_THRESHOLD as GLOBAL_MIN_PRAUC_THRESHOLD,
    MODEL_NAME,
    TIME_COL,
)

log = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD_DIR   = os.path.join(BASE_DIR, "data", "gold")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ─────────────────────────────────────────────────────────────────────────────
# 1. TrainingDataService
# ─────────────────────────────────────────────────────────────────────────────

class TrainingDataService:
    """Load and validate the training dataset from the Gold Delta table."""

    REQUIRED_COLS = [TIME_COL, LABEL_COL] + FEATURE_COLS

    def __init__(self, spark, table_path: str | None = None):
        self.spark = spark
        self.table_path = table_path or os.path.join(
            GOLD_DIR, "ml_fraud_detection_training"
        )

    def read_training_table(self) -> pd.DataFrame:
        """Read ml_fraud_detection_training Delta table → Pandas DataFrame."""
        log.info("[TrainingDataService] Reading from %s", self.table_path)
        df = self.spark.read.format("delta").load(self.table_path)
        log.info("[TrainingDataService] Total rows: %d", df.count())
        return df.select(self.REQUIRED_COLS).toPandas()

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Raise ValueError if any required column is missing."""
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"[TrainingDataService] Missing columns: {missing}"
            )
        if df.empty:
            raise RuntimeError(
                "[TrainingDataService] Training table is empty. "
                "Run the full lakehouse pipeline first."
            )
        log.info("[TrainingDataService] Schema validation passed.")

    def dedup_by_created_ts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep the latest record per (account_id, event_timestamp) if duplicates exist."""
        if "created_ts" in df.columns:
            before = len(df)
            df = (
                df.sort_values("created_ts", ascending=False)
                .drop_duplicates(subset=["account_id", TIME_COL], keep="first")
                .reset_index(drop=True)
            )
            log.info(
                "[TrainingDataService] Dedup: %d → %d rows", before, len(df)
            )
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. SplitService
# ─────────────────────────────────────────────────────────────────────────────

class SplitService:
    """Time-based train / validation split to avoid future data leakage."""

    def __init__(self, validation_ratio: float = 0.20, test_ratio: float = 0.0):
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

    def get_split_boundaries(self, df: pd.DataFrame) -> dict:
        """Return the cutoff timestamps for train/val/test boundaries."""
        df_sorted = df.sort_values(TIME_COL)
        n = len(df_sorted)
        train_end_idx = int(n * (1 - self.validation_ratio - self.test_ratio))
        val_end_idx   = int(n * (1 - self.test_ratio))

        boundaries = {
            "train_end_ts": str(df_sorted.iloc[train_end_idx - 1][TIME_COL]),
            "val_end_ts":   str(df_sorted.iloc[val_end_idx - 1][TIME_COL]),
            "total_rows":   n,
            "train_rows":   train_end_idx,
            "val_rows":     val_end_idx - train_end_idx,
            "test_rows":    n - val_end_idx,
        }
        log.info("[SplitService] Boundaries: %s", boundaries)
        return boundaries

    def split_by_time(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Sort by time and split into (train, val, test).
        test is empty when test_ratio == 0.
        """
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        n = len(df)
        train_end = int(n * (1 - self.validation_ratio - self.test_ratio))
        val_end   = int(n * (1 - self.test_ratio))

        train = df.iloc[:train_end]
        val   = df.iloc[train_end:val_end]
        test  = df.iloc[val_end:]

        log.info(
            "[SplitService] Split — train=%d | val=%d | test=%d",
            len(train), len(val), len(test),
        )
        return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 3. ModelService
# ─────────────────────────────────────────────────────────────────────────────

class ModelArtifact:
    """Lightweight wrapper around a trained XGBClassifier + metadata."""

    def __init__(self, model: XGBClassifier, params: dict, run_id: str = ""):
        self.model     = model
        self.params    = params
        self.run_id    = run_id
        self.version   = ""


class ModelService:
    """Train XGBoost, evaluate on validation set, and log/register via MLflow."""

    DEFAULT_PARAMS: dict = {
        "n_estimators":      300,
        "max_depth":         6,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "eval_metric":       "aucpr",
        "random_state":      42,
        "n_jobs":            -1,
        "tree_method":       "hist",
    }
    MIN_PRAUC_THRESHOLD = GLOBAL_MIN_PRAUC_THRESHOLD

    def __init__(self, params: dict | None = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
        mlflow.set_experiment(EXPERIMENT_NAME)

    # ── helpers ────────────────────────────────────────────────────────────

    def _compute_scale_pos_weight(self, y: np.ndarray) -> float:
        n_pos = max(int(y.sum()), 1)
        n_neg = len(y) - n_pos
        spw   = n_neg / n_pos
        log.info("[ModelService] scale_pos_weight=%.2f (neg=%d, pos=%d)", spw, n_neg, n_pos)
        return float(spw)

    def _arrays_from_df(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        X = df[FEATURE_COLS].fillna(0.0).astype("float32").values
        y = df[LABEL_COL].values.astype(int)
        return X, y

    # ── public interface ───────────────────────────────────────────────────

    def train(self, train_df: pd.DataFrame) -> ModelArtifact:
        """Fit XGBoost on training DataFrame; returns ModelArtifact."""
        X_train, y_train = self._arrays_from_df(train_df)
        spw  = self._compute_scale_pos_weight(y_train)
        full_params = {**self.params, "scale_pos_weight": spw}

        model = XGBClassifier(**full_params)
        model.fit(X_train, y_train, verbose=False)

        artifact = ModelArtifact(model=model, params=full_params)
        log.info("[ModelService] Training complete.")
        return artifact

    def evaluate(
        self, artifact: ModelArtifact, val_df: pd.DataFrame
    ) -> dict:
        """Compute Precision / Recall / PR-AUC on validation set."""
        X_val, y_val = self._arrays_from_df(val_df)
        y_prob = artifact.model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "val_precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "val_recall":    float(recall_score(y_val, y_pred, zero_division=0)),
            "val_pr_auc":    float(average_precision_score(y_val, y_prob)),
            "val_rows":      int(len(y_val)),
        }
        log.info(
            "[ModelService] Eval — PR-AUC=%.4f | P=%.4f | R=%.4f",
            metrics["val_pr_auc"], metrics["val_precision"], metrics["val_recall"],
        )
        return metrics

    def save_model(
        self, artifact: ModelArtifact, metrics: dict, model_version: str = ""
    ) -> str:
        """Log model + metrics to MLflow; return run_id."""
        run_name = (
            model_version
            or f"xgboost_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(artifact.params)
            mlflow.log_metrics(metrics)

            feat_path = "/tmp/feature_cols.txt"
            with open(feat_path, "w") as f:
                f.write("\n".join(FEATURE_COLS))
            mlflow.log_artifact(feat_path, artifact_path="metadata")

            mlflow.xgboost.log_model(
                artifact.model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )
            artifact.run_id = run.info.run_id
            log.info("[ModelService] MLflow run logged: %s", artifact.run_id)
            return artifact.run_id


# ─────────────────────────────────────────────────────────────────────────────
# 4. ScoringService
# ─────────────────────────────────────────────────────────────────────────────

class ScoringService:
    """Online / stream / batch fraud scoring using a loaded XGBoost model."""

    BLOCK_THRESHOLD = 0.60

    def __init__(self, model_artifact: ModelArtifact | None = None):
        self._artifact = model_artifact
        self._model_version = model_artifact.version if model_artifact else "unknown"

    # ── model loader ───────────────────────────────────────────────────────

    @classmethod
    def from_production_registry(cls) -> "ScoringService":
        """Load the Production model from the local MLflow registry."""
        mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            raise RuntimeError(
                f"No Production model for '{MODEL_NAME}'. "
                "Run evaluate_and_register_job first."
            )
        mv        = versions[0]
        raw_model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/Production")
        artifact  = ModelArtifact(model=raw_model, params={}, run_id=mv.run_id)
        artifact.version = str(mv.version)
        log.info(
            "[ScoringService] Loaded '%s' version=%s", MODEL_NAME, mv.version
        )
        svc = cls(artifact)
        svc._model_version = str(mv.version)
        return svc

    # ── scoring modes ──────────────────────────────────────────────────────

    def score_online(self, request: dict) -> dict:
        """
        Score a single dict request (online API use-case).
        Returns: {fraud_score, is_blocked, model_version, scored_at}
        """
        row = [float(request.get(c, 0.0)) for c in FEATURE_COLS]
        X   = np.array(row, dtype="float32").reshape(1, -1)
        prob = float(self._artifact.model.predict_proba(X)[0, 1])
        return {
            "fraud_score":   round(prob, 6),
            "is_blocked":    prob >= self.BLOCK_THRESHOLD,
            "model_version": self._model_version,
            "scored_at":     datetime.now(tz=timezone.utc).isoformat(),
        }

    def score_stream(self, event: dict) -> dict:
        """Score a single streaming event dict (same contract as score_online)."""
        return self.score_online(event)

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a Pandas DataFrame of features.
        Adds columns: fraud_score, is_blocked, model_version, scored_at.
        """
        X     = df[FEATURE_COLS].fillna(0.0).astype("float32").values
        probs = self._artifact.model.predict_proba(X)[:, 1].astype(float)
        now   = datetime.now(tz=timezone.utc).isoformat()
        result = df.copy()
        result["fraud_score"]   = probs
        result["is_blocked"]    = probs >= self.BLOCK_THRESHOLD
        result["model_version"] = self._model_version
        result["scored_at"]     = now
        return result

    def write_scores(
        self, score_df: pd.DataFrame, output_path: str, model_version: str = ""
    ) -> None:
        """Persist scored DataFrame as Parquet (batch sink)."""
        os.makedirs(output_path, exist_ok=True)
        ts   = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        ver  = model_version or self._model_version
        path = os.path.join(output_path, f"scores_{ver}_{ts}.parquet")
        score_df.to_parquet(path, index=False)
        log.info("[ScoringService] Scores written → %s (%d rows)", path, len(score_df))


# ─────────────────────────────────────────────────────────────────────────────
# 5. MonitoringService
# ─────────────────────────────────────────────────────────────────────────────

class MonitoringService:
    """Publish model metrics, compute feature drift, and trigger alerts."""

    PSI_THRESHOLD   = 0.15   # Population Stability Index – retrain trigger
    PRAUC_MIN       = 0.30   # Minimum acceptable PR-AUC in production
    ALERT_LOG_PATH  = os.path.join(OUTPUT_DIR, "monitoring_alerts.jsonl")

    def __init__(self, alert_log_path: str | None = None):
        self.alert_log_path = alert_log_path or self.ALERT_LOG_PATH
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── helpers ────────────────────────────────────────────────────────────

    def _compute_psi(
        self, expected: np.ndarray, actual: np.ndarray, bins: int = 10
    ) -> float:
        """Population Stability Index between two distributions."""
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints[0]  = -np.inf
        breakpoints[-1] =  np.inf

        expected_pct = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_pct   = np.histogram(actual,   breakpoints)[0] / len(actual)

        expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
        actual_pct   = np.where(actual_pct   == 0, 1e-4, actual_pct)

        psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
        return psi

    def _write_alert(self, alert: dict) -> None:
        with open(self.alert_log_path, "a") as f:
            f.write(json.dumps(alert) + "\n")

    # ── public interface ───────────────────────────────────────────────────

    def publish_model_metrics(self, metrics: dict) -> None:
        """
        Log model performance metrics to MLflow active run (or standalone).
        metrics keys: val_pr_auc, val_precision, val_recall, …
        """
        try:
            mlflow.log_metrics(metrics)
            log.info("[MonitoringService] Model metrics published: %s", metrics)
        except Exception:
            log.warning(
                "[MonitoringService] No active MLflow run; metrics logged locally only."
            )
        report_path = os.path.join(OUTPUT_DIR, "model_metrics.json")
        with open(report_path, "w") as f:
            json.dump(
                {
                    **metrics,
                    "reported_at": datetime.now(tz=timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )

    def publish_drift_metrics(self, drift_stats: dict) -> None:
        """
        Store feature drift statistics.
        drift_stats: {feature_name: psi_value, …}
        """
        drift_path = os.path.join(OUTPUT_DIR, "drift_metrics.json")
        record = {
            "computed_at": datetime.now(tz=timezone.utc).isoformat(),
            "drift_stats": drift_stats,
        }
        with open(drift_path, "w") as f:
            json.dump(record, f, indent=2)
        log.info("[MonitoringService] Drift metrics saved → %s", drift_path)

    def compute_feature_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df:  pd.DataFrame,
        features:    list[str] | None = None,
    ) -> dict[str, float]:
        """Compute PSI for each feature column between baseline and current windows."""
        features = features or FEATURE_COLS
        drift_stats: dict[str, float] = {}
        for feat in features:
            if feat in baseline_df.columns and feat in current_df.columns:
                psi = self._compute_psi(
                    baseline_df[feat].fillna(0).values,
                    current_df[feat].fillna(0).values,
                )
                drift_stats[feat] = round(psi, 6)
        return drift_stats

    def trigger_alerts(self, metrics: dict) -> None:
        """
        Check thresholds and emit alerts to the alert log file.
        Checks:
          - val_pr_auc  < PRAUC_MIN   → model quality alert
          - any PSI     > PSI_THRESHOLD → drift alert
        """
        alerts: list[dict] = []
        ts = datetime.now(tz=timezone.utc).isoformat()

        pr_auc = metrics.get("val_pr_auc")
        if pr_auc is not None and pr_auc < self.PRAUC_MIN:
            alerts.append({
                "ts":    ts,
                "type":  "model_quality",
                "msg":   f"PR-AUC {pr_auc:.4f} < threshold {self.PRAUC_MIN}",
                "value": pr_auc,
            })

        for key, val in metrics.items():
            if key.startswith("psi_") and isinstance(val, float):
                if val > self.PSI_THRESHOLD:
                    alerts.append({
                        "ts":      ts,
                        "type":    "feature_drift",
                        "feature": key.replace("psi_", ""),
                        "msg":     f"PSI {val:.4f} > threshold {self.PSI_THRESHOLD}",
                        "value":   val,
                    })

        for alert in alerts:
            self._write_alert(alert)
            log.warning("[MonitoringService] ALERT: %s", alert["msg"])

        if not alerts:
            log.info("[MonitoringService] No alerts triggered.")
