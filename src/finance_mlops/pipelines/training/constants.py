"""
Shared ML constants for training and serving.
"""

from __future__ import annotations

MODEL_NAME = "fraud_detection_model"
EXPERIMENT_NAME = "fraud_detection"

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

LABEL_COL = "label"
TIME_COL = "event_timestamp"

MIN_PRAUC_THRESHOLD = 0.30
BLOCK_THRESHOLD = 0.60
