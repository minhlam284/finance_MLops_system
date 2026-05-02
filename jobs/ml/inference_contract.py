"""
Helpers for request/response contract of fraud inference APIs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from jobs.ml.constants import BLOCK_THRESHOLD, FEATURE_COLS


def _to_float(value) -> float:
    if value is None:
        return 0.0
    return float(value)


def parse_instances_payload(payload: dict) -> np.ndarray:
    """
    Parse payload with key `instances` into a float32 matrix.

    Accepted instance forms:
    - list/tuple in exact FEATURE_COLS order
    - object keyed by FEATURE_COLS names
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    instances = payload.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("Payload must contain non-empty 'instances' list.")

    rows: list[list[float]] = []
    for idx, item in enumerate(instances):
        if isinstance(item, dict):
            row = [_to_float(item.get(col, 0.0)) for col in FEATURE_COLS]
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) != len(FEATURE_COLS):
                raise ValueError(
                    f"Instance at index {idx} must have {len(FEATURE_COLS)} values."
                )
            row = [_to_float(v) for v in item]
        else:
            raise ValueError(
                f"Instance at index {idx} must be object or list with features."
            )
        rows.append(row)

    return np.asarray(rows, dtype="float32")


def build_predictions(scores: np.ndarray, model_version: str) -> list[dict]:
    """Build response payload for a batch of fraud scores."""
    results: list[dict] = []
    for score in scores:
        value = float(score)
        results.append(
            {
                "fraud_score": value,
                "is_blocked": value >= BLOCK_THRESHOLD,
                "model_version": model_version,
            }
        )
    return results
