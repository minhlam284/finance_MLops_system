from __future__ import annotations

import numpy as np
import pytest

from finance_mlops.pipelines.training.constants import BLOCK_THRESHOLD, FEATURE_COLS
from finance_mlops.pipelines.training.inference_contract import build_predictions, parse_instances_payload


def test_parse_instances_payload_supports_dict_and_feature_order() -> None:
    payload = {
        "instances": [
            {col: idx + 1 for idx, col in enumerate(FEATURE_COLS)},
            [10.0] * len(FEATURE_COLS),
        ]
    }

    matrix = parse_instances_payload(payload)

    assert matrix.shape == (2, len(FEATURE_COLS))
    assert matrix.dtype == np.float32
    assert matrix[0].tolist() == [float(i + 1) for i in range(len(FEATURE_COLS))]
    assert matrix[1].tolist() == [10.0] * len(FEATURE_COLS)


def test_parse_instances_payload_fills_missing_or_null_with_zero() -> None:
    payload = {"instances": [{"f_account_total_tx_90d": None}]}

    matrix = parse_instances_payload(payload)

    assert matrix.shape == (1, len(FEATURE_COLS))
    assert matrix[0, 0] == 0.0
    assert np.all(matrix[0, 1:] == 0.0)


def test_parse_instances_payload_rejects_invalid_length() -> None:
    payload = {"instances": [[1.0, 2.0]]}
    with pytest.raises(ValueError):
        parse_instances_payload(payload)


def test_build_predictions_applies_block_threshold() -> None:
    scores = np.asarray([BLOCK_THRESHOLD - 1e-6, BLOCK_THRESHOLD + 1e-6], dtype=float)

    output = build_predictions(scores, model_version="12")

    assert len(output) == 2
    assert output[0]["is_blocked"] is False
    assert output[1]["is_blocked"] is True
    assert all(item["model_version"] == "12" for item in output)
