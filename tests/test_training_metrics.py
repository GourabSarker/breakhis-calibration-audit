from __future__ import annotations

import math

from src.training.metrics import (
    accuracy,
    balanced_accuracy,
    compute_metric_dict,
    f1,
    mcc,
    pr_auc,
    roc_auc,
    sensitivity,
    specificity,
)


def test_binary_metrics_are_correct_for_perfect_predictions() -> None:
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]

    assert accuracy(y_true, y_prob) == 1.0
    assert balanced_accuracy(y_true, y_prob) == 1.0
    assert sensitivity(y_true, y_prob) == 1.0
    assert specificity(y_true, y_prob) == 1.0
    assert f1(y_true, y_prob) == 1.0
    assert mcc(y_true, y_prob) == 1.0
    assert roc_auc(y_true, y_prob) == 1.0
    assert pr_auc(y_true, y_prob) == 1.0


def test_metric_dict_handles_single_class_auc_as_nan() -> None:
    metrics = compute_metric_dict([1, 1, 1], [0.6, 0.7, 0.9])
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["pr_auc"])
    assert metrics["sensitivity"] == 1.0
