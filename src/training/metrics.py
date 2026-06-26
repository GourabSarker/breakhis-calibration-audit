"""Training and evaluation metrics for binary classification."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def _as_numpy_1d(values) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")
    return array


def _validate_inputs(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y_true_array = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob_array = _as_numpy_1d(y_prob)
    if y_true_array.shape[0] != y_prob_array.shape[0]:
        raise ValueError("y_true and y_prob must have the same length.")
    return y_true_array, y_prob_array


def _predictions_from_probabilities(y_prob: np.ndarray) -> np.ndarray:
    return (y_prob >= 0.5).astype(int)


def accuracy(y_true, y_prob) -> float:
    y_true_array, y_prob_array = _validate_inputs(y_true, y_prob)
    return float(accuracy_score(y_true_array, _predictions_from_probabilities(y_prob_array)))


def balanced_accuracy(y_true, y_prob) -> float:
    y_true_array, y_prob_array = _validate_inputs(y_true, y_prob)
    return float(
        balanced_accuracy_score(y_true_array, _predictions_from_probabilities(y_prob_array))
    )


def sensitivity(y_true, y_prob) -> float:
    y_true_array, y_prob_array = _validate_inputs(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        _predictions_from_probabilities(y_prob_array),
        labels=[0, 1],
    ).ravel()
    denominator = tp + fn
    return float(tp / denominator) if denominator else math.nan


def specificity(y_true, y_prob) -> float:
    y_true_array, y_prob_array = _validate_inputs(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        _predictions_from_probabilities(y_prob_array),
        labels=[0, 1],
    ).ravel()
    denominator = tn + fp
    return float(tn / denominator) if denominator else math.nan


def f1(y_true, y_prob) -> float:
    y_true_array, y_prob_array = _validate_inputs(y_true, y_prob)
    return float(f1_score(y_true_array, _predictions_from_probabilities(y_prob_array), zero_division=0))


def mcc(y_true, y_prob) -> float:
    y_true_array, y_prob_array = _validate_inputs(y_true, y_prob)
    return float(matthews_corrcoef(y_true_array, _predictions_from_probabilities(y_prob_array)))


def roc_auc(y_true, y_prob) -> float:
    y_true_array, y_prob_array = _validate_inputs(y_true, y_prob)
    if np.unique(y_true_array).size < 2:
        return math.nan
    return float(roc_auc_score(y_true_array, y_prob_array))


def pr_auc(y_true, y_prob) -> float:
    y_true_array, y_prob_array = _validate_inputs(y_true, y_prob)
    if np.unique(y_true_array).size < 2:
        return math.nan
    return float(average_precision_score(y_true_array, y_prob_array))


def compute_metric_dict(y_true, y_prob) -> dict[str, float]:
    """Compute the full metric bundle expected by the training loop."""
    metrics: dict[str, Callable] = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
    return {name: float(fn(y_true, y_prob)) for name, fn in metrics.items()}
