"""Threshold-based decision metrics."""

from __future__ import annotations

import numpy as np


def _validate_inputs(probs, labels) -> tuple[np.ndarray, np.ndarray]:
    probs_array = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if probs_array.shape[0] != labels_array.shape[0]:
        raise ValueError("probs and labels must have the same length.")
    if probs_array.shape[0] == 0:
        raise ValueError("probs and labels must not be empty.")
    if np.any((probs_array < 0.0) | (probs_array > 1.0)):
        raise ValueError("probs must be in [0, 1].")
    if np.any((labels_array != 0) & (labels_array != 1)):
        raise ValueError("labels must be binary 0/1 values.")
    return probs_array, labels_array


def bayes_optimal_threshold(c_fp: float, c_fn: float) -> float:
    """Return the Bayes threshold c_fp / (c_fp + c_fn)."""
    if c_fp < 0 or c_fn < 0:
        raise ValueError("Costs must be non-negative.")
    if c_fp + c_fn <= 0:
        raise ValueError("At least one cost must be positive.")
    return float(c_fp / (c_fp + c_fn))


def metrics_at_threshold(
    probs,
    labels,
    tau: float,
    *,
    c_fp: float = 1.0,
    c_fn: float = 1.0,
) -> dict[str, float]:
    """Compute threshold metrics and expected cost per sample."""
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1].")
    if c_fp < 0 or c_fn < 0:
        raise ValueError("Costs must be non-negative.")

    probs_array, labels_array = _validate_inputs(probs, labels)
    predictions = (probs_array >= tau).astype(np.int64)

    tp = int(np.sum((predictions == 1) & (labels_array == 1)))
    tn = int(np.sum((predictions == 0) & (labels_array == 0)))
    fp = int(np.sum((predictions == 1) & (labels_array == 0)))
    fn = int(np.sum((predictions == 0) & (labels_array == 1)))
    n = len(labels_array)

    return {
        "accuracy": float((tp + tn) / n),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "ppv": float(tp / (tp + fp)) if (tp + fp) else np.nan,
        "npv": float(tn / (tn + fn)) if (tn + fn) else np.nan,
        "expected_cost": float(((c_fp * fp) + (c_fn * fn)) / n),
    }
