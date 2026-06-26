"""Reliability-curve data helpers."""

from __future__ import annotations

import numpy as np

from src.calibration.metrics import _bin_indices, _validate_probs_labels


def reliability_curve(probs, labels, n_bins: int = 15) -> dict[str, np.ndarray]:
    """Return per-bin confidence, empirical positive rate, and count arrays."""
    probs_array, labels_array = _validate_probs_labels(probs, labels)
    bin_ids = _bin_indices(probs_array, n_bins)

    mean_confidence = np.full(n_bins, np.nan, dtype=np.float64)
    empirical_accuracy = np.full(n_bins, np.nan, dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.int64)

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count[bin_id] = int(mask.sum())
        if count[bin_id] == 0:
            continue
        mean_confidence[bin_id] = float(probs_array[mask].mean())
        empirical_accuracy[bin_id] = float(labels_array[mask].mean())

    return {
        "mean_confidence": mean_confidence,
        "empirical_accuracy": empirical_accuracy,
        "count": count,
    }
