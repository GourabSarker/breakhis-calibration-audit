"""Calibration metrics for binary BreakHis classifiers."""

from __future__ import annotations

import numpy as np


def _positive_probabilities(probs) -> np.ndarray:
    array = np.asarray(probs, dtype=np.float64)
    if array.ndim == 2:
        if array.shape[1] != 2:
            raise ValueError("Two-dimensional probabilities must have shape (n, 2).")
        array = array[:, 1]
    array = array.reshape(-1)
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("Probabilities must be in [0, 1].")
    return array


def _labels(labels) -> np.ndarray:
    array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if np.any((array != 0) & (array != 1)):
        raise ValueError("Binary labels must be encoded as 0/1.")
    return array


def _validate_probs_labels(probs, labels) -> tuple[np.ndarray, np.ndarray]:
    probs_array = _positive_probabilities(probs)
    labels_array = _labels(labels)
    if probs_array.shape[0] != labels_array.shape[0]:
        raise ValueError("probs and labels must have the same length.")
    if probs_array.shape[0] == 0:
        raise ValueError("probs and labels must not be empty.")
    return probs_array, labels_array


def _bin_indices(values: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    indices = np.floor(values * n_bins).astype(int)
    return np.clip(indices, 0, n_bins - 1)


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """Equal-width ECE using max-confidence bin assignment."""
    probs_array, labels_array = _validate_probs_labels(probs, labels)
    confidences = np.maximum(probs_array, 1.0 - probs_array)
    predictions = (probs_array >= 0.5).astype(np.int64)
    correct = (predictions == labels_array).astype(np.float64)
    bin_ids = _bin_indices(confidences, n_bins)

    ece = 0.0
    n = len(labels_array)
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def maximum_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """Maximum equal-width calibration gap using max-confidence bins."""
    probs_array, labels_array = _validate_probs_labels(probs, labels)
    confidences = np.maximum(probs_array, 1.0 - probs_array)
    predictions = (probs_array >= 0.5).astype(np.int64)
    correct = (predictions == labels_array).astype(np.float64)
    bin_ids = _bin_indices(confidences, n_bins)

    gaps: list[float] = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if np.any(mask):
            gaps.append(float(abs(correct[mask].mean() - confidences[mask].mean())))
    return float(max(gaps, default=0.0))


def brier_score(probs, labels) -> float:
    """Binary Brier score for positive-class probabilities."""
    probs_array, labels_array = _validate_probs_labels(probs, labels)
    return float(np.mean((probs_array - labels_array) ** 2))


def brier_decomposition_direct(probs, labels, n_bins: int = 15) -> dict[str, float]:
    """Return the binned direct Murphy decomposition components.

    This is the Murphy (1973) reliability-resolution-uncertainty decomposition
    applied to equal-width probability bins: REL and RES are computed from
    per-bin forecast means and observed event rates, while UNC is the marginal
    event uncertainty pbar * (1 - pbar). As discussed by Bröcker (2009), this
    direct binned estimate is exact for discrete forecasters whose forecast
    values define the bins; continuous forecasts can retain a within-bin
    residual.
    """
    probs_array, labels_array = _validate_probs_labels(probs, labels)
    bin_ids = _bin_indices(probs_array, n_bins)
    n = len(labels_array)
    base_rate = float(labels_array.mean())

    reliability = 0.0
    resolution = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        mean_forecast = float(probs_array[mask].mean())
        empirical_rate = float(labels_array[mask].mean())
        reliability += (mask.sum() / n) * (mean_forecast - empirical_rate) ** 2
        resolution += (mask.sum() / n) * (empirical_rate - base_rate) ** 2

    uncertainty = base_rate * (1.0 - base_rate)
    brier = brier_score(probs_array, labels_array)

    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier": float(brier),
    }


def brier_decomposition(probs, labels, n_bins: int = 15) -> dict[str, float]:
    """Return the identity-closed binned Brier decomposition used in CSV outputs.

    UNC uses the direct Murphy (1973) marginal uncertainty pbar * (1 - pbar),
    and RES is the binned Murphy resolution term. REL is then defined by
    closure, REL = BS + RES - UNC, so BS = REL - RES + UNC holds exactly in
    finite samples. This identity-preserving convention is convenient for
    tabular analysis of continuous CNN probabilities, where the fully direct
    binned Murphy decomposition can otherwise leave a within-bin residual; see
    Bröcker (2009) for that distinction. Use brier_decomposition_direct() when
    the direct binned Murphy REL term is desired explicitly.
    """
    direct = brier_decomposition_direct(probs, labels, n_bins=n_bins)
    reliability = direct["brier"] + direct["resolution"] - direct["uncertainty"]
    return {
        "reliability": float(reliability),
        "resolution": direct["resolution"],
        "uncertainty": direct["uncertainty"],
        "brier": direct["brier"],
    }


def negative_log_likelihood(probs, labels) -> float:
    """Binary negative log-likelihood for positive-class probabilities."""
    probs_array, labels_array = _validate_probs_labels(probs, labels)
    eps = np.finfo(np.float64).eps
    clipped = np.clip(probs_array, eps, 1.0 - eps)
    losses = -(labels_array * np.log(clipped) + (1 - labels_array) * np.log(1.0 - clipped))
    return float(np.mean(losses))
