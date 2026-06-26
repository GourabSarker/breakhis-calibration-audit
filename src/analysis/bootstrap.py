"""Cluster bootstrap utilities for patient-level uncertainty."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def cluster_bootstrap(
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    logits,
    labels,
    patient_ids,
    B: int = 1000,
    seed: int | None = None,
) -> tuple[float, float, float, int]:
    """Bootstrap a statistic by resampling patients with replacement.

    ``logits`` may be raw logits, probabilities, or any first-dimension-aligned
    array consumed by ``statistic_fn``. Each bootstrap replicate samples patient
    IDs, reconstructs the image-level sample by concatenating all images for
    each sampled patient, and evaluates ``statistic_fn`` on that reconstructed
    image-level sample.
    """
    if B < 1:
        raise ValueError("B must be at least 1.")

    values = np.asarray(logits)
    labels_array = np.asarray(labels)
    patient_array = np.asarray(patient_ids).astype(str)

    if values.shape[0] != labels_array.shape[0] or values.shape[0] != patient_array.shape[0]:
        raise ValueError("logits, labels, and patient_ids must have matching first dimensions.")
    if values.shape[0] == 0:
        raise ValueError("Inputs must not be empty.")

    point_estimate = float(statistic_fn(values, labels_array))
    unique_patients = np.unique(patient_array)
    patient_indices = [np.flatnonzero(patient_array == patient_id) for patient_id in unique_patients]

    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(B, dtype=np.float64)
    for replicate_idx in range(B):
        sampled_patient_positions = rng.integers(0, len(unique_patients), size=len(unique_patients))
        sampled_indices = np.concatenate([patient_indices[position] for position in sampled_patient_positions])
        bootstrap_values[replicate_idx] = statistic_fn(values[sampled_indices], labels_array[sampled_indices])

    finite_values = bootstrap_values[np.isfinite(bootstrap_values)]
    n_finite = int(finite_values.size)
    if finite_values.size == 0:
        return point_estimate, np.nan, np.nan, n_finite

    lo, hi = np.percentile(finite_values, [2.5, 97.5])
    return point_estimate, float(lo), float(hi), n_finite
