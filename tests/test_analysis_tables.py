from __future__ import annotations

import numpy as np

from src.analysis.bootstrap import cluster_bootstrap
from src.analysis.tables import TABLE1_METRICS, architecture_cluster_bootstrap
from src.calibration.metrics import expected_calibration_error
from src.decision.net_benefit import net_benefit
from src.decision.thresholds import bayes_optimal_threshold


def test_architecture_ci_uses_architecture_bootstrap_distribution() -> None:
    runs = [
        (
            np.array([0.05, 0.10, 0.80, 0.90, 0.65, 0.70]),
            np.array([0, 0, 1, 1, 1, 1]),
            np.array(["a", "a", "b", "b", "c", "c"]),
        ),
        (
            np.array([0.20, 0.25, 0.40, 0.45, 0.85, 0.95]),
            np.array([0, 0, 0, 0, 1, 1]),
            np.array(["d", "d", "e", "e", "f", "f"]),
        ),
    ]
    B = 200
    seed = 77

    result = architecture_cluster_bootstrap(runs, B=B, seed=seed)["ECE"]

    rng = np.random.default_rng(seed)
    manual_distribution = []
    patient_universe = np.asarray(
        sorted({patient for _probs, _labels, patient_ids in runs for patient in patient_ids})
    )
    for _replicate in range(B):
        sampled_positions = rng.integers(
            0,
            len(patient_universe),
            size=len(patient_universe),
        )
        multiplicities = np.bincount(sampled_positions, minlength=len(patient_universe))
        cell_values = []
        for probs, labels, patient_ids in runs:
            indices = {
                patient: np.flatnonzero(patient_ids == patient)
                for patient in np.unique(patient_ids)
            }
            sampled_clusters = [
                indices[patient]
                for patient, multiplicity in zip(
                    patient_universe,
                    multiplicities,
                    strict=True,
                )
                for _repeat in range(int(multiplicity))
                if patient in indices
            ]
            if not sampled_clusters:
                continue
            sampled_indices = np.concatenate(sampled_clusters)
            cell_values.append(
                expected_calibration_error(probs[sampled_indices], labels[sampled_indices])
            )
        manual_distribution.append(np.mean(cell_values))
    expected_lo, expected_hi = np.percentile(manual_distribution, [2.5, 97.5])

    assert result[1] == expected_lo
    assert result[2] == expected_hi

    run_intervals = [
        cluster_bootstrap(
            expected_calibration_error,
            probs,
            labels,
            patient_ids,
            B=B,
            seed=seed + run_idx,
        )
        for run_idx, (probs, labels, patient_ids) in enumerate(runs)
    ]
    averaged_endpoint_lo = np.mean([interval[1] for interval in run_intervals])
    averaged_endpoint_hi = np.mean([interval[2] for interval in run_intervals])

    assert not np.allclose(
        [result[1], result[2]],
        [averaged_endpoint_lo, averaged_endpoint_hi],
    )


def test_r5_net_benefit_uses_architecture_bootstrap_distribution() -> None:
    runs = [
        (
            np.array([0.05, 0.10, 0.80, 0.90, 0.65, 0.70]),
            np.array([0, 0, 1, 1, 1, 1]),
            np.array(["a", "a", "b", "b", "c", "c"]),
        ),
        (
            np.array([0.20, 0.25, 0.40, 0.45, 0.85, 0.95]),
            np.array([0, 0, 0, 0, 1, 1]),
            np.array(["d", "d", "e", "e", "f", "f"]),
        ),
    ]
    B = 200
    seed = 91
    tau_r5 = bayes_optimal_threshold(1.0, 5.0)

    assert "Net benefit@r5" in TABLE1_METRICS
    result = architecture_cluster_bootstrap(runs, B=B, seed=seed)["Net benefit@r5"]

    rng = np.random.default_rng(seed)
    manual_distribution = []
    patient_universe = np.asarray(
        sorted({patient for _probs, _labels, patient_ids in runs for patient in patient_ids})
    )
    for _replicate in range(B):
        sampled_positions = rng.integers(
            0,
            len(patient_universe),
            size=len(patient_universe),
        )
        multiplicities = np.bincount(sampled_positions, minlength=len(patient_universe))
        cell_values = []
        for probs, labels, patient_ids in runs:
            indices = {
                patient: np.flatnonzero(patient_ids == patient)
                for patient in np.unique(patient_ids)
            }
            sampled_clusters = [
                indices[patient]
                for patient, multiplicity in zip(
                    patient_universe,
                    multiplicities,
                    strict=True,
                )
                for _repeat in range(int(multiplicity))
                if patient in indices
            ]
            if not sampled_clusters:
                continue
            sampled_indices = np.concatenate(sampled_clusters)
            cell_values.append(
                net_benefit(
                    probs[sampled_indices],
                    labels[sampled_indices],
                    tau_r5,
                )
            )
        manual_distribution.append(np.mean(cell_values))
    expected_lo, expected_hi = np.percentile(manual_distribution, [2.5, 97.5])

    assert result[1] == expected_lo
    assert result[2] == expected_hi

    run_intervals = [
        cluster_bootstrap(
            lambda sample_probs, sample_labels: net_benefit(
                sample_probs,
                sample_labels,
                tau_r5,
            ),
            probs,
            labels,
            patient_ids,
            B=B,
            seed=seed + run_idx,
        )
        for run_idx, (probs, labels, patient_ids) in enumerate(runs)
    ]
    averaged_endpoint_lo = np.mean([interval[1] for interval in run_intervals])
    averaged_endpoint_hi = np.mean([interval[2] for interval in run_intervals])

    assert not np.allclose(
        [result[1], result[2]],
        [averaged_endpoint_lo, averaged_endpoint_hi],
    )
