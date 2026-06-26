from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.bootstrap import cluster_bootstrap


def _image_level_bootstrap(statistic_fn, values, labels, B: int, seed: int):
    values_array = np.asarray(values)
    labels_array = np.asarray(labels)
    rng = np.random.default_rng(seed)
    point = float(statistic_fn(values_array, labels_array))
    estimates = np.empty(B, dtype=np.float64)
    for idx in range(B):
        sample_idx = rng.integers(0, len(labels_array), size=len(labels_array))
        estimates[idx] = statistic_fn(values_array[sample_idx], labels_array[sample_idx])
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return point, float(lo), float(hi)


def _mean_statistic(values, labels) -> float:
    return float(np.mean(values))


def test_cluster_and_image_bootstrap_are_similar_with_one_image_per_patient() -> None:
    rng = np.random.default_rng(10)
    labels = rng.binomial(1, 0.4, size=100)
    patient_ids = np.asarray([f"p{idx:03d}" for idx in range(100)])

    cluster_point, cluster_lo, cluster_hi, cluster_n_finite = cluster_bootstrap(
        _mean_statistic,
        labels,
        labels,
        patient_ids,
        B=500,
        seed=11,
    )
    image_point, image_lo, image_hi = _image_level_bootstrap(
        _mean_statistic,
        labels,
        labels,
        B=500,
        seed=11,
    )

    assert cluster_point == image_point
    assert cluster_n_finite == 500
    cluster_width = cluster_hi - cluster_lo
    image_width = image_hi - image_lo
    assert 0.75 <= cluster_width / image_width <= 1.25


def test_cluster_bootstrap_is_wider_with_strong_patient_clustering() -> None:
    rng = np.random.default_rng(12)
    patient_labels = rng.binomial(1, 0.45, size=20)
    labels = np.repeat(patient_labels, 50)
    patient_ids = np.repeat([f"p{idx:03d}" for idx in range(20)], 50)

    _point, cluster_lo, cluster_hi, cluster_n_finite = cluster_bootstrap(
        _mean_statistic,
        labels,
        labels,
        patient_ids,
        B=500,
        seed=13,
    )
    _point, image_lo, image_hi = _image_level_bootstrap(
        _mean_statistic,
        labels,
        labels,
        B=500,
        seed=13,
    )

    assert cluster_n_finite == 500
    assert (cluster_hi - cluster_lo) > 3.0 * (image_hi - image_lo)


def test_bootstrap_cis_script_runs_on_one_npz(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_root = tmp_path / "logits"
    npz_path = input_root / "custom_cnn" / "40X" / "seed0_fold0.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    cal_logits = np.array(
        [
            [0.0, -2.0],
            [0.0, -1.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, -1.5],
            [0.0, 1.5],
        ],
        dtype=np.float32,
    )
    cal_labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    test_logits = np.vstack([cal_logits, cal_logits]).astype(np.float32)
    test_labels = np.tile(cal_labels, 2)
    patient_ids_test = np.repeat(["p0", "p1", "p2", "p3", "p4", "p5"], 2)
    patient_ids_cal = np.asarray([f"cal{idx}" for idx in range(len(cal_labels))])
    np.savez_compressed(
        npz_path,
        logits_cal=cal_logits,
        labels_cal=cal_labels,
        patient_ids_cal=patient_ids_cal,
        logits_test=test_logits,
        labels_test=test_labels,
        patient_ids_test=patient_ids_test,
    )

    output_path = tmp_path / "results" / "bootstrap_cis.csv"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "06_bootstrap_cis.py"),
            "--input-root",
            str(input_root),
            "--output",
            str(output_path),
            "--B",
            "25",
            "--seed",
            "2026",
            "--only",
            "arch=custom_cnn,mag=40X,seed=0,fold=0",
            "--log-file",
            str(tmp_path / "results" / "bootstrap.log"),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    result = pd.read_csv(output_path)
    assert list(result.columns) == [
        "arch",
        "mag",
        "seed",
        "fold",
        "method",
        "metric",
        "point_estimate",
        "ci_lo",
        "ci_hi",
        "B",
        "n_finite_replicates",
    ]
    assert len(result) == 28
    assert set(result["method"]) == {"uncalibrated", "temperature", "platt", "isotonic"}
    assert set(result["metric"]) == {
        "ece",
        "brier",
        "accuracy_tau_0.5",
        "sensitivity_tau_0.5",
        "specificity_tau_0.5",
        "net_benefit_r10",
        "net_benefit_r5",
    }
    assert (result["B"] == 25).all()
    assert result["n_finite_replicates"].between(0, 25).all()
