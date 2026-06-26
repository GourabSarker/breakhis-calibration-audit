from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _rows_for_run(arch: str, mag: str, seed: int, fold: int, base_ece: float) -> list[dict]:
    rows: list[dict] = []
    for split in ("cal", "test"):
        split_shift = 0.0 if split == "cal" else 0.01
        method_values = {
            "uncalibrated": base_ece + split_shift,
            "temperature": base_ece - 0.01 + split_shift,
            "platt": base_ece - 0.005 + split_shift,
            "isotonic": base_ece + 0.002 + split_shift,
        }
        for method, ece in method_values.items():
            rows.append(
                {
                    "arch": arch,
                    "mag": mag,
                    "seed": seed,
                    "fold": fold,
                    "method": method,
                    "split": split,
                    "ece": ece,
                    "mce": ece + 0.02,
                    "brier": 0.18 + ece,
                    "rel": 0.02,
                    "res": 0.04,
                    "unc": 0.22,
                    "nll": 0.50 + ece,
                    "T": 1.4 if method == "temperature" else np.nan,
                    "T_ci_lo": 1.2 if method == "temperature" else np.nan,
                    "T_ci_hi": 1.6 if method == "temperature" else np.nan,
                    "fisher_info": 100.0 if method == "temperature" else np.nan,
                    "platt_a": 0.9 if method == "platt" else np.nan,
                    "platt_b": 0.1 if method == "platt" else np.nan,
                }
            )
    return rows


def _write_logits(logit_root: Path, arch: str, mag: str, seed: int, fold: int) -> None:
    path = logit_root / arch / mag / f"seed{seed}_fold{fold}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    logits = np.array(
        [
            [0.0, -2.0],
            [0.0, -0.4],
            [0.0, 0.5],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    patient_ids = np.array(["p0", "p0", "p1", "p1"])
    np.savez_compressed(
        path,
        logits_cal=logits,
        labels_cal=labels,
        patient_ids_cal=patient_ids,
        logits_test=logits,
        labels_test=labels,
        patient_ids_test=patient_ids,
    )


def test_calibration_diagnostics_script_runs_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    calibration_path = tmp_path / "calibration_results.csv"
    logit_root = tmp_path / "logits"
    results_root = tmp_path / "results"
    figure_root = tmp_path / "figures" / "exemplar_reliability"

    rows = []
    rows.extend(_rows_for_run("custom_cnn", "40X", 0, 0, 0.10))
    rows.extend(_rows_for_run("resnet50", "100X", 1, 2, 0.20))
    pd.DataFrame(rows).to_csv(calibration_path, index=False)

    _write_logits(logit_root, "custom_cnn", "40X", 0, 0)
    _write_logits(logit_root, "resnet50", "100X", 1, 2)

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "04b_calibration_diagnostics.py"),
            "--calibration-results",
            str(calibration_path),
            "--logit-root",
            str(logit_root),
            "--results-root",
            str(results_root),
            "--figure-root",
            str(figure_root),
            "--log-file",
            str(results_root / "diagnostics.log"),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (results_root / "calibration_significance.csv").exists()
    assert (results_root / "temperature_by_arch_mag.csv").exists()
    assert (results_root / "brier_decomposition_summary.csv").exists()
    assert (results_root / "generalization_gap.csv").exists()
    assert (figure_root / "worst.png").exists()
    assert (figure_root / "median.png").exists()
    assert (figure_root / "best.png").exists()
