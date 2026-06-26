from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ARCHITECTURES = ("custom_cnn", "efficientnet_b0", "resnet50")
MAGNIFICATIONS = ("40X", "100X", "200X", "400X")
METHODS = ("uncalibrated", "temperature", "platt", "isotonic")


def _write_logits_and_curves(logit_root: Path, curve_root: Path) -> None:
    taus = np.arange(1, 100, dtype=np.float64) / 100.0
    for arch_idx, arch in enumerate(ARCHITECTURES):
        for mag_idx, mag in enumerate(MAGNIFICATIONS):
            npz_path = logit_root / arch / mag / "seed0_fold0.npz"
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            logits = np.array(
                [
                    [1.5, -0.5],
                    [0.8, -0.2],
                    [-0.2, 0.8],
                    [-0.6, 1.2],
                    [0.0, 0.9],
                    [0.6, 0.1],
                ],
                dtype=np.float32,
            )
            labels = np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)
            patient_ids = np.array(["p0", "p0", "p1", "p1", "p2", "p2"])
            np.savez_compressed(
                npz_path,
                logits_cal=logits,
                labels_cal=labels,
                patient_ids_cal=patient_ids,
                logits_test=logits,
                labels_test=labels,
                patient_ids_test=patient_ids,
            )

            curve_dir = curve_root / arch / mag
            curve_dir.mkdir(parents=True, exist_ok=True)
            curve = np.column_stack(
                [
                    taus,
                    0.62 - (0.25 * taus) - (0.02 * arch_idx) + (0.005 * mag_idx),
                ]
            )
            np.save(curve_dir / "seed0_fold0_uncalibrated.npy", curve)


def _write_calibration_results(path: Path) -> None:
    rows: list[dict] = []
    for arch_idx, arch in enumerate(ARCHITECTURES):
        for mag_idx, mag in enumerate(MAGNIFICATIONS):
            for split in ("cal", "test"):
                split_shift = 0.02 if split == "test" else 0.0
                for method_idx, method in enumerate(METHODS):
                    rows.append(
                        {
                            "arch": arch,
                            "mag": mag,
                            "seed": 0,
                            "fold": 0,
                            "method": method,
                            "split": split,
                            "ece": 0.08 + split_shift + (0.01 * method_idx) + (0.005 * arch_idx),
                            "mce": 0.12 + split_shift,
                            "brier": 0.16 + (0.01 * arch_idx),
                            "rel": 0.03,
                            "res": 0.04,
                            "unc": 0.22,
                            "nll": 0.45,
                            "T": 1.0 + (0.2 * arch_idx) + (0.03 * mag_idx)
                            if method == "temperature"
                            else np.nan,
                            "T_ci_lo": 0.9 if method == "temperature" else np.nan,
                            "T_ci_hi": 1.1 if method == "temperature" else np.nan,
                            "fisher_info": 100.0 if method == "temperature" else np.nan,
                            "platt_a": 1.0 if method == "platt" else np.nan,
                            "platt_b": 0.0 if method == "platt" else np.nan,
                        }
                    )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_bootstrap_cis(path: Path) -> None:
    rows: list[dict] = []
    metrics = {
        "ece": 0.10,
        "brier": 0.18,
        "accuracy_tau_0.5": 0.82,
        "net_benefit_r10": 0.55,
    }
    for arch_idx, arch in enumerate(ARCHITECTURES):
        for mag in MAGNIFICATIONS:
            for metric, point in metrics.items():
                value = point - (0.01 * arch_idx)
                rows.append(
                    {
                        "arch": arch,
                        "mag": mag,
                        "seed": 0,
                        "fold": 0,
                        "method": "uncalibrated",
                        "metric": metric,
                        "point_estimate": value,
                        "ci_lo": value - 0.03,
                        "ci_hi": value + 0.03,
                        "B": 25,
                        "n_finite_replicates": 25,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_decision_results(path: Path) -> None:
    rows: list[dict] = []
    for arch_idx, arch in enumerate(ARCHITECTURES):
        for mag in MAGNIFICATIONS:
            for c_ratio, tau in (("fixed_0.5", 0.5), ("10", 1 / 11), ("5", 1 / 6)):
                rows.append(
                    {
                        "arch": arch,
                        "mag": mag,
                        "seed": 0,
                        "fold": 0,
                        "method": "uncalibrated",
                        "c_ratio": c_ratio,
                        "tau": tau,
                        "accuracy": 0.82 - (0.01 * arch_idx),
                        "sensitivity": 0.88 - (0.01 * arch_idx),
                        "specificity": 0.76 - (0.01 * arch_idx),
                        "ppv": 0.80,
                        "npv": 0.78,
                        "expected_cost": 0.12,
                        "net_benefit": 0.55 - (0.01 * arch_idx),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_architecture_significance(path: Path) -> None:
    pairs = [
        ("custom_cnn", "efficientnet_b0"),
        ("custom_cnn", "resnet50"),
        ("efficientnet_b0", "resnet50"),
    ]
    metrics = [
        "temperature_T",
        "ece_uncalibrated",
        "brier_uncalibrated",
        "net_benefit_r10_uncalibrated",
        "net_benefit_r5_uncalibrated",
        "accuracy_tau_0.5_uncalibrated",
    ]
    rows = []
    for arch_a, arch_b in pairs:
        for metric in metrics:
            rows.append(
                {
                    "arch_a": arch_a,
                    "arch_b": arch_b,
                    "metric": metric,
                    "n_pairs": 4,
                    "statistic": 1.0,
                    "p_value_raw": 0.01,
                    "p_value_bonferroni": 0.04 if metric == "temperature_T" else 0.5,
                    "median_diff_a_minus_b": -0.1,
                    "bonferroni_significant_005": metric == "temperature_T",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_brier_summary(path: Path) -> None:
    rows = []
    for arch_idx, arch in enumerate(ARCHITECTURES):
        for mag in MAGNIFICATIONS:
            for method in ("uncalibrated", "temperature"):
                rows.append(
                    {
                        "arch": arch,
                        "mag": mag,
                        "method": method,
                        "n": 1,
                        "rel": 0.02 + (0.002 * arch_idx),
                        "res": 0.04,
                        "unc": 0.22,
                        "brier": 0.20 + (0.01 * arch_idx),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_make_figures_script_runs_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_root = tmp_path / "results"
    figure_root = tmp_path / "figures"
    table_root = figure_root / "tables"
    logit_root = tmp_path / "logits"
    curve_root = results_root / "decision_curves"
    results_root.mkdir(parents=True)

    calibration_path = results_root / "calibration_results.csv"
    bootstrap_path = results_root / "bootstrap_cis.csv"
    decision_path = results_root / "decision_results.csv"
    architecture_path = results_root / "architecture_significance.csv"
    brier_path = results_root / "brier_decomposition_summary.csv"

    _write_logits_and_curves(logit_root, curve_root)
    _write_calibration_results(calibration_path)
    _write_bootstrap_cis(bootstrap_path)
    _write_decision_results(decision_path)
    _write_architecture_significance(architecture_path)
    _write_brier_summary(brier_path)

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "07_make_figures.py"),
            "--calibration-results",
            str(calibration_path),
            "--bootstrap-cis",
            str(bootstrap_path),
            "--decision-results",
            str(decision_path),
            "--architecture-significance",
            str(architecture_path),
            "--brier-summary",
            str(brier_path),
            "--logit-root",
            str(logit_root),
            "--decision-curve-root",
            str(curve_root),
            "--figure-root",
            str(figure_root),
            "--table-root",
            str(table_root),
            "--curve-bootstrap-B",
            "5",
            "--table-bootstrap-B",
            "5",
            "--log-file",
            str(results_root / "make_figures.log"),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    expected = [
        figure_root / "fig1_reliability_grid.png",
        figure_root / "fig1_reliability_grid.pdf",
        figure_root / "fig2_temperature_distribution.png",
        figure_root / "fig2_temperature_distribution.pdf",
        figure_root / "fig3_decision_curves.png",
        figure_root / "fig3_decision_curves.pdf",
        figure_root / "fig4_generalization_gap.png",
        figure_root / "fig4_generalization_gap.pdf",
        table_root / "table1_main.csv",
        table_root / "table1_main.tex",
        table_root / "table2_significance.csv",
        table_root / "table2_significance.tex",
        table_root / "table3_brier_decomposition.csv",
        table_root / "table3_brier_decomposition.tex",
    ]
    for path in expected:
        assert path.exists(), path
        assert path.stat().st_size > 0

    table1 = pd.read_csv(table_root / "table1_main.csv")
    assert "Net benefit@r5" in table1.columns
