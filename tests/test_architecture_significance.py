from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ARCHITECTURES = ("custom_cnn", "efficientnet_b0", "resnet50")


def _write_calibration_results(path: Path) -> None:
    rows: list[dict] = []
    arch_offsets = {"custom_cnn": 0.0, "efficientnet_b0": 0.1, "resnet50": 0.2}
    for arch in ARCHITECTURES:
        for mag in ("40X", "100X"):
            for seed in (0, 1):
                for fold in (0, 1):
                    base = arch_offsets[arch] + 0.01 * seed + 0.001 * fold
                    rows.append(
                        {
                            "arch": arch,
                            "mag": mag,
                            "seed": seed,
                            "fold": fold,
                            "method": "temperature",
                            "split": "test",
                            "ece": 0.2 + base,
                            "mce": 0.3 + base,
                            "brier": 0.1 + base,
                            "rel": 0.02,
                            "res": 0.03,
                            "unc": 0.21,
                            "nll": 0.4 + base,
                            "T": 1.0 + base,
                            "T_ci_lo": 0.9 + base,
                            "T_ci_hi": 1.1 + base,
                            "fisher_info": 100.0,
                            "platt_a": None,
                            "platt_b": None,
                        }
                    )
                    rows.append(
                        {
                            "arch": arch,
                            "mag": mag,
                            "seed": seed,
                            "fold": fold,
                            "method": "uncalibrated",
                            "split": "test",
                            "ece": 0.15 + base,
                            "mce": 0.25 + base,
                            "brier": 0.12 + base,
                            "rel": 0.02,
                            "res": 0.03,
                            "unc": 0.21,
                            "nll": 0.45 + base,
                            "T": None,
                            "T_ci_lo": None,
                            "T_ci_hi": None,
                            "fisher_info": None,
                            "platt_a": None,
                            "platt_b": None,
                        }
                    )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_decision_results(path: Path) -> None:
    rows: list[dict] = []
    arch_offsets = {"custom_cnn": 0.0, "efficientnet_b0": 0.1, "resnet50": 0.2}
    for arch in ARCHITECTURES:
        for mag in ("40X", "100X"):
            for seed in (0, 1):
                for fold in (0, 1):
                    base = arch_offsets[arch] + 0.01 * seed + 0.001 * fold
                    for c_ratio, tau in (("fixed_0.5", 0.5), ("10", 1 / 11), ("5", 1 / 6)):
                        rows.append(
                            {
                                "arch": arch,
                                "mag": mag,
                                "seed": seed,
                                "fold": fold,
                                "method": "uncalibrated",
                                "c_ratio": c_ratio,
                                "tau": tau,
                                "accuracy": 0.8 - base,
                                "sensitivity": 0.9 - base,
                                "specificity": 0.7 - base,
                                "ppv": 0.85 - base,
                                "npv": 0.75 - base,
                                "expected_cost": 0.2 + base,
                                "net_benefit": 0.6 - base,
                            }
                        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_architecture_significance_script_runs_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    calibration_path = tmp_path / "calibration_results.csv"
    decision_path = tmp_path / "decision_results.csv"
    output_path = tmp_path / "results" / "architecture_significance.csv"

    _write_calibration_results(calibration_path)
    _write_decision_results(decision_path)

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "06b_architecture_significance.py"),
            "--calibration-results",
            str(calibration_path),
            "--decision-results",
            str(decision_path),
            "--output",
            str(output_path),
            "--log-file",
            str(tmp_path / "results" / "architecture_significance.log"),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    result = pd.read_csv(output_path)
    assert list(result.columns) == [
        "arch_a",
        "arch_b",
        "metric",
        "n_pairs",
        "statistic",
        "p_value_raw",
        "p_value_bonferroni",
        "median_diff_a_minus_b",
        "bonferroni_significant_005",
    ]
    assert len(result) == 18
    assert set(result["n_pairs"]) == {8}
