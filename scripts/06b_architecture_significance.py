#!/usr/bin/env python3
"""Paired architecture-level significance tests across shared folds."""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.utils.io import ensure_parent_dir
from src.utils.logging import configure_logging
from src.utils.seeding import set_global_seed

ARCHITECTURES = ("custom_cnn", "efficientnet_b0", "resnet50")
PAIR_KEYS = ["mag", "seed", "fold"]
BONFERRONI_FACTOR = 18
OUTPUT_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-results",
        type=Path,
        default=Path("outputs/results/calibration_results.csv"),
    )
    parser.add_argument(
        "--decision-results",
        type=Path,
        default=Path("outputs/results/decision_results.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/results/architecture_significance.csv"),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("outputs/results/architecture_significance.log"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _metric_frame(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return df[["arch", *PAIR_KEYS, value_column]].rename(columns={value_column: "value"}).copy()


def build_metric_tables(
    calibration_df: pd.DataFrame,
    decision_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build one architecture-by-fold table per metric requested for testing."""
    calibration_test = calibration_df[calibration_df["split"] == "test"].copy()
    decision_uncalibrated = decision_df[decision_df["method"] == "uncalibrated"].copy()
    decision_uncalibrated["c_ratio"] = decision_uncalibrated["c_ratio"].astype(str)

    temperature = calibration_test[calibration_test["method"] == "temperature"]
    uncalibrated = calibration_test[calibration_test["method"] == "uncalibrated"]
    fixed_accuracy = decision_uncalibrated[decision_uncalibrated["c_ratio"] == "fixed_0.5"]
    net_benefit_r10 = decision_uncalibrated[decision_uncalibrated["c_ratio"] == "10"]
    net_benefit_r5 = decision_uncalibrated[decision_uncalibrated["c_ratio"] == "5"]

    return {
        "temperature_T": _metric_frame(temperature, "T"),
        "ece_uncalibrated": _metric_frame(uncalibrated, "ece"),
        "brier_uncalibrated": _metric_frame(uncalibrated, "brier"),
        "net_benefit_r10_uncalibrated": _metric_frame(
            net_benefit_r10,
            "net_benefit",
        ),
        "net_benefit_r5_uncalibrated": _metric_frame(
            net_benefit_r5,
            "net_benefit",
        ),
        "accuracy_tau_0.5_uncalibrated": _metric_frame(
            fixed_accuracy,
            "accuracy",
        ),
    }


def paired_differences(table: pd.DataFrame, arch_a: str, arch_b: str) -> np.ndarray:
    left = table[table["arch"] == arch_a][PAIR_KEYS + ["value"]].rename(
        columns={"value": "value_a"}
    )
    right = table[table["arch"] == arch_b][PAIR_KEYS + ["value"]].rename(
        columns={"value": "value_b"}
    )
    paired = left.merge(right, on=PAIR_KEYS, how="inner").dropna(subset=["value_a", "value_b"])
    return (paired["value_a"].to_numpy(dtype=np.float64) - paired["value_b"].to_numpy(dtype=np.float64))


def wilcoxon_row(arch_a: str, arch_b: str, metric: str, differences: np.ndarray) -> dict:
    if differences.size == 0:
        raise ValueError(f"No paired observations for {arch_a} vs {arch_b}, metric={metric}.")

    if np.allclose(differences, 0.0):
        statistic = 0.0
        p_value = 1.0
    else:
        result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)

    adjusted = min(float(p_value * BONFERRONI_FACTOR), 1.0)
    return {
        "arch_a": arch_a,
        "arch_b": arch_b,
        "metric": metric,
        "n_pairs": int(differences.size),
        "statistic": statistic,
        "p_value_raw": p_value,
        "p_value_bonferroni": adjusted,
        "median_diff_a_minus_b": float(np.median(differences)),
        "bonferroni_significant_005": bool(adjusted < 0.05),
    }


def architecture_significance(
    calibration_df: pd.DataFrame,
    decision_df: pd.DataFrame,
) -> pd.DataFrame:
    metric_tables = build_metric_tables(calibration_df, decision_df)
    rows: list[dict] = []

    for arch_a, arch_b in itertools.combinations(ARCHITECTURES, 2):
        for metric, table in metric_tables.items():
            differences = paired_differences(table, arch_a, arch_b)
            rows.append(wilcoxon_row(arch_a, arch_b, metric, differences))

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main() -> None:
    args = parse_args()
    configure_logging(log_path=args.log_file)
    set_global_seed(args.seed)

    calibration_df = pd.read_csv(args.calibration_results)
    decision_df = pd.read_csv(args.decision_results)
    logging.info(
        "Loaded %s calibration rows and %s decision rows",
        len(calibration_df),
        len(decision_df),
    )

    results = architecture_significance(calibration_df, decision_df)
    ensure_parent_dir(args.output)
    results.to_csv(args.output, index=False)
    logging.info("Wrote %s architecture significance rows to %s", len(results), args.output)


if __name__ == "__main__":
    main()
