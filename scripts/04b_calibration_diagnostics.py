#!/usr/bin/env python3
"""Generate diagnostics from Phase 4 calibration results."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import wilcoxon

from src.calibration.metrics import expected_calibration_error
from src.calibration.recalibrators import logit_gap, softmax_positive_probability
from src.calibration.reliability import reliability_curve
from src.utils.io import ensure_parent_dir
from src.utils.logging import configure_logging
from src.utils.seeding import set_global_seed

METHODS_COMPARED = ["temperature", "platt", "isotonic"]
METRICS_COMPARED = ["ece", "brier", "nll"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-results",
        type=Path,
        default=Path("outputs/results/calibration_results.csv"),
    )
    parser.add_argument("--logit-root", type=Path, default=Path("outputs/logits"))
    parser.add_argument("--results-root", type=Path, default=Path("outputs/results"))
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path("outputs/figures/exemplar_reliability"),
    )
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", type=Path, default=Path("outputs/results/calibration_diagnostics.log"))
    return parser.parse_args()


def run_wilcoxon_tests(df: pd.DataFrame) -> pd.DataFrame:
    test_df = df[df["split"] == "test"].copy()
    rows: list[dict] = []
    keys = ["arch", "mag", "seed", "fold"]

    for method in METHODS_COMPARED:
        for metric in METRICS_COMPARED:
            uncalibrated = test_df[test_df["method"] == "uncalibrated"][keys + [metric]]
            compared = test_df[test_df["method"] == method][keys + [metric]]
            paired = uncalibrated.merge(compared, on=keys, suffixes=("_uncalibrated", "_method"))
            differences = paired[f"{metric}_method"].to_numpy() - paired[
                f"{metric}_uncalibrated"
            ].to_numpy()

            if np.allclose(differences, 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            rows.append(
                {
                    "method_compared": method,
                    "metric": metric,
                    "n": int(len(differences)),
                    "statistic": statistic,
                    "p_value": p_value,
                    "median_difference": float(np.median(differences)),
                }
            )

    return pd.DataFrame(rows)


def temperature_by_arch_mag(df: pd.DataFrame) -> pd.DataFrame:
    temps = df[(df["method"] == "temperature") & (df["split"] == "test")].copy()
    grouped = (
        temps.groupby(["arch", "mag"], as_index=False)["T"]
        .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
        .sort_values(["arch", "mag"])
    )
    overall = pd.DataFrame(
        [
            {
                "arch": "overall",
                "mag": "overall",
                "n": int(temps["T"].count()),
                "mean": float(temps["T"].mean()),
                "median": float(temps["T"].median()),
                "std": float(temps["T"].std()),
                "min": float(temps["T"].min()),
                "max": float(temps["T"].max()),
            }
        ]
    )
    return pd.concat([grouped, overall], ignore_index=True)


def select_exemplar_runs(df: pd.DataFrame) -> dict[str, pd.Series]:
    uncal_test = (
        df[(df["method"] == "uncalibrated") & (df["split"] == "test")]
        .sort_values("ece", ascending=True)
        .reset_index(drop=True)
    )
    if uncal_test.empty:
        raise ValueError("No uncalibrated test rows found for exemplar selection.")

    best = uncal_test.iloc[0]
    worst = uncal_test.iloc[-1]
    percentile_90_index = int(round(0.9 * (len(uncal_test) - 1)))
    median_slot = uncal_test.iloc[percentile_90_index]
    return {
        "best": best,
        "median": median_slot,
        "worst": worst,
    }


def _temperature_for_run(df: pd.DataFrame, run: pd.Series) -> float:
    mask = (
        (df["arch"] == run["arch"])
        & (df["mag"] == run["mag"])
        & (df["seed"] == run["seed"])
        & (df["fold"] == run["fold"])
        & (df["method"] == "temperature")
        & (df["split"] == "test")
    )
    values = df.loc[mask, "T"].dropna()
    if values.empty:
        raise ValueError(f"No fitted T found for exemplar run {run.to_dict()}")
    return float(values.iloc[0])


def _plot_reliability_axis(ax, probs: np.ndarray, labels: np.ndarray, n_bins: int, title: str) -> None:
    curve = reliability_curve(probs, labels, n_bins=n_bins)
    mask = curve["count"] > 0
    sizes = 20.0 + 8.0 * np.sqrt(curve["count"][mask])

    ax.plot([0.0, 1.0], [0.0, 1.0], color="0.55", linewidth=1.0, linestyle="--")
    ax.scatter(
        curve["mean_confidence"][mask],
        curve["empirical_accuracy"][mask],
        s=sizes,
        color="#1f77b4",
        alpha=0.78,
        edgecolor="black",
        linewidth=0.4,
    )
    ece = expected_calibration_error(probs, labels, n_bins=n_bins)
    ax.set_title(f"{title}\nECE={ece:.4f}", fontsize=10)
    ax.set_xlabel("Mean predicted P(Y=1)")
    ax.set_ylabel("Observed event rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)


def write_exemplar_reliability_plots(
    df: pd.DataFrame,
    logit_root: Path,
    figure_root: Path,
    n_bins: int,
) -> None:
    exemplars = select_exemplar_runs(df)
    figure_root.mkdir(parents=True, exist_ok=True)

    for label, run in exemplars.items():
        npz_path = (
            logit_root
            / str(run["arch"])
            / str(run["mag"])
            / f"seed{int(run['seed'])}_fold{int(run['fold'])}.npz"
        )
        with np.load(npz_path) as data:
            logits = data["logits_test"]
            labels = data["labels_test"]

        temperature = _temperature_for_run(df, run)
        uncalibrated_probs = softmax_positive_probability(logits)
        temperature_probs = expit(logit_gap(logits) / temperature)

        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.25), sharex=True, sharey=True)
        fig.suptitle(
            f"{label.title()} exemplar: {run['arch']} {run['mag']} seed{int(run['seed'])}_fold{int(run['fold'])}",
            fontsize=11,
        )
        _plot_reliability_axis(axes[0], uncalibrated_probs, labels, n_bins, "Uncalibrated")
        _plot_reliability_axis(axes[1], temperature_probs, labels, n_bins, "Temperature")
        fig.tight_layout()
        fig.savefig(figure_root / f"{label}.png", dpi=180)
        plt.close(fig)


def brier_decomposition_summary(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[
        (df["split"] == "test") & (df["method"].isin(["uncalibrated", "temperature"]))
    ].copy()
    return (
        subset.groupby(["arch", "mag", "method"], as_index=False)
        .agg(
            n=("brier", "count"),
            rel=("rel", "mean"),
            res=("res", "mean"),
            unc=("unc", "mean"),
            brier=("brier", "mean"),
        )
        .sort_values(["arch", "mag", "method"])
    )


def generalization_gap(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["arch", "mag", "seed", "fold", "method"]
    pivot = df.pivot_table(index=keys, columns="split", values="ece", aggfunc="first").reset_index()
    pivot["test_minus_cal_ece"] = pivot["test"] - pivot["cal"]
    return (
        pivot.groupby(["arch", "mag", "method"], as_index=False)["test_minus_cal_ece"]
        .agg(n="count", mean="mean", std="std")
        .sort_values(["arch", "mag", "method"])
    )


def main() -> None:
    args = parse_args()
    configure_logging(log_path=args.log_file)
    set_global_seed(args.seed)

    df = pd.read_csv(args.calibration_results)
    logging.info("Loaded %s calibration rows from %s", len(df), args.calibration_results)

    outputs = {
        "significance": args.results_root / "calibration_significance.csv",
        "temperature": args.results_root / "temperature_by_arch_mag.csv",
        "brier": args.results_root / "brier_decomposition_summary.csv",
        "gap": args.results_root / "generalization_gap.csv",
    }

    tables = {
        outputs["significance"]: run_wilcoxon_tests(df),
        outputs["temperature"]: temperature_by_arch_mag(df),
        outputs["brier"]: brier_decomposition_summary(df),
        outputs["gap"]: generalization_gap(df),
    }

    for path, table in tables.items():
        ensure_parent_dir(path)
        table.to_csv(path, index=False)
        logging.info("Wrote %s rows to %s", len(table), path)

    write_exemplar_reliability_plots(
        df=df,
        logit_root=args.logit_root,
        figure_root=args.figure_root,
        n_bins=args.n_bins,
    )
    logging.info("Wrote exemplar reliability plots to %s", args.figure_root)


if __name__ == "__main__":
    main()
