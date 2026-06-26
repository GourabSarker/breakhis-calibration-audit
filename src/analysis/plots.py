"""Publication figure builders for Phase 7 analysis outputs."""

from __future__ import annotations

import re
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit

from src.calibration.metrics import expected_calibration_error
from src.calibration.recalibrators import logit_gap, softmax_positive_probability
from src.calibration.reliability import reliability_curve
from src.decision.net_benefit import treat_all_net_benefit
from src.decision.thresholds import bayes_optimal_threshold
from src.utils.io import ensure_parent_dir

ARCHITECTURES = ("custom_cnn", "efficientnet_b0", "resnet50")
MAGNIFICATIONS = ("40X", "100X", "200X", "400X")
ARCH_LABELS = {
    "custom_cnn": "Custom CNN",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet-50",
}
METHOD_LABELS = {
    "uncalibrated": "Uncalibrated",
    "temperature": "Temperature",
    "platt": "Platt",
    "isotonic": "Isotonic",
}
RUN_RE = re.compile(r"seed(?P<seed>\d+)_fold(?P<fold>\d+)(?:_(?P<method>[a-z_]+))?\.npy$")


def _set_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        palette="colorblind",
        font="sans-serif",
        rc={
            "axes.grid": True,
            "grid.color": "0.86",
            "grid.linewidth": 0.6,
            "axes.edgecolor": "0.25",
            "figure.dpi": 150,
        },
    )


def _save_figure(fig: plt.Figure, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return [png_path, pdf_path]


def _stable_seed(seed: int, *parts: object) -> int:
    key = "|".join(str(part) for part in parts)
    return int((seed + zlib.crc32(key.encode("utf-8"))) % (2**32 - 1))


def _average_reliability(curves: list[dict[str, np.ndarray]], n_bins: int) -> dict[str, np.ndarray]:
    total_count = np.zeros(n_bins, dtype=np.float64)
    confidence_sum = np.zeros(n_bins, dtype=np.float64)
    accuracy_sum = np.zeros(n_bins, dtype=np.float64)

    for curve in curves:
        count = curve["count"].astype(np.float64)
        valid = count > 0
        total_count[valid] += count[valid]
        confidence_sum[valid] += curve["mean_confidence"][valid] * count[valid]
        accuracy_sum[valid] += curve["empirical_accuracy"][valid] * count[valid]

    mean_confidence = np.full(n_bins, np.nan, dtype=np.float64)
    empirical_accuracy = np.full(n_bins, np.nan, dtype=np.float64)
    valid = total_count > 0
    mean_confidence[valid] = confidence_sum[valid] / total_count[valid]
    empirical_accuracy[valid] = accuracy_sum[valid] / total_count[valid]
    return {
        "mean_confidence": mean_confidence,
        "empirical_accuracy": empirical_accuracy,
        "count": total_count,
    }


def _temperature_for_run(
    calibration_df: pd.DataFrame,
    arch: str,
    mag: str,
    seed: int,
    fold: int,
) -> float:
    mask = (
        (calibration_df["arch"] == arch)
        & (calibration_df["mag"] == mag)
        & (calibration_df["seed"] == seed)
        & (calibration_df["fold"] == fold)
        & (calibration_df["method"] == "temperature")
        & (calibration_df["split"] == "test")
    )
    values = calibration_df.loc[mask, "T"].dropna()
    if values.empty:
        raise ValueError(f"Missing fitted T for {arch}/{mag}/seed{seed}_fold{fold}.")
    return float(values.iloc[0])


def _plot_curve_points(
    ax: plt.Axes,
    curve: dict[str, np.ndarray],
    *,
    color,
    label: str,
    marker: str,
) -> None:
    valid = curve["count"] > 0
    if not np.any(valid):
        return
    sizes = 18.0 + 4.5 * np.sqrt(curve["count"][valid])
    ax.scatter(
        curve["mean_confidence"][valid],
        curve["empirical_accuracy"][valid],
        s=sizes,
        marker=marker,
        color=color,
        alpha=0.78,
        edgecolor="black",
        linewidth=0.35,
        label=label,
    )


def plot_reliability_diagrams_grid(
    calibration_results_csv: Path,
    logit_root: Path,
    output_stem: Path = Path("outputs/figures/fig1_reliability_grid"),
    *,
    n_bins: int = 15,
) -> list[Path]:
    """Save a 3x4 grid of averaged reliability diagrams."""
    _set_style()
    calibration_df = pd.read_csv(calibration_results_csv)
    colors = sns.color_palette("colorblind", 4)
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 9.2), sharex=True, sharey=True)

    for row, arch in enumerate(ARCHITECTURES):
        for col, mag in enumerate(MAGNIFICATIONS):
            ax = axes[row, col]
            panel = calibration_df[
                (calibration_df["arch"] == arch)
                & (calibration_df["mag"] == mag)
                & (calibration_df["method"] == "uncalibrated")
                & (calibration_df["split"] == "test")
            ].sort_values(["seed", "fold"])

            uncalibrated_curves: list[dict[str, np.ndarray]] = []
            temperature_curves: list[dict[str, np.ndarray]] = []

            for run in panel.itertuples(index=False):
                npz_path = logit_root / arch / mag / f"seed{int(run.seed)}_fold{int(run.fold)}.npz"
                if not npz_path.exists():
                    continue
                with np.load(npz_path) as data:
                    logits = data["logits_test"]
                    labels = data["labels_test"]
                temperature = _temperature_for_run(calibration_df, arch, mag, int(run.seed), int(run.fold))
                uncalibrated_probs = softmax_positive_probability(logits)
                temperature_probs = expit(logit_gap(logits) / temperature)
                uncalibrated_curves.append(reliability_curve(uncalibrated_probs, labels, n_bins=n_bins))
                temperature_curves.append(reliability_curve(temperature_probs, labels, n_bins=n_bins))

            ax.plot([0.0, 1.0], [0.0, 1.0], color="0.45", linestyle="--", linewidth=0.9)
            _plot_curve_points(
                ax,
                _average_reliability(uncalibrated_curves, n_bins),
                color=colors[0],
                label="Uncalibrated",
                marker="o",
            )
            _plot_curve_points(
                ax,
                _average_reliability(temperature_curves, n_bins),
                color=colors[1],
                label="Temperature",
                marker="s",
            )

            temp_rows = calibration_df[
                (calibration_df["arch"] == arch)
                & (calibration_df["mag"] == mag)
                & (calibration_df["method"] == "temperature")
                & (calibration_df["split"] == "test")
            ]
            ece_uncal = float(panel["ece"].mean()) if not panel.empty else np.nan
            ece_temp = float(temp_rows["ece"].mean()) if not temp_rows.empty else np.nan
            median_t = float(temp_rows["T"].median()) if not temp_rows.empty else np.nan
            ax.text(
                0.04,
                0.96,
                f"ECE {ece_uncal:.3f}->{ece_temp:.3f}\nmedian T={median_t:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
                bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.82, "pad": 2.5},
            )
            ax.set_title(f"{ARCH_LABELS[arch]} | {mag}", fontsize=10)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            if row == 2:
                ax.set_xlabel("Predicted probability")
            if col == 0:
                ax.set_ylabel("Observed positive rate")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Reliability diagrams averaged across seed/fold cells", y=1.035, fontsize=13)
    fig.tight_layout()
    return _save_figure(fig, output_stem)


def plot_temperature_distribution(
    calibration_results_csv: Path,
    architecture_significance_csv: Path,
    output_stem: Path = Path("outputs/figures/fig2_temperature_distribution"),
) -> list[Path]:
    """Save fitted-temperature distribution by architecture and magnification."""
    _set_style()
    calibration_df = pd.read_csv(calibration_results_csv)
    sig_df = pd.read_csv(architecture_significance_csv)
    temps = calibration_df[
        (calibration_df["method"] == "temperature") & (calibration_df["split"] == "test")
    ].copy()
    temps["architecture"] = temps["arch"].map(ARCH_LABELS)

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    sns.violinplot(
        data=temps,
        x="architecture",
        y="T",
        hue="mag",
        order=[ARCH_LABELS[arch] for arch in ARCHITECTURES],
        hue_order=list(MAGNIFICATIONS),
        inner=None,
        linewidth=0.8,
        cut=0,
        alpha=0.45,
        ax=ax,
    )
    sns.stripplot(
        data=temps,
        x="architecture",
        y="T",
        hue="mag",
        order=[ARCH_LABELS[arch] for arch in ARCHITECTURES],
        hue_order=list(MAGNIFICATIONS),
        dodge=True,
        size=3.0,
        alpha=0.75,
        linewidth=0.2,
        edgecolor="black",
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[: len(MAGNIFICATIONS)], labels[: len(MAGNIFICATIONS)], title="Magnification")
    ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.1)
    ax.set_xlabel("")
    ax.set_ylabel("Fitted temperature T")
    ax.set_title("Architecture-dependent temperature scaling")

    temp_sig = sig_df[sig_df["metric"] == "temperature_T"].copy()
    lines = []
    for row in temp_sig.itertuples(index=False):
        lines.append(
            f"{ARCH_LABELS[row.arch_a]} vs {ARCH_LABELS[row.arch_b]}: "
            f"p_B={row.p_value_bonferroni:.2g}"
        )
    ax.text(
        0.99,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "0.82", "alpha": 0.86, "pad": 3.0},
    )
    fig.tight_layout()
    return _save_figure(fig, output_stem)


def _load_curve_files(decision_curve_root: Path, arch: str, mag: str, method: str) -> dict[tuple[int, int], np.ndarray]:
    curves: dict[tuple[int, int], np.ndarray] = {}
    for path in sorted((decision_curve_root / arch / mag).glob(f"seed*_fold*_{method}.npy")):
        match = RUN_RE.match(path.name)
        if match is None:
            continue
        curves[(int(match.group("seed")), int(match.group("fold")))] = np.load(path)
    return curves


def _net_benefit_curve(probs: np.ndarray, labels: np.ndarray, taus: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = probs[:, None] >= taus[None, :]
    positive = labels[:, None] == 1
    negative = ~positive
    tp = np.sum(predictions & positive, axis=0)
    fp = np.sum(predictions & negative, axis=0)
    n = labels.size
    return (tp / n) - (fp / n) * (taus / (1.0 - taus))


def _load_run_probabilities(logit_root: Path, arch: str, mag: str, runs: list[tuple[int, int]]) -> list[dict]:
    loaded: list[dict] = []
    for seed, fold in runs:
        npz_path = logit_root / arch / mag / f"seed{seed}_fold{fold}.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path) as data:
            patient_ids = data["patient_ids_test"].astype(str)
            loaded.append(
                {
                    "probs": softmax_positive_probability(data["logits_test"]),
                    "labels": data["labels_test"].astype(np.int64),
                    "patient_ids": patient_ids,
                    "patients": np.unique(patient_ids),
                }
            )
    return loaded


def _decision_curve_cluster_band(
    runs: list[dict],
    taus: np.ndarray,
    *,
    B: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if B < 1 or not runs:
        return np.full_like(taus, np.nan), np.full_like(taus, np.nan)

    rng = np.random.default_rng(seed)
    bootstrap_curves = np.empty((B, len(taus)), dtype=np.float64)
    patient_indices = []
    for run in runs:
        patient_indices.append(
            [np.flatnonzero(run["patient_ids"] == patient_id) for patient_id in run["patients"]]
        )

    for replicate_idx in range(B):
        curves = []
        for run, indices in zip(runs, patient_indices, strict=True):
            sampled_positions = rng.integers(0, len(indices), size=len(indices))
            sampled_indices = np.concatenate([indices[position] for position in sampled_positions])
            curves.append(
                _net_benefit_curve(run["probs"][sampled_indices], run["labels"][sampled_indices], taus)
            )
        bootstrap_curves[replicate_idx] = np.mean(curves, axis=0)

    lo, hi = np.percentile(bootstrap_curves, [2.5, 97.5], axis=0)
    return lo, hi


def plot_decision_curves(
    decision_curve_root: Path,
    logit_root: Path,
    output_stem: Path = Path("outputs/figures/fig3_decision_curves"),
    *,
    method: str = "uncalibrated",
    bootstrap_B: int = 300,
    seed: int = 2026,
) -> list[Path]:
    """Save magnification-stratified decision curves with patient-bootstrap bands."""
    _set_style()
    colors = dict(zip(ARCHITECTURES, sns.color_palette("colorblind", len(ARCHITECTURES)), strict=True))
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), sharex=True, sharey=True)
    tau_r10 = bayes_optimal_threshold(1.0, 10.0)
    tau_r5 = bayes_optimal_threshold(1.0, 5.0)

    for ax, mag in zip(axes.ravel(), MAGNIFICATIONS, strict=True):
        all_labels: list[np.ndarray] = []
        for arch in ARCHITECTURES:
            curve_map = _load_curve_files(decision_curve_root, arch, mag, method)
            if not curve_map:
                continue
            runs = sorted(curve_map)
            stacked = np.stack([curve_map[run][:, 1] for run in runs], axis=0)
            taus = curve_map[runs[0]][:, 0]
            mean_curve = stacked.mean(axis=0)
            run_data = _load_run_probabilities(logit_root, arch, mag, runs)
            all_labels.extend([run["labels"] for run in run_data])
            lo, hi = _decision_curve_cluster_band(
                run_data,
                taus,
                B=bootstrap_B,
                seed=_stable_seed(seed, arch, mag, method),
            )
            ax.plot(taus, mean_curve, color=colors[arch], linewidth=1.8, label=ARCH_LABELS[arch])
            ax.fill_between(taus, lo, hi, color=colors[arch], alpha=0.13, linewidth=0)

        if all_labels:
            labels = np.concatenate(all_labels)
            treat_all = np.asarray([treat_all_net_benefit(labels, tau) for tau in taus])
            ax.plot(taus, treat_all, color="0.35", linestyle="-.", linewidth=1.0, label="Treat all")
        ax.axhline(0.0, color="0.15", linestyle=":", linewidth=1.0, label="Treat none")
        ax.axvline(tau_r10, color="0.55", linestyle="--", linewidth=0.9)
        ax.axvline(tau_r5, color="0.55", linestyle="--", linewidth=0.9)
        ax.text(tau_r10, 0.96, "r=10", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        ax.text(tau_r5, 0.88, "r=5", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        ax.set_title(f"{mag} magnification")
        ax.set_xlim(0.01, 0.99)
        ax.set_ylim(-0.2, 1.0)
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Decision curves averaged across seed/fold cells", y=1.04, fontsize=13)
    fig.tight_layout()
    return _save_figure(fig, output_stem)


def plot_generalization_gap(
    calibration_results_csv: Path,
    output_stem: Path = Path("outputs/figures/fig4_generalization_gap"),
) -> list[Path]:
    """Save mean test-minus-calibration ECE gap by recalibration method."""
    _set_style()
    df = pd.read_csv(calibration_results_csv)
    keys = ["arch", "mag", "seed", "fold", "method"]
    pivot = df.pivot_table(index=keys, columns="split", values="ece", aggfunc="first").reset_index()
    pivot["test_minus_cal_ece"] = pivot["test"] - pivot["cal"]
    summary = (
        pivot.groupby("method", as_index=False)["test_minus_cal_ece"]
        .agg(mean="mean", std="std", n="count")
        .set_index("method")
        .reindex(["uncalibrated", "temperature", "platt", "isotonic"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = sns.color_palette("colorblind", len(summary))
    ax.bar(
        [METHOD_LABELS.get(method, method) for method in summary["method"]],
        summary["mean"],
        yerr=summary["std"],
        capsize=4,
        color=colors,
        edgecolor="0.25",
        linewidth=0.6,
    )
    ax.axhline(0.0, color="0.25", linewidth=0.9)
    ax.set_ylabel("Mean test ECE - calibration ECE")
    ax.set_xlabel("")
    ax.set_title("Calibration generalization gap across all architecture/magnification/fold cells")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return _save_figure(fig, output_stem)
