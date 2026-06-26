"""Publication table builders for Phase 7."""

from __future__ import annotations

import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.metrics import brier_score, expected_calibration_error
from src.calibration.recalibrators import softmax_positive_probability
from src.decision.net_benefit import net_benefit
from src.decision.thresholds import bayes_optimal_threshold, metrics_at_threshold
from src.utils.io import ensure_parent_dir

ARCHITECTURES = ("custom_cnn", "efficientnet_b0", "resnet50")
ARCH_LABELS = {
    "custom_cnn": "Custom CNN",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet-50",
}
RUN_RE = re.compile(r"^seed(?P<seed>\d+)_fold(?P<fold>\d+)\.npz$")


def _write_table(df: pd.DataFrame, output_dir: Path, name: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    tex_path = output_dir / f"{name}.tex"
    df.to_csv(csv_path, index=False)
    latex = df.to_latex(index=False, escape=False)
    ensure_parent_dir(tex_path)
    tex_path.write_text(latex, encoding="utf-8")
    return [csv_path, tex_path]


def _format_estimate_ci(point: float, lo: float, hi: float) -> str:
    if not np.isfinite(point) or not np.isfinite(lo) or not np.isfinite(hi):
        return "NA"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def _stable_seed(seed: int, *parts: object) -> int:
    key = "|".join(str(part) for part in parts)
    return int((seed + zlib.crc32(key.encode("utf-8"))) % (2**32 - 1))


def _iter_logit_files(logit_root: Path, arch: str) -> list[Path]:
    return sorted((logit_root / arch).glob("*/*.npz"))


def _metric_vector(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    tau_r10 = bayes_optimal_threshold(1.0, 10.0)
    tau_r5 = bayes_optimal_threshold(1.0, 5.0)
    at_half = metrics_at_threshold(probs, labels, 0.5)
    at_r10 = metrics_at_threshold(probs, labels, tau_r10)
    return np.asarray(
        [
            expected_calibration_error(probs, labels),
            brier_score(probs, labels),
            at_half["accuracy"],
            at_r10["sensitivity"],
            at_r10["specificity"],
            net_benefit(probs, labels, tau_r10),
            net_benefit(probs, labels, tau_r5),
        ],
        dtype=np.float64,
    )


TABLE1_METRICS = (
    "ECE",
    "Brier",
    "Accuracy@0.5",
    "Sensitivity@r10",
    "Specificity@r10",
    "Net benefit@r10",
    "Net benefit@r5",
)


def _finite_column_means(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    sums = np.where(finite, values, 0.0).sum(axis=0)
    means = np.full(values.shape[1], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means


def architecture_cluster_bootstrap(
    runs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    B: int,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap the equal-cell-weighted architecture mean.

    Each replicate samples from the union of patient IDs once and applies the
    resulting patient multiplicities consistently to every fixed
    magnification/seed/fold cell. This preserves patient dependence across
    repeated cells. Cell metrics are recomputed and then averaged; percentile
    limits are taken from that architecture-level bootstrap distribution,
    never from averaged run-level endpoints.
    """
    if B < 1:
        raise ValueError("B must be at least 1.")
    if not runs:
        raise ValueError("At least one run is required.")

    prepared = []
    point_cells = []
    all_patients: set[str] = set()
    for probs, labels, patient_ids in runs:
        probs_array = np.asarray(probs, dtype=np.float64).reshape(-1)
        labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
        patient_array = np.asarray(patient_ids).astype(str).reshape(-1)
        if not (len(probs_array) == len(labels_array) == len(patient_array)):
            raise ValueError("Run probabilities, labels, and patient IDs must align.")
        unique_patients = np.unique(patient_array)
        all_patients.update(unique_patients.tolist())
        patient_indices = {
            patient_id: np.flatnonzero(patient_array == patient_id)
            for patient_id in unique_patients
        }
        prepared.append((probs_array, labels_array, patient_indices))
        point_cells.append(_metric_vector(probs_array, labels_array))

    point_estimates = _finite_column_means(np.stack(point_cells))
    rng = np.random.default_rng(seed)
    distribution = np.full((B, len(TABLE1_METRICS)), np.nan, dtype=np.float64)
    patient_universe = np.asarray(sorted(all_patients))

    for replicate_idx in range(B):
        sampled_positions = rng.integers(
            0,
            len(patient_universe),
            size=len(patient_universe),
        )
        multiplicities = np.bincount(
            sampled_positions,
            minlength=len(patient_universe),
        )
        cell_values = []
        for probs_array, labels_array, patient_indices in prepared:
            sampled_clusters = [
                patient_indices[patient_id]
                for patient_id, multiplicity in zip(
                    patient_universe,
                    multiplicities,
                    strict=True,
                )
                for _repeat in range(int(multiplicity))
                if patient_id in patient_indices
            ]
            if not sampled_clusters:
                cell_values.append(np.full(len(TABLE1_METRICS), np.nan))
                continue
            sampled_indices = np.concatenate(sampled_clusters)
            cell_values.append(
                _metric_vector(
                    probs_array[sampled_indices],
                    labels_array[sampled_indices],
                )
            )
        distribution[replicate_idx] = _finite_column_means(np.stack(cell_values))

    results = {}
    for metric_idx, metric in enumerate(TABLE1_METRICS):
        finite = distribution[np.isfinite(distribution[:, metric_idx]), metric_idx]
        if finite.size == 0:
            results[metric] = (float(point_estimates[metric_idx]), np.nan, np.nan)
            continue
        lo, hi = np.percentile(finite, [2.5, 97.5])
        results[metric] = (
            float(point_estimates[metric_idx]),
            float(lo),
            float(hi),
        )
    return results


def _load_architecture_runs(
    logit_root: Path,
    arch: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    runs = []
    for npz_path in _iter_logit_files(logit_root, arch):
        if RUN_RE.match(npz_path.name) is None:
            continue
        with np.load(npz_path) as data:
            runs.append(
                (
                    softmax_positive_probability(data["logits_test"]),
                    data["labels_test"].astype(np.int64),
                    data["patient_ids_test"].astype(str),
                )
            )
    return runs


def build_main_table(
    bootstrap_cis_csv: Path,
    decision_results_csv: Path,
    output_dir: Path = Path("outputs/figures/tables"),
    *,
    logit_root: Path | None = Path("outputs/logits"),
    B: int = 1000,
    seed: int = 2026,
) -> tuple[pd.DataFrame, list[Path]]:
    """Build Table 1: magnification-averaged architecture performance."""
    pd.read_csv(bootstrap_cis_csv)  # Provenance/schema guard for the Phase 6 output.
    pd.read_csv(decision_results_csv)  # Provenance/schema guard for the Phase 5 output.
    if logit_root is None:
        raise ValueError("logit_root is required for architecture-level cluster bootstrap CIs.")

    rows: list[dict[str, str]] = []
    for arch in ARCHITECTURES:
        row: dict[str, str] = {"Architecture": ARCH_LABELS[arch]}
        estimates = architecture_cluster_bootstrap(
            _load_architecture_runs(logit_root, arch),
            B=B,
            seed=_stable_seed(seed, arch, "table1"),
        )
        for metric in TABLE1_METRICS:
            row[metric] = _format_estimate_ci(*estimates[metric])
        rows.append(row)

    table = pd.DataFrame(rows)
    ordered = [
        "Architecture",
        *TABLE1_METRICS,
    ]
    table = table[ordered]
    return table, _write_table(table, output_dir, "table1_main")


def _format_p_value(p_value: float, significant: bool) -> str:
    if not np.isfinite(p_value):
        return "NA"
    text = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
    return f"{text}*" if significant else text


def build_significance_table(
    architecture_significance_csv: Path,
    output_dir: Path = Path("outputs/figures/tables"),
) -> tuple[pd.DataFrame, list[Path]]:
    """Build Table 2: Bonferroni-corrected architecture-pair p-values."""
    sig_df = pd.read_csv(architecture_significance_csv)
    rows = []
    for row in sig_df.itertuples(index=False):
        rows.append(
            {
                "Architecture pair": f"{ARCH_LABELS[row.arch_a]} vs {ARCH_LABELS[row.arch_b]}",
                "Metric": row.metric,
                "n pairs": int(row.n_pairs),
                "Bonferroni p": _format_p_value(
                    float(row.p_value_bonferroni),
                    bool(row.bonferroni_significant_005),
                ),
                "Median diff": f"{float(row.median_diff_a_minus_b):.3f}",
            }
        )
    table = pd.DataFrame(rows)
    return table, _write_table(table, output_dir, "table2_significance")


def build_brier_decomposition_table(
    brier_summary_csv: Path,
    output_dir: Path = Path("outputs/figures/tables"),
) -> tuple[pd.DataFrame, list[Path]]:
    """Build Table 3: Brier decomposition by architecture and magnification."""
    df = pd.read_csv(brier_summary_csv)
    subset = df[df["method"].isin(["uncalibrated", "temperature"])].copy()
    rows: list[dict[str, object]] = []
    for (arch, mag), group in subset.groupby(["arch", "mag"], sort=False):
        row: dict[str, object] = {
            "Architecture": ARCH_LABELS.get(arch, arch),
            "Magnification": mag,
        }
        for method in ("uncalibrated", "temperature"):
            method_rows = group[group["method"] == method]
            prefix = "Uncal" if method == "uncalibrated" else "Temp"
            if method_rows.empty:
                for component in ("REL", "RES", "UNC", "Brier"):
                    row[f"{prefix} {component}"] = np.nan
                continue
            method_row = method_rows.iloc[0]
            row[f"{prefix} REL"] = float(method_row["rel"])
            row[f"{prefix} RES"] = float(method_row["res"])
            row[f"{prefix} UNC"] = float(method_row["unc"])
            row[f"{prefix} Brier"] = float(method_row["brier"])
        rows.append(row)

    table = pd.DataFrame(rows)
    numeric_columns = [column for column in table.columns if column not in {"Architecture", "Magnification"}]
    table[numeric_columns] = table[numeric_columns].round(4)
    return table, _write_table(table, output_dir, "table3_brier_decomposition")
