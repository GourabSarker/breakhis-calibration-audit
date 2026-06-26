#!/usr/bin/env python3
"""Run cost-sensitive decision analysis from cached BreakHis logits."""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.recalibrators import (
    IsotonicCalibration,
    PlattScaling,
    Recalibrator,
    TemperatureScaling,
    softmax_positive_probability,
)
from src.decision.cost_matrix import cost_ratio_sweep
from src.decision.net_benefit import decision_curve, net_benefit
from src.decision.thresholds import metrics_at_threshold
from src.training.train import format_seconds, magnification_tag, normalize_magnification
from src.utils.cli import require_nonaggregate_output_for_partial_run
from src.utils.io import ensure_parent_dir
from src.utils.logging import configure_logging
from src.utils.seeding import set_global_seed

RUN_RE = re.compile(r"^seed(?P<seed>\d+)_fold(?P<fold>\d+)\.npz$")
DEFAULT_OUTPUT = Path("outputs/results/decision_results.csv")
METHODS = ("uncalibrated", "temperature", "platt", "isotonic")
OUTPUT_COLUMNS = [
    "arch",
    "mag",
    "seed",
    "fold",
    "method",
    "c_ratio",
    "tau",
    "accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "expected_cost",
    "net_benefit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/logits"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--curve-root",
        type=Path,
        default=Path("outputs/results/decision_curves"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--log-file", type=Path, default=Path("outputs/results/run_decision.log"))
    return parser.parse_args()


def parse_logits_path(path: Path, input_root: Path) -> tuple[str, str, int, int]:
    relative = path.resolve().relative_to(input_root.resolve())
    if len(relative.parts) != 3:
        raise ValueError(f"Logit path must match {{arch}}/{{mag}}/seed{{s}}_fold{{f}}.npz: {path}")
    arch, mag, filename = relative.parts
    match = RUN_RE.match(filename)
    if match is None:
        raise ValueError(f"Logit filename does not match seed/fold naming: {filename}")
    return arch, mag, int(match.group("seed")), int(match.group("fold"))


def parse_only_filter(value: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for segment in value.split(","):
        if "=" not in segment:
            raise ValueError(f"Invalid --only segment '{segment}'. Expected key=value.")
        key, raw_value = segment.split("=", maxsplit=1)
        pairs[key.strip()] = raw_value.strip()
    required = {"arch", "mag", "seed", "fold"}
    missing = required - set(pairs)
    if missing:
        raise ValueError(f"--only is missing required keys: {sorted(missing)}")
    return pairs


def matches_only_filter(
    arch: str,
    mag: str,
    seed: int,
    fold: int,
    only_filter: dict[str, str] | None,
) -> bool:
    if only_filter is None:
        return True
    return (
        arch == only_filter["arch"]
        and mag == magnification_tag(normalize_magnification(only_filter["mag"]))
        and seed == int(only_filter["seed"])
        and fold == int(only_filter["fold"])
    )


def discover_logit_files(input_root: Path, only: str | None = None) -> list[Path]:
    only_filter = parse_only_filter(only) if only else None
    files: list[Path] = []
    for path in sorted(input_root.glob("*/*/seed*_fold*.npz")):
        arch, mag, seed, fold = parse_logits_path(path, input_root)
        if matches_only_filter(arch, mag, seed, fold, only_filter):
            files.append(path)
    if only_filter is not None and not files:
        raise FileNotFoundError(f"No logit file matched --only={only!r} under {input_root.resolve()}")
    return files


def fitted_recalibrators(logits_cal: np.ndarray, labels_cal: np.ndarray) -> dict[str, Recalibrator]:
    return {
        "temperature": TemperatureScaling().fit(logits_cal, labels_cal),
        "platt": PlattScaling().fit(logits_cal, labels_cal),
        "isotonic": IsotonicCalibration().fit(logits_cal, labels_cal),
    }


def probabilities_by_method(
    logits_cal: np.ndarray,
    labels_cal: np.ndarray,
    logits_test: np.ndarray,
) -> dict[str, np.ndarray]:
    probs = {
        "uncalibrated": softmax_positive_probability(logits_test),
    }
    recalibrators = fitted_recalibrators(logits_cal, labels_cal)
    for method, recalibrator in recalibrators.items():
        probs[method] = recalibrator.transform(logits_test)
    return probs


def threshold_grid() -> list[dict]:
    rows = [
        {
            "c_ratio": "fixed_0.5",
            "c_fp": 1.0,
            "c_fn": 1.0,
            "tau": 0.5,
        }
    ]
    rows.extend(cost_ratio_sweep([1, 2, 5, 10, 20]))
    return rows


def decision_rows_for_method(
    *,
    arch: str,
    mag: str,
    seed: int,
    fold: int,
    method: str,
    probs: np.ndarray,
    labels: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    for threshold in threshold_grid():
        metrics = metrics_at_threshold(
            probs,
            labels,
            tau=float(threshold["tau"]),
            c_fp=float(threshold["c_fp"]),
            c_fn=float(threshold["c_fn"]),
        )
        rows.append(
            {
                "arch": arch,
                "mag": mag,
                "seed": seed,
                "fold": fold,
                "method": method,
                "c_ratio": str(threshold["c_ratio"]),
                "tau": float(threshold["tau"]),
                **metrics,
                "net_benefit": net_benefit(probs, labels, float(threshold["tau"])),
            }
        )
    return rows


def curve_path_for_run(
    curve_root: Path,
    arch: str,
    mag: str,
    seed: int,
    fold: int,
    method: str,
) -> Path:
    return curve_root / arch / mag / f"seed{seed}_fold{fold}_{method}.npy"


def rows_for_logit_file(path: Path, input_root: Path, curve_root: Path) -> list[dict]:
    arch, mag, seed, fold = parse_logits_path(path, input_root)
    with np.load(path) as data:
        logits_cal = data["logits_cal"]
        labels_cal = data["labels_cal"]
        logits_test = data["logits_test"]
        labels_test = data["labels_test"]

    method_probs = probabilities_by_method(logits_cal, labels_cal, logits_test)
    taus = np.arange(1, 100, dtype=np.float64) / 100.0
    rows: list[dict] = []

    for method in METHODS:
        probs = method_probs[method]
        rows.extend(
            decision_rows_for_method(
                arch=arch,
                mag=mag,
                seed=seed,
                fold=fold,
                method=method,
                probs=probs,
                labels=labels_test,
            )
        )
        curve = decision_curve(probs, labels_test, taus)
        curve_path = curve_path_for_run(curve_root, arch, mag, seed, fold, method)
        ensure_parent_dir(curve_path)
        np.save(curve_path, curve)

    return rows


def main() -> None:
    args = parse_args()
    require_nonaggregate_output_for_partial_run(
        only=args.only,
        output=args.output,
        aggregate_output=DEFAULT_OUTPUT,
    )
    configure_logging(log_path=args.log_file)
    set_global_seed(args.seed)

    input_root = args.input_root.resolve()
    curve_root = args.curve_root.resolve()
    logit_files = discover_logit_files(input_root, only=args.only)
    all_rows: list[dict] = []
    start = time.perf_counter()

    logging.info("Running decision analysis on %s cached logit files from %s", len(logit_files), input_root)
    for index, path in enumerate(logit_files, start=1):
        file_start = time.perf_counter()
        rows = rows_for_logit_file(path, input_root, curve_root)
        all_rows.extend(rows)
        arch, mag, seed, fold = parse_logits_path(path, input_root)
        logging.info(
            "[%s/%s] decision %s/%s/seed%s_fold%s | rows=%s | wall=%s",
            index,
            len(logit_files),
            arch,
            mag,
            seed,
            fold,
            len(rows),
            format_seconds(time.perf_counter() - file_start),
        )

    results = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
    ensure_parent_dir(args.output)
    results.to_csv(args.output, index=False)
    logging.info(
        "Saved %s rows to %s | total_wall=%s",
        len(results),
        args.output,
        format_seconds(time.perf_counter() - start),
    )


if __name__ == "__main__":
    main()
