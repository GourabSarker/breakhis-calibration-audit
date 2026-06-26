#!/usr/bin/env python3
"""Run calibration analysis from cached BreakHis logits."""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.metrics import (
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
)
from src.calibration.recalibrators import (
    IsotonicCalibration,
    PlattScaling,
    Recalibrator,
    TemperatureScaling,
    softmax_positive_probability,
)
from src.training.train import format_seconds, magnification_tag, normalize_magnification
from src.utils.cli import require_nonaggregate_output_for_partial_run
from src.utils.io import ensure_parent_dir
from src.utils.logging import configure_logging
from src.utils.seeding import set_global_seed

RUN_RE = re.compile(r"^seed(?P<seed>\d+)_fold(?P<fold>\d+)\.npz$")
DEFAULT_OUTPUT = Path("outputs/results/calibration_results.csv")
OUTPUT_COLUMNS = [
    "arch",
    "mag",
    "seed",
    "fold",
    "method",
    "split",
    "ece",
    "mce",
    "brier",
    "rel",
    "res",
    "unc",
    "nll",
    "T",
    "T_ci_lo",
    "T_ci_hi",
    "fisher_info",
    "platt_a",
    "platt_b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/logits"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--log-file", type=Path, default=Path("outputs/results/run_calibration.log"))
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


def metric_record(
    *,
    arch: str,
    mag: str,
    seed: int,
    fold: int,
    method: str,
    split: str,
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
    params: dict,
) -> dict[str, float | int | str | None]:
    decomposition = brier_decomposition(probs, labels, n_bins=n_bins)
    return {
        "arch": arch,
        "mag": mag,
        "seed": seed,
        "fold": fold,
        "method": method,
        "split": split,
        "ece": expected_calibration_error(probs, labels, n_bins=n_bins),
        "mce": maximum_calibration_error(probs, labels, n_bins=n_bins),
        "brier": brier_score(probs, labels),
        "rel": decomposition["reliability"],
        "res": decomposition["resolution"],
        "unc": decomposition["uncertainty"],
        "nll": negative_log_likelihood(probs, labels),
        "T": params.get("T"),
        "T_ci_lo": params.get("T_ci_lo"),
        "T_ci_hi": params.get("T_ci_hi"),
        "fisher_info": params.get("fisher_info"),
        "platt_a": params.get("platt_a"),
        "platt_b": params.get("platt_b"),
    }


def fitted_recalibrators(logits_cal: np.ndarray, labels_cal: np.ndarray) -> dict[str, Recalibrator]:
    return {
        "temperature": TemperatureScaling().fit(logits_cal, labels_cal),
        "platt": PlattScaling().fit(logits_cal, labels_cal),
        "isotonic": IsotonicCalibration().fit(logits_cal, labels_cal),
    }


def rows_for_logit_file(path: Path, input_root: Path, n_bins: int) -> list[dict]:
    arch, mag, seed, fold = parse_logits_path(path, input_root)
    with np.load(path) as data:
        logits_cal = data["logits_cal"]
        labels_cal = data["labels_cal"]
        logits_test = data["logits_test"]
        labels_test = data["labels_test"]

    split_logits = {
        "cal": logits_cal,
        "test": logits_test,
    }
    split_labels = {
        "cal": labels_cal,
        "test": labels_test,
    }

    rows: list[dict] = []
    for split in ("cal", "test"):
        probs = softmax_positive_probability(split_logits[split])
        rows.append(
            metric_record(
                arch=arch,
                mag=mag,
                seed=seed,
                fold=fold,
                method="uncalibrated",
                split=split,
                probs=probs,
                labels=split_labels[split],
                n_bins=n_bins,
                params={},
            )
        )

    recalibrators = fitted_recalibrators(logits_cal, labels_cal)
    for method, recalibrator in recalibrators.items():
        for split in ("cal", "test"):
            probs = recalibrator.transform(split_logits[split])
            rows.append(
                metric_record(
                    arch=arch,
                    mag=mag,
                    seed=seed,
                    fold=fold,
                    method=method,
                    split=split,
                    probs=probs,
                    labels=split_labels[split],
                    n_bins=n_bins,
                    params=recalibrator.params,
                )
            )
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
    logit_files = discover_logit_files(input_root, only=args.only)
    all_rows: list[dict] = []
    start = time.perf_counter()

    logging.info("Running calibration on %s cached logit files from %s", len(logit_files), input_root)
    for index, path in enumerate(logit_files, start=1):
        file_start = time.perf_counter()
        rows = rows_for_logit_file(path, input_root, args.n_bins)
        all_rows.extend(rows)
        arch, mag, seed, fold = parse_logits_path(path, input_root)
        logging.info(
            "[%s/%s] calibrated %s/%s/seed%s_fold%s | rows=%s | wall=%s",
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
