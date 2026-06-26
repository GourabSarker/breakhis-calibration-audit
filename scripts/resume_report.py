#!/usr/bin/env python3
"""Report checkpoint progress for the configured training grid."""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from itertools import product
from pathlib import Path

from src.training.completion import RunExpectation, classify_training_run
from src.training.train import (
    checkpoint_dir_for_run,
    format_seconds,
    load_training_config,
    magnification_tag,
    normalize_magnification,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/checkpoints"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)

    grid = list(
        product(
            config["architectures"],
            config["magnifications"],
            config["seeds"],
            config["folds"],
        )
    )
    total_runs = len(grid)
    classifications = []
    runtimes: list[float] = []

    for architecture, magnification, seed, fold in grid:
        run_dir = checkpoint_dir_for_run(
            args.output_root,
            architecture,
            magnification,
            int(seed),
            int(fold),
        )
        classification = classify_training_run(
            run_dir,
            RunExpectation(
                architecture=str(architecture),
                magnification=normalize_magnification(magnification),
                seed=int(seed),
                fold=int(fold),
                epochs=int(config["epochs"]),
                patience=int(config["early_stopping_patience"]),
            ),
        )
        classifications.append((architecture, magnification, seed, fold, classification))
        if classification.is_complete and classification.summary is not None:
            runtime = classification.summary.get("runtime_seconds")
            if runtime is not None:
                runtimes.append(float(runtime))

    completed_runs = sum(item[-1].is_complete for item in classifications)
    if runtimes:
        remaining_seconds = statistics.mean(runtimes) * (total_runs - completed_runs)
        eta = format_seconds(remaining_seconds)
    else:
        eta = "unknown"

    print(f"completed {completed_runs} / {total_runs} runs, estimated remaining time {eta}.")
    counts = Counter(item[-1].status for item in classifications)
    for status in (
        "complete",
        "incomplete",
        "interrupted",
        "smoke/diagnostic",
        "invalid metadata",
    ):
        print(f"{status}: {counts.get(status, 0)}")
    for architecture, magnification, seed, fold, classification in classifications:
        if not classification.is_complete:
            print(
                f"- {classification.status}: {architecture}/{magnification_tag(magnification)}/"
                f"seed{int(seed)}_fold{int(fold)}: {classification.reason}"
            )


if __name__ == "__main__":
    main()
