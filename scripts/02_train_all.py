#!/usr/bin/env python3
"""Run the configured BreakHis training grid with resumable checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from itertools import product
from pathlib import Path
from typing import Any

import torch

from src.data.patient_parser import collect_metadata, resolve_breakhis_root
from src.training.completion import RunExpectation, classify_training_run
from src.training.train import (
    TrainingConfig,
    checkpoint_dir_for_run,
    checkpoint_path_for_run,
    format_seconds,
    load_training_config,
    magnification_tag,
    normalize_magnification,
    run_training,
)
from src.utils.io import append_jsonl
from src.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--folds-path", type=Path, default=Path("data/folds/folds.json"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=Path("outputs/results/train_all.log"))
    parser.add_argument(
        "--summary-log",
        type=Path,
        default=Path("outputs/results/train_run_summaries.jsonl"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--rerun-incomplete",
        action="store_true",
        help="Allow intentional overwrite of incomplete run directories after they have been backed up.",
    )
    return parser.parse_args()


def build_run_grid(config: dict[str, Any], smoke_test: bool) -> list[dict[str, int | str]]:
    if smoke_test:
        return [
            {
                "architecture": "custom_cnn",
                "magnification": 40,
                "seed": 0,
                "fold": 0,
            }
        ]

    return [
        {
            "architecture": architecture,
            "magnification": normalize_magnification(magnification),
            "seed": int(seed),
            "fold": int(fold),
        }
        for architecture, magnification, seed, fold in product(
            config["architectures"],
            config["magnifications"],
            config["seeds"],
            config["folds"],
        )
    ]


def average_completed_runtime_seconds(output_root: Path) -> float | None:
    runtimes: list[float] = []
    for summary_path in output_root.glob("*/*/seed*_fold*/summary.json"):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        if "runtime_seconds" in data:
            runtimes.append(float(data["runtime_seconds"]))
    if not runtimes:
        return None
    return statistics.mean(runtimes)


def main() -> None:
    args = parse_args()
    configure_logging(log_path=args.log_file)

    config = load_training_config(args.config)
    run_grid = build_run_grid(config, smoke_test=args.smoke_test)
    total_requested = len(run_grid)
    output_root = args.output_root or (
        Path("outputs/checkpoints_diagnostic")
        if args.smoke_test
        else Path("outputs/checkpoints")
    )

    if args.dry_run:
        logging.info("Planned %s runs.", total_requested)
        for cell in run_grid:
            logging.info(
                "plan | arch=%s mag=%s seed=%s fold=%s",
                cell["architecture"],
                magnification_tag(cell["magnification"]),
                cell["seed"],
                cell["fold"],
            )
        return

    classifications = {}
    for cell in run_grid:
        run_dir = checkpoint_dir_for_run(
            output_root,
            str(cell["architecture"]),
            int(cell["magnification"]),
            int(cell["seed"]),
            int(cell["fold"]),
        )
        expected_epochs = 2 if args.smoke_test else int(config["epochs"])
        classification = classify_training_run(
            run_dir,
            RunExpectation(
                architecture=str(cell["architecture"]),
                magnification=int(cell["magnification"]),
                seed=int(cell["seed"]),
                fold=int(cell["fold"]),
                epochs=expected_epochs,
                patience=min(int(config["early_stopping_patience"]), expected_epochs),
            ),
        )
        classifications[
            (
                str(cell["architecture"]),
                int(cell["magnification"]),
                int(cell["seed"]),
                int(cell["fold"]),
            )
        ] = classification

    preexisting_completed = sum(
        classification.is_complete for classification in classifications.values()
    )
    unsafe_incomplete = [
        (cell, classifications[
            (
                str(cell["architecture"]),
                int(cell["magnification"]),
                int(cell["seed"]),
                int(cell["fold"]),
            )
        ])
        for cell in run_grid
        if classifications[
            (
                str(cell["architecture"]),
                int(cell["magnification"]),
                int(cell["seed"]),
                int(cell["fold"]),
            )
        ].status
        in {"interrupted", "smoke/diagnostic", "invalid metadata"}
        or (
            classifications[
                (
                    str(cell["architecture"]),
                    int(cell["magnification"]),
                    int(cell["seed"]),
                    int(cell["fold"]),
                )
            ].status
            == "incomplete"
            and checkpoint_dir_for_run(
                output_root,
                str(cell["architecture"]),
                int(cell["magnification"]),
                int(cell["seed"]),
                int(cell["fold"]),
            ).exists()
            and any(
                checkpoint_dir_for_run(
                    output_root,
                    str(cell["architecture"]),
                    int(cell["magnification"]),
                    int(cell["seed"]),
                    int(cell["fold"]),
                ).iterdir()
            )
        )
    ]
    if unsafe_incomplete and not args.rerun_incomplete:
        details = "\n".join(
            f"- {cell['architecture']}/{magnification_tag(cell['magnification'])}/"
            f"seed{cell['seed']}_fold{cell['fold']}: {classification.status} "
            f"({classification.reason})"
            for cell, classification in unsafe_incomplete
        )
        raise RuntimeError(
            "Non-complete run artifacts require a backup and explicit --rerun-incomplete:\n"
            f"{details}"
        )

    metadata_df = collect_metadata(resolve_breakhis_root(args.data_root))
    folds_data = json.loads(Path(args.folds_path).read_text(encoding="utf-8"))

    completed_runs = preexisting_completed
    executed_runtimes: list[float] = []
    overall_start = time.perf_counter()
    logging.info(
        "Starting training grid | requested=%s | existing_completed=%s | log_file=%s | summary_log=%s",
        total_requested,
        preexisting_completed,
        args.log_file,
        args.summary_log,
    )

    try:
        for cell in run_grid:
            checkpoint_path = checkpoint_path_for_run(
                output_root,
                str(cell["architecture"]),
                int(cell["magnification"]),
                int(cell["seed"]),
                int(cell["fold"]),
            )
            key = (
                str(cell["architecture"]),
                int(cell["magnification"]),
                int(cell["seed"]),
                int(cell["fold"]),
            )
            classification = classifications[key]
            if classification.is_complete:
                logging.info(
                    "skip | %s/%s | arch=%s mag=%s seed=%s fold=%s | validated complete",
                    completed_runs,
                    total_requested,
                    cell["architecture"],
                    magnification_tag(cell["magnification"]),
                    cell["seed"],
                    cell["fold"],
                )
                continue

            cell_epochs = 2 if args.smoke_test else int(config["epochs"])
            cell_patience = min(int(config["early_stopping_patience"]), cell_epochs)
            train_config = TrainingConfig(
                architecture=str(cell["architecture"]),
                magnification=int(cell["magnification"]),
                seed=int(cell["seed"]),
                fold=int(cell["fold"]),
                data_root=args.data_root,
                folds_path=args.folds_path,
                output_root=output_root,
                batch_size=int(config["batch_size"]),
                epochs=cell_epochs,
                learning_rate=float(config["learning_rate"]),
                patience=cell_patience,
                pretrained=bool(config.get("pretrained", True)),
                num_workers=int(config.get("num_workers", 0)),
                run_kind="smoke" if args.smoke_test else "full",
            )

            run_dir = checkpoint_path.parent
            overwrite_existing = (
                args.rerun_incomplete and run_dir.exists() and any(run_dir.iterdir())
            )
            summary = run_training(
                train_config,
                metadata_df=metadata_df,
                folds_data=folds_data,
                overwrite_existing=overwrite_existing,
            )
            completed_runs += 1
            run_wall = float(summary["runtime_seconds"])
            executed_runtimes.append(run_wall)

            cumulative_wall = time.perf_counter() - overall_start
            running_average = statistics.mean(executed_runtimes)
            remaining = total_requested - completed_runs
            eta_seconds = running_average * remaining

            heartbeat = (
                f"heartbeat | {completed_runs}/{total_requested} | "
                f"arch={train_config.architecture} mag={magnification_tag(train_config.magnification)} "
                f"seed={train_config.seed} fold={train_config.fold} | "
                f"run_wall={format_seconds(run_wall)} | "
                f"cumulative_wall={format_seconds(cumulative_wall)} | "
                f"running_avg={format_seconds(running_average)} | "
                f"eta={format_seconds(eta_seconds)}"
            )
            logging.info(heartbeat)

            append_jsonl(
                {
                    "progress_index": completed_runs,
                    "progress_total": total_requested,
                    "run_wall_seconds": run_wall,
                    "cumulative_wall_seconds": cumulative_wall,
                    "running_average_wall_seconds": running_average,
                    "eta_seconds": eta_seconds,
                    **summary,
                },
                args.summary_log,
            )

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    except KeyboardInterrupt as exc:
        logging.warning(
            "Interrupted at %s/%s. The current cell will be classified as interrupted until a "
            "validated completion summary is written.",
            completed_runs,
            total_requested,
        )
        raise SystemExit(130) from exc

    logging.info(
        "Finished %s requested runs in %s.",
        total_requested,
        format_seconds(time.perf_counter() - overall_start),
    )


if __name__ == "__main__":
    main()
