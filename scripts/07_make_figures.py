#!/usr/bin/env python3
"""Regenerate Phase 7 figures and publication tables from cached outputs."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from src.analysis.plots import (
    plot_decision_curves,
    plot_generalization_gap,
    plot_reliability_diagrams_grid,
    plot_temperature_distribution,
)
from src.analysis.tables import (
    build_brier_decomposition_table,
    build_main_table,
    build_significance_table,
)
from src.training.train import format_seconds
from src.utils.logging import configure_logging
from src.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-results",
        type=Path,
        default=Path("outputs/results/calibration_results.csv"),
    )
    parser.add_argument(
        "--bootstrap-cis",
        type=Path,
        default=Path("outputs/results/bootstrap_cis.csv"),
    )
    parser.add_argument(
        "--decision-results",
        type=Path,
        default=Path("outputs/results/decision_results.csv"),
    )
    parser.add_argument(
        "--architecture-significance",
        type=Path,
        default=Path("outputs/results/architecture_significance.csv"),
    )
    parser.add_argument(
        "--brier-summary",
        type=Path,
        default=Path("outputs/results/brier_decomposition_summary.csv"),
    )
    parser.add_argument("--logit-root", type=Path, default=Path("outputs/logits"))
    parser.add_argument(
        "--decision-curve-root",
        type=Path,
        default=Path("outputs/results/decision_curves"),
    )
    parser.add_argument("--figure-root", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--table-root", type=Path, default=None)
    parser.add_argument("--curve-bootstrap-B", type=int, default=300)
    parser.add_argument("--table-bootstrap-B", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-file", type=Path, default=Path("outputs/results/make_figures.log"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(log_path=args.log_file)
    set_global_seed(args.seed)
    start = time.perf_counter()

    figure_root = args.figure_root
    table_root = args.table_root if args.table_root is not None else figure_root / "tables"
    figure_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    generated.extend(
        plot_reliability_diagrams_grid(
            args.calibration_results,
            args.logit_root,
            figure_root / "fig1_reliability_grid",
        )
    )
    logging.info("Generated Figure 1")
    generated.extend(
        plot_temperature_distribution(
            args.calibration_results,
            args.architecture_significance,
            figure_root / "fig2_temperature_distribution",
        )
    )
    logging.info("Generated Figure 2")
    generated.extend(
        plot_decision_curves(
            args.decision_curve_root,
            args.logit_root,
            figure_root / "fig3_decision_curves",
            bootstrap_B=args.curve_bootstrap_B,
            seed=args.seed,
        )
    )
    logging.info("Generated Figure 3")
    generated.extend(
        plot_generalization_gap(
            args.calibration_results,
            figure_root / "fig4_generalization_gap",
        )
    )
    logging.info("Generated Figure 4")

    _table, paths = build_main_table(
        args.bootstrap_cis,
        args.decision_results,
        table_root,
        logit_root=args.logit_root,
        B=args.table_bootstrap_B,
        seed=args.seed,
    )
    generated.extend(paths)
    logging.info("Generated Table 1")
    _table, paths = build_significance_table(args.architecture_significance, table_root)
    generated.extend(paths)
    logging.info("Generated Table 2")
    _table, paths = build_brier_decomposition_table(args.brier_summary, table_root)
    generated.extend(paths)
    logging.info("Generated Table 3")

    for path in generated:
        logging.info("Wrote %s", path)
    logging.info("Generated %s files | total_wall=%s", len(generated), format_seconds(time.perf_counter() - start))


if __name__ == "__main__":
    main()
