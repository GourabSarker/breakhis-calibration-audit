#!/usr/bin/env python3
"""Build patient-level train/cal/test folds for BreakHis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.folds import assert_no_patient_leakage, build_patient_folds, serialize_folds
from src.data.folds import validate_patient_ids_against_folds_csv
from src.data.patient_parser import collect_metadata, resolve_breakhis_root
from src.utils.io import save_json
from src.utils.logging import configure_logging
from src.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/folds/folds.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    set_global_seed(args.seed)

    breakhis_root = resolve_breakhis_root(args.data_root)
    metadata_df = collect_metadata(breakhis_root)
    csv_patient_ids = validate_patient_ids_against_folds_csv(breakhis_root)

    patient_count = metadata_df["patient_id"].nunique()
    class_counts = metadata_df[["patient_id", "class"]].drop_duplicates()["class"].value_counts().to_dict()

    if csv_patient_ids is not None and patient_count != len(csv_patient_ids):
        raise ValueError(
            "Directory metadata and Folds.csv disagree on the number of patient ids: "
            f"{patient_count} vs {len(csv_patient_ids)}"
        )

    if csv_patient_ids is not None and patient_count != 82:
        raise ValueError(f"Expected 82 patients when Folds.csv is present, found {patient_count}.")

    folds = build_patient_folds(
        metadata_df=metadata_df,
        k=args.k,
        n_seeds=args.n_seeds,
        stratify_by="class",
        seed_offset=args.seed,
    )
    assert_no_patient_leakage(folds)
    save_json(serialize_folds(folds), args.output)

    logging.info("BreakHis root: %s", breakhis_root)
    logging.info("Metadata rows: %s", len(metadata_df))
    logging.info("Patient count: %s", patient_count)
    logging.info("Patient class counts: %s", class_counts)
    logging.info("Saved folds to %s", args.output)


if __name__ == "__main__":
    main()
