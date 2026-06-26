"""Patient-level fold construction for BreakHis."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.patient_parser import derive_patient_ids_from_directory_tree, resolve_breakhis_root


def extract_patient_ids_from_folds_csv(csv_path: str | Path) -> set[str]:
    """Read the official BreakHis ``Folds.csv`` and extract patient ids."""
    csv_path = Path(csv_path)
    patient_ids: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_ids.add(Path(row["filename"]).parent.parent.name)
    return patient_ids


def validate_patient_ids_against_folds_csv(breakhis_root: str | Path) -> set[str] | None:
    """Ensure filesystem patient ids match the official CSV grouping if it is available."""
    dataset_root = resolve_breakhis_root(breakhis_root)
    csv_path = dataset_root / "Folds.csv"
    if not csv_path.exists():
        return None

    csv_patient_ids = extract_patient_ids_from_folds_csv(csv_path)
    directory_patient_ids = derive_patient_ids_from_directory_tree(dataset_root)

    if csv_patient_ids != directory_patient_ids:
        missing_from_csv = sorted(directory_patient_ids - csv_patient_ids)
        missing_from_dirs = sorted(csv_patient_ids - directory_patient_ids)
        raise ValueError(
            "Folds.csv patient ids do not match the directory-derived ids. "
            f"Missing from CSV: {missing_from_csv}. Missing from directories: {missing_from_dirs}."
        )

    return csv_patient_ids


def _patient_level_metadata(metadata_df: pd.DataFrame, stratify_by: str) -> pd.DataFrame:
    required_columns = {"patient_id", stratify_by}
    missing = required_columns - set(metadata_df.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    patient_df = metadata_df[["patient_id", stratify_by]].drop_duplicates().sort_values("patient_id")
    duplicated = patient_df["patient_id"].duplicated().any()
    if duplicated:
        raise ValueError("Patient-level metadata contains conflicting rows after deduplication.")
    return patient_df.reset_index(drop=True)


def build_patient_folds(
    metadata_df: pd.DataFrame,
    k: int = 5,
    n_seeds: int = 3,
    stratify_by: str = "class",
    seed_offset: int = 0,
) -> dict[int, dict[int, dict[str, list[str]]]]:
    """Build deterministic patient-level train/cal/test splits.

    The split proportions are approximately ``(k-2)/k`` train, ``1/k`` calibration,
    and ``1/k`` test. For the default ``k=5`` this yields 60/20/20 splits.
    """
    if k < 3:
        raise ValueError("k must be at least 3 to create train/cal/test partitions.")
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1.")

    patient_df = _patient_level_metadata(metadata_df, stratify_by=stratify_by)
    folds: dict[int, dict[int, dict[str, list[str]]]] = {}

    features = np.zeros((len(patient_df), 1), dtype=np.float32)
    labels = patient_df[stratify_by].to_numpy()
    groups = patient_df["patient_id"].to_numpy()

    for seed_idx in range(n_seeds):
        actual_seed = seed_offset + seed_idx
        outer_splitter = StratifiedGroupKFold(
            n_splits=k,
            shuffle=True,
            random_state=actual_seed,
        )
        folds[actual_seed] = {}
        outer_splits = list(outer_splitter.split(features, labels, groups))

        for fold_idx, (train_val_idx, test_idx) in enumerate(outer_splits):
            train_val_df = patient_df.iloc[train_val_idx].reset_index(drop=True)
            inner_features = np.zeros((len(train_val_df), 1), dtype=np.float32)
            inner_labels = train_val_df[stratify_by].to_numpy()
            inner_groups = train_val_df["patient_id"].to_numpy()

            inner_splitter = StratifiedGroupKFold(
                n_splits=k - 1,
                shuffle=True,
                random_state=(actual_seed * 1000) + fold_idx,
            )
            inner_splits = list(inner_splitter.split(inner_features, inner_labels, inner_groups))
            train_idx, cal_idx = inner_splits[fold_idx % (k - 1)]

            folds[actual_seed][fold_idx] = {
                "train": sorted(train_val_df.iloc[train_idx]["patient_id"].tolist()),
                "cal": sorted(train_val_df.iloc[cal_idx]["patient_id"].tolist()),
                "test": sorted(patient_df.iloc[test_idx]["patient_id"].tolist()),
            }

    return folds


def assert_no_patient_leakage(folds: dict[int, dict[int, dict[str, list[str]]]]) -> None:
    """Raise if any split contains overlapping patients."""
    for seed, seed_folds in folds.items():
        for fold_idx, split in seed_folds.items():
            train = set(split["train"])
            cal = set(split["cal"])
            test = set(split["test"])

            if train & cal:
                raise AssertionError(f"Train/cal overlap for seed={seed}, fold={fold_idx}")
            if train & test:
                raise AssertionError(f"Train/test overlap for seed={seed}, fold={fold_idx}")
            if cal & test:
                raise AssertionError(f"Cal/test overlap for seed={seed}, fold={fold_idx}")


def serialize_folds(folds: dict[int, dict[int, dict[str, list[str]]]]) -> dict[str, Any]:
    """Convert fold keys to JSON-friendly strings."""
    serialized: dict[str, Any] = {}
    for seed, seed_folds in folds.items():
        serialized[str(seed)] = {}
        for fold_idx, split in seed_folds.items():
            serialized[str(seed)][str(fold_idx)] = split
    return serialized
