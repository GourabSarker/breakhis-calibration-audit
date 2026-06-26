from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.train import (
    build_class_weights,
    checkpoint_path_for_run,
    load_fold_assignments,
    magnification_tag,
    normalize_magnification,
    resolve_device,
)


def test_training_helpers_normalize_magnification_and_paths() -> None:
    assert normalize_magnification("40X") == 40
    assert normalize_magnification(200) == 200
    assert magnification_tag(40) == "40X"
    path = checkpoint_path_for_run(Path("outputs/checkpoints"), "custom_cnn", 40, 0, 1)
    assert path.as_posix().endswith("outputs/checkpoints/custom_cnn/40X/seed0_fold1/best.pt")


def test_build_class_weights_uses_inverse_frequency(synthetic_metadata_df) -> None:
    patient_ids = synthetic_metadata_df["patient_id"].unique().tolist()[:30]
    weights = build_class_weights(synthetic_metadata_df, patient_ids, 100)
    benign_count = 25
    malignant_count = 5
    assert weights.tolist() == pytest.approx([1.0 / benign_count, 1.0 / malignant_count])


def test_load_fold_assignments_reads_expected_split(tmp_path: Path) -> None:
    folds_path = tmp_path / "folds.json"
    folds_path.write_text(
        json.dumps({"0": {"1": {"train": ["p1"], "cal": ["p2"], "test": ["p3"]}}}),
        encoding="utf-8",
    )
    split = load_fold_assignments(folds_path, seed=0, fold=1)
    assert split == {"train": ["p1"], "cal": ["p2"], "test": ["p3"]}


def test_resolve_device_returns_supported_backend() -> None:
    assert resolve_device().type in {"cpu", "mps"}
