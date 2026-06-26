from __future__ import annotations

from pathlib import Path

import pytest

from src.data.folds import build_patient_folds, validate_patient_ids_against_folds_csv


def test_build_patient_folds_has_expected_structure_and_stratification(synthetic_metadata_df) -> None:
    folds = build_patient_folds(synthetic_metadata_df, k=5, n_seeds=3)
    expected_patients = set(synthetic_metadata_df["patient_id"].unique())
    overall_ratio = (
        synthetic_metadata_df[["patient_id", "label"]].drop_duplicates()["label"].mean()
    )

    assert len(folds) == 3
    for seed_folds in folds.values():
        assert len(seed_folds) == 5
        for split in seed_folds.values():
            union = set(split["train"]) | set(split["cal"]) | set(split["test"])
            assert union == expected_patients

            for split_name in ("train", "cal", "test"):
                split_patient_ids = split[split_name]
                split_ratio = (
                    synthetic_metadata_df.loc[
                        synthetic_metadata_df["patient_id"].isin(split_patient_ids),
                        ["patient_id", "label"],
                    ]
                    .drop_duplicates()["label"]
                    .mean()
                )
                assert abs(split_ratio - overall_ratio) <= 0.2


def test_validate_patient_ids_against_folds_csv_raises_on_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "BreaKHis_v1"
    patient_dir = (
        root
        / "histology_slides"
        / "breast"
        / "benign"
        / "SOB"
        / "adenosis"
        / "SOB_B_A_14-11111"
    )
    patient_dir.mkdir(parents=True)

    csv_path = root / "Folds.csv"
    csv_path.write_text(
        "fold,mag,grp,filename\n"
        "1,100,train,BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/"
        "SOB_B_A_14-22222/100X/SOB_B_A-14-22222-100-001.png\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not match"):
        validate_patient_ids_against_folds_csv(root)


def test_validate_patient_ids_against_real_breakhis_folds_csv_if_available() -> None:
    data_root = Path("data/raw/BreaKHis_v1")
    if not data_root.exists():
        pytest.skip("Real BreakHis dataset not available in test environment.")

    patient_ids = validate_patient_ids_against_folds_csv(data_root)
    assert patient_ids is not None
    assert len(patient_ids) == 82
