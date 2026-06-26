from __future__ import annotations

from src.data.folds import assert_no_patient_leakage, build_patient_folds


def test_every_seed_and_fold_has_zero_patient_overlap(synthetic_metadata_df) -> None:
    folds = build_patient_folds(synthetic_metadata_df, k=5, n_seeds=3)
    assert_no_patient_leakage(folds)

    for seed_folds in folds.values():
        for split in seed_folds.values():
            train = set(split["train"])
            cal = set(split["cal"])
            test = set(split["test"])
            assert not (train & cal)
            assert not (train & test)
            assert not (cal & test)
