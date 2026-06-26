from __future__ import annotations

from pathlib import Path

import pytest

from src.data.patient_parser import parse_sample


@pytest.mark.parametrize(
    ("path_str", "expected_class", "expected_subtype", "expected_patient_id", "expected_slide_id"),
    [
        (
            "BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/"
            "SOB_B_A_14-22549AB/100X/SOB_B_A-14-22549AB-100-011.png",
            "benign",
            "adenosis",
            "SOB_B_A_14-22549AB",
            "14-22549AB",
        ),
        (
            "BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/"
            "SOB_B_F_14-29960AB/200X/SOB_B_F-14-29960AB-200-009.png",
            "benign",
            "fibroadenoma",
            "SOB_B_F_14-29960AB",
            "14-29960AB",
        ),
        (
            "BreaKHis_v1/histology_slides/breast/benign/SOB/phyllodes_tumor/"
            "SOB_B_PT_14-22704/400X/SOB_B_PT-14-22704-400-028.png",
            "benign",
            "phyllodes_tumor",
            "SOB_B_PT_14-22704",
            "14-22704",
        ),
        (
            "BreaKHis_v1/histology_slides/breast/benign/SOB/tubular_adenoma/"
            "SOB_B_TA_14-16184CD/40X/SOB_B_TA-14-16184CD-40-029.png",
            "benign",
            "tubular_adenoma",
            "SOB_B_TA_14-16184CD",
            "14-16184CD",
        ),
        (
            "BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/"
            "SOB_M_DC_14-11951/100X/SOB_M_DC-14-11951-100-017.png",
            "malignant",
            "ductal_carcinoma",
            "SOB_M_DC_14-11951",
            "14-11951",
        ),
        (
            "BreaKHis_v1/histology_slides/breast/malignant/SOB/lobular_carcinoma/"
            "SOB_M_LC_14-16196/200X/SOB_M_LC-14-16196-200-003.png",
            "malignant",
            "lobular_carcinoma",
            "SOB_M_LC_14-16196",
            "14-16196",
        ),
        (
            "BreaKHis_v1/histology_slides/breast/malignant/SOB/mucinous_carcinoma/"
            "SOB_M_MC_14-13418DE/400X/SOB_M_MC-14-13418DE-400-009.png",
            "malignant",
            "mucinous_carcinoma",
            "SOB_M_MC_14-13418DE",
            "14-13418DE",
        ),
        (
            "BreaKHis_v1/histology_slides/breast/malignant/SOB/papillary_carcinoma/"
            "SOB_M_PC_15-190EF/40X/SOB_M_PC-15-190EF-40-009.png",
            "malignant",
            "papillary_carcinoma",
            "SOB_M_PC_15-190EF",
            "15-190EF",
        ),
    ],
)
def test_parse_sample_covers_all_breakhis_subtypes(
    path_str: str,
    expected_class: str,
    expected_subtype: str,
    expected_patient_id: str,
    expected_slide_id: str,
) -> None:
    parsed = parse_sample(Path(path_str))
    assert parsed["class"] == expected_class
    assert parsed["subtype"] == expected_subtype
    assert parsed["patient_id"] == expected_patient_id
    assert parsed["slide_id"] == expected_slide_id
    assert parsed["magnification"] in {40, 100, 200, 400}
    assert parsed["seq"] > 0


def test_parse_sample_keeps_duplicate_slide_ids_separate_by_patient_directory() -> None:
    ductal_path = Path(
        "BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/"
        "SOB_M_DC_14-13412/100X/SOB_M_DC-14-13412-100-001.png"
    )
    lobular_path = Path(
        "BreaKHis_v1/histology_slides/breast/malignant/SOB/lobular_carcinoma/"
        "SOB_M_LC_14-13412/100X/SOB_M_LC-14-13412-100-001.png"
    )

    ductal = parse_sample(ductal_path)
    lobular = parse_sample(lobular_path)

    assert ductal["slide_id"] == lobular["slide_id"] == "14-13412"
    assert ductal["patient_id"] != lobular["patient_id"]
