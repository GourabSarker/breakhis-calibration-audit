from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def synthetic_metadata_df() -> pd.DataFrame:
    rows = []
    for idx in range(50):
        class_name = "benign" if idx < 25 else "malignant"
        patient_id = f"SOB_{'B' if class_name == 'benign' else 'M'}_X_{idx:03d}"
        for magnification in (100, 200):
            rows.append(
                {
                    "patient_id": patient_id,
                    "slide_id": f"14-{idx:05d}",
                    "class": class_name,
                    "subtype": "synthetic_subtype",
                    "magnification": magnification,
                    "seq": 1,
                    "image_path": f"/tmp/{patient_id}-{magnification}.png",
                    "label": 0 if class_name == "benign" else 1,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def toy_image_metadata(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for patient_id, label, magnification in (
        ("SOB_B_A_14-10000", 0, 100),
        ("SOB_B_A_14-10000", 0, 200),
        ("SOB_M_DC_14-20000", 1, 100),
    ):
        image_path = tmp_path / f"{patient_id}-{magnification}.png"
        Image.new("RGB", (32, 32), color=(120, 30, 200)).save(image_path)
        rows.append(
            {
                "patient_id": patient_id,
                "slide_id": patient_id.split("_")[-1],
                "class": "malignant" if label else "benign",
                "subtype": "synthetic_subtype",
                "magnification": magnification,
                "seq": 1,
                "image_path": str(image_path),
                "label": label,
            }
        )
    return pd.DataFrame(rows)
