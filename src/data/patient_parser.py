"""BreakHis patient- and image-level metadata parsing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

CLASS_CODE_TO_NAME = {
    "B": "benign",
    "M": "malignant",
}

SUBTYPE_CODE_TO_NAME = {
    "A": "adenosis",
    "F": "fibroadenoma",
    "PT": "phyllodes_tumor",
    "TA": "tubular_adenoma",
    "DC": "ductal_carcinoma",
    "LC": "lobular_carcinoma",
    "MC": "mucinous_carcinoma",
    "PC": "papillary_carcinoma",
}

FILENAME_RE = re.compile(
    r"^(?P<prefix>SOB_(?P<class_code>[BM])_(?P<subtype_code>A|F|PT|TA|DC|LC|MC|PC))"
    r"-(?P<year>\d+)-(?P<slide>[A-Za-z0-9]+)-(?P<magnification>\d+)-(?P<seq>\d+)$"
)

SAMPLE_DIR_RE = re.compile(
    r"^(?P<prefix>SOB_(?P<class_code>[BM])_(?P<subtype_code>A|F|PT|TA|DC|LC|MC|PC))"
    r"_(?P<year>\d+)-(?P<slide>[A-Za-z0-9]+)$"
)


def resolve_breakhis_root(data_root: str | Path) -> Path:
    """Resolve either ``data/raw`` or ``data/raw/BreaKHis_v1`` to the dataset root."""
    candidate = Path(data_root)
    if candidate.name == "BreaKHis_v1":
        return candidate
    return candidate / "BreaKHis_v1"


def parse_sample(path: Path) -> dict[str, Any]:
    """Parse a BreakHis image path into audit-ready metadata."""
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError(f"Expected a PNG image path, received: {path}")

    try:
        class_dir = path.parts[-6]
        sob_dir = path.parts[-5]
        subtype_dir = path.parts[-4]
        patient_id = path.parts[-3]
        magnification_dir = path.parts[-2]
    except IndexError as exc:
        raise ValueError(f"Path is too short to be a BreakHis sample: {path}") from exc

    if sob_dir != "SOB":
        raise ValueError(f"Expected 'SOB' directory in BreakHis path: {path}")

    filename_match = FILENAME_RE.match(path.stem)
    if filename_match is None:
        raise ValueError(f"Filename does not match BreakHis convention: {path.name}")

    sample_match = SAMPLE_DIR_RE.match(patient_id)
    if sample_match is None:
        raise ValueError(f"Sample directory does not match BreakHis convention: {patient_id}")

    class_code = filename_match.group("class_code")
    subtype_code = filename_match.group("subtype_code")
    expected_class = CLASS_CODE_TO_NAME[class_code]
    expected_subtype = SUBTYPE_CODE_TO_NAME[subtype_code]
    expected_patient_id = (
        f"{filename_match.group('prefix')}_{filename_match.group('year')}-{filename_match.group('slide')}"
    )
    slide_id = f"{filename_match.group('year')}-{filename_match.group('slide')}"
    magnification = int(filename_match.group("magnification"))

    if class_dir != expected_class:
        raise ValueError(
            f"Class directory '{class_dir}' disagrees with filename class '{expected_class}' for {path}"
        )

    if subtype_dir != expected_subtype:
        raise ValueError(
            f"Subtype directory '{subtype_dir}' disagrees with filename subtype '{expected_subtype}' for {path}"
        )

    if patient_id != expected_patient_id:
        raise ValueError(
            f"Patient directory '{patient_id}' disagrees with filename-derived patient id "
            f"'{expected_patient_id}' for {path}"
        )

    if magnification_dir.upper() != f"{magnification}X":
        raise ValueError(
            f"Magnification directory '{magnification_dir}' disagrees with filename magnification "
            f"'{magnification}' for {path}"
        )

    return {
        "patient_id": patient_id,
        "slide_id": slide_id,
        "class": expected_class,
        "subtype": expected_subtype,
        "magnification": magnification,
        "seq": int(filename_match.group("seq")),
    }


def derive_patient_ids_from_directory_tree(breakhis_root: str | Path) -> set[str]:
    """Extract patient ids from the sample-directory level of the BreakHis tree."""
    dataset_root = resolve_breakhis_root(breakhis_root)
    slides_root = dataset_root / "histology_slides" / "breast"
    patient_dirs = slides_root.glob("*/*/*/*")
    patient_ids = {path.name for path in patient_dirs if path.is_dir() and SAMPLE_DIR_RE.match(path.name)}
    if not patient_ids:
        raise FileNotFoundError(f"No patient directories found under: {slides_root}")
    return patient_ids


def collect_metadata(breakhis_root: str | Path) -> pd.DataFrame:
    """Scan the BreakHis tree and build image-level metadata."""
    dataset_root = resolve_breakhis_root(breakhis_root)
    slides_root = dataset_root / "histology_slides" / "breast"
    if not slides_root.exists():
        raise FileNotFoundError(f"BreakHis slides directory not found: {slides_root}")

    rows: list[dict[str, Any]] = []
    for image_path in sorted(slides_root.rglob("*.png")):
        parsed = parse_sample(image_path)
        rows.append(
            {
                **parsed,
                "image_path": str(image_path.resolve()),
                "label": 1 if parsed["class"] == "malignant" else 0,
            }
        )

    if not rows:
        raise FileNotFoundError(f"No PNG files found under: {slides_root}")

    return pd.DataFrame(rows).sort_values(
        by=["patient_id", "magnification", "seq", "image_path"]
    ).reset_index(drop=True)
