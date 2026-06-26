"""PyTorch dataset wrapper for BreakHis."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BreakHisDataset(Dataset):
    """Filter BreakHis metadata by patient ids and magnification."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        patient_ids: list[str] | set[str] | tuple[str, ...],
        magnification: int | str,
        transform: Callable | None = None,
    ) -> None:
        if not patient_ids:
            raise ValueError("patient_ids must not be empty.")

        mag_value = int(str(magnification).upper().removesuffix("X"))
        patient_id_set = set(patient_ids)
        mask = metadata_df["patient_id"].isin(patient_id_set) & (metadata_df["magnification"] == mag_value)
        self.metadata = metadata_df.loc[mask].sort_values(
            by=["patient_id", "seq", "image_path"]
        ).reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        image = Image.open(Path(row["image_path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"]), row["patient_id"]
