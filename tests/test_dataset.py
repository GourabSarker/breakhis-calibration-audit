from __future__ import annotations

import torch

from src.data.dataset import BreakHisDataset
from src.data.transforms import build_eval_transform


def test_breakhis_dataset_filters_by_patient_and_magnification(toy_image_metadata) -> None:
    dataset = BreakHisDataset(
        metadata_df=toy_image_metadata,
        patient_ids=["SOB_B_A_14-10000", "SOB_M_DC_14-20000"],
        magnification=100,
        transform=build_eval_transform(),
    )

    assert len(dataset) == 2

    image, label, patient_id = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert label in {0, 1}
    assert patient_id in {"SOB_B_A_14-10000", "SOB_M_DC_14-20000"}
