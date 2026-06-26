from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.architectures import build_model


def _write_breakhis_image(
    dataset_root: Path,
    *,
    class_name: str,
    subtype_name: str,
    patient_id: str,
    filename_stem: str,
    magnification: int,
    seq: int,
    color: tuple[int, int, int],
) -> None:
    image_dir = (
        dataset_root
        / "histology_slides"
        / "breast"
        / class_name
        / "SOB"
        / subtype_name
        / patient_id
        / f"{magnification}X"
    )
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{filename_stem}-{magnification}-{seq:03d}.png"
    Image.new("RGB", (24, 24), color=color).save(image_path)


def _make_constant_logit_checkpoint(checkpoint_path: Path) -> None:
    model = build_model("custom_cnn", num_classes=2, pretrained=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier[-1].bias.copy_(torch.tensor([-1.5, 2.5]))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    history = [
        {"epoch": epoch, "val_patient_loss": 1.0 / epoch}
        for epoch in range(1, 31)
    ]
    (checkpoint_path.parent / "history.json").write_text(
        json.dumps(history),
        encoding="utf-8",
    )
    (checkpoint_path.parent / "summary.json").write_text(
        json.dumps(
            {
                "completion_status": "complete",
                "run_kind": "full",
                "architecture": "custom_cnn",
                "magnification": 40,
                "seed": 0,
                "fold": 0,
                "epochs_requested": 30,
                "epochs_completed": 30,
                "best_epoch": 30,
                "best_val_patient_loss": 1.0 / 30,
                "runtime_seconds": 1.0,
                "checkpoint_path": str(checkpoint_path),
                "history_path": str(checkpoint_path.parent / "history.json"),
                "stopped_early": False,
            }
        ),
        encoding="utf-8",
    )


def test_cache_logits_script_writes_raw_logits_npz(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = tmp_path / "data" / "raw"
    dataset_root = data_root / "BreaKHis_v1"

    _write_breakhis_image(
        dataset_root,
        class_name="benign",
        subtype_name="adenosis",
        patient_id="SOB_B_A_14-10000",
        filename_stem="SOB_B_A-14-10000",
        magnification=40,
        seq=1,
        color=(255, 0, 0),
    )
    _write_breakhis_image(
        dataset_root,
        class_name="benign",
        subtype_name="adenosis",
        patient_id="SOB_B_A_14-10000",
        filename_stem="SOB_B_A-14-10000",
        magnification=40,
        seq=2,
        color=(200, 0, 0),
    )
    _write_breakhis_image(
        dataset_root,
        class_name="malignant",
        subtype_name="ductal_carcinoma",
        patient_id="SOB_M_DC_14-20000",
        filename_stem="SOB_M_DC-14-20000",
        magnification=40,
        seq=1,
        color=(0, 255, 0),
    )
    _write_breakhis_image(
        dataset_root,
        class_name="malignant",
        subtype_name="ductal_carcinoma",
        patient_id="SOB_M_DC_14-20000",
        filename_stem="SOB_M_DC-14-20000",
        magnification=40,
        seq=2,
        color=(0, 200, 0),
    )
    _write_breakhis_image(
        dataset_root,
        class_name="malignant",
        subtype_name="lobular_carcinoma",
        patient_id="SOB_M_LC_14-30000",
        filename_stem="SOB_M_LC-14-30000",
        magnification=40,
        seq=1,
        color=(0, 0, 255),
    )
    _write_breakhis_image(
        dataset_root,
        class_name="malignant",
        subtype_name="lobular_carcinoma",
        patient_id="SOB_M_LC_14-30000",
        filename_stem="SOB_M_LC-14-30000",
        magnification=40,
        seq=2,
        color=(0, 0, 200),
    )

    folds_path = tmp_path / "folds.json"
    folds_path.write_text(
        json.dumps(
            {
                "0": {
                    "0": {
                        "train": ["SOB_M_LC_14-30000"],
                        "cal": ["SOB_B_A_14-10000"],
                        "test": ["SOB_M_DC_14-20000"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_path = checkpoint_root / "custom_cnn" / "40X" / "seed0_fold0" / "best.pt"
    _make_constant_logit_checkpoint(checkpoint_path)

    output_root = tmp_path / "logits"
    log_file = tmp_path / "cache_logits.log"
    script_path = repo_root / "scripts" / "03_cache_logits.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data-root",
            str(data_root),
            "--folds-path",
            str(folds_path),
            "--checkpoint-root",
            str(checkpoint_root),
            "--output-root",
            str(output_root),
            "--only",
            "arch=custom_cnn,mag=40X,seed=0,fold=0",
            "--log-file",
            str(log_file),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    output_path = output_root / "custom_cnn" / "40X" / "seed0_fold0.npz"
    assert output_path.exists()

    with np.load(output_path) as cached:
        assert cached["logits_cal"].shape == (2, 2)
        assert cached["labels_cal"].shape == (2,)
        assert cached["patient_ids_cal"].shape == (2,)
        assert cached["logits_test"].shape == (2, 2)
        assert cached["labels_test"].shape == (2,)
        assert cached["patient_ids_test"].shape == (2,)

        assert set(cached["patient_ids_cal"].tolist()).isdisjoint(cached["patient_ids_test"].tolist())
        assert np.allclose(cached["logits_cal"], np.array([[-1.5, 2.5], [-1.5, 2.5]], dtype=np.float32))
        assert np.allclose(cached["logits_test"], np.array([[-1.5, 2.5], [-1.5, 2.5]], dtype=np.float32))
        assert np.any(cached["logits_cal"] < 0.0)
        assert np.any(cached["logits_cal"] > 1.0)

    assert log_file.exists()
