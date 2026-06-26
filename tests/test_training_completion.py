from __future__ import annotations

import json
from pathlib import Path

from src.training.completion import RunExpectation, classify_training_run


EXPECTED = RunExpectation(
    architecture="custom_cnn",
    magnification=40,
    seed=0,
    fold=0,
    epochs=30,
    patience=5,
)


def _write_history(run_dir: Path, epochs: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "history.json").write_text(
        json.dumps(
            [
                {"epoch": epoch, "val_patient_loss": 1.0 / epoch}
                for epoch in range(1, epochs + 1)
            ]
        ),
        encoding="utf-8",
    )


def _write_summary(
    run_dir: Path,
    *,
    architecture: str = "custom_cnn",
    seed: int = 0,
    epochs_requested: int = 30,
    epochs_completed: int = 30,
    best_epoch: int = 30,
    stopped_early: bool = False,
    run_kind: str = "full",
) -> None:
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "completion_status": "complete",
                "run_kind": run_kind,
                "architecture": architecture,
                "magnification": 40,
                "seed": seed,
                "fold": 0,
                "epochs_requested": epochs_requested,
                "epochs_completed": epochs_completed,
                "best_epoch": best_epoch,
                "best_val_patient_loss": 0.1,
                "runtime_seconds": 1.0,
                "checkpoint_path": str(
                    Path("outputs/checkpoints")
                    / architecture
                    / "40X"
                    / f"seed{seed}_fold0"
                    / "best.pt"
                ),
                "history_path": str(
                    Path("outputs/checkpoints")
                    / architecture
                    / "40X"
                    / f"seed{seed}_fold0"
                    / "history.json"
                ),
                "stopped_early": stopped_early,
            }
        ),
        encoding="utf-8",
    )


def test_checkpoint_without_summary_is_incomplete(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best.pt").touch()

    result = classify_training_run(run_dir, EXPECTED)

    assert result.status == "incomplete"


def test_smoke_checkpoint_is_not_a_main_grid_completion(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_history(run_dir, 2)
    (run_dir / "best.pt").touch()
    _write_summary(
        run_dir,
        epochs_requested=2,
        epochs_completed=2,
        best_epoch=2,
        run_kind="smoke",
    )

    result = classify_training_run(run_dir, EXPECTED)

    assert result.status == "smoke/diagnostic"


def test_interrupted_history_without_summary_is_interrupted(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_history(run_dir, 10)
    (run_dir / "best.pt").touch()

    result = classify_training_run(run_dir, EXPECTED)

    assert result.status == "interrupted"


def test_valid_completed_run_is_complete(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_history(run_dir, 30)
    (run_dir / "best.pt").touch()
    _write_summary(run_dir)

    result = classify_training_run(run_dir, EXPECTED)

    assert result.status == "complete"


def test_mismatched_cell_metadata_is_invalid(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_history(run_dir, 30)
    (run_dir / "best.pt").touch()
    _write_summary(run_dir, architecture="resnet50", seed=2)

    result = classify_training_run(run_dir, EXPECTED)

    assert result.status == "invalid metadata"
