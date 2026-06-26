from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_run(
    output_root: Path,
    seed: int,
    *,
    summary: bool,
    epochs_requested: int = 30,
    architecture: str = "custom_cnn",
) -> None:
    run_dir = output_root / "custom_cnn" / "40X" / f"seed{seed}_fold0"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").touch()
    history = [{"epoch": epoch} for epoch in range(1, 31)]
    (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    if not summary:
        return
    run_kind = "smoke" if epochs_requested != 30 else "full"
    completed = 2 if epochs_requested == 2 else 30
    if completed == 2:
        (run_dir / "history.json").write_text(
            json.dumps([{"epoch": 1}, {"epoch": 2}]),
            encoding="utf-8",
        )
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
                "epochs_completed": completed,
                "best_epoch": completed,
                "best_val_patient_loss": 0.1,
                "runtime_seconds": 10.0,
                "checkpoint_path": str(run_dir / "best.pt"),
                "history_path": str(run_dir / "history.json"),
                "stopped_early": False,
            }
        ),
        encoding="utf-8",
    )


def test_resume_report_distinguishes_run_states(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "checkpoints"
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        "\n".join(
            [
                "architectures: [custom_cnn]",
                "magnifications: [40]",
                "seeds: [0, 1, 2, 3, 4]",
                "folds: [0]",
                "epochs: 30",
                "early_stopping_patience: 5",
            ]
        ),
        encoding="utf-8",
    )
    _write_run(output_root, 0, summary=True)
    _write_run(output_root, 2, summary=False)
    _write_run(output_root, 3, summary=True, epochs_requested=2)
    _write_run(output_root, 4, summary=True, architecture="resnet50")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "resume_report.py"),
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "complete: 1" in result.stdout
    assert "incomplete: 1" in result.stdout
    assert "interrupted: 1" in result.stdout
    assert "smoke/diagnostic: 1" in result.stdout
    assert "invalid metadata: 1" in result.stdout
