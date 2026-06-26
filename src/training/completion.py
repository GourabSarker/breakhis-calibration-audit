"""Validation and classification of persisted training runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RunStatus = Literal[
    "complete",
    "incomplete",
    "interrupted",
    "smoke/diagnostic",
    "invalid metadata",
]


def _normalize_magnification(magnification: int | str) -> int:
    return int(str(magnification).upper().removesuffix("X"))


def _magnification_tag(magnification: int | str) -> str:
    return f"{_normalize_magnification(magnification)}X"


@dataclass(frozen=True, slots=True)
class RunExpectation:
    architecture: str
    magnification: int
    seed: int
    fold: int
    epochs: int
    patience: int


@dataclass(frozen=True, slots=True)
class RunClassification:
    status: RunStatus
    reason: str
    summary: dict | None = None

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc


def classify_training_run(run_dir: Path, expected: RunExpectation) -> RunClassification:
    """Classify a run without treating checkpoint existence as completion."""
    checkpoint_path = run_dir / "best.pt"
    history_path = run_dir / "history.json"
    summary_path = run_dir / "summary.json"

    present = [path.name for path in (checkpoint_path, history_path, summary_path) if path.exists()]
    if not present:
        return RunClassification("incomplete", "no checkpoint, history, or summary exists")

    if not summary_path.exists():
        if history_path.exists():
            return RunClassification("interrupted", "history exists but summary.json is missing")
        return RunClassification("incomplete", "checkpoint exists but summary.json is missing")

    try:
        summary = _load_json(summary_path)
    except ValueError as exc:
        return RunClassification("invalid metadata", str(exc))
    if not isinstance(summary, dict):
        return RunClassification("invalid metadata", "summary.json must contain a JSON object")

    completion_status = summary.get("completion_status")
    if completion_status in {"running", "interrupted"}:
        return RunClassification("interrupted", f"completion_status={completion_status}", summary)
    if completion_status not in {None, "complete"}:
        return RunClassification(
            "invalid metadata",
            f"unsupported completion_status={completion_status!r}",
            summary,
        )

    run_kind = summary.get("run_kind")
    if run_kind not in {None, "full"}:
        return RunClassification("smoke/diagnostic", f"run_kind={run_kind!r}", summary)

    identity = {
        "architecture": expected.architecture,
        "magnification": _normalize_magnification(expected.magnification),
        "seed": expected.seed,
        "fold": expected.fold,
    }
    mismatches = [
        f"{key}: expected {value!r}, found {summary.get(key)!r}"
        for key, value in identity.items()
        if summary.get(key) != value
    ]
    if mismatches:
        return RunClassification("invalid metadata", "; ".join(mismatches), summary)

    epochs_requested = summary.get("epochs_requested")
    if epochs_requested != expected.epochs:
        return RunClassification(
            "smoke/diagnostic",
            f"expected {expected.epochs} requested epochs, found {epochs_requested!r}",
            summary,
        )

    if not checkpoint_path.exists():
        return RunClassification("incomplete", "summary exists but best.pt is missing", summary)
    if not history_path.exists():
        return RunClassification("incomplete", "summary exists but history.json is missing", summary)

    try:
        history = _load_json(history_path)
    except ValueError as exc:
        return RunClassification("invalid metadata", str(exc), summary)
    if not isinstance(history, list) or not history:
        return RunClassification("invalid metadata", "history.json must contain at least one epoch", summary)

    epochs_completed = summary.get("epochs_completed")
    best_epoch = summary.get("best_epoch")
    stopped_early = summary.get("stopped_early")
    if not isinstance(epochs_completed, int) or epochs_completed != len(history):
        return RunClassification(
            "invalid metadata",
            f"epochs_completed={epochs_completed!r} does not match history length {len(history)}",
            summary,
        )
    history_epochs = [record.get("epoch") for record in history if isinstance(record, dict)]
    if history_epochs != list(range(1, epochs_completed + 1)):
        return RunClassification("invalid metadata", "history epochs are not contiguous from 1", summary)
    if not isinstance(best_epoch, int) or not 1 <= best_epoch <= epochs_completed:
        return RunClassification("invalid metadata", f"invalid best_epoch={best_epoch!r}", summary)
    if not math.isfinite(float(summary.get("best_val_patient_loss", math.nan))):
        return RunClassification("invalid metadata", "best_val_patient_loss is not finite", summary)

    if epochs_completed == expected.epochs:
        if stopped_early is not False:
            return RunClassification(
                "invalid metadata",
                "full-length run must have stopped_early=false",
                summary,
            )
    else:
        if stopped_early is not True:
            return RunClassification(
                "interrupted",
                "run ended before requested epochs without early-stopping evidence",
                summary,
            )
        if epochs_completed - best_epoch < expected.patience:
            return RunClassification(
                "invalid metadata",
                "early-stopped run lacks the configured patience after its best epoch",
                summary,
            )

    checkpoint_record = Path(str(summary.get("checkpoint_path", "")))
    history_record = Path(str(summary.get("history_path", "")))
    expected_checkpoint_suffix = Path(expected.architecture) / _magnification_tag(
        expected.magnification
    ) / f"seed{expected.seed}_fold{expected.fold}" / "best.pt"
    expected_history_suffix = expected_checkpoint_suffix.with_name("history.json")
    if not str(checkpoint_record).endswith(str(expected_checkpoint_suffix)):
        return RunClassification("invalid metadata", "checkpoint_path does not match the cell", summary)
    if not str(history_record).endswith(str(expected_history_suffix)):
        return RunClassification("invalid metadata", "history_path does not match the cell", summary)

    return RunClassification("complete", "validated completion record", summary)
