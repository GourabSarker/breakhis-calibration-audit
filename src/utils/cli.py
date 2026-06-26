"""Command-line safety checks shared by aggregate analysis scripts."""

from __future__ import annotations

from pathlib import Path


def require_nonaggregate_output_for_partial_run(
    *,
    only: str | None,
    output: Path,
    aggregate_output: Path,
) -> None:
    """Prevent a one-cell run from replacing a complete aggregate CSV."""
    if only is None:
        return
    if output.resolve() == aggregate_output.resolve():
        raise ValueError(
            "--only cannot write to the default aggregate output. "
            "Provide an explicit diagnostic --output path."
        )
