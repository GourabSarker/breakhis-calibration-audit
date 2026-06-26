"""Small IO helpers shared across scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for a file path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Path, *, indent: int = 2) -> None:
    """Persist JSON with deterministic formatting."""
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, sort_keys=True)
        handle.write("\n")


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    """Append one JSON object per line to a JSONL file."""
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")
