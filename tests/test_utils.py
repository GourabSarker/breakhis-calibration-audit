from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from src.utils.io import append_jsonl, save_json
from src.utils.logging import configure_logging
from src.utils.seeding import set_global_seed


def test_set_global_seed_is_reproducible() -> None:
    set_global_seed(123)
    python_draw_1 = random.random()
    numpy_draw_1 = np.random.rand(3)
    torch_draw_1 = torch.rand(3)

    set_global_seed(123)
    python_draw_2 = random.random()
    numpy_draw_2 = np.random.rand(3)
    torch_draw_2 = torch.rand(3)

    assert python_draw_1 == python_draw_2
    np.testing.assert_allclose(numpy_draw_1, numpy_draw_2)
    assert torch.equal(torch_draw_1, torch_draw_2)


def test_save_json_creates_parent_directory(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "result.json"
    save_json({"beta": 2, "alpha": 1}, output_path)

    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == {"alpha": 1, "beta": 2}


def test_append_jsonl_appends_records(tmp_path: Path) -> None:
    output_path = tmp_path / "logs" / "runs.jsonl"
    append_jsonl({"run": 1}, output_path)
    append_jsonl({"run": 2}, output_path)

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line) for line in lines] == [{"run": 1}, {"run": 2}]


def test_configure_logging_is_safe_to_call() -> None:
    configure_logging()
