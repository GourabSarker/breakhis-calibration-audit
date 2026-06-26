from pathlib import Path

import pytest

from src.utils.cli import require_nonaggregate_output_for_partial_run


def test_partial_run_cannot_replace_default_aggregate_output(tmp_path: Path) -> None:
    aggregate = tmp_path / "results.csv"

    with pytest.raises(ValueError, match="--only"):
        require_nonaggregate_output_for_partial_run(
            only="arch=custom_cnn,mag=40X,seed=0,fold=0",
            output=aggregate,
            aggregate_output=aggregate,
        )


def test_partial_run_can_use_an_explicit_diagnostic_output(tmp_path: Path) -> None:
    require_nonaggregate_output_for_partial_run(
        only="arch=custom_cnn,mag=40X,seed=0,fold=0",
        output=tmp_path / "diagnostic.csv",
        aggregate_output=tmp_path / "results.csv",
    )
