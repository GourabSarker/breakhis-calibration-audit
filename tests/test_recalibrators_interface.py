from __future__ import annotations

import numpy as np
import pytest

from src.calibration.recalibrators import IsotonicCalibration, PlattScaling, TemperatureScaling


@pytest.mark.parametrize(
    "recalibrator",
    [
        TemperatureScaling(),
        PlattScaling(),
        IsotonicCalibration(),
    ],
)
def test_recalibrators_share_fit_transform_params_interface(recalibrator) -> None:
    rng = np.random.default_rng(7)
    gaps = rng.normal(size=200)
    logits = np.column_stack([np.zeros_like(gaps), gaps])
    labels = rng.binomial(1, 1.0 / (1.0 + np.exp(-gaps)))

    fitted = recalibrator.fit(logits, labels)
    probs = fitted.transform(logits[:25])
    params = fitted.params

    assert fitted is recalibrator
    assert probs.shape == (25,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert isinstance(params, dict)
