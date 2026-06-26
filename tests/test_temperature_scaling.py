from __future__ import annotations

import numpy as np

from src.calibration.metrics import expected_calibration_error
from src.calibration.recalibrators import TemperatureScaling, softmax_positive_probability


def test_temperature_scaling_reduces_ece_on_overconfident_synthetic_logits() -> None:
    rng = np.random.default_rng(42)
    true_gap = rng.normal(loc=0.0, scale=1.0, size=5000)
    true_probs = 1.0 / (1.0 + np.exp(-true_gap))
    labels = rng.binomial(1, true_probs)

    overconfident_gap = 3.0 * true_gap
    logits = np.column_stack([np.zeros_like(overconfident_gap), overconfident_gap])

    pre_probs = softmax_positive_probability(logits)
    recalibrator = TemperatureScaling().fit(logits, labels)
    post_probs = recalibrator.transform(logits)

    assert recalibrator.T is not None
    assert recalibrator.T > 1.0
    assert expected_calibration_error(post_probs, labels) < expected_calibration_error(pre_probs, labels)

    fisher_info = recalibrator.fisher_information()
    params = recalibrator.params
    assert fisher_info > 0.0
    assert params["T_ci_lo"] <= recalibrator.T <= params["T_ci_hi"]
