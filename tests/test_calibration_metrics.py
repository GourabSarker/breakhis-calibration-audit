from __future__ import annotations

import numpy as np
import pytest

from src.calibration.metrics import (
    brier_decomposition,
    brier_decomposition_direct,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)


def test_brier_decomposition_identity_holds_on_random_data() -> None:
    rng = np.random.default_rng(123)
    for _ in range(1000):
        probs = rng.uniform(0.001, 0.999, size=250)
        labels = rng.binomial(1, probs)
        decomposition = brier_decomposition(probs, labels, n_bins=15)
        reconstructed = (
            decomposition["reliability"]
            - decomposition["resolution"]
            + decomposition["uncertainty"]
        )
        assert abs(decomposition["brier"] - reconstructed) < 1e-8


def test_brier_decomposition_matches_direct_murphy_for_discrete_forecaster() -> None:
    rng = np.random.default_rng(456)
    probs = rng.choice(np.array([0.1, 0.3, 0.7, 0.9]), size=10_000)
    labels = rng.binomial(1, 0.55, size=10_000)
    n_bins = 10

    base_rate = labels.mean()
    expected_rel = 0.0
    expected_res = 0.0
    for forecast_value in (0.1, 0.3, 0.7, 0.9):
        mask = probs == forecast_value
        empirical_rate = labels[mask].mean()
        expected_rel += (mask.sum() / len(labels)) * (forecast_value - empirical_rate) ** 2
        expected_res += (mask.sum() / len(labels)) * (empirical_rate - base_rate) ** 2
    expected_unc = base_rate * (1.0 - base_rate)
    expected_brier = np.mean((probs - labels) ** 2)

    for decomposition in (
        brier_decomposition(probs, labels, n_bins=n_bins),
        brier_decomposition_direct(probs, labels, n_bins=n_bins),
    ):
        assert decomposition["reliability"] == pytest.approx(expected_rel, abs=1e-6)
        assert decomposition["resolution"] == pytest.approx(expected_res, abs=1e-6)
        assert decomposition["uncertainty"] == pytest.approx(expected_unc, abs=1e-6)
        assert decomposition["brier"] == pytest.approx(expected_brier, abs=1e-6)


def test_ece_is_zero_on_perfectly_calibrated_max_confidence_bins() -> None:
    probs = np.array([0.8] * 5 + [0.2] * 5)
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0)


def test_brier_score_matches_manual_computation() -> None:
    probs = np.array([0.1, 0.7, 0.8, 0.2])
    labels = np.array([0, 1, 1, 0])
    expected = ((0.1 - 0) ** 2 + (0.7 - 1) ** 2 + (0.8 - 1) ** 2 + (0.2 - 0) ** 2) / 4
    assert brier_score(probs, labels) == pytest.approx(expected)


def test_negative_log_likelihood_matches_manual_computation() -> None:
    probs = np.array([0.25, 0.75])
    labels = np.array([0, 1])
    expected = -0.5 * (np.log(1.0 - 0.25) + np.log(0.75))
    assert negative_log_likelihood(probs, labels) == pytest.approx(expected)
