from __future__ import annotations

import numpy as np
import pytest

from src.decision.cost_matrix import cost_ratio_sweep
from src.decision.net_benefit import net_benefit
from src.decision.thresholds import bayes_optimal_threshold, metrics_at_threshold


def test_bayes_optimal_threshold_values() -> None:
    assert bayes_optimal_threshold(1, 1) == pytest.approx(0.5)
    assert bayes_optimal_threshold(1, 10) == pytest.approx(1 / 11)
    assert bayes_optimal_threshold(1, 20) == pytest.approx(1 / 21)


def test_metrics_at_threshold_expected_cost() -> None:
    probs = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 0, 1, 0])
    metrics = metrics_at_threshold(probs, labels, tau=0.5, c_fp=1, c_fn=10)

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["sensitivity"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["ppv"] == pytest.approx(0.5)
    assert metrics["npv"] == pytest.approx(0.5)
    assert metrics["expected_cost"] == pytest.approx((1 + 10) / 4)


def test_net_benefit_is_zero_when_no_positives_are_predicted() -> None:
    probs = np.array([0.1, 0.2, 0.3])
    labels = np.array([0, 1, 1])
    assert net_benefit(probs, labels, tau=0.9) == pytest.approx(0.0)


def test_net_benefit_at_zero_equals_positive_prevalence() -> None:
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 1, 1, 1])
    assert net_benefit(probs, labels, tau=0.0) == pytest.approx(0.75)


def test_cost_ratio_sweep_returns_expected_thresholds() -> None:
    sweep = cost_ratio_sweep([1, 2])
    assert sweep == [
        {"c_ratio": 1, "c_fp": 1.0, "c_fn": 1.0, "tau": 0.5, "label": "r=1"},
        {"c_ratio": 2, "c_fp": 1.0, "c_fn": 2.0, "tau": pytest.approx(1 / 3), "label": "r=2"},
    ]
