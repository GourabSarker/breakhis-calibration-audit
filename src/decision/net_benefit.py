"""Decision curve analysis net-benefit utilities."""

from __future__ import annotations

import numpy as np

from src.decision.thresholds import _validate_inputs


def _validate_tau(tau: float) -> None:
    if not 0.0 <= tau < 1.0:
        raise ValueError("tau must be in [0, 1).")


def net_benefit(probs, labels, tau: float) -> float:
    """Compute Vickers and Elkin (2006) net benefit at threshold tau."""
    _validate_tau(tau)
    probs_array, labels_array = _validate_inputs(probs, labels)
    predictions = (probs_array >= tau).astype(np.int64)
    tp = np.sum((predictions == 1) & (labels_array == 1))
    fp = np.sum((predictions == 1) & (labels_array == 0))
    n = len(labels_array)
    return float((tp / n) - (fp / n) * (tau / (1.0 - tau)))


def treat_all_net_benefit(labels, tau: float) -> float:
    """Reference net benefit for treating every sample as positive."""
    _validate_tau(tau)
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels_array.size == 0:
        raise ValueError("labels must not be empty.")
    prevalence = float(labels_array.mean())
    return float(prevalence - (1.0 - prevalence) * (tau / (1.0 - tau)))


def treat_none_net_benefit(labels, tau: float) -> float:
    """Reference net benefit for treating every sample as negative."""
    _validate_tau(tau)
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels_array.size == 0:
        raise ValueError("labels must not be empty.")
    return 0.0


def decision_curve(probs, labels, taus) -> np.ndarray:
    """Return a two-column array of tau and model net benefit values."""
    taus_array = np.asarray(taus, dtype=np.float64).reshape(-1)
    values = np.asarray([net_benefit(probs, labels, tau) for tau in taus_array])
    return np.column_stack([taus_array, values])


def reference_decision_curves(labels, taus) -> dict[str, np.ndarray]:
    """Return treat-all and treat-none reference curves as two-column arrays."""
    taus_array = np.asarray(taus, dtype=np.float64).reshape(-1)
    treat_all = np.asarray([treat_all_net_benefit(labels, tau) for tau in taus_array])
    treat_none = np.asarray([treat_none_net_benefit(labels, tau) for tau in taus_array])
    return {
        "treat_all": np.column_stack([taus_array, treat_all]),
        "treat_none": np.column_stack([taus_array, treat_none]),
    }
