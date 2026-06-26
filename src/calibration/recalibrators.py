"""Post-hoc recalibration methods for cached BreakHis logits."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from src.calibration.metrics import negative_log_likelihood


def logit_gap(logits) -> np.ndarray:
    array = np.asarray(logits, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("logits must have shape (n, 2).")
    return array[:, 1] - array[:, 0]


def softmax_positive_probability(logits) -> np.ndarray:
    """Compute P(Y=1) for a two-logit classifier without mutating logits."""
    return expit(logit_gap(logits))


class Recalibrator(ABC):
    @abstractmethod
    def fit(self, logits_cal, labels_cal):
        """Fit the recalibrator on calibration logits and labels."""

    @abstractmethod
    def transform(self, logits_new) -> np.ndarray:
        """Return positive-class probabilities for new logits."""

    @property
    @abstractmethod
    def params(self) -> dict:
        """Return fitted parameters."""


class TemperatureScaling(Recalibrator):
    """Single-scalar temperature scaling for two-class softmax logits."""

    def __init__(self, bounds: tuple[float, float] = (0.05, 20.0)) -> None:
        self.bounds = bounds
        self.T: float | None = None
        self.pre_nll: float | None = None
        self.post_nll: float | None = None
        self._fisher_info: float | None = None
        self._ci: tuple[float, float] | None = None

    def fit(self, logits_cal, labels_cal):
        gaps = logit_gap(logits_cal)
        self._fit_gaps = gaps.copy()
        labels = np.asarray(labels_cal, dtype=np.int64).reshape(-1)

        def objective(temperature: float) -> float:
            return negative_log_likelihood(expit(gaps / temperature), labels)

        self.pre_nll = negative_log_likelihood(expit(gaps), labels)
        result = minimize_scalar(objective, bounds=self.bounds, method="bounded")
        if not result.success:
            raise RuntimeError(f"Temperature scaling failed to converge: {result.message}")

        self.T = float(result.x)
        self.post_nll = objective(self.T)
        self._fisher_info = self.fisher_information(self.T, gaps=gaps)
        if self._fisher_info > 0.0:
            half_width = 1.96 / np.sqrt(self._fisher_info)
            self._ci = (float(self.T - half_width), float(self.T + half_width))
        else:
            self._ci = (np.nan, np.nan)
        return self

    def fisher_information(self, T: float | None = None, *, gaps=None) -> float:
        if T is None:
            if self.T is None:
                raise RuntimeError("TemperatureScaling must be fitted before fisher_information().")
            T = self.T
        if gaps is None:
            if not hasattr(self, "_fit_gaps"):
                raise RuntimeError("No fitted logits are available for fisher_information().")
            gaps = self._fit_gaps
        gaps_array = np.asarray(gaps, dtype=np.float64).reshape(-1)
        probs = expit(gaps_array / T)
        info = (1.0 / (T**2)) * np.sum(probs * (1.0 - probs) * ((gaps_array / T) ** 2))
        return float(info)

    def transform(self, logits_new) -> np.ndarray:
        if self.T is None:
            raise RuntimeError("TemperatureScaling must be fitted before transform().")
        return expit(logit_gap(logits_new) / self.T)

    @property
    def params(self) -> dict:
        ci_lo, ci_hi = self._ci if self._ci is not None else (None, None)
        return {
            "T": self.T,
            "T_ci_lo": ci_lo,
            "T_ci_hi": ci_hi,
            "fisher_info": self._fisher_info,
            "pre_nll": self.pre_nll,
            "post_nll": self.post_nll,
        }


class PlattScaling(Recalibrator):
    """Fit sigmoid(a * z + b) on the two-logit gap."""

    def __init__(self) -> None:
        self.a: float | None = None
        self.b: float | None = None
        self.pre_nll: float | None = None
        self.post_nll: float | None = None

    def fit(self, logits_cal, labels_cal):
        gaps = logit_gap(logits_cal)
        labels = np.asarray(labels_cal, dtype=np.int64).reshape(-1)
        self.pre_nll = negative_log_likelihood(expit(gaps), labels)

        def objective(parameters: np.ndarray) -> float:
            a, b = parameters
            return negative_log_likelihood(expit((a * gaps) + b), labels)

        result = minimize(objective, x0=np.array([1.0, 0.0]), method="BFGS")
        if not result.success:
            result = minimize(objective, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
        if not result.success:
            raise RuntimeError(f"Platt scaling failed to converge: {result.message}")

        self.a = float(result.x[0])
        self.b = float(result.x[1])
        self.post_nll = objective(result.x)
        return self

    def transform(self, logits_new) -> np.ndarray:
        if self.a is None or self.b is None:
            raise RuntimeError("PlattScaling must be fitted before transform().")
        return expit((self.a * logit_gap(logits_new)) + self.b)

    @property
    def params(self) -> dict:
        return {
            "platt_a": self.a,
            "platt_b": self.b,
            "pre_nll": self.pre_nll,
            "post_nll": self.post_nll,
        }


class IsotonicCalibration(Recalibrator):
    """Monotone calibration map fitted on raw sigmoid probabilities."""

    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.pre_nll: float | None = None
        self.post_nll: float | None = None
        self._fitted = False

    def fit(self, logits_cal, labels_cal):
        probs = softmax_positive_probability(logits_cal)
        labels = np.asarray(labels_cal, dtype=np.int64).reshape(-1)
        self.pre_nll = negative_log_likelihood(probs, labels)
        self.model.fit(probs, labels)
        fitted_probs = np.clip(self.model.transform(probs), 0.0, 1.0)
        self.post_nll = negative_log_likelihood(fitted_probs, labels)
        self._fitted = True
        return self

    def transform(self, logits_new) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("IsotonicCalibration must be fitted before transform().")
        probs = softmax_positive_probability(logits_new)
        return np.clip(self.model.transform(probs), 0.0, 1.0)

    @property
    def params(self) -> dict:
        return {
            "pre_nll": self.pre_nll,
            "post_nll": self.post_nll,
        }
