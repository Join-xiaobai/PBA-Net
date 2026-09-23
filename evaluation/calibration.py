"""Probability calibration metrics and reliability curve analysis."""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across equally spaced bins."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes Mean Squared Error of probabilistic predictions (Brier Score)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_intercept_and_slope(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-5) -> tuple[float, float]:
    """Computes calibration intercept and calibration slope via logistic calibration."""
    y_true = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    logit_p = np.log(p / (1.0 - p)).reshape(-1, 1)

    reg = LogisticRegression(penalty=None, solver="lbfgs")
    reg.fit(logit_p, y_true)

    slope = float(reg.coef_[0, 0])
    intercept = float(reg.intercept_[0])
    return intercept, slope
