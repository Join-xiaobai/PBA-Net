"""Dice-stratified tertile analysis for evaluating robustness under segmentation errors."""

from __future__ import annotations
import numpy as np
from sklearn import metrics as sk_metrics


def evaluate_by_segmentation_tertiles(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    dices: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Partitions evaluation data into Low-Dice, Mid-Dice, and High-Dice cohorts."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    dices = np.asarray(dices, dtype=float)

    finite_mask = np.isfinite(dices)
    valid_dices = dices[finite_mask]

    if len(valid_dices) == 0:
        return {}

    lo_cut = float(np.quantile(valid_dices, 1.0 / 3.0))
    hi_cut = float(np.quantile(valid_dices, 2.0 / 3.0))

    tertiles = {
        "low_dice": (dices <= lo_cut) & finite_mask,
        "mid_dice": (dices > lo_cut) & (dices <= hi_cut) & finite_mask,
        "high_dice": (dices > hi_cut) & finite_mask,
    }

    out: dict[str, dict[str, float]] = {}
    for name, mask in tertiles.items():
        sub_y = y_true[mask]
        sub_p = y_prob[mask]
        n_samples = int(np.sum(mask))

        if len(np.unique(sub_y)) > 1:
            auroc = float(sk_metrics.roc_auc_score(sub_y, sub_p))
        else:
            auroc = float("nan")

        out[name] = {
            "n_samples": n_samples,
            "mean_dice": float(np.mean(dices[mask])),
            "auroc": auroc,
        }

    return out
