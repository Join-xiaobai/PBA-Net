"""Paired DeLong test for comparing two correlated ROC curves."""

from __future__ import annotations
import numpy as np
import scipy.stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    a = 0
    while a < n:
        b = a
        while b < n - 1 and z[b] == z[b + 1]:
            b += 1
        rank = 0.5 * (a + b) + 1
        t[a : b + 1] = rank
        a = b + 1
    out = np.empty(n, dtype=float)
    out[j] = t
    return out


def _fast_delong_structural_components(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)

    x = y_prob[pos_idx]
    y = y_prob[neg_idx]

    all_scores = np.concatenate([x, y])
    ranks = _compute_midrank(all_scores)
    tx = ranks[:m]
    ty = ranks[m:]

    v10 = (tx - _compute_midrank(x)) / float(n)
    v01 = 1.0 - (ty - _compute_midrank(y)) / float(m)
    auc = float(np.mean(v10))
    return v10, v01, auc


def paired_delong_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> dict[str, float]:
    """Evaluates whether the AUROC difference between Model A and Model B is statistically significant."""
    y_true = np.asarray(y_true, dtype=int)
    preds_a = np.asarray(preds_a, dtype=float)
    preds_b = np.asarray(preds_b, dtype=float)

    v10_a, v01_a, auc_a = _fast_delong_structural_components(y_true, preds_a)
    v10_b, v01_b, auc_b = _fast_delong_structural_components(y_true, preds_b)

    m = len(v10_a)
    n = len(v01_a)

    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    sigma = s10 / float(m) + s01 / float(n)

    var_diff = sigma[0, 0] + sigma[1, 1] - 2.0 * sigma[0, 1]
    diff = auc_a - auc_b

    if var_diff <= 1e-12:
        return {"auc_a": auc_a, "auc_b": auc_b, "diff": diff, "z": 0.0, "p_value": 1.0}

    se = np.sqrt(var_diff)
    z = diff / se
    p_value = 2.0 * float(scipy.stats.norm.sf(np.abs(z)))

    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "diff": diff,
        "se": float(se),
        "z": float(z),
        "p_value": p_value,
    }
