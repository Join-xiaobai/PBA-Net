"""Comprehensive evaluation metrics for breast lesion classification and segmentation."""

from __future__ import annotations
import numpy as np
from sklearn import metrics as sk_metrics


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Computes clinical diagnostic metrics: AUROC, AUPRC, Sensitivity, Specificity, Balanced Acc, F1."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) < 2:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "balanced_acc": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
        }

    auroc = float(sk_metrics.roc_auc_score(y_true, y_prob))
    precision_arr, recall_arr, _ = sk_metrics.precision_recall_curve(y_true, y_prob)
    auprc = float(sk_metrics.auc(recall_arr, precision_arr))

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = sk_metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_acc = float(0.5 * (sensitivity + specificity))
    f1 = float(sk_metrics.f1_score(y_true, y_pred, zero_division=0))
    acc = float(sk_metrics.accuracy_score(y_true, y_pred))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_acc": balanced_acc,
        "f1": f1,
        "accuracy": acc,
    }


def compute_dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Computes 2D binary Dice similarity coefficient."""
    p = (pred_mask >= threshold).astype(np.float32)
    g = (gt_mask >= threshold).astype(np.float32)
    inter = np.sum(p * g)
    total = np.sum(p) + np.sum(g)
    if total == 0:
        return 1.0
    return float((2.0 * inter + eps) / (total + eps))
