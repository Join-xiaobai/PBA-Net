"""Loss functions for PBA-Net multi-task learning."""

from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from ..models.pba_net import PBAOutput


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for binary lesion segmentation."""
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def _class_weight(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Inverse frequency class weights for handling benign-malignant imbalance."""
    counts = torch.stack([(labels == 0).sum(), (labels == 1).sum()]).float().clamp_min(1.0)
    return (counts.sum() / (2.0 * counts)).to(logits.dtype)


def class_balanced_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Class-balanced Cross-Entropy loss."""
    weight = _class_weight(labels, logits)
    return F.cross_entropy(logits, labels, weight=weight)


def pba_losses(
    out: PBAOutput,
    mask: torch.Tensor | None,
    labels: torch.Tensor,
    lambda_cls: float = 1.0,
    lambda_seg: float = 1.0,
    lambda_aux: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Total multi-task training objective (Paper Eq. 17).
    
    L_total = lambda_seg * L_seg + lambda_cls * L_cls + lambda_aux * (L_cls_g + L_cls_p)
    where:
      L_seg = Dice(m_pred, m_gt) + BCE(m_pred, m_gt)
      L_cls = CE(z_comb, y)
      L_cls_g = CE(z_whole, y)
      L_cls_p = CE(z_phys, y)
    """
    losses: dict[str, torch.Tensor] = {}

    # Auxiliary Lesion Segmentation Loss
    if mask is not None:
        losses["seg"] = dice_loss(out.seg_logits, mask) + F.binary_cross_entropy_with_logits(out.seg_logits, mask)
    else:
        losses["seg"] = out.cls_logits.new_zeros(())

    # Primary Stage 1 Combined Classification Loss
    losses["cls"] = class_balanced_ce(out.cls_logits, labels)

    # Dedicated Branch Classification Losses
    losses["cls_g"] = class_balanced_ce(out.cls_global, labels)
    losses["cls_p"] = class_balanced_ce(out.cls_physics, labels)

    # Multi-task Joint Loss
    losses["total"] = (
        lambda_seg * losses["seg"]
        + lambda_cls * losses["cls"]
        + lambda_aux * losses["cls_g"]
        + lambda_aux * losses["cls_p"]
    )
    return losses
