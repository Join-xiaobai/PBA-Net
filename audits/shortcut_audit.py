"""Lesion-dependence and background shortcut perturbation audit."""

from __future__ import annotations
import torch


def erase_lesion_features(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Interventional lesion erasure: zeros out lesion interior to test causal feature reliance."""
    m = (mask > 0.5).float()
    return image * (1.0 - m)


def mask_peripheral_background(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Interventional background suppression: zeros out peripheral background outside lesion."""
    m = (mask > 0.5).float()
    return image * m
