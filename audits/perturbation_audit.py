"""Controlled image-domain robustness stress test approximating ultrasound acquisition and post-processing variations."""

from __future__ import annotations
import numpy as np
import torch


def apply_gain_perturbation(image: torch.Tensor, scale: float = 1.20) -> torch.Tensor:
    """Applies global gain scaling (e.g. +20% or -20%)."""
    return torch.clamp(image * scale, 0.0, 1.0)


def apply_tgc_drift(image: torch.Tensor, slope_range: tuple[float, float] = (0.75, 1.25)) -> torch.Tensor:
    """Applies depth-dependent Time Gain Compensation (TGC) linear slope."""
    b, c, h, w = image.shape
    ys = torch.linspace(slope_range[0], slope_range[1], h, device=image.device, dtype=image.dtype)
    ramp = ys.view(1, 1, h, 1)
    return torch.clamp(image * ramp, 0.0, 1.0)


def apply_gamma_compression(image: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
    """Applies non-linear dynamic range / log compression curve."""
    return torch.clamp(torch.pow(image.clamp_min(1e-6), gamma), 0.0, 1.0)


def apply_speckle_noise(image: torch.Tensor, sigma: float = 0.15) -> torch.Tensor:
    """Applies multiplicative Rayleigh/Gaussian acoustic speckle noise."""
    noise = torch.randn_like(image) * sigma
    return torch.clamp(image + image * noise, 0.0, 1.0)


PERTURBATION_SUITE = {
    "gain_plus_20": lambda x: apply_gain_perturbation(x, 1.20),
    "gain_minus_20": lambda x: apply_gain_perturbation(x, 0.80),
    "tgc_depth_drift": lambda x: apply_tgc_drift(x, (0.75, 1.25)),
    "gamma_compression": lambda x: apply_gamma_compression(x, 1.4),
    "speckle_noise": lambda x: apply_speckle_noise(x, 0.15),
}
