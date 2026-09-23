"""PBA-Net: Physics-Grounded Beam Attention and Acoustic Shadow Network.

Paper: Physics-Grounded Beam Attention Network for Breast Ultrasound Benign-Malignant Lesion Stratification
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from .backbones import build_encoder


def _soft_pool(feat: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Channel-wise soft spatial pooling weighted by soft acoustic zones."""
    weight = weight.clamp(0.0, 1.0)
    numerator = (feat * weight).sum(dim=(2, 3))
    denominator = weight.sum(dim=(2, 3)) + eps
    return numerator / denominator


class AxialLateralAttention(nn.Module):
    """Anisotropic Low-Rank Factorized (ALF) Beam Attention.
    
    Factorizes dense 2D spatial attention into:
      1. 1D Axial Beam Attention along vertical acoustic propagation depth (H).
      2. 1D Lateral Context Attention along horizontal tissue stratum (W).
    Reduces spatial attention complexity from O((HW)^2) to O(HW(H + W)).
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.axial_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.lateral_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ax1 = nn.LayerNorm(dim)
        self.norm_lat1 = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, C, H, W)
        B, C, H, W = x.shape

        # Step 1: 1D Axial Beam Attention (along depth H for each scanline W)
        # (B, C, H, W) -> (B, W, H, C) -> (B * W, H, C)
        x_ax = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        x_ax_norm = self.norm_ax1(x_ax)
        ax_out, _ = self.axial_attn(x_ax_norm, x_ax_norm, x_ax_norm)
        x_ax = x_ax + ax_out

        # Step 2: 1D Lateral Context Attention (along width W for each depth H)
        # (B * W, H, C) -> (B, W, H, C) -> (B, H, W, C) -> (B * H, W, C)
        x_lat = x_ax.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B * H, W, C)
        x_lat_norm = self.norm_lat1(x_lat)
        lat_out, _ = self.lateral_attn(x_lat_norm, x_lat_norm, x_lat_norm)
        x_lat = x_lat + lat_out

        # Step 3: Position-wise Feed-Forward Network
        out = x_lat + self.ffn(self.norm_ffn(x_lat))

        # Reshape back to standard BCHW format
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


@dataclass
class PBAOutput:
    seg_logits: torch.Tensor
    mask_prob: torch.Tensor
    cls_logits: torch.Tensor       # Stage 1: z_comb = z_whole + gamma * z_phys
    cls_global: torch.Tensor       # Whole-image branch logits: z_whole
    cls_physics: torch.Tensor      # Physics acoustic branch logits: z_phys
    gamma: torch.Tensor            # Presence reliability gating weight in [0, 1]
    delta_shadow: torch.Tensor     # Posterior shadow differential invariant
    delta_lateral: torch.Tensor    # Lateral differential invariant
    feat: torch.Tensor
    zones: dict[str, torch.Tensor]


class PBANet(nn.Module):
    """PBA-Net Architecture.
    
    Integrates:
      1. Multi-scale feature encoder (default ResNet-50)
      2. Axial-Lateral Factorized (ALF) Beam Attention
      3. Auxiliary Lesion Segmentation Head
      4. Continuous Scanline Acoustic Zone Extraction
      5. Soft Differential Invariant Pooling (Delta_shadow, Delta_lateral, AR_diff)
      6. Adaptive Presence Reliability Gate (gamma)
      7. Dedicated Dual-Pathway Classifiers (Whole-image + Physics)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.1,
        empty_tau: float = 0.002,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.empty_tau = empty_tau
        self.encoder = build_encoder(backbone, pretrained=pretrained and backbone != "tiny")
        c = self.encoder.out_ch

        # ALF-Attention
        self.beam_attn = AxialLateralAttention(dim=c, num_heads=4, dropout=dropout)

        # Auxiliary Segmentation Head
        self.seg_head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, 1),
        )

        # Dedicated Whole-Image Classifier
        self.cls_global_head = nn.Sequential(
            nn.Linear(c, c // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(c // 2, 2),
        )

        # Dedicated Physics Acoustic Classifier
        # Input features: g_lesion (c) + delta_shadow (c) + delta_lateral (c) + AR_diff (1)
        phys_in = 3 * c + 1
        self.cls_physics_head = nn.Sequential(
            nn.Linear(phys_in, c),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(c, c // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(c // 2, 2),
        )

        # Adaptive Presence Reliability Gate (gamma)
        self.gate_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, -0.5)

    def extract_acoustic_zones(self, mask_prob: torch.Tensor, hw: tuple[int, int]) -> dict[str, torch.Tensor]:
        """Derive continuous differential acoustic zones along vertical scanlines."""
        m = F.interpolate(mask_prob, size=hw, mode="bilinear", align_corners=False)
        m = m.clamp(0.0, 1.0)

        # Acoustic ray envelope along vertical depth (dim=2)
        col_has_lesion, _ = torch.max(m, dim=2, keepdim=True)  # (B, 1, 1, W)
        cum_depth, _ = torch.cummax(m, dim=2)                 # (B, 1, H, W)

        w_lesion = m
        # Posterior shadow corridor: strictly below lesion along active beam
        w_shadow = cum_depth * (1.0 - m)
        # Superficial corridor: strictly above lesion along active beam
        w_superficial = col_has_lesion * (1.0 - cum_depth) * (1.0 - m)
        # Lateral parenchyma: tissue outside the active acoustic beam
        w_lateral = (1.0 - col_has_lesion).expand_as(m)
        # Global reference
        w_global = torch.ones_like(m)

        return {
            "lesion": w_lesion,
            "shadow": w_shadow,
            "superficial": w_superficial,
            "lateral": w_lateral,
            "global": w_global,
            "col_active": col_has_lesion,
        }

    def compute_aspect_ratio(self, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Compute differentiable 2nd-order moment aspect ratio AR_diff = 0.5 * ln(var_y / var_x)."""
        b, _, h, w = mask.shape
        ys = torch.linspace(0.0, 1.0, h, device=mask.device, dtype=mask.dtype).view(1, 1, h, 1)
        xs = torch.linspace(0.0, 1.0, w, device=mask.device, dtype=mask.dtype).view(1, 1, 1, w)
        mass = mask.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
        cy = (mask * ys).sum(dim=(2, 3), keepdim=True) / mass
        cx = (mask * xs).sum(dim=(2, 3), keepdim=True) / mass
        mass_b = mass.view(b)
        var_y = ((mask * (ys - cy) ** 2).sum(dim=(1, 2, 3)) / mass_b).clamp_min(eps)
        var_x = ((mask * (xs - cx) ** 2).sum(dim=(1, 2, 3)) / mass_b).clamp_min(eps)
        log_ratio = 0.5 * (torch.log(var_y) - torch.log(var_x))
        return log_ratio.view(b, 1).clamp(-3.0, 3.0)

    def forward(self, x: torch.Tensor, pool_mask: torch.Tensor | None = None) -> PBAOutput:
        # 1. Base Multi-scale Feature Extraction
        feat_raw = self.encoder(x)

        # 2. ALF Beam Attention
        feat = self.beam_attn(feat_raw)

        # 3. Auxiliary Lesion Segmentation
        seg_low = self.seg_head(feat)
        seg_logits = F.interpolate(seg_low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        mask_prob = torch.sigmoid(seg_logits)

        # Use pool_mask if explicitly provided during audits, else predicted mask
        m_src = pool_mask if pool_mask is not None else mask_prob

        # 4. Acoustic Zone Extraction
        hw = feat.shape[-2:]
        zones = self.extract_acoustic_zones(m_src, hw)

        # 5. Soft Feature Pooling over Acoustic Zones
        g_global = _soft_pool(feat, zones["global"])
        g_lesion = _soft_pool(feat, zones["lesion"])
        g_shadow = _soft_pool(feat, zones["shadow"])
        g_superficial = _soft_pool(feat, zones["superficial"])
        g_lateral = _soft_pool(feat, zones["lateral"])

        # 6. Physical Differential Invariants
        delta_shadow = g_shadow - g_superficial
        delta_lateral = g_lesion - g_lateral
        aspect_token = self.compute_aspect_ratio(zones["lesion"])

        # 7. Acoustic Physics Representation
        h_phys = torch.cat([g_lesion, delta_shadow, delta_lateral, aspect_token], dim=1)

        # 8. Adaptive Presence Reliability Gating
        area = zones["lesion"].mean(dim=(1, 2, 3)).view(-1, 1)
        mean_p = (zones["lesion"] * zones["lesion"].detach()).sum(dim=(1, 2, 3)).view(-1, 1) / zones["lesion"].sum(dim=(1, 2, 3)).view(-1, 1).clamp_min(1e-6)
        active_w = zones["col_active"].mean(dim=(1, 2, 3)).view(-1, 1)
        empty = (area < self.empty_tau).float()

        gate_input = torch.cat([area, mean_p, active_w], dim=1)
        gamma = torch.sigmoid(self.gate_mlp(gate_input))
        gamma = gamma * (1.0 - empty)  # zero out if lesion absent

        # 9. Dedicated Branch Predictions
        cls_global = self.cls_global_head(g_global)
        cls_physics = self.cls_physics_head(h_phys)

        # Stage 1 combination
        cls_logits = cls_global + gamma * cls_physics

        return PBAOutput(
            seg_logits=seg_logits,
            mask_prob=mask_prob,
            cls_logits=cls_logits,
            cls_global=cls_global,
            cls_physics=cls_physics,
            gamma=gamma,
            delta_shadow=delta_shadow,
            delta_lateral=delta_lateral,
            feat=feat,
            zones=zones,
        )

    def predict_blended(self, x: torch.Tensor, beta: float = 0.3) -> torch.Tensor:
        """Stage 2 Fixed Whole-Physics Safety Blending for inference.
        
        P_blend = (1 - beta) * sigma(z_whole) + beta * sigma(z_phys)
        """
        out = self.forward(x)
        prob_whole = torch.softmax(out.cls_global, dim=1)[:, 1]
        prob_phys = torch.softmax(out.cls_physics, dim=1)[:, 1]
        prob_blend = (1.0 - beta) * prob_whole + beta * prob_phys
        return prob_blend
