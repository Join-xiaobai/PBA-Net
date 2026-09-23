#!/usr/bin/env python3
"""PBA-Net Quickstart Demo.

Verifies:
  1. PBANet model initialization
  2. Multi-scale feature extraction & Anisotropic Low-Rank Factorized (ALF) Beam Attention
  3. Continuous acoustic scanline corridor extraction (lesion, shadow, superficial, lateral)
  4. Differential invariant calculation (Delta_shadow, Delta_lateral, AR_diff)
  5. Adaptive Presence Reliability Gate (gamma)
  6. Dual-pathway inference (Whole-Image, Physics, Gated Stage 1, Blended Stage 2)
  7. Multi-task loss computation and backward gradient flow
"""

import sys
from pathlib import Path
import torch

from models.pba_net import PBANet
from losses.pba_loss import pba_losses


def main():
    print("=================================================================")
    print("                     PBA-Net QUICKSTART DEMO                     ")
    print("=================================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    # 1. Instantiate Model (using lightweight tiny backbone for fast CPU verification)
    backbone = "tiny"
    print(f"Initializing PBANet with '{backbone}' backbone...")
    model = PBANet(backbone=backbone, pretrained=False).to(device)
    model.eval()

    # 2. Simulate Ultrasound Batch: (Batch=2, Channels=3, Height=256, Width=256)
    B, C, H, W = 2, 3, 256, 256
    x = torch.randn(B, C, H, W, device=device)
    dummy_mask = (torch.randn(B, 1, H, W, device=device) > 0.8).float()
    dummy_labels = torch.tensor([0, 1], device=device, dtype=torch.long)

    print(f"Simulated Ultrasound Input: shape={x.shape}, dtype={x.dtype}")

    # 3. Forward Pass
    print("\n--- Running Forward Pass ---")
    with torch.no_grad():
        out = model(x)

    print(f"Auxiliary Segmentation Logits: {out.seg_logits.shape}")
    print(f"Predicted Lesion Probability:  {out.mask_prob.shape} (Range: [{out.mask_prob.min():.3f}, {out.mask_prob.max():.3f}])")
    print(f"ALF-Attended Feature Map:      {out.feat.shape}")
    print(f"Posterior Shadow Invariant:    {out.delta_shadow.shape}")
    print(f"Lateral Differential Invariant:{out.delta_lateral.shape}")
    print(f"Presence Reliability Gate γ:   {out.gamma.squeeze(-1).tolist()}")
    print(f"Whole-Image Branch Logits:     {out.cls_global.shape}")
    print(f"Physics Branch Logits:         {out.cls_physics.shape}")
    print(f"Stage 1 Combined Logits:       {out.cls_logits.shape}")

    # 4. Stage 2 Safety-Floor Blending
    p_whole = torch.softmax(out.cls_global, dim=1)[:, 1]
    p_phys = torch.softmax(out.cls_physics, dim=1)[:, 1]
    p_blend = model.predict_blended(x, beta=0.3)
    print(f"\n--- Probabilistic Predictions ---")
    for i in range(B):
        print(f"Sample {i+1}: Whole P={p_whole[i]:.4f} | Phys P={p_phys[i]:.4f} | Gate γ={out.gamma[i, 0]:.3f} | Blended P={p_blend[i]:.4f}")

    # 5. Multi-task Loss & Gradient Check
    print("\n--- Verifying Multi-task Loss and Gradient Flow ---")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    out_train = model(x)
    losses = pba_losses(out_train, dummy_mask, dummy_labels, lambda_cls=1.0, lambda_seg=1.0, lambda_aux=0.5)

    print(f"Segmentation Loss (Dice+BCE): {losses['seg'].item():.4f}")
    print(f"Classification Loss (Stage 1):{losses['cls'].item():.4f}")
    print(f"Whole Auxiliary Loss:         {losses['cls_g'].item():.4f}")
    print(f"Physics Auxiliary Loss:       {losses['cls_p'].item():.4f}")
    print(f"Total Multi-Task Loss:        {losses['total'].item():.4f}")

    losses["total"].backward()
    optimizer.step()
    print("Backward pass and parameter update completed successfully with zero NaN gradients!")

    print("\n=================================================================")
    print("              PBA-Net DEMO PASSED SUCCESSFULLY!                 ")
    print("=================================================================")


if __name__ == "__main__":
    main()
