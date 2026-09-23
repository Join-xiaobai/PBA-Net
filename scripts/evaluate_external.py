#!/usr/bin/env python3
"""External generalization evaluation on BUSI benchmark dataset."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pba_net import PBANet
from datasets.busi import BUSIDataset, build_busi_manifest
from evaluation.metrics import compute_classification_metrics, compute_dice_score
from evaluation.calibration import expected_calibration_error, brier_score, calibration_intercept_and_slope
from evaluation.tertiles import evaluate_by_segmentation_tertiles


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PBA-Net on BUSI")
    parser.add_argument("--busi-root", type=str, required=True, help="Root path to BUSI dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--beta", type=float, default=0.3, help="Fixed Whole-Physics Safety Blending factor (0.3)")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. Load Dataset
    records = build_busi_manifest(args.busi_root, include_normal=False)
    print(f"Loaded {len(records)} lesion scans from BUSI (normal scans excluded).")

    dataset = BUSIDataset(records, size=args.size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 2. Load Model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    backbone = checkpoint.get("args", {}).get("backbone", "resnet50")
    model = PBANet(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint["model"])
    model.eval()

    y_true = []
    p_whole = []
    p_phys = []
    p_gated = []
    p_blended = []
    dices = []
    gammas = []

    print("Running external evaluation...")
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()
        has_mask = batch["has_mask"].cpu().numpy()

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out = model(images)

        prob_g = torch.softmax(out.cls_global.float(), dim=1)[:, 1].cpu().numpy()
        prob_p = torch.softmax(out.cls_physics.float(), dim=1)[:, 1].cpu().numpy()
        prob_comb = torch.softmax(out.cls_logits.float(), dim=1)[:, 1].cpu().numpy()
        prob_blend = (1.0 - args.beta) * prob_g + args.beta * prob_p
        gamma = out.gamma.float().squeeze(-1).cpu().numpy()

        pred_masks = out.mask_prob.float().cpu().numpy()
        gt_masks = masks.float().cpu().numpy()

        y_true.extend(labels)
        p_whole.extend(prob_g)
        p_phys.extend(prob_p)
        p_gated.extend(prob_comb)
        p_blended.extend(prob_blend)
        gammas.extend(gamma)

        for i in range(len(labels)):
            if int(has_mask[i]) == 1:
                dices.append(compute_dice_score(pred_masks[i, 0], gt_masks[i, 0]))

    y = np.array(y_true)
    pw = np.array(p_whole)
    pp = np.array(p_phys)
    pg = np.array(p_gated)
    pb = np.array(p_blended)
    d = np.array(dices)

    print("\n================== EXTERNAL BENCHMARK RESULTS (BUSI) ==================")
    print("--- 1. Whole-Image Branch ---")
    mw = compute_classification_metrics(y, pw)
    print(f"AUROC: {mw['auroc']:.4f} | AUPRC: {mw['auprc']:.4f} | Sens: {mw['sensitivity']*100:.2f}% | Spec: {mw['specificity']*100:.2f}% | ECE: {expected_calibration_error(y, pw):.4f}")

    print("\n--- 2. Physics Acoustic Branch ---")
    mp = compute_classification_metrics(y, pp)
    print(f"AUROC: {mp['auroc']:.4f} | AUPRC: {mp['auprc']:.4f} | Sens: {mp['sensitivity']*100:.2f}% | Spec: {mp['specificity']*100:.2f}% | ECE: {expected_calibration_error(y, pp):.4f}")

    print("\n--- 3. Stage 1 Reliability Gated Combination ---")
    mg = compute_classification_metrics(y, pg)
    print(f"AUROC: {mg['auroc']:.4f} | AUPRC: {mg['auprc']:.4f} | Sens: {mg['sensitivity']*100:.2f}% | Spec: {mg['specificity']*100:.2f}% | Mean Gamma: {np.mean(gammas):.3f}")

    print(f"\n--- 4. PBA-Net Full Model (Stage 2 Safety Blend, beta={args.beta}) ---")
    mb = compute_classification_metrics(y, pb)
    ece_b = expected_calibration_error(y, pb)
    brier_b = brier_score(y, pb)
    intercept, slope = calibration_intercept_and_slope(y, pb)
    print(f"AUROC:        {mb['auroc']:.4f}")
    print(f"AUPRC:        {mb['auprc']:.4f}")
    print(f"Sensitivity:  {mb['sensitivity']*100:.2f}%")
    print(f"Specificity:  {mb['specificity']*100:.2f}%")
    print(f"Balanced Acc: {mb['balanced_acc']*100:.2f}%")
    print(f"F1 Score:     {mb['f1']:.4f}")
    print(f"ECE:          {ece_b:.4f}")
    print(f"Brier Score:  {brier_b:.4f}")
    print(f"Calib Slope:  {slope:.2f} (Intercept: {intercept:.2f})")
    print(f"Mean Dice:    {np.mean(d):.4f}")

    print("\n--- Segmentation Tertile Robustness Analysis ---")
    tertiles = evaluate_by_segmentation_tertiles(y, pb, d)
    for name, res in tertiles.items():
        print(f"  {name.upper():12s}: Samples={res['n_samples']:3d}, Mean Dice={res['mean_dice']:.4f}, AUROC={res['auroc']:.4f}")


if __name__ == "__main__":
    main()
