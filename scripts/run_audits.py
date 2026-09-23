#!/usr/bin/env python3
"""Run controlled image-domain stress tests and shortcut perturbation audits."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pba_net import PBANet
from datasets.busi import BUSIDataset, build_busi_manifest
from evaluation.metrics import compute_classification_metrics
from evaluation.calibration import expected_calibration_error
from audits.perturbation_audit import PERTURBATION_SUITE
from audits.shortcut_audit import erase_lesion_features, mask_peripheral_background


def parse_args():
    parser = argparse.ArgumentParser(description="Audit Robustness and Shortcut Dependence")
    parser.add_argument("--busi-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    records = build_busi_manifest(args.busi_root, include_normal=False)
    dataset = BUSIDataset(records, size=args.size)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    backbone = checkpoint.get("args", {}).get("backbone", "resnet50")
    model = PBANet(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint["model"])
    model.eval()

    print("================== 1. CONTROLLED IMAGE-DOMAIN STRESS AUDIT ==================")
    for pert_name, pert_fn in [("Clean Reference", lambda x: x)] + list(PERTURBATION_SUITE.items()):
        y_true, p_whole, p_blend = [], [], []
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()

            # Apply perturbation
            pert_images = pert_fn(images)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(pert_images)

            pw = torch.softmax(out.cls_global.float(), dim=1)[:, 1].cpu().numpy()
            pp = torch.softmax(out.cls_physics.float(), dim=1)[:, 1].cpu().numpy()
            pb = (1.0 - args.beta) * pw + args.beta * pp

            y_true.extend(labels)
            p_whole.extend(pw)
            p_blend.extend(pb)

        y = np.array(y_true)
        mw = compute_classification_metrics(y, np.array(p_whole))
        mb = compute_classification_metrics(y, np.array(p_blend))
        ece_b = expected_calibration_error(y, np.array(p_blend))
        print(f"[{pert_name:20s}] Whole AUROC: {mw['auroc']:.4f} | PBA-Net AUROC: {mb['auroc']:.4f} | ECE: {ece_b:.4f}")

    print("\n================== 2. LESION DEPENDENCE & SHORTCUT AUDIT ==================")
    for condition in ["Intact", "Lesion Erased", "Lesion Only (Black BG)"]:
        y_true, p_whole, p_phys, p_blend = [], [], []
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].cpu().numpy()

            if condition == "Lesion Erased":
                cur_images = erase_lesion_features(images, masks)
            elif condition == "Lesion Only (Black BG)":
                cur_images = mask_peripheral_background(images, masks)
            else:
                cur_images = images

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(cur_images)

            pw = torch.softmax(out.cls_global.float(), dim=1)[:, 1].cpu().numpy()
            pp = torch.softmax(out.cls_physics.float(), dim=1)[:, 1].cpu().numpy()
            pb = (1.0 - args.beta) * pw + args.beta * pp

            y_true.extend(labels)
            p_whole.extend(pw)
            p_phys.extend(pp)
            p_blend.extend(pb)

        y = np.array(y_true)
        mw = compute_classification_metrics(y, np.array(p_whole))
        mp = compute_classification_metrics(y, np.array(p_phys))
        mb = compute_classification_metrics(y, np.array(p_blend))
        print(f"[{condition:25s}] Whole AUROC: {mw['auroc']:.4f} | PBA Physics AUROC: {mp['auroc']:.4f} | PBA Full AUROC: {mb['auroc']:.4f}")


if __name__ == "__main__":
    main()
