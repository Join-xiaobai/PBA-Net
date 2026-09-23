#!/usr/bin/env python3
"""5-Fold Patient-Level Stratified Cross-Validation Training for PBA-Net."""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pba_net import PBANet
from losses.pba_loss import pba_losses
from datasets.busbra import BUSBRADataset, build_busbra_splits
from evaluation.metrics import compute_classification_metrics, compute_dice_score


def parse_args():
    parser = argparse.ArgumentParser(description="PBA-Net 5-Fold Training")
    parser.add_argument("--data-manifest", type=str, required=True, help="Path to BUS-BRA dataset manifest JSON or CSV")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--outer-fold", type=int, default=0, choices=[0, 1, 2, 3, 4], help="Fold index to evaluate as test")
    parser.add_argument("--seed", type=int, default=20260613, help="Random seed")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "convnext_tiny", "tiny"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def evaluate_epoch(model: PBANet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_y, all_p, all_dices = [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()
        has_mask = batch["has_mask"].cpu().numpy()

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out = model(images)

        probs = torch.softmax(out.cls_logits.float(), dim=1)[:, 1].cpu().numpy()
        pred_masks = out.mask_prob.float().cpu().numpy()
        gt_masks = masks.float().cpu().numpy()

        all_y.extend(labels)
        all_p.extend(probs)

        for i in range(len(labels)):
            if int(has_mask[i]) == 1:
                all_dices.append(compute_dice_score(pred_masks[i, 0], gt_masks[i, 0]))

    metrics = compute_classification_metrics(np.array(all_y), np.array(all_p))
    metrics["dice"] = float(np.mean(all_dices)) if all_dices else float("nan")
    return metrics


def train_epoch(
    model: PBANet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = (scaler is not None and device.type == "cuda")

        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(images)
            losses = pba_losses(out, masks, labels, lambda_cls=1.0, lambda_seg=1.0, lambda_aux=0.5)
            loss = losses["total"]

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs

    return total_loss / max(1, total_samples)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data_manifest, "r", encoding="utf-8") as f:
        records = json.load(f)

    splits = build_busbra_splits(records, outer_fold=args.outer_fold, seed=args.seed)

    train_ds = BUSBRADataset(splits["train"], size=args.size, is_training=True)
    val_ds = BUSBRADataset(splits["inner_val"], size=args.size, is_training=False)
    test_ds = BUSBRADataset(splits["outer_test"], size=args.size, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = PBANet(backbone=args.backbone, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    best_val_auroc = -math.inf
    stale_epochs = 0
    ckpt_name = f"pba_net_{args.backbone}_fold{args.outer_fold}_seed{args.seed}.pt"
    best_ckpt_path = out_dir / ckpt_name

    print(f"=== Starting Fold {args.outer_fold} Training (Seed: {args.seed}, Backbone: {args.backbone}) ===")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, scaler)
        val_metrics = evaluate_epoch(model, val_loader, device)
        val_auroc = val_metrics["auroc"]

        if not math.isnan(val_auroc):
            scheduler.step(val_auroc)

        if val_auroc > best_val_auroc + 1e-4:
            best_val_auroc = val_auroc
            stale_epochs = 0
            torch.save({
                "epoch": epoch,
                "fold": args.outer_fold,
                "seed": args.seed,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, best_ckpt_path)
            mark = "(* Best Saved)"
        else:
            stale_epochs += 1
            mark = ""

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Loss: {loss:.4f} | Val AUROC: {val_auroc:.4f} | Val Dice: {val_metrics['dice']:.4f} {mark}")

        if epoch >= args.min_epochs and stale_epochs >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"Fold {args.outer_fold} complete. Evaluating on outer test fold...")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate_epoch(model, test_loader, device)
    print(f"Outer Test Performance: AUROC={test_metrics['auroc']:.4f}, AUPRC={test_metrics['auprc']:.4f}, Dice={test_metrics['dice']:.4f}")


if __name__ == "__main__":
    main()
