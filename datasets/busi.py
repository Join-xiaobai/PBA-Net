"""BUSI external validation benchmark dataset loader.

Dataset: Breast Ultrasound Images (BUSI)
Cohort: 647 lesion images (437 benign, 210 malignant; excluding 133 normal scans)
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .transforms import UltrasoundTransform


def merge_busi_masks(image_path: Path) -> Path | None:
    """Merges multiple lesion annotation masks for a single scan into a unified union mask."""
    parent = image_path.parent
    stem = image_path.stem
    primary_mask = parent / f"{stem}_mask{image_path.suffix}"
    extra_masks = sorted(parent.glob(f"{stem}_mask_*{image_path.suffix}"))

    if primary_mask.exists() and not extra_masks:
        return primary_mask

    if primary_mask.exists() or extra_masks:
        arrays = []
        if primary_mask.exists():
            arrays.append(np.array(Image.open(primary_mask).convert("L")) > 0)
        for extra in extra_masks:
            arrays.append(np.array(Image.open(extra).convert("L")) > 0)

        merged = np.logical_or.reduce(arrays).astype(np.uint8) * 255
        cache_dir = parent / ".cache_merged_masks"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{stem}_merged.png"
        if not cache_path.exists():
            Image.fromarray(merged).save(cache_path)
        return cache_path
    return None


def build_busi_manifest(dataset_root: Path | str, include_normal: bool = False) -> list[dict[str, Any]]:
    """Builds a verified manifest of the BUSI benchmark dataset."""
    root = Path(dataset_root)
    candidates = [
        root / "Dataset_BUSI_with_GT",
        root / "busi" / "Dataset_BUSI_with_GT",
        root / "data" / "raw" / "busi" / "Dataset_BUSI_with_GT",
        root,
    ]
    base = None
    for cand in candidates:
        if cand.exists() and (cand / "benign").exists() and (cand / "malignant").exists():
            base = cand
            break

    if base is None:
        raise FileNotFoundError(f"Could not locate BUSI dataset directory under: {dataset_root}")

    categories = [("benign", 0), ("malignant", 1)]
    if include_normal:
        categories.append(("normal", 0))

    records = []
    for cat_name, label in categories:
        folder = base / cat_name
        if not folder.exists():
            continue
        for img_path in sorted(folder.glob("*.png")):
            if "_mask" in img_path.name:
                continue
            mask_path = merge_busi_masks(img_path)
            records.append({
                "dataset": "busi",
                "sample_id": f"busi_{cat_name}_{img_path.stem}",
                "patient_id": f"busi_{cat_name}_{img_path.stem}",
                "image_path": str(img_path),
                "mask_path": str(mask_path) if mask_path else "",
                "label": label,
                "category": cat_name,
            })

    return records


class BUSIDataset(Dataset):
    """BUSI External Validation Dataset."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        transform: Callable | None = None,
        size: int = 256,
    ) -> None:
        self.records = records
        self.transform = transform if transform is not None else UltrasoundTransform(size=size, is_training=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        image_path = Path(rec["image_path"])
        mask_path = Path(rec["mask_path"]) if rec.get("mask_path") else None

        img = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") if mask_path and mask_path.exists() else None

        img_t, mask_t = self.transform(img, mask)

        has_mask = mask_t is not None
        if mask_t is None:
            mask_t = torch.zeros((1, img_t.shape[1], img_t.shape[2]), dtype=torch.float32)

        return {
            "image": img_t,
            "mask": mask_t,
            "label": torch.tensor(rec["label"], dtype=torch.long),
            "sample_id": rec.get("sample_id", image_path.stem),
            "patient_id": rec.get("patient_id", "unknown"),
            "has_mask": torch.tensor(1.0 if has_mask else 0.0, dtype=torch.float32),
        }
