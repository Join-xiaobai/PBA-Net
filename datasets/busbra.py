"""BUS-BRA multi-center breast ultrasound dataset loader and patient-level CV split."""

from __future__ import annotations
import csv
import random
from pathlib import Path
from typing import Callable, Any
from PIL import Image
import torch
from torch.utils.data import Dataset

from .transforms import UltrasoundTransform


class BUSBRADataset(Dataset):
    """BUS-BRA Dataset: Multi-center cohort for breast lesion segmentation and benign-malignant classification."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        transform: Callable | None = None,
        size: int = 256,
        is_training: bool = False,
    ) -> None:
        self.records = records
        self.transform = transform if transform is not None else UltrasoundTransform(size=size, is_training=is_training)

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


def build_busbra_splits(
    records: list[dict[str, Any]],
    outer_fold: int,
    seed: int = 20260613,
    inner_val_ratio: float = 0.20,
) -> dict[str, list[dict[str, Any]]]:
    """Generates patient-level stratified train/val/test splits without patient identity leakage."""
    if outer_fold not in range(5):
        raise ValueError("outer_fold must be between 0 and 4.")

    outer_test = [r for r in records if int(r.get("fold", 0)) == outer_fold]
    remainder = [r for r in records if int(r.get("fold", 0)) != outer_fold]

    # Partition patients strictly
    patients = sorted({r["patient_id"] for r in remainder})
    rng = random.Random(seed + 17 * outer_fold)
    rng.shuffle(patients)

    n_val = max(1, int(round(len(patients) * inner_val_ratio)))
    val_patients = set(patients[:n_val])

    inner_val = [r for r in remainder if r["patient_id"] in val_patients]
    train = [r for r in remainder if r["patient_id"] not in val_patients]

    # Audit for zero leakage
    train_pts = {r["patient_id"] for r in train}
    val_pts = {r["patient_id"] for r in inner_val}
    test_pts = {r["patient_id"] for r in outer_test}

    assert not (train_pts & val_pts), "Patient overlap between train and inner validation sets!"
    assert not (train_pts & test_pts), "Patient overlap between train and test sets!"
    assert not (val_pts & test_pts), "Patient overlap between validation and test sets!"

    return {"train": train, "inner_val": inner_val, "outer_test": outer_test}
