"""Image preprocessing and augmentation pipelines for ultrasound images and masks."""

from __future__ import annotations
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF


class UltrasoundTransform:
    """Synchronized image and mask transformations."""

    def __init__(self, size: int = 256, is_training: bool = True) -> None:
        self.size = size
        self.is_training = is_training

    def __call__(self, image: Image.Image, mask: Image.Image | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Resize both image and mask
        image = TF.resize(image, [self.size, self.size], interpolation=TF.InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.resize(mask, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST)

        if self.is_training:
            # Random horizontal flip (p=0.5)
            if random.random() > 0.5:
                image = TF.hflip(image)
                if mask is not None:
                    mask = TF.hflip(mask)

            # Random slight rotation (-10 to +10 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-10.0, 10.0)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                if mask is not None:
                    mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Convert image to RGB tensor in [0, 1]
        img_tensor = TF.to_tensor(image)
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)

        # Standard ImageNet normalization for pre-trained backbones
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask_tensor = None
        if mask is not None:
            mask_arr = np.array(mask, dtype=np.float32)
            if mask_arr.max() > 1.0:
                mask_arr = mask_arr / 255.0
            mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).clamp(0.0, 1.0)

        return img_tensor, mask_tensor
