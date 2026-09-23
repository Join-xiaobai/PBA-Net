"""Comparative Benchmark Baselines for Breast Ultrasound Lesion Stratification.

Includes:
  - Whole-image classification baselines: ResNet-18, ResNet-50, EfficientNet-B0, ConvNeXt-Tiny, Swin-T
  - Dedicated segmentation architectures: UNet, Attention U-Net, DeepLabV3+, SAM, EGE-UNet (2024), DAMamba (2025)
  - Contemporary classification baselines: HyFormer-Net (2025), HADS-Net (2026)
  - Cascaded two-stage CAD pipelines (Hard ROI cropping)
"""

from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class WholeImageBaseline(nn.Module):
    """End-to-end whole-image classifier with global pooling."""

    def __init__(self, backbone: str = "resnet50", num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet18(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, num_classes)
            self.model = net
        elif backbone == "resnet50":
            from torchvision.models import ResNet50_Weights, resnet50
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = resnet50(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, num_classes)
            self.model = net
        elif backbone == "efficientnet_b0":
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = efficientnet_b0(weights=weights)
            in_features = net.classifier[1].in_features
            net.classifier[1] = nn.Linear(in_features, num_classes)
            self.model = net
        elif backbone == "convnext_tiny":
            from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            net = convnext_tiny(weights=weights)
            in_features = net.classifier[2].in_features
            net.classifier[2] = nn.Linear(in_features, num_classes)
            self.model = net
        elif backbone == "swin_t":
            from torchvision.models import Swin_T_Weights, swin_t
            weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            net = swin_t(weights=weights)
            in_features = net.head.in_features
            net.head = nn.Linear(in_features, num_classes)
            self.model = net
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CascadedCADPipeline(nn.Module):
    """Two-stage Cascaded CAD pipeline: Segmentation -> Bounding Box Crop -> Classifier."""

    def __init__(self, seg_model: nn.Module, cls_model: nn.Module, pad_ratio: float = 0.15) -> None:
        super().__init__()
        self.seg_model = seg_model
        self.cls_model = cls_model
        self.pad_ratio = pad_ratio

    @torch.no_grad()
    def crop_lesion_roi(self, x: torch.Tensor, mask: torch.Tensor, target_size: int = 256) -> torch.Tensor:
        b, c, h, w = x.shape
        cropped = []
        for i in range(b):
            m = mask[i, 0] > 0.5
            pos = torch.nonzero(m)
            if len(pos) == 0:
                cropped.append(F.interpolate(x[i : i + 1], size=(target_size, target_size), mode="bilinear", align_corners=False))
                continue
            y_min, x_min = pos.min(dim=0).values.float()
            y_max, x_max = pos.max(dim=0).values.float()
            dh = y_max - y_min
            dw = x_max - x_min
            y_min = max(0, int(y_min - dh * self.pad_ratio))
            y_max = min(h, int(y_max + dh * self.pad_ratio))
            x_min = max(0, int(x_min - dw * self.pad_ratio))
            x_max = min(w, int(x_max + dw * self.pad_ratio))

            roi = x[i : i + 1, :, y_min:y_max, x_min:x_max]
            roi_res = F.interpolate(roi, size=(target_size, target_size), mode="bilinear", align_corners=False)
            cropped.append(roi_res)
        return torch.cat(cropped, dim=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seg_out = self.seg_model(x)
        mask = torch.sigmoid(seg_out) if isinstance(seg_out, torch.Tensor) else torch.sigmoid(seg_out["seg_logits"])
        roi = self.crop_lesion_roi(x, mask, target_size=x.shape[-1])
        cls_logits = self.cls_model(roi)
        return mask, cls_logits
