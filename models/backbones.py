"""Multi-scale Feature Extraction Backbones for PBA-Net."""

from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TinyEncoder(nn.Module):
    """Lightweight convolutional encoder for smoke testing and CPU environments."""

    def __init__(self, out_ch: int = 128) -> None:
        super().__init__()
        self.stem = _conv_block(3, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), _conv_block(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), _conv_block(64, 96))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), _conv_block(96, out_ch))
        self.out_ch = out_ch
        self.out_stride = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        return self.down3(x)


class ResNet50Encoder(nn.Module):
    """ResNet-50 feature encoder with multi-scale lateral connections (stride 8 output)."""

    def __init__(self, pretrained: bool = True, out_ch: int = 256) -> None:
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = resnet50(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.lat2 = nn.Conv2d(512, out_ch, 1, bias=False)
        self.lat3 = nn.Conv2d(1024, out_ch, 1, bias=False)
        self.lat4 = nn.Conv2d(2048, out_ch, 1, bias=False)
        self.smooth = _conv_block(out_ch, out_ch)
        self.out_ch = out_ch
        self.out_stride = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        s2 = self.layer2(x)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)

        f2 = self.lat2(s2)
        f3 = F.interpolate(self.lat3(s3), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f4 = F.interpolate(self.lat4(s4), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return self.smooth(f2 + f3 + f4)


class ConvNeXtTinyEncoder(nn.Module):
    """ConvNeXt-Tiny encoder with multi-scale feature pyramid (stride 8 output)."""

    def __init__(self, pretrained: bool = True, out_ch: int = 256) -> None:
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        net = convnext_tiny(weights=weights)

        stages = list(net.features)
        self.stage0 = stages[0]
        self.stage1 = stages[1]
        self.stage2 = stages[2]
        self.stage3 = stages[3]
        self.stage4 = stages[4] if len(stages) > 4 else nn.Identity()
        self.stage5 = stages[5] if len(stages) > 5 else nn.Identity()
        self.stage6 = stages[6] if len(stages) > 6 else nn.Identity()
        self.stage7 = stages[7] if len(stages) > 7 else nn.Identity()

        self.lat2 = nn.Conv2d(192, out_ch, 1, bias=False)
        self.lat3 = nn.Conv2d(384, out_ch, 1, bias=False)
        self.lat4 = nn.Conv2d(768, out_ch, 1, bias=False)
        self.smooth = _conv_block(out_ch, out_ch)
        self.out_ch = out_ch
        self.out_stride = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage0(x)
        x = self.stage1(x)
        s2 = self.stage2(x)
        s2 = self.stage3(s2)
        s3 = self.stage4(s2)
        s3 = self.stage5(s3)
        s4 = self.stage6(s3)
        s4 = self.stage7(s4)

        f2 = self.lat2(s2)
        f3 = F.interpolate(self.lat3(s3), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f4 = F.interpolate(self.lat4(s4), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return self.smooth(f2 + f3 + f4)


def build_encoder(name: str = "resnet50", pretrained: bool = True) -> nn.Module:
    if name == "tiny":
        return TinyEncoder()
    elif name == "resnet50":
        return ResNet50Encoder(pretrained=pretrained)
    elif name == "convnext_tiny":
        return ConvNeXtTinyEncoder(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported encoder backbone: {name}")
