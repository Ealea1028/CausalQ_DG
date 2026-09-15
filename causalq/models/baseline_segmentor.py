"""Frozen-DINOv3 source-only semantic-segmentation baseline."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .dinov3_wrapper import DINOv3Backbone


class BaselineDecoder(nn.Module):
    """Fuse same-resolution ViT layers and predict Cityscapes logits."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_feature_maps: int,
        decoder_channels: int = 256,
        num_classes: int = 19,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_feature_maps < 1:
            raise ValueError("num_feature_maps must be positive")
        if decoder_channels % 32:
            raise ValueError("decoder_channels must be divisible by 32")
        input_channels = hidden_size * num_feature_maps
        self.fuse = nn.Sequential(
            nn.Conv2d(input_channels, decoder_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, decoder_channels),
            nn.GELU(),
            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(32, decoder_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def forward(self, feature_maps: Sequence[Tensor]) -> Tensor:
        return self.classifier(self.fuse(torch.cat(tuple(feature_maps), dim=1)))


class BaselineSegmentor(nn.Module):
    """Keep DINOv3 frozen and train only a lightweight dense decoder."""

    def __init__(
        self,
        backbone: DINOv3Backbone,
        *,
        decoder_channels: int = 256,
        num_classes: int = 19,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not backbone.freeze:
            raise ValueError("Phase 5 requires a frozen DINOv3 backbone")
        self.backbone = backbone
        feature_count = max(1, len(backbone.intermediate_indices))
        self.decoder = BaselineDecoder(
            backbone.hidden_size,
            num_feature_maps=feature_count,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    def train(self, mode: bool = True) -> "BaselineSegmentor":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        features = self.backbone(images)
        maps = features.intermediate_maps or (features.patch_map,)
        logits = self.decoder(maps)
        return F.interpolate(
            logits,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
