"""Streaming Cityscapes mean intersection-over-union."""

from __future__ import annotations

import torch
from torch import Tensor


class MeanIoU:
    def __init__(self, num_classes: int = 19, *, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, logits_or_predictions: Tensor, labels: Tensor) -> None:
        predictions = (
            logits_or_predictions.argmax(dim=1)
            if logits_or_predictions.ndim == 4
            else logits_or_predictions
        )
        predictions = predictions.detach().to("cpu").long().reshape(-1)
        labels = labels.detach().to("cpu").long().reshape(-1)
        valid = (
            (labels != self.ignore_index)
            & (labels >= 0)
            & (labels < self.num_classes)
        )
        encoded = labels[valid] * self.num_classes + predictions[valid]
        self.confusion += torch.bincount(
            encoded, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, object]:
        matrix = self.confusion.double()
        intersection = matrix.diag()
        union = matrix.sum(0) + matrix.sum(1) - intersection
        valid = union > 0
        class_iou = torch.full((self.num_classes,), float("nan"), dtype=torch.float64)
        class_iou[valid] = intersection[valid] / union[valid]
        mean_iou = class_iou[valid].mean().item() if valid.any() else float("nan")
        return {
            "miou": mean_iou,
            "class_iou": [None if torch.isnan(value) else value.item() for value in class_iou],
            "valid_classes": int(valid.sum().item()),
        }
