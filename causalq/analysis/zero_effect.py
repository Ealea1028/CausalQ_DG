"""Localization diagnostics for the zero-ablation query effect."""

from __future__ import annotations

import torch
from torch import Tensor

from .query_behavior import present_class_mask


def zero_effect_localization_values(
    effects: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = 255,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    """Compare each present class effect inside and outside its GT region.

    ``effects`` is the factual-minus-zero logit effect with shape ``BxCxHxW``.
    One value per ground-truth-present class map is returned. Invalid pixels are
    excluded from both regions. Classes without either an inside or outside
    valid pixel are skipped.
    """
    if effects.ndim != 4:
        raise ValueError("effects must have shape BxCxHxW")
    batch_size, num_classes, height, width = effects.shape
    if labels.shape != (batch_size, height, width):
        raise ValueError("labels must match effect batch/spatial dimensions")
    if eps <= 0:
        raise ValueError("eps must be positive")

    present = present_class_mask(
        labels,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    valid = labels != ignore_index
    inside_absolute: list[Tensor] = []
    outside_absolute: list[Tensor] = []
    inside_signed: list[Tensor] = []
    outside_signed: list[Tensor] = []

    values = effects.float()
    for batch_index, class_index in present.nonzero(as_tuple=False):
        inside = valid[batch_index] & (labels[batch_index] == class_index)
        outside = valid[batch_index] & ~inside
        if not inside.any() or not outside.any():
            continue
        class_effect = values[batch_index, class_index]
        inside_values = class_effect[inside]
        outside_values = class_effect[outside]
        inside_absolute.append(inside_values.abs().mean())
        outside_absolute.append(outside_values.abs().mean())
        inside_signed.append(inside_values.mean())
        outside_signed.append(outside_values.mean())

    if not inside_absolute:
        raise ValueError("batch contains no comparable present class maps")

    inside_abs = torch.stack(inside_absolute)
    outside_abs = torch.stack(outside_absolute)
    return {
        "inside_absolute_mean": inside_abs,
        "outside_absolute_mean": outside_abs,
        "absolute_ratio": inside_abs / outside_abs.clamp_min(eps),
        "normalized_absolute_contrast": (
            (inside_abs - outside_abs) / (inside_abs + outside_abs).clamp_min(eps)
        ),
        "inside_signed_mean": torch.stack(inside_signed),
        "outside_signed_mean": torch.stack(outside_signed),
    }
