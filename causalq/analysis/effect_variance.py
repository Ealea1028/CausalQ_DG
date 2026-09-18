"""Cross-style variance of normalized causal-query-effect maps."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def cross_style_effect_variance_values(
    effects: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> Tensor:
    """Return one variance value per GT-present class and sample.

    ``effects`` has shape ``B x V x C x H x W``. Each view/class map is
    masked to valid pixels and L2-normalized over space, matching the Phase-9
    CQE definition. Population variance is then computed across views and
    summed over space. Classes absent from a sample are excluded.
    """
    if effects.ndim != 5:
        raise ValueError("effects must have shape BxVxCxHxW")
    if effects.shape[1] < 2:
        raise ValueError("At least two style views are required")
    if labels.ndim != 3 or labels.shape != (
        effects.shape[0],
        effects.shape[3],
        effects.shape[4],
    ):
        raise ValueError("labels must match the effect batch/spatial dimensions")
    if eps <= 0:
        raise ValueError("eps must be positive")

    batch_size, _, num_classes, _, _ = effects.shape
    valid = labels != ignore_index
    if not valid.any():
        raise ValueError("Effect-variance batch contains no valid pixels")

    safe_labels = labels.masked_fill(~valid, 0)
    present = F.one_hot(safe_labels, num_classes=num_classes).bool()
    present = (present & valid.unsqueeze(-1)).any(dim=(1, 2))
    if not present.any():
        raise ValueError("Effect-variance batch contains no present classes")

    valid_flat = valid.reshape(batch_size, 1, 1, -1)
    flattened = effects.float().reshape(batch_size, effects.shape[1], num_classes, -1)
    flattened = flattened.masked_fill(~valid_flat, 0.0)
    normalized = F.normalize(flattened, p=2, dim=-1, eps=eps)
    centered = normalized - normalized.mean(dim=1, keepdim=True)
    per_class = centered.square().mean(dim=1).sum(dim=-1)
    return per_class[present]


def cross_style_effect_variance(
    effects: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> Tensor:
    """Average normalized effect-map variance across present classes."""
    return cross_style_effect_variance_values(
        effects,
        labels,
        ignore_index=ignore_index,
        eps=eps,
    ).mean()
