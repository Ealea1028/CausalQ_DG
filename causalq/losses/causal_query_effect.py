"""Cross-style causal-query-effect distillation for Phase 9."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def causal_query_effect_loss(
    view_effect: Tensor,
    reference_effect: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = 255,
    beta: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """Match normalized effect maps for GT-present classes only."""
    if view_effect.ndim != 4 or reference_effect.ndim != 4:
        raise ValueError("Expected BCHW effect tensors")
    if view_effect.shape != reference_effect.shape:
        raise ValueError("View and reference effects must have identical shapes")
    if labels.ndim != 3 or labels.shape != (
        view_effect.shape[0],
        view_effect.shape[2],
        view_effect.shape[3],
    ):
        raise ValueError("Labels must match the effect batch/spatial dimensions")
    if beta <= 0 or eps <= 0:
        raise ValueError("beta and eps must be positive")

    batch_size, num_classes, _, _ = view_effect.shape
    valid = labels != ignore_index
    if not valid.any():
        raise ValueError("CQE batch contains no valid pixels")
    safe_labels = labels.masked_fill(~valid, 0)
    present = F.one_hot(safe_labels, num_classes=num_classes).bool()
    present = (present & valid.unsqueeze(-1)).any(dim=(1, 2))
    if not present.any():
        raise ValueError("CQE batch contains no present training classes")

    valid_flat = valid.reshape(batch_size, 1, -1)
    view_flat = view_effect.float().reshape(batch_size, num_classes, -1)
    reference_flat = reference_effect.detach().float().reshape(
        batch_size, num_classes, -1
    )
    view_flat = view_flat.masked_fill(~valid_flat, 0.0)
    reference_flat = reference_flat.masked_fill(~valid_flat, 0.0)
    view_normalized = F.normalize(view_flat, p=2, dim=-1, eps=eps)
    reference_normalized = F.normalize(reference_flat, p=2, dim=-1, eps=eps)

    # L2-normalized maps make the spatial sum resolution-stable. Average only
    # across semantic classes that occur in each sample.
    per_element = F.smooth_l1_loss(
        view_normalized,
        reference_normalized,
        reduction="none",
        beta=beta,
    )
    per_class = per_element.sum(dim=-1)
    return per_class[present].mean()
