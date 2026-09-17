"""Prediction-level consistency control objective for Phase 8."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def prediction_consistency_kl(
    view_logits: Tensor,
    reference_logits: Tensor,
    *,
    labels: Tensor | None = None,
    ignore_index: int = 255,
    temperature: float = 1.0,
) -> Tensor:
    """Return KL(reference || view) with a detached original-view teacher."""
    if view_logits.ndim != 4 or reference_logits.ndim != 4:
        raise ValueError("Expected BCHW view and reference logits")
    if view_logits.shape != reference_logits.shape:
        raise ValueError("View and reference logits must have identical shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if labels is not None:
        if labels.ndim != 3 or labels.shape != (
            view_logits.shape[0],
            view_logits.shape[2],
            view_logits.shape[3],
        ):
            raise ValueError("Labels must match the logits batch/spatial dimensions")

    scaled_view = view_logits.float() / float(temperature)
    scaled_reference = reference_logits.detach().float() / float(temperature)
    reference_probabilities = F.softmax(scaled_reference, dim=1)
    per_class = F.kl_div(
        F.log_softmax(scaled_view, dim=1),
        reference_probabilities,
        reduction="none",
    )
    per_pixel = per_class.sum(dim=1) * float(temperature) ** 2
    if labels is None:
        return per_pixel.mean()
    valid = labels != ignore_index
    if not valid.any():
        raise ValueError("Prediction consistency batch contains no valid pixels")
    return per_pixel[valid].mean()
