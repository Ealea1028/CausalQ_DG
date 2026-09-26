"""Calibration objective for the class-agnostic learned-null query bank."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def null_query_centroid_loss(
    null_query_states: Tensor,
    factual_query_states: Tensor,
    *,
    stop_gradient_target: bool = True,
) -> Tensor:
    """Fit shared null slots to the class-agnostic factual-query centroid."""
    if null_query_states.ndim != 4 or factual_query_states.ndim != 4:
        raise ValueError("query states must have shape BxCxRxD")
    if null_query_states.shape != factual_query_states.shape:
        raise ValueError("null and factual query states must have matching shapes")
    target = factual_query_states.mean(dim=1, keepdim=True).expand_as(
        factual_query_states
    )
    if stop_gradient_target:
        target = target.detach()
    return F.smooth_l1_loss(null_query_states, target)
