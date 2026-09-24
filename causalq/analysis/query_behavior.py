"""Metrics for within-class query similarity and logsumexp responsibilities."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def within_class_query_similarity_values(
    query_states: Tensor,
    *,
    present_classes: Tensor | None = None,
) -> Tensor:
    """Return unordered within-class cosine similarities.

    ``query_states`` has shape ``B x C x R x D``. When supplied,
    ``present_classes`` has shape ``B x C`` and excludes classes absent from
    each sample. R=1 has no within-class pairs and returns an empty tensor.
    """
    if query_states.ndim != 4:
        raise ValueError("query_states must have shape BxCxRxD")
    batch_size, num_classes, query_count, hidden_size = query_states.shape
    if not all(value > 0 for value in query_states.shape):
        raise ValueError("query_states dimensions must be positive")
    if present_classes is None:
        present_classes = torch.ones(
            batch_size,
            num_classes,
            dtype=torch.bool,
            device=query_states.device,
        )
    elif present_classes.shape != (batch_size, num_classes):
        raise ValueError("present_classes must have shape BxC")
    else:
        present_classes = present_classes.to(
            device=query_states.device,
            dtype=torch.bool,
        )
    if query_count < 2:
        return query_states.new_empty((0,), dtype=torch.float32)

    normalized = F.normalize(query_states.float(), p=2, dim=-1)
    similarities = torch.einsum("bcrd,bcsd->bcrs", normalized, normalized)
    upper = torch.triu(
        torch.ones(
            query_count,
            query_count,
            dtype=torch.bool,
            device=query_states.device,
        ),
        diagonal=1,
    )
    selected = similarities[:, :, upper]
    return selected[present_classes].reshape(-1)


def present_class_mask(
    labels: Tensor,
    *,
    num_classes: int,
    ignore_index: int = 255,
) -> Tensor:
    """Return the classes represented by valid pixels in each sample."""
    if labels.ndim != 3:
        raise ValueError("labels must have shape BxHxW")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    valid = labels != ignore_index
    invalid = valid & ((labels < 0) | (labels >= num_classes))
    if invalid.any():
        raise ValueError("labels contain class indices outside the configured range")
    safe = labels.masked_fill(~valid, 0)
    encoded = F.one_hot(safe, num_classes=num_classes).bool()
    return (encoded & valid.unsqueeze(-1)).any(dim=(1, 2))


def active_query_behavior_values(
    query_scores: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = 255,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    """Measure how queries share logsumexp responsibility on GT-class pixels.

    ``query_scores`` has shape ``B x C x R x H x W`` and contains the
    pre-aggregation score maps. Responsibilities are the softmax over R. One
    value per ground-truth-present class map is returned for each metric.
    """
    if query_scores.ndim != 5:
        raise ValueError("query_scores must have shape BxCxRxHxW")
    batch_size, num_classes, query_count, height, width = query_scores.shape
    if labels.shape != (batch_size, height, width):
        raise ValueError("labels must match query-score batch/spatial dimensions")
    if query_count < 1:
        raise ValueError("query count must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    present = present_class_mask(
        labels,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    if not present.any():
        raise ValueError("active-query batch contains no present classes")

    responsibilities = query_scores.float().softmax(dim=2)
    effective_counts: list[Tensor] = []
    effective_fractions: list[Tensor] = []
    dominant_shares: list[Tensor] = []
    pixel_top1: list[Tensor] = []

    for batch_index, class_index in present.nonzero(as_tuple=False):
        class_pixels = labels[batch_index] == class_index
        values = responsibilities[batch_index, class_index].reshape(
            query_count, -1
        )[:, class_pixels.reshape(-1)]
        mean_usage = values.mean(dim=1)
        entropy = -(mean_usage * mean_usage.clamp_min(eps).log()).sum()
        effective = entropy.exp()
        effective_counts.append(effective)
        effective_fractions.append(effective / query_count)
        dominant_shares.append(mean_usage.max())
        pixel_top1.append(values.max(dim=0).values.mean())

    return {
        "effective_query_count": torch.stack(effective_counts),
        "effective_query_fraction": torch.stack(effective_fractions),
        "dominant_query_share": torch.stack(dominant_shares),
        "pixel_top1_responsibility": torch.stack(pixel_top1),
    }
