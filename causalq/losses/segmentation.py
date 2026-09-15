"""Supervised semantic-segmentation objective."""

from torch import Tensor
from torch.nn import functional as F


def segmentation_cross_entropy(
    logits: Tensor, labels: Tensor, *, ignore_index: int = 255
) -> Tensor:
    if logits.ndim != 4 or labels.ndim != 3:
        raise ValueError("Expected BCHW logits and BHW labels")
    if logits.shape[0] != labels.shape[0] or logits.shape[-2:] != labels.shape[-2:]:
        raise ValueError("Logits and labels must have matching batch/spatial dimensions")
    return F.cross_entropy(logits.float(), labels, ignore_index=ignore_index)
