"""Lightweight within-class query-residual diversity for Phase 10."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def query_diversity_loss(residuals: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Penalize squared off-diagonal cosine similarity within each class."""
    if residuals.ndim != 3:
        raise ValueError("residuals must have shape CxRxD")
    if residuals.shape[0] < 1 or residuals.shape[1] < 1 or residuals.shape[2] < 1:
        raise ValueError("query residual dimensions must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if residuals.shape[1] == 1:
        return residuals.sum() * 0.0

    normalized = F.normalize(residuals.float(), p=2, dim=-1, eps=eps)
    cosine = torch.matmul(normalized, normalized.transpose(-1, -2))
    count = residuals.shape[1]
    off_diagonal = ~torch.eye(count, dtype=torch.bool, device=residuals.device)
    return cosine[:, off_diagonal].square().mean()
