"""Class-specific query residual bridge for a fixed semantic segmentor."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ReinQueryOutput:
    """Keep the fixed base prediction and intervention term explicit."""

    base_logits: torch.Tensor
    query_residual: torch.Tensor
    alpha: torch.Tensor

    @property
    def logits(self) -> torch.Tensor:
        return self.base_logits + self.alpha * self.query_residual

    def counterfactual_logits(self, class_id: int) -> torch.Tensor:
        if not 0 <= class_id < self.base_logits.shape[1]:
            raise IndexError("class_id outside semantic class range")
        result = self.logits.clone()
        result[:, class_id] = self.base_logits[:, class_id]
        return result

    def class_effect(self, class_id: int) -> torch.Tensor:
        if not 0 <= class_id < self.base_logits.shape[1]:
            raise IndexError("class_id outside semantic class range")
        return self.alpha * self.query_residual[:, class_id]


class ReinClassQueryResidual(nn.Module):
    """Map frozen class logits to an isolated grouped-query residual.

    Native Mask2Former queries remain untouched and are not assigned semantic
    identities. This separate bridge has explicit ``[class, route, channel]``
    queries, so intervening on class ``c`` changes only output channel ``c``.
    """

    def __init__(self, num_classes: int = 19, queries_per_class: int = 2,
                 hidden_channels: int = 64, temperature: float = 0.07,
                 alpha_init: float = 0.0) -> None:
        super().__init__()
        if num_classes <= 1 or queries_per_class <= 0 or hidden_channels <= 0:
            raise ValueError("Positive class/query/channel dimensions required")
        if not temperature > 0:
            raise ValueError("temperature must be positive")
        self.num_classes = num_classes
        self.queries_per_class = queries_per_class
        self.hidden_channels = hidden_channels
        self.temperature = float(temperature)
        self.pixel_projection = nn.Conv2d(num_classes, hidden_channels, 1)
        self.query_bank = nn.Parameter(
            torch.empty(num_classes, queries_per_class, hidden_channels)
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        nn.init.kaiming_uniform_(self.pixel_projection.weight, a=math.sqrt(5))
        nn.init.zeros_(self.pixel_projection.bias)
        nn.init.normal_(self.query_bank, std=0.02)

    def forward(self, base_logits: torch.Tensor) -> ReinQueryOutput:
        if base_logits.ndim != 4 or base_logits.shape[1] != self.num_classes:
            raise ValueError("Expected B x num_classes x H x W base logits")
        if not torch.is_floating_point(base_logits):
            raise TypeError("Base logits must be floating point")
        pixels = F.normalize(self.pixel_projection(base_logits), dim=1)
        queries = F.normalize(self.query_bank, dim=-1)
        similarities = torch.einsum("bdhw,crd->bcrhw", pixels, queries)
        residual = torch.logsumexp(similarities / self.temperature, dim=2)
        residual = residual - math.log(self.queries_per_class)
        return ReinQueryOutput(base_logits=base_logits, query_residual=residual,
                               alpha=self.alpha)
