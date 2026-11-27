"""Shared class queries and selected-layer Read–Write injection."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SCSConfig:
    backbone_dim: int = 1024
    bottleneck_dim: int = 128
    num_classes: int = 19
    queries_per_class: int = 2
    num_heads: int = 4
    selected_layers: tuple[int, ...] = (12, 16, 20, 24)
    aux_temperature: float = 0.1

    def __post_init__(self) -> None:
        if min(self.backbone_dim, self.bottleneck_dim, self.num_classes,
               self.queries_per_class, self.num_heads) < 1:
            raise ValueError("SCS dimensions, queries and heads must be positive")
        if self.bottleneck_dim % self.num_heads:
            raise ValueError("SCS bottleneck dimension must be divisible by the number of heads")
        if (not self.selected_layers or self.selected_layers[0] < 1
                or tuple(sorted(set(self.selected_layers))) != self.selected_layers):
            raise ValueError("SCS layers must be distinct, sorted and one-based")
        if not math.isfinite(self.aux_temperature) or self.aux_temperature <= 0:
            raise ValueError("SCS auxiliary temperature must be finite and positive")

    @property
    def parameter_count(self) -> int:
        dim, width = self.backbone_dim, self.bottleneck_dim
        bridge = 2 * dim * width + 2 * dim + 2 * width
        shared = 8 * width**2 + (self.num_classes * self.queries_per_class + 4) * width
        return len(self.selected_layers) * bridge + shared


class LayerBridge(nn.Module):
    def __init__(self, dim: int, width: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, width, bias=False)
        self.latent_norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, dim, bias=False)
        nn.init.zeros_(self.up.weight)


class SCSInjector(nn.Module):
    def __init__(self, config: SCSConfig):
        super().__init__()
        self.config = config
        width = config.bottleneck_dim
        self.query_bank = nn.Parameter(torch.empty(config.num_classes, config.queries_per_class, width))
        nn.init.normal_(self.query_bank, std=0.02)
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.read = nn.MultiheadAttention(width, config.num_heads, bias=False, batch_first=True)
        self.write = nn.MultiheadAttention(width, config.num_heads, bias=False, batch_first=True)
        self.bridges = nn.ModuleDict({
            str(layer): LayerBridge(config.backbone_dim, width) for layer in config.selected_layers
        })

    def forward(
        self, layer: int, patches: Tensor, *, patch_valid: Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if patches.ndim != 3 or patches.shape[-1] != self.config.backbone_dim or not patches.shape[1]:
            raise ValueError("Expected a nonempty [B, N, D] spatial patch sequence")
        if patch_valid is not None and (
            patch_valid.shape != patches.shape[:2] or patch_valid.dtype != torch.bool
            or patch_valid.device != patches.device or not patch_valid.any(dim=1).all()
        ):
            raise ValueError("Expected a boolean patch mask on the patch device with a valid token per image")

        bridge = self.bridges[str(layer)]
        latent = bridge.latent_norm(bridge.down(bridge.input_norm(patches)))
        query = self.query_bank.flatten(0, 1).unsqueeze(0).expand(patches.shape[0], -1, -1)
        read, _ = self.read(self.query_norm(query), latent, latent,
                            key_padding_mask=None if patch_valid is None else ~patch_valid,
                            need_weights=False)
        memory = self.memory_norm(query + read)
        write, _ = self.write(latent, memory, memory, need_weights=False)
        delta = bridge.up(write).to(patches.dtype)
        if patch_valid is not None:
            delta = delta.masked_fill(~patch_valid.unsqueeze(-1), 0)
        aux = self._aux_scores(latent, memory) if return_aux else None
        return patches + delta, aux

    def _aux_scores(self, latent: Tensor, memory: Tensor) -> Tensor:
        config = self.config
        with torch.autocast(latent.device.type, enabled=False):
            scores = F.normalize(latent.float(), dim=-1) @ F.normalize(memory.float(), dim=-1).transpose(1, 2)
            scores = (scores / config.aux_temperature).unflatten(
                -1, (config.num_classes, config.queries_per_class)
            )
            return scores.logsumexp(dim=-1) - math.log(config.queries_per_class)


def scs_auxiliary_loss(logits: Tensor, labels: Tensor, ignore_index: int = 255) -> Tensor:
    """Resize scores to the label grid and average CE over labeled pixels."""
    if logits.ndim != 4 or labels.ndim != 3 or logits.shape[0] != labels.shape[0]:
        raise ValueError("Expected auxiliary logits [B, C, h, w] and labels [B, H, W]")
    resized = F.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
    total = F.cross_entropy(resized, labels, ignore_index=ignore_index, reduction="sum")
    return total / (labels != ignore_index).sum().clamp_min(1)
