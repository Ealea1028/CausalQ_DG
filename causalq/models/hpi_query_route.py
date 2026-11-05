"""B2 class-anchor readout and writeback in the final patch feature space."""
from dataclasses import dataclass
import math

import torch
from torch import nn

from .dinov3_wrapper import DINOv3Features
from .query_segmentor import QuerySegmentor


@dataclass(frozen=True)
class RouteOutput:
    tokens: torch.Tensor
    injected: torch.Tensor
    gate: torch.Tensor
    gamma: torch.Tensor
    match_logits: torch.Tensor
    patch_attention: torch.Tensor


class QueryFeatureRouter(nn.Module):
    def __init__(self, width, *, heads=8, route_width=256, gate_width=32):
        super().__init__()
        if route_width % heads:
            raise ValueError('Route width must be divisible by head count')
        self.heads = heads
        self.head_width = route_width // heads
        self.anchor_norm = nn.LayerNorm(width)
        self.patch_norm = nn.LayerNorm(width)
        self.delta_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, route_width)
        self.key = nn.Linear(width, route_width)
        self.value = nn.Linear(width, route_width, bias=False)
        self.output = nn.Linear(route_width, width, bias=False)
        self.gate = nn.Sequential(nn.Conv2d(2 * width, gate_width, 1), nn.GELU(),
                                  nn.Conv2d(gate_width, 1, 3, padding=1))
        self.gamma_logit = nn.Parameter(torch.tensor(math.log(.1 / .9)))

    def forward(self, patches, anchors, grid, *, active_classes=None):
        batch, count, width = patches.shape
        if count != grid[0] * grid[1] or anchors.ndim != 2 or anchors.shape[-1] != width:
            raise ValueError('Unexpected anchor or patch grid shape')
        norm_patches = self.patch_norm(patches)
        query = self.query(self.anchor_norm(anchors)).reshape(-1, self.heads, self.head_width).transpose(0, 1)
        key = self.key(norm_patches).reshape(batch, count, self.heads, self.head_width).transpose(1, 2)
        values = self.value(patches).reshape(batch, count, self.heads, self.head_width).transpose(1, 2)
        matches = (query @ key.transpose(-1, -2)).float() / math.sqrt(self.head_width)
        attention = matches.softmax(dim=-1)
        if active_classes is not None:
            attention = attention * active_classes[:, None, :, None]
        attention = attention.to(values.dtype)
        readout = attention @ values
        writeback = attention.transpose(-1, -2) @ readout
        delta = self.output(writeback.transpose(1, 2).reshape(batch, count, -1))
        gate_features = torch.cat((norm_patches, self.delta_norm(delta)), dim=-1)
        gate = self.gate(gate_features.transpose(1, 2).reshape(batch, 2 * width, *grid)).sigmoid()
        gamma = self.gamma_logit.sigmoid()
        injected = gamma * gate.flatten(2).transpose(1, 2) * delta
        return RouteOutput(patches + injected, injected, gate, gamma, matches.mean(1), attention)


class RoutedQuerySegmentor(QuerySegmentor):
    """Use the same injected update for raw layer 24 and normalized readout."""
    def forward_routed(self, features, output_size, *, include_anchor=False, component_protocol=None,
                       closed_classes=None):
        grid = features.patch_map.shape[-2:]
        active = None
        if closed_classes is not None:
            active = torch.arange(self.num_classes, device=closed_classes.device)[None] != closed_classes[:, None]
        route = self.query_router(features.patch_tokens, self.get_anchors(), grid, active_classes=active)
        batch, _, width = route.tokens.shape
        if len(features.intermediate_maps) != 4:
            raise ValueError('Routing expects four backbone maps including raw layer 24')
        token_map = route.tokens.transpose(1, 2).reshape(batch, width, *grid)
        update_map = route.injected.transpose(1, 2).reshape(batch, width, *grid)
        routed = DINOv3Features(token_map, route.tokens,
            (*features.intermediate_maps[:-1], features.intermediate_maps[-1] + update_map))
        output = super().forward_features(routed, output_size, include_anchor=include_anchor,
                                          component_protocol=component_protocol)
        return output, route

    def forward_features(self, features, output_size, *, include_anchor=False, component_protocol=None):
        return self.forward_routed(features, output_size, include_anchor=include_anchor,
                                   component_protocol=component_protocol)[0]

    def closed_class_logits(self, features, output_size, closed_classes):
        output, _ = self.forward_routed(features, output_size, closed_classes=closed_classes)
        selected = torch.arange(self.num_classes, device=closed_classes.device)[None] == closed_classes[:, None]
        return torch.where(selected[:, :, None, None], output.base_logits, output.logits)
