"""Frozen HPI-CLIP DINOv2-L/14 -> patch16, without the VLM branch.

Architecture/state names follow Meta DINOv2 (Apache-2.0); see
third_party/hpi_dinov2/LICENSE and experiments/HPI_DINOV2_20261031.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .dinov3_wrapper import DINOv3Features


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False,
                                           scale=(d // self.heads) ** -0.5)
        return self.proj(x.transpose(1, 2).reshape(b, n, d))


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), 1e-5))

    def forward(self, x):
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class DinoVisionTransformer(nn.Module):
    """Small constructor overrides exist only for synthetic CPU tests."""
    def __init__(self, *, dim=1024, depth=24, heads=16, patch_size=16, img_size=512):
        super().__init__()
        self.dim, self.patch_size = dim, patch_size
        self.patch_embed = PatchEmbed(dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def position(self, h, w, dtype):
        gh, gw = h // self.patch_size, w // self.patch_size
        n = math.isqrt(self.pos_embed.shape[1] - 1)
        if (gh, gw) == (n, n):
            return self.pos_embed.to(dtype)
        # Preserve DINO/HPI's historical +0.1 runtime interpolation offset.
        spatial = self.pos_embed[:, 1:].float().reshape(1, n, n, self.dim).permute(0, 3, 1, 2)
        spatial = F.interpolate(spatial, scale_factor=((gh + .1) / n, (gw + .1) / n),
                                mode='bicubic', align_corners=False, antialias=False)
        if spatial.shape[-2:] != (gh, gw):
            raise ValueError('Position interpolation shape mismatch')
        return torch.cat((self.pos_embed[:, :1].float(), spatial.flatten(2).transpose(1, 2)), 1).to(dtype)

    def forward(self, images, indices):
        h, w = images.shape[-2:]
        x = self.patch_embed(images)
        x = torch.cat((self.cls_token.to(x.dtype).expand(x.shape[0], -1, -1), x), 1)
        x = x + self.position(h, w, x.dtype)
        maps = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in indices:
                maps.append(x[:, 1:].transpose(1, 2).reshape(x.shape[0], self.dim,
                            h // self.patch_size, w // self.patch_size).contiguous())
        tokens = self.norm(x)[:, 1:]
        return DINOv3Features(tokens.transpose(1, 2).reshape(x.shape[0], self.dim,
                              h // self.patch_size, w // self.patch_size).contiguous(), tokens, tuple(maps))


def convert_hpi_state(state, *, dim=1024, original_grid=37, target_grid=32, patch_size=16):
    """HPI FP32 bicubic conversion; all other tensor objects remain unchanged."""
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected unwrapped official tensor state')
    if tuple(state['patch_embed.proj.weight'].shape) != (dim, 3, 14, 14):
        raise ValueError('Expected non-register DINOv2-L/14 patch projection')
    if tuple(state['pos_embed'].shape) != (1, original_grid ** 2 + 1, dim) or 'register_tokens' in state:
        raise ValueError('Unexpected official position grid/register variant')
    result = dict(state)
    result['patch_embed.proj.weight'] = F.interpolate(state['patch_embed.proj.weight'].float(),
        size=(patch_size, patch_size), mode='bicubic', align_corners=False)
    pos = state['pos_embed']
    spatial = pos[:, 1:].reshape(1, original_grid, original_grid, dim).permute(0, 3, 1, 2).float()
    spatial = F.interpolate(spatial, size=(target_grid, target_grid), mode='bicubic', align_corners=False)
    result['pos_embed'] = torch.cat((pos[:, :1], spatial.flatten(2).transpose(1, 2)), 1)
    return result


class DINOv2Backbone(nn.Module):
    def __init__(self, model=None, *, intermediate_indices=(5, 11, 17, 23), freeze=True):
        super().__init__()
        self.model = model if model is not None else DinoVisionTransformer()
        self.freeze = freeze
        self.hidden_size = self.model.dim
        self.patch_size = (self.model.patch_size,) * 2
        self.intermediate_indices = tuple(intermediate_indices)
        if tuple(sorted(set(self.intermediate_indices))) != self.intermediate_indices or any(
                i < 0 or i >= len(self.model.blocks) for i in self.intermediate_indices):
            raise ValueError('Expected ordered zero-based block indices')
        if freeze:
            self.requires_grad_(False)
            self.eval()

    @classmethod
    def from_pretrained(cls, weights: str | Path, *, dtype=None):
        # True weights are loaded only on AutoDL; converted state is strict.
        with torch.random.fork_rng(devices=[]):
            backbone = cls()
        backbone.model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        return backbone.to(dtype=dtype) if dtype else backbone

    def train(self, mode=True):
        super().train(False if self.freeze else mode)
        return self

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or any(
                s % p for s, p in zip(images.shape[-2:], self.patch_size)):
            raise ValueError('Expected BCHW RGB images divisible by patch16')
        with torch.no_grad() if self.freeze else nullcontext():
            return self.model(images, self.intermediate_indices)
