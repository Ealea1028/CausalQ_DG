"""Opt-in Q/V LoRA for DINO; load base weights before injection."""
from contextlib import contextmanager
import math

import torch
from torch import nn


class LowRankDelta(nn.Module):
    def __init__(self, width, rank=4, alpha=4., *, device=None):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('Expected positive rank and alpha')
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, width, device=device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(width, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return nn.functional.linear(nn.functional.linear(x.to(self.A.dtype), self.A), self.B) * self.scale


class LoRALinear(nn.Module):
    def __init__(self, base, *, fused=False, rank=4, alpha=4.):
        super().__init__()
        if not isinstance(base, nn.Linear) or base.out_features != base.in_features * (3 if fused else 1):
            raise ValueError('Expected square Q/V or fused 3D QKV linear')
        self.base = base
        self.base.requires_grad_(False)
        self.enabled = True
        self.fused = fused
        self.q = LowRankDelta(base.in_features, rank, alpha).to(base.weight.device)
        self.v = LowRankDelta(base.in_features, rank, alpha).to(base.weight.device) if fused else None

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        delta = self.q(x).to(y.dtype)
        if self.fused:
            delta = torch.cat((delta, torch.zeros_like(delta), self.v(x).to(y.dtype)), dim=-1)
        return y + delta


def inject_dino_lora(backbone, *, blocks=tuple(range(18, 24)), rank=4, alpha=4., seed=20261114):
    """Validate all targets before mutation; support D2 and HF D3 explicit layouts."""
    if any(isinstance(m, LoRALinear) for m in backbone.modules()):
        raise ValueError('LoRA already injected')
    indices = tuple(blocks)
    if not indices or tuple(sorted(set(indices))) != indices or min(indices) < 0:
        raise ValueError('Expected ordered unique block indices')
    model = backbone.model
    if hasattr(model, 'blocks'):
        layers, prefix, attention_name = model.blocks, 'blocks', 'attn'
    elif hasattr(model, 'layer'):
        layers, prefix, attention_name = model.layer, 'layer', 'attention'
    else:
        raise ValueError('Unsupported DINO block layout; inspect actual model')
    if max(indices) >= len(layers):
        raise ValueError('LoRA block index out of range')
    targets = []
    for i in indices:
        attn = getattr(layers[i], attention_name)
        names = ('qkv',) if hasattr(attn, 'qkv') else ('q_proj', 'v_proj')
        for name in names:
            module = getattr(attn, name, None)
            fused = name == 'qkv'
            if not isinstance(module, nn.Linear) or module.in_features != backbone.hidden_size or module.out_features != backbone.hidden_size * (3 if fused else 1):
                raise ValueError(f'Unexpected projection {prefix}.{i}.{attention_name}.{name}')
            targets.append((attn, name, module, fused, f'model.{prefix}.{i}.{attention_name}.{name}'))
    # CPU initialization then device transfer preserves caller CPU/CUDA RNG.
    replacements = []
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for owner, name, base, fused, path in targets:
            # Create deltas on CPU; no device RNG is consumed.
            wrapper = LoRALinear(base, fused=fused, rank=rank, alpha=alpha)
            replacements.append((owner, name, wrapper, path))
    model.requires_grad_(False)
    for owner, name, wrapper, _ in replacements:
        wrapper.q.requires_grad_(True)
        if wrapper.v is not None:
            wrapper.v.requires_grad_(True)
        setattr(owner, name, wrapper)
    backbone.lora_enabled = True
    model.eval()
    manifest = {'blocks_zero_based': list(indices), 'rank': rank, 'alpha_lora': alpha,
                'targets': [{'path': p, 'components': ['q', 'v'] if w.fused else [n.split('_')[0]]}
                            for _, n, w, p in replacements],
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    backbone.lora_manifest = manifest
    return manifest


@contextmanager
def adapters_disabled(backbone):
    modules = [m for m in backbone.modules() if isinstance(m, LoRALinear)]
    previous = [m.enabled for m in modules]
    try:
        for m in modules:
            m.enabled = False
        yield
    finally:
        for m, enabled in zip(modules, previous):
            m.enabled = enabled
