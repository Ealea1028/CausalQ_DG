"""Independent heads share frozen features, never gradients or recovery cursors."""
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch
from causalq.losses.causal_query_effect import causal_query_effect_loss
from causalq.losses.source_ce import deterministic_source_ce
from .feature_pipeline import feature_tensors
from .identity import file_sha, write_json


def head_state(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith('backbone.')}


def capture_rng():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            'python': random.getstate(), 'numpy': np.random.get_state()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python']); np.random.set_state(state['numpy'])


@dataclass
class IndependentHead:
    name: str
    model: object
    optimizer: object
    rng: dict
    cursor: int = 0
    anchor_cqe: bool = False

    @contextmanager
    def random_scope(self):
        outer = capture_rng()
        restore_rng(self.rng)
        try:
            yield
        finally:
            self.rng = capture_rng()
            restore_rng(outer)


def baseline_backward(head, original, photo, label, output_size, *, amp=False):
    """Two backwards, original Anchor detached; no optimizer in this helper."""
    device_type = label.device.type
    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=amp):
        result = head.model.forward_features(original, output_size, include_anchor=head.anchor_cqe,
                                              component_protocol='hpi_dinov2_20261031')
        original_ce = deterministic_source_ce(result.logits, label)
    reference = result.anchor_scaled_delta_logits.detach().float() if head.anchor_cqe else None
    original_ce.backward()
    del result
    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=amp):
        result = head.model.forward_features(photo, output_size, include_anchor=head.anchor_cqe,
                                              component_protocol='hpi_dinov2_20261031')
        photo_ce = deterministic_source_ce(result.logits, label)
    cqe = (causal_query_effect_loss(result.anchor_scaled_delta_logits.float(), reference, label, beta=1.)
           if head.anchor_cqe else photo_ce.new_zeros(()))
    (photo_ce + cqe).backward()
    return torch.stack((original_ce.detach(), photo_ce.detach(), cqe.detach()))


class CohortRunner:
    def __init__(self, heads, *, horizon=40000, lr=1e-4, power=.9):
        self.heads, self.horizon, self.lr, self.power = list(heads), horizon, lr, power
        seen = set()
        for head in self.heads:
            parameters = {id(p) for p in head.model.parameters() if p.requires_grad}
            if parameters & seen:
                raise ValueError('Cohort heads share trainable parameters')
            seen |= parameters
            optimizer_params = {id(p) for group in head.optimizer.param_groups for p in group['params']}
            if optimizer_params != parameters:
                raise ValueError('Optimizer must contain exactly its own trainable head')

    def aligned_start(self, views_per_window):
        size = views_per_window // 2
        return min(head.cursor for head in self.heads) // size * size

    def update(self, sample, original, photo, *, amp=False, update_fn=baseline_backward):
        step = sample['metadata']['step_zero_based']
        if step >= self.horizon:
            raise ValueError('Cohort step exceeds horizon')
        tensors = (*feature_tensors(original), *feature_tensors(photo))
        versions = tuple(x._version for x in tensors)
        records = {}
        for head in self.heads:
            if head.cursor > step:
                continue
            if head.cursor != step:
                raise ValueError('Missing input before head recovery cursor')
            label = sample['label'][None].to(original.patch_tokens.device, non_blocking=True)
            with head.random_scope():
                events=tuple(torch.cuda.Event(enable_timing=True) for _ in range(2)) if label.is_cuda else ()
                if events:events[0].record()
                head.model.train()
                head.optimizer.zero_grad(set_to_none=True)
                values = update_fn(head, original, photo, label, label.shape[-2:], amp=amp)
                norm = torch.nn.utils.clip_grad_norm_([p for p in head.model.parameters() if p.requires_grad],
                                                      1., error_if_nonfinite=True)
                if not torch.isfinite(values).all():
                    raise FloatingPointError('Nonfinite cohort loss')
                head.optimizer.step()
                if events:events[1].record()
                head.cursor += 1
                for group in head.optimizer.param_groups:
                    group['lr'] = self.lr * (1 - head.cursor / self.horizon) ** self.power
                # Keep metrics on device; caller flushes at controlled intervals.
                records[head.name] = {'optimizer_updates': head.cursor, 'losses': values, 'gradient_norm': norm.detach(), 'timing_events':events}
            if tuple(x._version for x in tensors) != versions:
                raise ValueError('Head modified shared frozen features in-place')
        return {'cohort_tick': step + 1, 'heads': records}


def commit_head(head, root, identity):
    """Synchronous fixed CPU snapshot, hash then atomic complete marker."""
    import copy
    root = Path(root) / head.name / f'{head.cursor:05d}_{uuid.uuid4().hex}'
    root.mkdir(parents=True)
    def cpu(x):
        if isinstance(x, torch.Tensor): return x.detach().cpu().clone()
        if isinstance(x, dict): return {k: cpu(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)): return type(x)(cpu(v) for v in x)
        return copy.deepcopy(x)
    snapshot = cpu({'model': head_state(head.model), 'optimizer': head.optimizer.state_dict(),
                    'rng': head.rng, 'cursor': head.cursor, 'identity': identity})
    stage = root / 'state.pth.partial'
    torch.save(snapshot, stage)
    os.replace(stage, root / 'state.pth')
    write_json(root / 'complete.json', {'cursor': head.cursor, 'identity': identity,
                                       'state_sha256': file_sha(root / 'state.pth')})
    return root


def restore_head(head, root, identity):
    import json
    records = []
    for marker in (Path(root) / head.name).glob('*/complete.json'):
        record = json.loads(marker.read_text())
        if record['identity'] != identity or file_sha(marker.parent / 'state.pth') != record['state_sha256']:
            raise ValueError('Recovery identity/hash mismatch')
        records.append((record['cursor'], marker.parent))
    if not records:
        return 0
    cursor, directory = max(records, key=lambda x: x[0])
    if sum(c == cursor for c, _ in records) != 1:
        raise ValueError('Ambiguous committed head cursor')
    state = torch.load(directory / 'state.pth', map_location='cpu', weights_only=False)
    if state['identity'] != identity or state['cursor'] != cursor or set(state['model']) != set(head_state(head.model)):
        raise ValueError('Compact head state coverage mismatch')
    incompatible = head.model.load_state_dict(state['model'], strict=False)
    if incompatible.unexpected_keys or any(not k.startswith('backbone.') for k in incompatible.missing_keys):
        raise ValueError('Unexpected missing head state')
    head.optimizer.load_state_dict(state['optimizer'])
    head.rng, head.cursor = state['rng'], cursor
    return cursor
