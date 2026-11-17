"""Independent LoRA core models; reuse the accepted image-level CausalQ update."""
import hashlib
import json
from pathlib import Path

import torch

from causalq.models.lora import LowRankDelta, inject_dino_lora
from causalq.models.query_segmentor import QuerySegmentor
from causalq.runtime.cohort_runner import capture_rng, restore_rng
from causalq.runtime.identity import file_sha, tree_digest
from causalq.runtime.lora_training import (compact_state, learning_rate, load_state_file,
                                          make_optimizer, save_checkpoint, update)

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT/'experiments/LORA_CAUSALQ_1114'
VARIANTS = ('U0', 'UA', 'C0', 'CA')
NAME = 'dinov3_vitl16'


def load_contract():
    freeze = json.loads((DIRECTORY/'phase42_freeze.json').read_text())
    for name, digest in freeze['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError('Phase42 frozen source changed: '+name)
    config = json.loads((ROOT/'configs/lora_causalq_1114/phase42_core_d3_l1.json').read_text())
    scope = json.loads((DIRECTORY/'lora_only_scope_20261117.json').read_text())
    retired = json.loads((DIRECTORY/'phase41_retirement_apply_acceptance.json').read_text())
    if (config['phase'] != 42 or config['backbones'] != [NAME] or config['variants'] != list(VARIANTS)
            or config['lora_enabled'] is not True or config['formal_steps'] != 40000
            or config['effective_source_batch'] != 8 or config['physical_source_batch'] != 4
            or config['target_access_authorized'] is not False or scope['train_lora_states'] != [1]
            or not retired['accepted'] or retired['remaining_candidate_bytes'] != 0):
        raise ValueError('Phase42 contract/scope/retirement mismatch')
    return config, freeze


def core_l1_head(backbone, variant, *, decoder_channels=256, num_classes=19, dropout=.1, blocks=None):
    if variant not in VARIANTS:
        raise ValueError('Only core U0/UA/C0/CA L1 models are released')
    if getattr(backbone, 'lora_enabled', False) or any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('Build each L1 model from an independent original frozen backbone')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = QuerySegmentor(backbone, decoder_channels=decoder_channels, num_classes=num_classes,
                               dropout=dropout, queries_per_class=2, interaction='static',
                               cross_attention_layers=0, centered_residuals=variant in ('C0', 'CA'),
                               temperature=.07, alpha_init=0.)
    model.to(next(backbone.parameters()).device)
    inject_dino_lora(backbone, **({} if blocks is None else {'blocks': blocks}))
    allowed = {id(p) for module in backbone.modules() if isinstance(module, LowRankDelta)
               for p in module.parameters()}
    if not allowed or allowed != {id(p) for p in backbone.parameters() if p.requires_grad}:
        raise ValueError('Only LoRA A/B may train inside DINO')
    return model


def checkpoint_rows(root, identity, horizon=40000):
    """Reject incomplete/redirected histories; validate identities before loading pickle."""
    root = Path(root).absolute()
    if root.resolve() != root:
        raise ValueError('Redirected checkpoint root')
    rows = []
    for path in sorted(root.rglob('*')):
        if path.resolve() != path or path.is_symlink():
            raise ValueError('Redirected checkpoint path')
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError('Hard-linked checkpoint path')
        if path.name.endswith('.partial') or (path.name == 'state.pt' and not (path.parent/'complete.json').exists()):
            raise ValueError('Uncommitted checkpoint needs review')
        if path.name != 'complete.json':
            continue
        marker = json.loads(path.read_text())
        step = marker['step']
        if (marker['identity'] != identity or type(step) is not int or not 0 < step <= horizon
                or path.parent.parent != root or path.parent.name != f'{step:08d}'
                or not (path.parent/'state.pt').is_file()):
            raise ValueError('Checkpoint identity/step/payload mismatch')
        rows.append((step, path.parent, marker))
    if len(rows) != len({step for step, _, _ in rows}):
        raise ValueError('Ambiguous checkpoint step')
    return sorted(rows, reverse=True, key=lambda row: row[0])


def validate_snapshot(state, marker, model, optimizer):
    step = marker['step']
    expected = compact_state(model)
    if (state['schema'] != 1 or state['identity'] != marker['identity'] or state['step'] != step
            or state['source_cursor'] != step*8 or state['next_lr'] != learning_rate(step)
            or set(state['model']) != set(expected)):
        raise ValueError('Incomplete head/LoRA identity, source cursor or schedule')
    for key, tensor in state['model'].items():
        if tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype or not torch.isfinite(tensor).all():
            raise ValueError('Invalid model tensor: '+key)
    groups = state['optimizer']['param_groups']
    reference = optimizer.state_dict()['param_groups']
    if len(groups) != len(reference):
        raise ValueError('Optimizer group coverage mismatch')
    ids = []
    for group, ref, actual in zip(groups, reference, optimizer.param_groups):
        if (set(group) != set(ref) or group['params'] != ref['params']
                or group['lr'] != learning_rate(step-1)
                or any(group[k] != ref[k] for k in ref if k not in ('params', 'lr'))):
            raise ValueError('Optimizer recipe/order mismatch')
        for pid, parameter in zip(group['params'], actual['params']):
            ids.append(pid)
            slot = state['optimizer']['state'][pid]
            if int(slot['step']) != step:
                raise ValueError('Optimizer absolute step mismatch')
            for key in ('exp_avg', 'exp_avg_sq'):
                if slot[key].shape != parameter.shape or not torch.isfinite(slot[key]).all():
                    raise ValueError('Optimizer moment shape/value mismatch')
    if len(set(ids)) != len(ids) or set(ids) != set(state['optimizer']['state']):
        raise ValueError('Incomplete optimizer slots')
    if set(state['rng']) != {'torch', 'cuda', 'python', 'numpy'}:
        raise ValueError('Incomplete RNG providers')
    import random
    import numpy as np
    rng = state['rng']
    random.Random().setstate(rng['python']); np.random.RandomState().set_state(rng['numpy'])
    torch.Generator().set_state(rng['torch'])
    if any(t.dtype != torch.uint8 or t.ndim != 1 or not t.numel() for t in rng['cuda']):
        raise ValueError('Invalid CUDA RNG state')


class L1Trainer:
    def __init__(self, model, variant, *, horizon=40000):
        if variant not in VARIANTS or not getattr(model.backbone, 'lora_enabled', False) or not 0 < horizon <= 40000:
            raise ValueError('Expected one independent adapted core model')
        if model.query_bank.centered_residuals != (variant in ('C0', 'CA')):
            raise ValueError('Variant centering mismatch')
        self.model, self.variant, self.horizon = model, variant, horizon
        self.optimizer = make_optimizer(model)
        self.step = 0
        self.tainted = False
        self.frozen = tuple(p for p in model.backbone.parameters() if not p.requires_grad)
        self.versions = tuple(p._version for p in self.frozen)
        initial = compact_state(model)
        self.initial_head_sha256 = tree_digest({k: v for k, v in initial.items() if not k.startswith('backbone.')})
        self.initial_adapter_sha256 = tree_digest({k: v for k, v in initial.items() if k.startswith('backbone.')})

    def check_base(self):
        if tuple(p._version for p in self.frozen) != self.versions or any(p.grad is not None for p in self.frozen):
            raise ValueError('Original DINO weights mutated or received gradients')

    def advance(self, packets, *, amp=False):
        positions = tuple(s['metadata']['step_zero_based'] for s in packets)
        if positions != tuple(range(self.step*8, self.step*8+8)) or self.step >= self.horizon or self.tainted:
            raise ValueError('Input/optimizer cursor mismatch or tainted update')
        self.check_base()
        self.tainted = True
        values = update(self.model, self.optimizer, packets, self.step, microbatch=4, amp=amp,
                        variant=self.variant, return_tensors=True, prevalidated_source=True)
        self.step += 1
        self.check_base()
        self.tainted = False
        return values

    def commit(self, root, identity):
        if self.step == 0 or self.tainted:
            return None
        self.check_base()
        path = Path(root).absolute()/f'{self.step:08d}'
        if path.resolve() != path:
            raise ValueError('Redirected checkpoint destination')
        if not (path/'complete.json').exists():
            if path.exists() and any(path.iterdir()):
                raise ValueError('Uncommitted checkpoint destination needs review')
            def cpu_copy(value):
                if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
                if isinstance(value, dict): return {k: cpu_copy(v) for k, v in value.items()}
                if isinstance(value, (tuple, list)): return type(value)(cpu_copy(v) for v in value)
                return value
            snapshot = {'schema': 1, 'identity': identity, 'step': self.step, 'source_cursor': self.step*8,
                        'next_lr': learning_rate(self.step), 'model': compact_state(self.model),
                        'optimizer': cpu_copy(self.optimizer.state_dict()), 'rng': capture_rng()}
            validate_snapshot(snapshot, {'step': self.step, 'identity': identity}, self.model, self.optimizer)
            del snapshot
            save_checkpoint(path, self.model, self.optimizer, self.step, identity)
        marker = json.loads((path/'complete.json').read_text())
        if marker['identity'] != identity or marker['step'] != self.step or marker['sha256'] != file_sha(path/'state.pt'):
            raise ValueError('Existing immutable checkpoint differs')
        return {'step': self.step, 'path': str(path), 'sha256': marker['sha256'], 'source_cursor': self.step*8}

    def restore(self, root, identity):
        rows = checkpoint_rows(root, identity, self.horizon)
        if not rows:
            # A run with previous commits must never silently initialize again.
            if Path(root).exists() and any(Path(root).rglob('*')):
                raise ValueError('Nonempty history without a complete checkpoint')
            return 0
        for _, folder, marker in rows[:2]:
            if file_sha(folder/'state.pt') != marker['sha256']:
                raise ValueError('Checkpoint SHA mismatch')
            state = load_state_file(folder/'state.pt')
            validate_snapshot(state, marker, self.model, self.optimizer)
        step, folder, marker = rows[0]
        state = load_state_file(folder/'state.pt')
        self.model.load_state_dict(state['model'], strict=False)
        self.optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])
        self.step = step
        self.check_base()
        return step
