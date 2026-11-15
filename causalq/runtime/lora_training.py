"""LoRA study update/checkpoint primitives; Phase41 adds frozen L0 cohorts."""
from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import dataclass
import copy
import hashlib
import os
from pathlib import Path

import torch
from torch import nn

from causalq.losses.source_ce import deterministic_source_ce
from causalq.losses.causal_query_effect import causal_query_effect_loss
from causalq.runtime.cohort_runner import capture_rng, restore_rng
from causalq.runtime.feature_pipeline import slice_features
from causalq.runtime.identity import file_sha, write_json


def seed_training_rng(seed=0):
    """Initialize every checkpointed RNG, including independent process defaults."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def learning_rate(step):
    if not 0 <= step <= 40000:
        raise ValueError('Step outside HPI schedule')
    poly = 1e-6 + (1e-4-1e-6)*(1-step/40000)**.9
    return poly * (1e-6+(1-1e-6)*step/1500 if step<1500 else 1.)


def rng_digest(state):
    import numpy as np
    from causalq.runtime.identity import tree_digest
    def canonical(x):
        if isinstance(x,np.ndarray): return torch.from_numpy(x.copy())
        if isinstance(x,dict): return {k:canonical(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)): return type(x)(canonical(v) for v in x)
        return x
    return tree_digest(canonical(state))


def make_optimizer(model):
    norm_ids={id(p) for m in model.modules() if isinstance(m,(nn.LayerNorm,nn.GroupNorm,nn.modules.batchnorm._BatchNorm))
              for p in m.parameters(recurse=False)}
    groups=[]
    for decay in (0.,1e-5):
        params=[p for p in model.parameters() if p.requires_grad and ((id(p) in norm_ids)==(decay==0))]
        if params: groups.append({'params':params,'weight_decay':decay,'lr':learning_rate(0)})
    return torch.optim.AdamW(groups,lr=learning_rate(0),betas=(.9,.999),eps=1e-8)


@contextmanager
def sample_rng(index,view):
    """Head dropout is tied to absolute source position, not microbatch layout."""
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        seed=int.from_bytes(hashlib.sha256(f'lora1114:{index}:{view}'.encode()).digest()[:8],'big')%(2**63-1)
        torch.manual_seed(seed)
        yield


@dataclass
class FrozenFeatureBatch:
    """One logical batch shared by L0 heads only; bound to exact input objects."""
    backbone: object
    microbatch: int
    source_identity: tuple
    entries: dict
    versions: tuple

    @staticmethod
    def source_signature(packets):
        return tuple((s['metadata']['step_zero_based'],id(s['views']),s['views']._version,
                      id(s['label']),s['label']._version) for s in packets)

    def tensors(self):
        return tuple(t for f in self.entries.values() for t in
            (f.patch_map,f.patch_tokens,*f.intermediate_maps,f.cls_token,f.register_tokens) if t is not None)

    def verify(self,model,packets,microbatch):
        if model.backbone is not self.backbone or self.microbatch!=microbatch or self.source_signature(packets)!=self.source_identity:
            raise ValueError('Frozen cache/input/backbone mismatch')
        if any(p.requires_grad for p in model.backbone.parameters()) or getattr(model.backbone,'lora_enabled',False):
            raise ValueError('Adapted/trainable backbone cannot use frozen cache')
        if self.versions!=tuple(t._version for t in self.tensors()): raise ValueError('Frozen features mutated')

    @classmethod
    def extract(cls,backbone,packets,microbatch=4,amp=False):
        if len(packets)!=8 or microbatch not in (1,2,4,8): raise ValueError('Invalid frozen batch')
        if any(p.requires_grad for p in backbone.parameters()) or getattr(backbone,'lora_enabled',False):
            raise ValueError('Frozen cache requires an unadapted frozen backbone')
        device=next(backbone.parameters()).device
        entries={}
        backbone.eval()
        with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
            for start in range(0,8,microbatch):
                for view in (0,1):
                    images=torch.stack([s['views'][view] for s in packets[start:start+microbatch]]).to(device,non_blocking=True)
                    entries[start,view]=backbone(images)
        cache=cls(backbone,microbatch,cls.source_signature(packets),entries,())
        cache.versions=tuple(t._version for t in cache.tensors())
        return cache


def update(model, optimizer, packets, step, *, microbatch=1, amp=False,variant='UA',timer=None,return_tensors=False,prevalidated_source=False,frozen_features=None):
    """Eight source images, two summed CE terms, detached original Anchor CQE."""
    if len(packets)!=8 or microbatch not in (1,2,4,8):
        raise ValueError('Effective source batch must be eight')
    if variant not in ('U0','UA','C0','CA','B3'): raise ValueError('Unreleased objective')
    if frozen_features is not None: frozen_features.verify(model,packets,microbatch)
    timed=timer if timer is not None else lambda name:nullcontext()
    if prevalidated_source:
        from causalq.losses.lora_source import source_ce_prevalidated as ce_fn,anchor_cqe_prevalidated as cqe_fn
    else:
        ce_fn,cqe_fn=deterministic_source_ce,causal_query_effect_loss
    for group in optimizer.param_groups: group['lr']=learning_rate(step)
    device=next(model.parameters()).device
    model.train(); optimizer.zero_grad(set_to_none=True)
    values=[]
    for start in range(0,8,microbatch):
        rows=packets[start:start+microbatch]
        # L1 builds a fresh graph for each view; only L0 may reuse no-grad inputs.
        refs=[]
        for view in (0,1):
            images=torch.stack([s['views'][view] for s in rows]).to(device,non_blocking=True) if frozen_features is None else None
            labels=torch.stack([s['label'] for s in rows]).to(device,non_blocking=True)
            terms=[]
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
                with timed('backbone_forward'):
                    features=model.backbone(images) if frozen_features is None else frozen_features.entries[start,view]
                with timed('head_and_loss'):
                    for j,sample in enumerate(rows):
                        local=slice_features(features,j,j+1)
                        with sample_rng(sample['metadata']['step_zero_based'],view):
                            if variant=='B3':
                                out,route=model.forward_routed(local,labels.shape[-2:])
                            else:
                                out=model.forward_features(local, labels.shape[-2:],
                                    include_anchor=variant in ('UA','CA'),component_protocol='lora_causalq_1114')
                        ce=ce_fn(out.logits,labels[j:j+1])
                        if variant=='B3':
                            from causalq.losses.hpi_localization import patch_localization_loss
                            loss=ce+.05*patch_localization_loss(route.match_logits,labels[j:j+1],local.patch_map.shape[-2:])
                        elif variant in ('U0','C0'):
                            loss=ce
                        elif view==0:
                            refs.append(out.anchor_scaled_delta_logits.detach().float())
                            loss=ce
                        else:
                            loss=ce+cqe_fn(out.anchor_scaled_delta_logits.float(),refs[j],labels[j:j+1])
                        terms.append(loss)
                        values.append(loss.detach())
                total=torch.stack(terms).sum()/8
            with timed('backward'):
                total.backward()
            del out,terms,total,features,images,labels
    with timed('clip_and_optimizer'):
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        if not torch.isfinite(torch.stack(values)).all(): raise FloatingPointError('Nonfinite source objective')
        optimizer.step()
    if frozen_features is not None: frozen_features.verify(model,packets,microbatch)
    if return_tensors:
        return torch.stack((torch.stack(values).sum()/8,norm.detach()))
    return {'loss':float(torch.stack(values).sum()/8),'gradient_norm':float(norm),'lr':learning_rate(step)}


def compact_state(model):
    # Keep the complete head/buffers and all adapters, including backbone keys.
    trainable={n for n,p in model.named_parameters() if p.requires_grad}
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()
            if not k.startswith('backbone.') or k in trainable}


def save_checkpoint(directory,model,optimizer,step,identity):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists(): raise ValueError('Immutable checkpoint already committed')
    state={'schema':1,'identity':copy.deepcopy(identity),'step':step,'source_cursor':step*8,
           'model':compact_state(model),'optimizer':optimizer.state_dict(),'rng':capture_rng(),
           'next_lr':learning_rate(step)}
    tmp=directory/'state.pt.partial'
    torch.save(state,tmp); os.replace(tmp,directory/'state.pt')
    write_json(directory/'complete.json',{'sha256':file_sha(directory/'state.pt'),'identity':identity,'step':step})


def load_state_file(path):
    # NumPy RNG uses legacy pickle globals: only locally hash-verified own artifacts.
    return torch.load(path,map_location='cpu',weights_only=False)


def restore_checkpoint(directory,model,optimizer,identity):
    import json
    directory=Path(directory)
    marker=json.loads((directory/'complete.json').read_text())
    if marker['identity']!=identity or marker['sha256']!=file_sha(directory/'state.pt'):
        raise ValueError('Checkpoint identity/hash mismatch')
    state=load_state_file(directory/'state.pt')
    if (state['schema']!=1 or state['identity']!=identity or state['step']!=marker['step']
            or state['source_cursor']!=state['step']*8 or state['next_lr']!=learning_rate(state['step'])
            or set(state['model'])!=set(compact_state(model))):
        raise ValueError('Checkpoint coverage/cursor mismatch')
    model.load_state_dict(state['model'],strict=False)
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    return state['step']
