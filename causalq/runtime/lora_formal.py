"""Phase41 independent L0 heads sharing one frozen DINO; no target interfaces."""
import hashlib
import json
from pathlib import Path
import torch

from causalq.models.query_segmentor import QuerySegmentor
from causalq.runtime.identity import file_sha,tree_digest
from causalq.runtime.lora_training import (FrozenFeatureBatch,update,make_optimizer,compact_state,
    save_checkpoint,restore_checkpoint)

VARIANTS=('U0','UA','C0','CA')
ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=ROOT/'experiments/LORA_CAUSALQ_1114'


def load_contract():
    freeze=json.loads((DIRECTORY/'phase41_freeze.json').read_text())
    if freeze['file_hash_scheme']!='sha256_lf_text_v1': raise ValueError('Unknown frozen file hash scheme')
    for name,digest in freeze['files'].items():
        raw=(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        if hashlib.sha256(raw).hexdigest()!=digest: raise ValueError(f'Phase41 frozen file changed: {name}')
    evidence=(DIRECTORY/'phase40_resume_acceptance.json').read_bytes().replace(b'\r\n',b'\n')
    if hashlib.sha256(evidence).hexdigest()!=freeze['resume_acceptance_git_blob_sha256'] or not json.loads(evidence)['accepted']:
        raise ValueError('Recovery acceptance changed')
    config=json.loads((ROOT/'configs/lora_causalq_1114/phase41_core_l0.json').read_text())
    if (config['variants']!=list(VARIANTS) or config['formal_steps']!=40000 or config['effective_source_batch']!=8
        or config['lora_enabled'] is not False or config['target_access_authorized'] is not False):
        raise ValueError('Phase41 scope mismatch')
    return config,freeze


def core_head(backbone,variant,*,decoder_channels=256,num_classes=19,dropout=.1):
    if variant not in VARIANTS: raise ValueError('Phase41 core L0 only')
    if any(p.requires_grad for p in backbone.parameters()) or getattr(backbone,'lora_enabled',False):
        raise ValueError('Phase41 requires unadapted frozen backbone')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model=QuerySegmentor(backbone,decoder_channels=decoder_channels,num_classes=num_classes,dropout=dropout,
            queries_per_class=2,interaction='static',cross_attention_layers=0,centered_residuals=variant in ('C0','CA'),
            temperature=.07,alpha_init=0.)
    return model.to(next(backbone.parameters()).device)


def build_backbone(name,root):
    from causalq.models.backbone_factory import build_backbone as factory
    accepted=json.loads((DIRECTORY/'phase39_report.json').read_text())
    rows=[m for m in accepted['models'] if m['model']==name]
    if len(rows)!=1:
        raise ValueError(f'Backbone {name!r} is not uniquely present in Phase39 accepted models: '
                         f'{[m["model"] for m in accepted["models"]]}')
    row=rows[0]
    weights=root/'dinov3_vitl16' if name=='dinov3_vitl16' else root/'dinov2_vitl14/dinov2_vitl14_hpi_p16.pth'
    for filename,digest in row['weight_files'].items():
        path=weights/filename if weights.is_dir() else weights
        if file_sha(path)!=digest: raise ValueError(f'Base weight changed: {path}')
    opts={'intermediate_indices':(6,12,18,24)} if name=='dinov3_vitl16' else {}
    return factory(name,weights=weights,dtype=torch.bfloat16,**opts).cuda()


class FormalCohort:
    def __init__(self,backbone,heads,*,horizon=40000,microbatch=4):
        if tuple(heads)!=VARIANTS or microbatch!=4 or not 0<horizon<=40000:
            raise ValueError('Invalid core cohort')
        self.backbone,self.heads,self.horizon,self.microbatch=backbone,heads,horizon,microbatch
        self.optimizers={name:make_optimizer(model) for name,model in heads.items()}
        self.cursors={name:0 for name in heads}
        self.initial_head_digests={name:tree_digest(compact_state(model)) for name,model in heads.items()}
        if len(set(self.initial_head_digests.values()))!=1: raise ValueError('Common head initializations differ')
        seen=set()
        for model in heads.values():
            if model.backbone is not backbone: raise ValueError('Backbone must be shared')
            ids={id(p) for p in model.parameters() if p.requires_grad}
            if seen&ids: raise ValueError('Trainable heads overlap')
            seen|=ids
        self.tainted_head=None

    def tick(self,packets,*,amp=False):
        if len(packets)!=8: raise ValueError('Expected eight source samples')
        positions=tuple(s['metadata']['step_zero_based'] for s in packets)
        step=positions[0]//8
        if positions!=tuple(range(step*8,step*8+8)) or not 0<=step<self.horizon:
            raise ValueError('Input source cursor mismatch')
        cache=FrozenFeatureBatch.extract(self.backbone,packets,self.microbatch,amp)
        results={}
        for name,model in self.heads.items():
            if self.cursors[name]>step: continue
            if self.cursors[name]!=step: raise ValueError('Missing source update before recovery cursor')
            self.tainted_head=name
            results[name]=update(model,self.optimizers[name],packets,step,microbatch=self.microbatch,amp=amp,
                variant=name,return_tensors=True,prevalidated_source=True,frozen_features=cache)
            self.cursors[name]+=1
            self.tainted_head=None
        cache.entries.clear()
        return results

    def commit(self,root,identities,records=None):
        if records is None: records={}
        for name,model in self.heads.items():
            if name==self.tainted_head: continue
            step=self.cursors[name]
            if step==0: continue
            path=Path(root)/name/f'{step:08d}'
            marker=path/'complete.json'
            if not marker.exists(): save_checkpoint(path,model,self.optimizers[name],step,identities[name])
            m=json.loads(marker.read_text())
            if m['identity']!=identities[name] or m['step']!=step or m['sha256']!=file_sha(path/'state.pt'):
                raise ValueError('Existing commit identity/hash mismatch')
            records[name]={'step':step,'path':str(path),'sha256':m['sha256']}
        return records

    def restore(self,root,identities):
        for name,model in self.heads.items():
            rows=[]
            for marker in (Path(root)/name).glob('*/complete.json'):
                record=json.loads(marker.read_text())
                if record['identity']!=identities[name]: raise ValueError('Recovery identity mismatch')
                rows.append((record['step'],marker.parent))
            if rows:
                step,path=max(rows,key=lambda r:r[0])
                if sum(s==step for s,_ in rows)!=1 or not 0<step<=self.horizon: raise ValueError('Ambiguous recovery step')
                self.cursors[name]=restore_checkpoint(path,model,self.optimizers[name],identities[name])
        return min(self.cursors.values())
