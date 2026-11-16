"""Read-only Phase41 stop inventory. No training, migration or deletion."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

from causalq.runtime.identity import file_sha,tree_digest,write_json
from causalq.runtime.lora_formal import VARIANTS,core_head
from causalq.runtime.lora_training import compact_state,make_optimizer,learning_rate,rng_digest

PRODUCER='1af6b2e91e54083be433341f9fc8958954536272'
BACKBONES=('dinov3_vitl16','dinov2_vitl14_hpi_p16')


def safe(root,path):
    root,path=Path(root).absolute(),Path(path).absolute()
    if root.resolve()!=root or path.resolve()!=path: raise ValueError('Redirected or noncanonical path')
    if root.is_symlink() or path!=root and root not in path.parents:
        raise ValueError('Path outside run root')
    for p in (path,*path.parents):
        if p.is_symlink(): raise ValueError('Symlink in inventory path')
    return path


def protected(steps):
    """Provisional protections only; every payload remains pinned until migration."""
    common=set.intersection(*(set(steps[n]) for n in VARIANTS))
    common=sorted(common)[-2:]
    return {n:sorted(set(sorted(steps[n])[-2:])|set(common)|({40000} if 40000 in steps[n] else set())) for n in VARIANTS},common


def validate_state(state,marker,model):
    step=marker['step'];identity=marker['identity']
    if (type(step) is not int or not 0<step<=40000 or state['schema']!=1 or state['identity']!=identity
        or state['step']!=step or state['source_cursor']!=step*8 or state['next_lr']!=learning_rate(step)):
        raise ValueError('Schema/identity/source cursor/schedule mismatch')
    expected=compact_state(model)
    if set(state['model'])!=set(expected): raise ValueError('Incomplete head coverage')
    for name,t in state['model'].items():
        if t.shape!=expected[name].shape or t.dtype!=expected[name].dtype or not torch.isfinite(t).all():
            raise ValueError('Model shape/dtype/nonfinite: '+name)
    opt=make_optimizer(model)
    groups=state['optimizer']['param_groups']
    if len(groups)!=len(opt.param_groups): raise ValueError('Optimizer groups missing')
    ids=[]
    for group,ref in zip(groups,opt.param_groups):
        if len(group['params'])!=len(ref['params']): raise ValueError('Optimizer parameter coverage')
        for key in ('betas','eps','weight_decay','amsgrad','maximize'):
            if group[key]!=ref[key]: raise ValueError('Optimizer configuration changed: '+key)
        if group['lr']!=learning_rate(step-1): raise ValueError('Optimizer absolute LR mismatch')
        for pid,param in zip(group['params'],ref['params']):
            ids.append(pid);slot=state['optimizer']['state'][pid]
            if int(slot['step'])!=step: raise ValueError('AdamW step mismatch')
            for key in ('exp_avg','exp_avg_sq'):
                if slot[key].shape!=param.shape or not torch.isfinite(slot[key]).all():
                    raise ValueError('Optimizer moment mismatch')
    if len(set(ids))!=len(ids) or set(ids)!=set(state['optimizer']['state']): raise ValueError('Optimizer slot coverage')
    rng=state['rng']
    if set(rng)!={'python','numpy','torch','cuda'}: raise ValueError('RNG provider coverage')
    random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy'])
    torch.Generator().set_state(rng['torch'])
    if any(t.dtype!=torch.uint8 or t.ndim!=1 or not t.numel() for t in rng['cuda']): raise ValueError('CUDA RNG structure')
    return {'model_sha256':tree_digest(state['model']),'optimizer_sha256':tree_digest(state['optimizer']),
            'rng_sha256':rng_digest(rng),'source_cursor':state['source_cursor']}


class ShapeOnlyBackbone(torch.nn.Module):
    """Head schema reconstruction only; no DINO weights or forward implementation."""
    def __init__(self):
        super().__init__();self.placeholder=torch.nn.Parameter(torch.zeros(1),requires_grad=False)
        self.freeze=True;self.hidden_size=1024;self.intermediate_indices=(6,12,18,24)


def git_bytes(path):
    return subprocess.check_output(['git','show',PRODUCER+':'+path])


def inventory(root):
    root=safe(root,root)
    if not root.is_dir(): raise ValueError('Original run directory missing; do not infer zero progress')
    evidence={};uncommitted=[]
    for path in root.rglob('*'):
        safe(root,path)
        if path.is_file() and path.name in ('report.json','training_report.json','resolved_configs.json','progress.json'):
            evidence[str(path.relative_to(root))]={'sha256':file_sha(path),'content':json.loads(path.read_text())}
        if path.is_file() and (path.name.endswith('.partial') or path.name=='state.pt' and not (path.parent/'complete.json').exists()):
            uncommitted.append({'path':str(path),'bytes':path.stat().st_size,'protected':True})
    expected_hash={key:__import__('hashlib').sha256(git_bytes(path)).hexdigest() for key,path in {
        'freeze_sha256':'experiments/LORA_CAUSALQ_1114/phase41_freeze.json',
        'config_sha256':'configs/lora_causalq_1114/phase41_core_l0.json',
        'execution_profile_sha256':'experiments/LORA_CAUSALQ_1114/execution_profile.json'}.items()}
    report={'phase':41,'action':'stopped-inventory','ok':True,'run_root':str(root),'producer':PRODUCER,
        'formal_updates_performed':0,'payloads_deleted':0,'migration_accepted':False,
        'all_existing_payloads_pinned':True,'backbones':{},'errors':[]}
    report['existing_evidence']=evidence;report['uncommitted_files']=uncommitted
    for backbone in BACKBONES:
        folder=safe(root,root/backbone);rows={n:[] for n in VARIANTS};steps={n:[] for n in VARIANTS}
        for name in VARIANTS:
            model=core_head(ShapeOnlyBackbone(),name)
            initial=tree_digest(compact_state(model))
            for marker_path in sorted((folder/'checkpoints'/name).glob('*/complete.json')):
                row={'marker':str(marker_path),'ok':False}
                try:
                    safe(root,marker_path);m=json.loads(marker_path.read_text());identity=m['identity']
                    required={'study':'lora_causalq_1114','phase':41,'variant':name,'backbone':backbone,
                        'lora_enabled':False,'training_producer_git_sha':PRODUCER,'microbatch':4,
                        'effective_source_batch':8,'horizon':40000,'centered':name in ('C0','CA'),
                        'anchor_cqe':name in ('UA','CA'),'initial_head_sha256':initial,**expected_hash}
                    if any(identity.get(k)!=v for k,v in required.items()): raise ValueError('Producer/config/head identity mismatch')
                    if marker_path.parent.name!=f"{m['step']:08d}": raise ValueError('Directory step mismatch')
                    payload=safe(root,marker_path.parent/'state.pt')
                    if file_sha(payload)!=m['sha256']: raise ValueError('Checkpoint SHA mismatch')
                    state=torch.load(payload,map_location='cpu',weights_only=False)
                    row.update(validate_state(state,m,model));del state
                    row.update(ok=True,step=m['step'],identity=identity,sha256=m['sha256'],bytes=payload.stat().st_size)
                    steps[name].append(m['step'])
                    print(backbone,name,m['step'],'verified',flush=True)
                except Exception as exc:
                    row['error']=str(exc);report['ok']=False;report['errors'].append(row.copy())
                rows[name].append(row)
        keep,common=protected(steps)
        cursors={n:max(steps[n],default=0) for n in VARIANTS}
        report['backbones'][backbone]={'checkpoints':rows,'verified_cursors':cursors,
            'provisional_protection_steps':keep,'latest_common_boundaries':common,
            'no_committed_checkpoint':[n for n in VARIANTS if not steps[n]],
            'skewed':len(set(cursors.values()))>1}
    report['verified_committed_updates']=sum(sum(v['verified_cursors'].values()) for v in report['backbones'].values())
    report['in_memory_progress_recovered']=False
    report['next']='local review before batched migration or any cleanup; all payloads stay pinned'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    allowed=Path('/root/autodl-tmp/outputs/CausalQ_DG/lora_causalq_1114/phase41_core_1af6b2e')
    if a.run_root.absolute()!=allowed: p.error('Only the original Phase41 run root is authorized')
    if a.output.exists() or allowed==a.output.absolute() or allowed in a.output.absolute().parents:
        p.error('Output must be a fresh file outside the original run')
    if sys.platform!='linux': p.error('Run this inventory on AutoDL Linux')
    import fcntl
    # Hold original run and GPU locks during the read-only checkpoint inventory.
    with ExitStack() as stack:
        for lock in (allowed/'run.lock',allowed.parent/'gpu0.lock'):
            safe(allowed.parent,lock)
            handle=stack.enter_context(lock.open('r+'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=[]
        for proc in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                cmd=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace')
                if 'tools.train_lora_core_l0' in cmd and str(allowed) in cmd and proc.parent.name!=str(os.getpid()):
                    active.append({'pid':proc.parent.name,'command':cmd})
            except (OSError,PermissionError): pass
        if active: raise RuntimeError('Original training processes still present: '+repr(active))
        r=inventory(allowed)
        r['audit_git_sha']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        r['environment']={'python':sys.version,'torch':str(torch.__version__),'cuda_initialized':torch.cuda.is_initialized()}
        r['nvidia_smi']=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout
        r['disk_free_bytes']=__import__('shutil').disk_usage(allowed).free
        write_json(a.output,r)
    print(json.dumps({'ok':r['ok'],'verified_committed_updates':r['verified_committed_updates'],'report':str(a.output)},indent=2))
    return 0 if r['ok'] else 2


if __name__=='__main__': raise SystemExit(main())
