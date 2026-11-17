"""Read-only existing Phase41 engineering-prefix analysis. No model forward."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import torch
from causalq.runtime.identity import file_sha,tree_digest,write_json
from causalq.runtime.lora_formal import DIRECTORY,VARIANTS,core_head
from causalq.runtime.lora_training import make_optimizer,rng_digest
from tools.audit_lora_stopped import safe,ShapeOnlyBackbone,validate_state
from tools.check_lora_l0_batching import RUN,NAME


def tensor_difference(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Tensor shape/dtype mismatch')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Nonfinite state')
    x,y=a.double(),b.double();delta=(x-y).abs()
    failed=~torch.isclose(x,y,atol=1e-6,rtol=1e-5)
    return {'shape':list(a.shape),'max_abs':float(delta.max()) if a.numel() else 0.,
        'relative_l2':float(delta.norm()/x.norm().clamp_min(1e-30)),
        'failed_elements':int(failed.sum()),'elements':a.numel(),
        'max_abs_flat_index':int(delta.reshape(-1).argmax()) if a.numel() else None,
        'within_original_allclose':not bool(failed.any())}


def compare_prefixes(old,new,original,model):
    params=dict(model.named_parameters());names={id(p):n for n,p in params.items()}
    opt=make_optimizer(model);mapping={};groups=[]
    for index,ref in enumerate(opt.param_groups):
        ga,gb,g0=[s['optimizer']['param_groups'][index] for s in (old,new,original)]
        if ga!=gb:raise ValueError('Engineering optimizer groups differ')
        if ga['params']!=g0['params']:raise ValueError('Original optimizer mapping differs')
        for pid,p in zip(ga['params'],ref['params']):mapping[pid]=names[id(p)]
        groups.append({'index':index,'lr':ga['lr'],'weight_decay':ga['weight_decay'],'betas':ga['betas']})
    result={'parameters':{},'adamw':{},'groups':groups,
        'rng_equal':rng_digest(old['rng'])==rng_digest(new['rng']),
        'source_cursor_equal':old['source_cursor']==new['source_cursor'],
        'original_step':original['step'],'prefix_step':old['step']}
    if old['step']!=new['step'] or old['step']!=original['step']+1:raise ValueError('Expected same one-step prefix')
    if set(old['model'])!=set(new['model']):raise ValueError('Model keys differ')
    for n,a in old['model'].items():
        result['parameters'][n]=tensor_difference(a,new['model'][n])
    for pid,n in mapping.items():
        sa,sb,s0=[s['optimizer']['state'][pid] for s in (old,new,original)]
        if set(sa)!=set(sb):raise ValueError('AdamW slot keys differ')
        row={k:tensor_difference(v,sb[k]) if isinstance(v,torch.Tensor) else {'equal':v==sb[k]} for k,v in sa.items()}
        # Difference of recovered first-moment inputs, not saved actual gradients.
        beta=next(g['betas'][0] for g in old['optimizer']['param_groups'] if pid in g['params'])
        inferred_a=(sa['exp_avg'].double()-beta*s0['exp_avg'].double())/(1-beta)
        inferred_b=(sb['exp_avg'].double()-beta*s0['exp_avg'].double())/(1-beta)
        row['first_moment_inferred_clipped_gradient']=tensor_difference(inferred_a,inferred_b)
        result['adamw'][n]=row
    result['model_allclose']=all(v['within_original_allclose'] for v in result['parameters'].values())
    result['optimizer_allclose']=all(v.get('within_original_allclose',v.get('equal',False)) for row in result['adamw'].values() for k,v in row.items() if k!='first_moment_inferred_clipped_gradient')
    return result


def diagnose(source):
    failure=json.loads((DIRECTORY/'phase41_batch_failure_ee70c3e.json').read_text())
    stop=json.loads((DIRECTORY/'phase41_stop_acceptance.json').read_text())['latest'][NAME]
    source=safe(source,source)
    r={'phase':41,'action':'read-only-prefix-diagnosis','ok':True,'migration_accepted':False,
       'formal_updates':0,'engineering_updates':0,'payloads_deleted':0,'heads':{},'errors':[],
       'source':str(source),'causal_attribution':'not established; inferred gradients are derived from rounded first moments'}
    for n in VARIANTS:
        try:
            model=core_head(ShapeOnlyBackbone(),n);states=[];ids={}
            for mode in ('baseline','batch8','original'):
                folder=RUN/NAME/'checkpoints'/n/'00027000' if mode=='original' else source/mode/'prefix'/n
                safe(RUN if mode=='original' else source,folder)
                marker=safe(folder,folder/'complete.json');payload=safe(folder,folder/'state.pt')
                m=json.loads(marker.read_text())
                expected=stop[n] if mode=='original' else failure['prefix_markers'][mode][n]
                if mode=='original':
                    if m['identity']!=expected['identity'] or m['step']!=expected['step'] or m['sha256']!=expected['sha256']:raise ValueError('Formal snapshot identity changed')
                elif m!=expected:raise ValueError('Engineering prefix marker changed')
                if file_sha(payload)!=m['sha256']:raise ValueError('Payload SHA mismatch')
                state=torch.load(payload,map_location='cpu',weights_only=False)
                validate_state(state,m,model);states.append(state)
                ids[mode]={'sha256':m['sha256'],'step':m['step'],'identity':m['identity'],
                    'model_sha256':tree_digest(state['model']),'rng_sha256':rng_digest(state['rng'])}
            r['heads'][n]={'identities':ids,'comparison':compare_prefixes(*states,model)}
            print(n,'complete',flush=True)
        except Exception as exc:r['ok']=False;r['errors'].append({'head':n,'error':str(exc)})
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if sys.platform!='linux':p.error('Real payload inventory belongs on AutoDL Linux')
    import fcntl,subprocess
    failure=json.loads((DIRECTORY/'phase41_batch_failure_ee70c3e.json').read_text());source=Path(failure['remote_engineering_root'])
    if not source.is_dir():p.error('Original engineering run missing; return paths instead of creating it')
    if a.output.exists() or RUN in a.output.absolute().parents or source in a.output.absolute().parents:p.error('Use a fresh output outside original runs')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():p.error('Tracked source changed')
    with ExitStack() as stack:
        for lock in (RUN/'run.lock',RUN.parent/'gpu0.lock'):
            safe(RUN.parent,lock);f=stack.enter_context(lock.open('r+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        r=diagnose(source);r['git_sha']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        r['environment']={'python':sys.version,'torch':str(torch.__version__),'cuda_initialized':torch.cuda.is_initialized()}
        write_json(a.output,r)
    print(json.dumps({'ok':r['ok'],'report':str(a.output),'errors':r['errors']},indent=2));return 0 if r['ok'] else 2


if __name__=='__main__':raise SystemExit(main())
