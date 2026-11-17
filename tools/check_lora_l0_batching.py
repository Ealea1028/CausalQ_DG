"""Phase41 protected D2/27000 L0 candidate comparison; zero formal updates."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import torch
from causalq.runtime.lora_formal import ROOT,DIRECTORY,FormalCohort,VARIANTS,core_head,build_backbone
from causalq.runtime.lora_training import restore_checkpoint,save_checkpoint,compact_state,seed_training_rng,rng_digest
from causalq.runtime.lora_training import FrozenFeatureBatch
from causalq.runtime.cohort_runner import capture_rng
from causalq.runtime.lora_batched import features,update_batched
from causalq.runtime.identity import file_sha,write_json,tree_digest
from causalq.runtime.hpi_prepare import configure,Telemetry
from causalq.runtime.lora_performance import InputPrefetch,telemetry
from causalq.runtime.hpi_packets import packet_loader
from causalq.runtime.input_stream import CanonicalInputStream
from causalq.datasets.strict_source import build_source_datasets,SourceEvalTransform
from causalq.datasets.segmentation import TrainTransform
from tools.check_lora_resume import compare
from tools.check_lora_backbones import frozen_digest
from tools.train_lora_core_l0 import runtime

NAME='dinov2_vitl14_hpi_p16'
RUN=Path('/root/autodl-tmp/outputs/CausalQ_DG/lora_causalq_1114/phase41_core_1af6b2e')
MODES=('baseline','batch8','combined16')


def load(base,rows):
    c=FormalCohort(base,{n:core_head(base,n) for n in VARIANTS})
    for n,row in rows.items():
        p=RUN/NAME/'checkpoints'/n/'00027000'
        if file_sha(p/'state.pt')!=row['sha256']:raise ValueError('Pinned snapshot changed')
        c.cursors[n]=restore_checkpoint(p,c.heads[n],c.optimizers[n],row['identity'])
    return c


def tick(c,packets,step,mode):
    if mode=='baseline':return c.tick(packets,amp=True)
    f=features(c.backbone,packets,8 if mode=='batch8' else 16,True)
    out={}
    for n in VARIANTS:
        if c.cursors[n]!=step:raise ValueError('Engineering cursor mismatch')
        out[n]=update_batched(c.heads[n],c.optimizers[n],packets,step,n,f,amp=True)
        c.cursors[n]+=1
    return out


def snapshot(c):
    def cpu(x):
        if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
        if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
        if isinstance(x,list):return [cpu(v) for v in x]
        if isinstance(x,tuple):return tuple(cpu(v) for v in x)
        return x
    return {n:{'model':compact_state(m),'optimizer':cpu(c.optimizers[n].state_dict()),
            'gradients':{k:p.grad.detach().cpu().clone() for k,p in m.named_parameters() if p.requires_grad and p.grad is not None}}
            for n,m in c.heads.items()}


def errors(ref,actual):
    rows={}
    for n in VARIANTS:
        model=compare(ref[n]['model'],actual[n]['model']);opt=compare(ref[n]['optimizer'],actual[n]['optimizer'])
        a,b=ref[n]['gradients'],actual[n]['gradients']
        if set(a)!=set(b):raise ValueError('Gradient coverage changed')
        energy=sum(float(v.double().square().sum()) for v in a.values())
        delta=sum(float((v.double()-b[k].double()).square().sum()) for k,v in a.items())
        rel=(delta/max(energy,1e-24))**.5
        if model>1e-6 or opt>1e-6 or rel>.02:raise ValueError(f'Numerical mismatch {n}: {model} {opt} {rel}')
        for k,v in a.items():torch.testing.assert_close(v,b[k],atol=.1,rtol=.05)
        rows[n]={'model_max_abs':model,'optimizer_max_abs':opt,'clipped_gradient_relative_l2':rel}
    return rows


def feature_check(base,packets,mode):
    old=FrozenFeatureBatch.extract(base,packets,4,True)
    new=features(base,packets,8 if mode=='batch8' else 16,True)
    result=[]
    def fields(f):return (f.patch_map,f.patch_tokens,f.cls_token,f.register_tokens,*f.intermediate_maps)
    for view in (0,1):
        for i,b in enumerate(fields(new[view])):
            chunks=[fields(old.entries[start,view])[i] for start in (0,4)]
            if b is None:
                if any(x is not None for x in chunks):raise ValueError('Feature metadata missing')
                continue
            a=torch.cat(chunks)
            torch.testing.assert_close(a,b,atol=.1,rtol=.05)
            rel=float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-12))
            if rel>.02:raise ValueError('Feature relative L2 exceeds .02')
            result.append({'view':view,'field':i,'shape':list(b.shape),'relative_l2':rel})
    return result


def worker(a):
    worker_started=time.monotonic()
    configure();seed_training_rng(0);a.output.mkdir(parents=True,exist_ok=True)
    r={'ok':False,'mode':a.mode,'formal_updates':0,'engineering_completed_head_updates':0,'engineering_ticks':0,
       'errors':[],'old_run_readonly':True,'selection_authorized':False}
    try:
        acceptance=json.loads((DIRECTORY/'phase41_stop_acceptance.json').read_text());rows=acceptance['latest'][NAME]
        if any(row['step']!=27000 for row in rows.values()):raise ValueError('Expected protected common27000')
        r['runtime']=runtime()
        for key in ('python','torch','cuda','transformers','gpu','visible_memory_bytes'):
            if r['runtime'][key]!=rows['U0']['identity']['runtime'][key]:raise ValueError('Runtime changed: '+key)
        train,_=build_source_datasets(Path('/root/autodl-tmp/datasets/gta5'),TrainTransform((512,512)),SourceEvalTransform())
        profile=json.loads((DIRECTORY/'execution_profile.json').read_text())
        base=build_backbone(NAME,Path('/root/autodl-tmp/pretrained'));digest=frozen_digest(base);r['base_digest']=digest
        versions=tuple(p._version for p in base.parameters())
        def packets():
            stream=CanonicalInputStream(train.pairs,start=27000*8,stop=(27000+4096)*8)
            loader,policy=packet_loader(stream,profile);r['loader_policy']=policy
            pipeline=InputPrefetch(loader,'cuda',2)
            return pipeline,iter(pipeline)
        pipe,it=packets();first=[]
        for _ in range(2):
            d=next(it);first.append([{'views':d['views'][j],'label':d['labels'][j],'metadata':m} for j,m in enumerate(d['metadata'])])
        it.close();del it,pipe
        if a.mode!='baseline':r['feature_comparison']=feature_check(base,first[0],a.mode)
        r['input_sha256']=tree_digest([(p['views'],p['label'],p['metadata']) for batch in first for p in batch])
        def step(c,p,index):
            # Persist observed completed engineering work; hard-killed work remains unknown in parent.
            result=tick(c,p,index,a.mode);r['engineering_ticks']+=1;r['engineering_completed_head_updates']+=4
            return result
        c=load(base,rows);step(c,first[0],27000);step(c,first[1],27001)
        ref=snapshot(c);rng=rng_digest(capture_rng());del c
        c=load(base,rows);step(c,first[0],27000)
        identity={'engineering_only':True,'mode':a.mode,'source_producer':rows['U0']['identity']['training_producer_git_sha']}
        for n in VARIANTS:save_checkpoint(a.output/'prefix'/n,c.heads[n],c.optimizers[n],27001,identity)
        del c;c=load(base,rows)
        for n in VARIANTS:c.cursors[n]=restore_checkpoint(a.output/'prefix'/n,c.heads[n],c.optimizers[n],identity)
        step(c,first[1],27001);r['self_recovery']=errors(ref,snapshot(c))
        if rng!=rng_digest(capture_rng()):raise ValueError('Self recovery RNG mismatch')
        r['self_recovery_rng_equal']=True;del c
        refpath=a.output.parent/'baseline/reference.pt'
        if a.mode=='baseline':
            torch.save({'heads':ref,'input_sha256':r['input_sha256'],'rng':rng},refpath)
            write_json(refpath.with_suffix('.json'),{'sha256':file_sha(refpath)})
        else:
            meta=json.loads(refpath.with_suffix('.json').read_text())
            if meta['sha256']!=file_sha(refpath):raise ValueError('Engineering reference changed')
            old=torch.load(refpath,map_location='cpu',weights_only=False)
            if old['input_sha256']!=r['input_sha256'] or old['rng']!=rng:raise ValueError('Old/new input or RNG mismatch')
            r['old_new']=errors(old['heads'],ref);del old
        del ref,first
        c=load(base,rows);pipe,it=packets();windows=[];warm_end=None;start_wait=0.
        r['loading_hash_and_correctness_seconds']=time.monotonic()-worker_started
        with Telemetry(a.output/'telemetry.csv',1) as monitor:
            for offset in range(4096-r['engineering_ticks']):
                d=next(it);p=[{'views':d['views'][j],'label':d['labels'][j],'metadata':m} for j,m in enumerate(d['metadata'])]
                step(c,p,27000+offset);del p,d
                if offset==19:
                    torch.cuda.synchronize();warm_end=time.monotonic()-monitor.started;start_wait=pipe.total_data_wait_seconds
                    r['warmup_seconds']=warm_end
                if warm_end is not None and time.monotonic()-monitor.started-warm_end>=120:
                    torch.cuda.synchronize();end=time.monotonic()-monitor.started;duration=end-warm_end
                    windows=[telemetry(a.output/'telemetry.csv',start=warm_end,end=warm_end+duration/2),telemetry(a.output/'telemetry.csv',start=warm_end+duration/2,end=end)]
                    r.update(timed_seconds=duration,timed_ticks=offset+1-20,cohort_ticks_per_second=(offset+1-20)/duration,
                        data_wait_fraction=(pipe.total_data_wait_seconds-start_wait)/duration,
                        telemetry=telemetry(a.output/'telemetry.csv',start=warm_end,end=end),subwindows=windows)
                    break
        it.close();del it,pipe
        if not windows:raise ValueError('4096 tick limit before steady window')
        r['peak_reserved_bytes']=torch.cuda.max_memory_reserved();r['visible_memory_bytes']=torch.cuda.get_device_properties(0).total_memory
        r['base_unchanged']=versions==tuple(p._version for p in base.parameters()) and frozen_digest(base)==digest
        if not r['base_unchanged']:raise ValueError('Frozen base mutated')
        for n,row in rows.items():
            if file_sha(RUN/NAME/'checkpoints'/n/'00027000/state.pt')!=row['sha256']:raise ValueError('Original payload mutated')
        r['ok']=True
    except Exception:
        r['errors'].append(traceback.format_exc());r['engineering_counts_are_completed_tick_lower_bounds']=True
    write_json(a.output/'report.json',r);return 0 if r['ok'] else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=MODES);a=p.parse_args()
    if a.mode:return worker(a)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Tracked code modified')
    if a.output.exists():raise ValueError('Fresh engineering output required')
    a.output.mkdir(parents=True);reports=[]
    import fcntl
    with ExitStack() as locks:
        for path in (RUN/'run.lock',RUN.parent/'gpu0.lock'):
            f=locks.enter_context(path.open('r+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Measure availability before any engineering updates; no replacement utilization claims.
        subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
        for mode in MODES:
            out=a.output/mode;out.mkdir()
            process_started=time.monotonic()
            with (out/'worker.log').open('w') as log:
                proc=subprocess.Popen([sys.executable,'-u','-m','tools.check_lora_l0_batching','--mode',mode,'--output',str(out)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:proc.wait(timeout=300)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                finally:
                    if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            row=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else {'ok':False,'mode':mode,'engineering_ticks':None,'errors':['Worker failed or hit300-second process cap; incomplete engineering work unknown']}
            row['process_wall_seconds']=time.monotonic()-process_started
            reports.append(row)
            if not row['ok']:break
    write_json(a.output/'report.json',{'phase':41,'formal_updates':0,'migration_accepted':False,'candidates':reports,
        'ok':len(reports)==3 and all(r['ok'] for r in reports),
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'selection':'await local numerical/telemetry/throughput review; do not resume formal training'})
    return 0 if len(reports)==3 and all(r['ok'] for r in reports) else 2


if __name__=='__main__':raise SystemExit(main())
