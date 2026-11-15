"""Phase40 bounded source-only UA/B3 performance calibration; discard all heads."""
import argparse
import gc
import hashlib
import json
import os
import platform
import signal
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from causalq.datasets.segmentation import TrainTransform
from causalq.datasets.strict_source import build_source_datasets,SourceEvalTransform,MANIFEST_HASHES
from causalq.models.hpi_anchor_only import freeze_query_residuals,verify_frozen_residuals
from causalq.models.hpi_query_route import QueryFeatureRouter
from causalq.models.lora_routing import LoRARoutedQuerySegmentor
from causalq.runtime.hpi_packets import packet_loader,PacketDataset
from causalq.runtime.input_stream import CanonicalInputStream
from causalq.runtime.hpi_prepare import configure,Telemetry
from causalq.runtime.identity import file_sha,tree_digest,write_json
from causalq.runtime.lora_training import update,make_optimizer,compact_state,save_checkpoint,seed_training_rng
from causalq.runtime.lora_performance import CANDIDATES,BF16_BATCH_TOLERANCES,InputPrefetch,CudaTimings,telemetry,select
from tools.check_lora_resume import ROOT,PROFILE,build,compare
from tools.check_lora_backbones import frozen_digest

WARMUP=20
TIMED_SECONDS=120
MAX_CANDIDATE_SECONDS=300
MAX_UPDATES=4096


def make_model(name,weights,variant):
    if variant=='UA': return build(name,weights)
    if variant!='B3': raise ValueError('Only UA/B3 engineering workloads released')
    m=build(name,weights,head_type=LoRARoutedQuerySegmentor)
    freeze_query_residuals(m)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261031)
        m.query_router=QueryFeatureRouter(m.backbone.hidden_size).cuda()
    return m


@torch.no_grad()
def check_batch_features(backbone,views,microbatch):
    """BF16 physical batch drift, separate from exact zero-adapter validation."""
    fields=lambda f:(f.patch_tokens,f.cls_token,*f.intermediate_maps)
    def run(size):
        chunks=[]
        for start in range(0,8,size):
            with torch.autocast('cuda',dtype=torch.bfloat16): chunks.append(fields(backbone(views[start:start+size])))
        return [torch.cat([row[j] for row in chunks]) for j in range(len(chunks[0]))]
    reference,actual=run(1),run(microbatch)
    rows=[]; tol=BF16_BATCH_TOLERANCES
    for a,b in zip(reference,actual):
        relative=float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-12))
        if not torch.allclose(a.float(),b.float(),atol=tol['atol'],rtol=tol['rtol']) or relative>tol['relative_l2']:
            raise ValueError(f'Physical batch feature drift: relative L2 {relative}')
        rows.append({'max_abs':float((a.float()-b.float()).abs().max()),'relative_l2':relative})
    return rows


def first_update_check(model,packet,output,candidate):
    grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
    state={'model':compact_state(model),'gradients':grads,
           'input_sha256':tree_digest((packet['views'],packet['labels']))}
    if candidate=='micro1_serial':
        torch.save(state,output/'first_update.pt')
        write_json(output/'first_update_identity.json',{'sha256':file_sha(output/'first_update.pt'),'input_sha256':state['input_sha256'],
            'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
        return {'model_max_abs':0.,'gradient_relative_l2':0.,'gradient_max_abs':0.}
    base=output.parent/'micro1_serial'
    identity=json.loads((base/'first_update_identity.json').read_text())
    if identity['git_sha']!=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip():
        raise ValueError('Baseline snapshot producer differs')
    if file_sha(base/'first_update.pt')!=identity['sha256']: raise ValueError('Baseline snapshot hash changed')
    ref=torch.load(base/'first_update.pt',map_location='cpu',weights_only=True)
    if ref['input_sha256']!=state['input_sha256'] or set(ref['gradients'])!=set(grads):
        raise ValueError('Input/gradient coverage differs from baseline')
    model_error=compare(ref['model'],state['model'])
    squared=0.; energy=0.; max_abs=0.; max_ref=0.
    for key,a in ref['gradients'].items():
        b=grads[key]; diff=a.double()-b.double()
        squared+=float(diff.square().sum()); energy+=float(a.double().square().sum())
        max_abs=max(max_abs,float(diff.abs().max())); max_ref=max(max_ref,float(a.abs().max()))
    relative=(squared/max(energy,1e-24))**.5
    tol=BF16_BATCH_TOLERANCES
    if relative>tol['relative_l2'] or max_abs>tol['atol']+tol['rtol']*max_ref:
        raise ValueError(f'Batched gradient drift: {relative}')
    return {'model_max_abs':model_error,'gradient_relative_l2':relative,'gradient_max_abs':max_abs}


def candidate_worker(args):
    started=time.monotonic(); deadline=started+MAX_CANDIDATE_SECONDS
    configure(); seed_training_rng(0)
    args.output.mkdir(parents=True,exist_ok=True)
    policy=next(p for p in CANDIDATES if p['name']==args.candidate)
    row={'ok':False,**policy,'model':args.model,'variant':args.variant,'formal_optimizer_updates':0,
         'target_data_accessed':False,'engineering_updates':0,'numerical_ok':False,
         'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'source_loss_backend':'canonical_cpu_validated_fp32_static_reduction_v1',
         'diagnostic_backbone_views':16,'loader_prefetch_may_decode_unused_source_samples':True}
    row['visible_memory_bytes']=torch.cuda.get_device_properties(0).total_memory
    try:
        import transformers
        accepted=json.loads((ROOT/'experiments/LORA_CAUSALQ_1114/phase39_report.json').read_text())['environment']
        if (str(torch.__version__)!=accepted['torch'] or transformers.__version__!=accepted['transformers']
            or torch.cuda.get_device_name(0)!=accepted['gpu'] or sys.version!=accepted['python'] or torch.version.cuda!=accepted['cuda']
            or row['visible_memory_bytes']!=accepted['visible_memory_bytes']):
            raise ValueError('Accepted runtime changed')
        train,_=build_source_datasets(args.data_root/'gta5',TrainTransform((512,512)),SourceEvalTransform())
        m=make_model(args.model,args.pretrained_root,args.variant); opt=make_optimizer(m)
        frozen_before=frozen_digest(m)
        stream=CanonicalInputStream(train.pairs,stop=MAX_UPDATES*8)
        # Numerical checks use the first real source packet, outside timed windows.
        first=PacketDataset(stream,8)[0]
        views=first['views'][:,0].cuda()
        m.eval(); row['feature_numerics']=check_batch_features(m.backbone,views,policy['microbatch'])
        del views
        loader,loader_policy=packet_loader(stream,PROFILE)
        pipeline=InputPrefetch(loader,'cuda',policy['buffer_slots'])
        it=iter(pipeline); torch.cuda.reset_peak_memory_stats()
        timers=CudaTimings(); timings_samples=0; metrics=[]; copy_events=[]
        trace=hashlib.sha256(); wait_total=0.; stage_total=0.; warm_start=time.monotonic()
        def step_once(step,timed=False):
            nonlocal wait_total,stage_total,timings_samples
            waited,staged=pipeline.total_data_wait_seconds,pipeline.total_staging_seconds
            data=next(it)
            samples=[{'views':data['views'][j],'label':data['labels'][j],'metadata':meta} for j,meta in enumerate(data['metadata'])]
            values=update(m,opt,samples,step,microbatch=policy['microbatch'],amp=True,variant=args.variant,
                          timer=timers if timed and (step-WARMUP)%10==0 else None,return_tensors=True,prevalidated_source=True)
            if step==0:
                row['first_update_comparison']=first_update_check(m,first,args.output,args.candidate)
                row['numerical_ok']=True
            if timed:
                metrics.append(values.detach()); copy_events.append(data['copy_events'])
                wait_total+=pipeline.total_data_wait_seconds-waited; stage_total+=pipeline.total_staging_seconds-staged
                trace.update('|'.join(s['metadata']['id_parameters_sha256'] for s in samples).encode())
                if (step-WARMUP)%10==0: timings_samples+=1
            samples.clear(); del data,samples,values
            row['engineering_updates']+=1
        for step in range(WARMUP):
            if time.monotonic()>=deadline: raise TimeoutError('Warmup exceeded candidate wall-clock cap')
            step_once(step)
        torch.cuda.synchronize(); row['warmup_seconds']=time.monotonic()-warm_start
        row['cold_and_warmup_seconds']=time.monotonic()-started
        count=0
        with Telemetry(args.output/'telemetry.csv',1):
            timed_start=time.monotonic()
            while time.monotonic()-timed_start<TIMED_SECONDS and time.monotonic()<deadline and WARMUP+count<MAX_UPDATES:
                step_once(WARMUP+count,True); count+=1
                if count%50==0: print(args.model,args.variant,args.candidate,'timed updates',count,flush=True)
            torch.cuda.synchronize(); elapsed=time.monotonic()-timed_start
        # Input wait for a prefetched packet was charged when fetched: steady includes all refills.
        row.update(timed_seconds=elapsed,timed_updates=count,source_images_per_second=count*8/max(elapsed,1e-9),
            views_per_second=count*16/max(elapsed,1e-9),updates_per_second=count/max(elapsed,1e-9),
            data_wait_seconds=wait_total,data_wait_fraction=wait_total/max(elapsed,1e-9),staging_seconds=stage_total,
            sampled_gpu_milliseconds=timers.summarize(),gpu_timing_sampled_updates=timings_samples,
            h2d_milliseconds=sum(a.elapsed_time(b) for a,b in copy_events),
            telemetry=telemetry(args.output/'telemetry.csv'),input_trace_sha256=trace.hexdigest(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            loader_policy=loader_policy,peak_live_input_slots=pipeline.peak_live_slots,
            effective_source_batch=8,physical_source_batch=policy['microbatch'],
            gradient_accumulation=8//policy['microbatch'],view_forward='original_then_photo_separate_graphs',
            gpu_activity_is_not_sm_arithmetic_utilization=True)
        if metrics:
            summary=torch.stack(metrics).float().cpu()
            row['loss_mean']=float(summary[:,0].mean()); row['gradient_norm_mean']=float(summary[:,1].mean())
        # Close loader before CPU hashing/snapshot cost; no adapted features are cached.
        it.close(); del loader,it,pipeline,first
        if frozen_digest(m)!=frozen_before: raise ValueError('Original backbone weights changed')
        if args.variant=='B3': verify_frozen_residuals(m)
        row['base_unchanged']=True
        checkpoint_start=time.monotonic()
        save_checkpoint(args.output/'disposable_checkpoint',m,opt,row['engineering_updates'],
                        {'study':'lora_causalq_1114','action':'performance-check','formal':False,'model':args.model,'variant':args.variant,
                         'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'execution':policy})
        row['checkpoint_seconds']=time.monotonic()-checkpoint_start
        row['ok']=elapsed>=TIMED_SECONDS
    except Exception:
        row['error']=traceback.format_exc()
    row['total_seconds']=time.monotonic()-started
    write_json(args.output/'candidate.json',row)
    print(json.dumps(row,indent=2),flush=True)
    gc.collect()


def run(args):
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip(): raise ValueError('Tracked source is modified')
    acceptance=ROOT/'experiments/LORA_CAUSALQ_1114/phase40_resume_acceptance.json'
    if not json.loads(acceptance.read_text())['accepted']: raise ValueError('Recovery not accepted')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists(): raise ValueError('Return existing report before retry')
    report={'study':'lora_causalq_1114','phase':40,'action':'performance-check','ok':False,'workloads':[],
            'formal_optimizer_updates':0,'target_data_accessed':False,'source_data_accessed':True,
            'formal_training_authorized':False,'performance_profile_frozen':False,
            'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'resume_acceptance_sha256':file_sha(acceptance),'batch_numerical_tolerances':BF16_BATCH_TOLERANCES,
            'bounds':{'candidates_per_workload':3,'warmup_updates':20,'timed_seconds':120,'candidate_cap_seconds':300,'max_updates_per_candidate':MAX_UPDATES}}
    try:
        report['runtime']={'python':sys.version,'platform':platform.platform(),'torch':str(torch.__version__),
            'cuda_runtime':torch.version.cuda,'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),
            'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                                                '--format=csv,noheader,nounits','--id=0'],text=True).strip()}
        report['source_manifest_sha256']=MANIFEST_HASHES
        report['data_root']=str(args.data_root.resolve()); report['pretrained_root']=str(args.pretrained_root.resolve())
        report['phase39_report_sha256']=file_sha(ROOT/'experiments/LORA_CAUSALQ_1114/phase39_report.json')
        for name in ('dinov3_vitl16','dinov2_vitl14_hpi_p16'):
            for variant in ('UA','B3'):
                rows=[]
                for c in CANDIDATES:
                    out=args.output/name/variant/c['name']; out.mkdir(parents=True,exist_ok=True)
                    if not (out/'candidate.json').exists():
                        cmd=[sys.executable,'-u','-m','tools.check_lora_performance','--worker','--model',name,'--variant',variant,
                             '--candidate',c['name'],'--output',str(out),'--data-root',str(args.data_root),'--pretrained-root',str(args.pretrained_root)]
                        child_started=time.monotonic()
                        with (out/'worker.log').open('a') as log:
                            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
                            try:
                                code=child.wait(timeout=MAX_CANDIDATE_SECONDS)
                                if code: raise subprocess.CalledProcessError(code,cmd)
                            except subprocess.TimeoutExpired:
                                # Kill only this isolated candidate group, including its loader workers.
                                try: os.killpg(child.pid,signal.SIGTERM)
                                except ProcessLookupError: pass
                                try: child.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    os.killpg(child.pid,signal.SIGKILL); child.wait()
                                write_json(out/'candidate.json',{'name':c['name'],'ok':False,'numerical_ok':False,
                                    'engineering_updates':None,'error':'Candidate process group exceeded 300-second bound',
                                    'git_sha':report['git_sha'],'model':name,'variant':variant,
                                    'visible_memory_bytes':json.loads((ROOT/'experiments/LORA_CAUSALQ_1114/phase39_report.json').read_text())['environment']['visible_memory_bytes']})
                        write_json(out/'process_timing.json',{'wall_seconds':time.monotonic()-child_started,'hard_cap_seconds':MAX_CANDIDATE_SECONDS})
                    result=json.loads((out/'candidate.json').read_text())
                    if result['git_sha']!=report['git_sha'] or result['model']!=name or result['variant']!=variant or result['name']!=c['name']:
                        raise ValueError('Existing candidate identity mismatch')
                    rows.append(result)
                    print(name,variant,c['name'],'complete',rows[-1]['ok'],flush=True)
                memory=rows[0]['visible_memory_bytes']
                if any(r['visible_memory_bytes']!=memory for r in rows): raise ValueError('Hardware changed between candidates')
                report['workloads'].append({'model':name,'variant':variant,'candidates':rows,'selection':select(rows,memory)})
                write_json(args.output/'progress.json',report)
        report['ok']=all(w['selection']['selected'] is not None for w in report['workloads'])
        report['performance_target_met']=all(w['selection']['performance_target_met'] for w in report['workloads'])
        counts=[r['engineering_updates'] for w in report['workloads'] for r in w['candidates']]
        report['engineering_update_counts_complete']=all(isinstance(x,int) for x in counts)
        report['engineering_updates']=sum(counts) if report['engineering_update_counts_complete'] else None
        if not report['engineering_update_counts_complete']:
            report['ok']=False; report['performance_target_met']=False
    except Exception: report['error']=traceback.format_exc()
    report['exit_code']=0 if report['ok'] else 1
    write_json(args.output/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report['exit_code']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true'); p.add_argument('--model',choices=('dinov3_vitl16','dinov2_vitl14_hpi_p16'))
    p.add_argument('--variant',choices=('UA','B3')); p.add_argument('--candidate',choices=[x['name'] for x in CANDIDATES])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/datasets'))
    p.add_argument('--pretrained-root',type=Path,default=Path('/root/autodl-tmp/pretrained'))
    a=p.parse_args()
    if a.worker:
        if not all((a.model,a.variant,a.candidate)): p.error('Worker needs model, variant and candidate')
        candidate_worker(a); return 0
    return run(a)


if __name__=='__main__': raise SystemExit(main())
