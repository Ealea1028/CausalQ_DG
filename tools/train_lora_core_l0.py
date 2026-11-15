"""Phase41 exact-contract source-only formal training, 8 fresh L0 core models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import torch

from causalq.datasets.segmentation import TrainTransform
from causalq.datasets.strict_source import build_source_datasets,SourceEvalTransform
from causalq.runtime.hpi_packets import packet_loader
from causalq.runtime.input_stream import CanonicalInputStream
from causalq.runtime.hpi_prepare import configure,Telemetry
from causalq.runtime.lora_training import seed_training_rng
from causalq.runtime.lora_performance import InputPrefetch,telemetry
from causalq.runtime.lora_formal import ROOT,DIRECTORY,VARIANTS,FormalCohort,core_head,build_backbone,load_contract
from causalq.runtime.identity import write_json,file_sha
from tools.check_lora_backbones import frozen_digest


def runtime():
    import platform,transformers
    return {'python':sys.version,'platform':platform.platform(),'torch':str(torch.__version__),
            'cuda':torch.version.cuda,'transformers':transformers.__version__,
            'gpu':torch.cuda.get_device_name(0),'visible_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
            'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader,nounits','--id=0'],text=True).strip(),
            'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()}


def worker(args):
    config,freeze=load_contract(); configure(); seed_training_rng(0)
    args.output.mkdir(parents=True,exist_ok=True)
    attempt=args.output/'attempts'/uuid.uuid4().hex
    attempt.mkdir(parents=True)
    env=runtime(); accepted=json.loads((DIRECTORY/'phase39_report.json').read_text())['environment']
    for key in ('python','torch','cuda','transformers','gpu','visible_memory_bytes'):
        if env[key]!=accepted[key]: raise ValueError(f'Runtime changed: {key}')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    profile=json.loads((DIRECTORY/'execution_profile.json').read_text())
    report={'study':'lora_causalq_1114','phase':41,'backbone':args.backbone,'git_sha':sha,'runtime':env,
        'ok':False,'target_data_accessed':False,'source_miou':None,'target_miou':None,'head_horizon':40000,
        'source_diagnostics_deferred_to_phase45':True,'formal_optimizer_updates_this_invocation':0,
        'cohort_ticks_this_invocation':0,'actual_backbone_image_views_this_invocation':0,'peak_live_input_slots':0,
        'attempt_directory':str(attempt),'telemetry_path':str(attempt/'telemetry.csv'),
        'input_trace_kind':'absolute source positions, IDs and canonical CPU metadata hash; no GPU tensor readback',
        'execution_profile_origin':profile['measured_scope']}
    cohort=None; identities={}; final_records={}; completed_at_start={}; telemetry_windows=[]
    started=time.monotonic()
    trace=None
    try:
        train,_=build_source_datasets(args.data_root/'gta5',TrainTransform((512,512)),SourceEvalTransform())
        b=build_backbone(args.backbone,args.pretrained_root)
        heads={name:core_head(b,name) for name in VARIANTS}
        cohort=FormalCohort(b,heads)
        report['trainable_parameters']={n:sum(p.numel() for p in m.parameters() if p.requires_grad) for n,m in heads.items()}
        report['lora_enabled']=False;report['lora_targets']=[]
        for name,m in heads.items():
            identities[name]={'study':'lora_causalq_1114','phase':41,'variant':name,'backbone':args.backbone,
                'lora_enabled':False,'centered':name in ('C0','CA'),'anchor_cqe':name in ('UA','CA'),
                'training_producer_git_sha':sha,'freeze_sha256':file_sha(DIRECTORY/'phase41_freeze.json'),
                'config_sha256':file_sha(ROOT/'configs/lora_causalq_1114/phase41_core_l0.json'),
                'execution_profile_sha256':file_sha(DIRECTORY/'execution_profile.json'),
                'data_root':str(args.data_root.resolve()),'runtime':env,
                'effective_source_batch':8,'microbatch':4,'horizon':40000,
                'initial_head_sha256':cohort.initial_head_digests[name],
                'source_stream':'hpi_step_sha256_v1','rng_initialization':'python_numpy_torch_seed0_v1'}
        first=cohort.restore(args.output/'checkpoints',identities)
        completed_at_start=cohort.cursors.copy(); report['restored_cursors']=completed_at_start
        cohort.commit(args.output/'checkpoints',identities,final_records)
        write_json(args.output/'resolved_configs.json',{'base_config':config,'heads':identities,'execution_profile':profile})
        base_digest=frozen_digest(b); versions=tuple(p._version for p in b.parameters())
        report['base_digest']=base_digest; report['initial_head_digests']=cohort.initial_head_digests
        if first==40000:
            cohort.commit(args.output/'checkpoints',identities,final_records)
        else:
            stream=CanonicalInputStream(train.pairs,start=first*8,stop=40000*8)
            loader,policy=packet_loader(stream,profile); report['loader_policy']=policy
            pipeline=InputPrefetch(loader,'cuda',profile['buffer_slots'])
            it=iter(pipeline); ticks=0; recent=[]; window_start=None; eligible_warmup=20
            window_wait_start=0.; pauses=[]; checkpoint_seconds=0.
            trace=(attempt/'input_trace.jsonl').open('w')
            with Telemetry(attempt/'telemetry.csv',1) as monitor:
                step=first
                while step<40000:
                    data=next(it)
                    samples=[{'views':data['views'][j],'label':data['labels'][j],'metadata':meta} for j,meta in enumerate(data['metadata'])]
                    tick_start=time.monotonic(); values=cohort.tick(samples,amp=True); ticks+=1; step+=1
                    report['cohort_ticks_this_invocation']=ticks
                    report['actual_backbone_image_views_this_invocation']=ticks*16
                    payload=json.dumps(data['metadata'],sort_keys=True,separators=(',',':')).encode()
                    trace.write(json.dumps({'step_zero_based':step-1,'first_source_position':(step-1)*8,
                        'ids':[m['id'] for m in data['metadata']],
                        'canonical_metadata_sha256':hashlib.sha256(payload).hexdigest(),
                        'updated_variants':list(values)},separators=(',',':'))+'\n')
                    recent.append((time.monotonic()-tick_start,values))
                    samples.clear(); del samples,data
                    if versions!=tuple(p._version for p in b.parameters()): raise ValueError('Frozen base mutated')
                    if ticks==eligible_warmup:
                        torch.cuda.synchronize(); window_start=time.monotonic()-monitor.started
                        window_wait_start=pipeline.total_data_wait_seconds
                    if ticks%50==0:
                        tensor_values={name:torch.stack([v[name] for _,v in recent if name in v]).float().mean(0).cpu().tolist()
                            for name in VARIANTS if any(name in v for _,v in recent)}
                        recent.clear()
                        status={'status':'running','git_sha':sha,'cursors':cohort.cursors.copy(),
                            'attempt_directory':str(attempt),
                            'updates_in_memory_total':sum(cohort.cursors.values()),
                            'committed_updates_total':sum(r['step'] for r in final_records.values()),
                            'optimizer_calls_this_invocation':sum(cohort.cursors[n]-completed_at_start[n] for n in VARIANTS),
                            'loss_and_gradient_norm_mean':tensor_values,'seconds':time.monotonic()-started,
                            'cohort_source_images_per_second':ticks*8/max(time.monotonic()-started,1e-9)}
                        write_json(args.output/'progress.json',status)
                        print(args.backbone,'tick',step,'head updates',cohort.cursors,flush=True)
                    end=time.monotonic()-monitor.started
                    active_seconds=end-window_start-sum(b-a for a,b in pauses) if window_start is not None else 0.
                    if window_start is not None and active_seconds>=profile['steady_seconds']:
                        util=telemetry(attempt/'telemetry.csv',start=window_start,end=end,exclude_intervals=pauses)
                        wait_fraction=(pipeline.total_data_wait_seconds-window_wait_start)/active_seconds
                        reserved_fraction=torch.cuda.max_memory_reserved()/env['visible_memory_bytes']
                        passed=(util['samples']>=profile['telemetry_samples_min'] and util['median']>=profile['gpu_median_min']
                            and util['p10']>=profile['gpu_p10_min'] and wait_fraction<.1 and reserved_fraction<=.85)
                        telemetry_windows.append({'start':window_start,'end':end,'active_seconds':active_seconds,
                            'checkpoint_exclusions':pauses.copy(),'data_wait_fraction':wait_fraction,
                            'reserved_fraction':reserved_fraction,'summary':util,'target_met':passed})
                        write_json(attempt/'steady_windows.json',telemetry_windows)
                        if not passed:
                            cohort.commit(args.output/'checkpoints',identities,final_records)
                            report['status']='needs_performance_review'
                            break
                        window_start=end;window_wait_start=pipeline.total_data_wait_seconds;pauses=[]
                    if step%1000==0:
                        trace.flush()
                        pause_start=time.monotonic()-monitor.started
                        cohort.commit(args.output/'checkpoints',identities,final_records)
                        pause_end=time.monotonic()-monitor.started;checkpoint_seconds+=pause_end-pause_start
                        if window_start is not None: pauses.append((pause_start,pause_end))
                report['peak_live_input_slots']=pipeline.peak_live_slots
                it.close(); del it,loader,pipeline
            cohort.commit(args.output/'checkpoints',identities,final_records)
            report['cohort_ticks_this_invocation']=ticks
            report['actual_backbone_image_views_this_invocation']=ticks*16
            report['checkpoint_seconds_in_training']=checkpoint_seconds
        if frozen_digest(b)!=base_digest: raise ValueError('Original base hash changed')
        report['base_unchanged']=True
        report['ok']=all(c==40000 for c in cohort.cursors.values()) and all(final_records.get(n,{}).get('step')==40000 for n in VARIANTS)
        report['status']='complete' if report['ok'] else report.get('status','incomplete')
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    except Exception:
        report['error']=traceback.format_exc(); report['status']='failed'
        report['backbone_views_successful_ticks_lower_bound']=report['actual_backbone_image_views_this_invocation']
        report['actual_backbone_image_views_this_invocation']=None
        if cohort is not None:
            try: cohort.commit(args.output/'checkpoints',identities,final_records)
            except Exception: report['checkpoint_error']=traceback.format_exc()
    if cohort is not None:
        report['cursors']=cohort.cursors
        report['committed_cursors']={n:final_records.get(n,{}).get('step',0) for n in VARIANTS}
        report['formal_optimizer_updates_total']=sum(report['committed_cursors'].values())
        report['formal_optimizer_updates_this_invocation']=sum(report['committed_cursors'][n]-completed_at_start.get(n,0) for n in VARIANTS)
        report['optimizer_calls_this_invocation']=sum(cohort.cursors[n]-completed_at_start.get(n,0) for n in VARIANTS)
        report['logical_model_image_views_this_invocation']=report['formal_optimizer_updates_this_invocation']*16
        report['tainted_head']=cohort.tainted_head
    if trace is not None:
        trace.close()
        report['input_trace_path']=str(attempt/'input_trace.jsonl')
        report['input_trace_sha256']=file_sha(attempt/'input_trace.jsonl')
    report['checkpoints']=final_records;report['steady_windows']=telemetry_windows
    report['seconds']=time.monotonic()-started;report['exit_code']=0 if report['ok'] else 2
    write_json(attempt/'training_report.json',report)
    write_json(args.output/'training_report.json',report)
    return report['exit_code']


def status(root):
    result={}
    for name in ('dinov3_vitl16','dinov2_vitl14_hpi_p16'):
        folder=Path(root)/name
        paths=[p for p in (folder/'training_report.json',folder/'progress.json') if p.exists()]
        r=json.loads(max(paths,key=lambda p:p.stat().st_mtime_ns).read_text()) if paths else {'status':'not_started'}
        result[name]={k:r[k] for k in ('status','ok','git_sha','cursors','committed_cursors',
            'committed_updates_total','formal_optimizer_updates_total','seconds','attempt_directory','error') if k in r}
    print(json.dumps(result,indent=2))


def run(args):
    config,freeze=load_contract()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip(): raise ValueError('Tracked source modified')
    args.output.mkdir(parents=True,exist_ok=True)
    # Forty immutable recovery segments per head need about 15GiB for all eight.
    remaining=0
    for name in config['backbones']:
        for variant in VARIANTS:
            markers=(args.output/name/'checkpoints'/variant).glob('*/complete.json')
            cursor=max([json.loads(p.read_text())['step'] for p in markers]+[0])
            remaining+=(40000-cursor+999)//1000
    required=remaining*(50<<20)+(2<<30)
    report={'study':'lora_causalq_1114','phase':41,'ok':False,'backbones':{},'target_data_accessed':False,
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'expected_total_formal_optimizer_updates':320000,'checkpoint_budget_bytes':required}
    try:
        if remaining and shutil.disk_usage(args.output).free<required:
            raise RuntimeError(f'Need {required/(1<<30):.1f}GiB free for remaining immutable checkpoints; expand data disk or return disk usage')
        for name in config['backbones']:
            out=args.output/name;out.mkdir(exist_ok=True)
            existing=out/'training_report.json'
            r=json.loads(existing.read_text()) if existing.exists() else None
            if r and r.get('status')=='needs_performance_review':
                report['backbones'][name]=r
                raise RuntimeError('Existing steady-window stop: return evidence for local review before another attempt')
            if r is None or not r['ok']:
                previous_mtime=existing.stat().st_mtime_ns if existing.exists() else -1
                cmd=[sys.executable,'-u','-m','tools.train_lora_core_l0','--worker','--backbone',name,
                    '--output',str(out),'--data-root',str(args.data_root),'--pretrained-root',str(args.pretrained_root)]
                with (out/'worker.log').open('a') as log:
                    child=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                if existing.exists() and existing.stat().st_mtime_ns!=previous_mtime: r=json.loads(existing.read_text())
                else: raise RuntimeError(f'Worker exited {child.returncode}; see {out}/worker.log')
            if r['git_sha']!=report['git_sha']: raise ValueError('Existing run producer differs')
            report['backbones'][name]=r
            if not r['ok']: break
        report['ok']=len(report['backbones'])==2 and all(r['ok'] for r in report['backbones'].values())
    except Exception: report['error']=traceback.format_exc()
    report['formal_optimizer_updates_total']=sum(r.get('formal_optimizer_updates_total',0) for r in report['backbones'].values())
    report['exit_code']=0 if report['ok'] else 2
    write_json(args.output/'report.json',report)
    print(json.dumps({'ok':report['ok'],'formal_optimizer_updates_total':report['formal_optimizer_updates_total'],'error':report.get('error')},indent=2),flush=True)
    return report['exit_code']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true');p.add_argument('--status',action='store_true')
    p.add_argument('--backbone',choices=('dinov3_vitl16','dinov2_vitl14_hpi_p16'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/datasets'))
    p.add_argument('--pretrained-root',type=Path,default=Path('/root/autodl-tmp/pretrained'))
    a=p.parse_args()
    if a.status: status(a.output);return 0
    if a.worker:
        if not a.backbone:p.error('Worker needs backbone')
        return worker(a)
    return run(a)


if __name__=='__main__':raise SystemExit(main())
