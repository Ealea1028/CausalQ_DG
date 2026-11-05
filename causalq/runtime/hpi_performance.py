"""Bounded v3 execution experiment, isolated from all formal checkpoints."""
import csv
import gc
import json
import statistics
import tempfile
import time
from pathlib import Path
import torch
import yaml
from causalq.datasets.segmentation import TrainTransform
from causalq.datasets.strict_source import SourceEvalTransform, build_source_datasets
from causalq.models.dinov2_wrapper import DINOv2Backbone
from causalq.protocols.hpi_dinov2_20261031 import ROOT, STUDY, build_head, initialization_identity
from .cohort_runner import CohortRunner, IndependentHead, capture_rng, restore_rng, commit_head, head_state
from .feature_pipeline import FeaturePipeline, slice_features, feature_tensors
from .hpi_double_buffer import DoubleBuffer
from .hpi_packets import packet_loader, packet_policy
from .hpi_prepare import configure, feature_error, Telemetry
from .hpi_training import DIRECTORY, verify_freeze
from .identity import write_json, file_sha, tree_digest
from .input_stream import CanonicalInputStream, make_loader, loader_policy


def settings():
    return yaml.safe_load((ROOT/'configs'/STUDY/'performance_v3.yaml').read_bytes())


def choose(rows,metric,tie):
    best=max(row[metric] for row in rows)
    return min((r for r in rows if r[metric]>=best*(1-tie)),
               key=lambda r:(r['memory_bytes'],r['workers'],-r[metric]))


def update_budget(candidates,options):
    if len(candidates)>3:raise ValueError('At most three full-cohort candidates')
    updates=sum((options['warmup_windows']+options['timed_windows'])*p['backbone_microbatch_views']//2 for p in candidates)
    if updates>min(1024,options['maximum_updates_per_head']):raise ValueError('Disposable update cap exceeded')
    return updates


def flush_metrics(records,*,batched,stream=None):
    """One small D2H per head/flush for v3; the control retains v2 transfers."""
    result={}
    for name in ('U0','UA'):
        rows=[r['heads'][name] for r in records if name in r['heads']]
        if not rows:continue
        if batched:
            numbers=torch.stack([torch.cat((r['losses'],r['gradient_norm'].reshape(1))) for r in rows]).cpu().tolist()
        else:
            numbers=[r['losses'].cpu().tolist()+[float(r['gradient_norm'].cpu())] for r in rows]
        if rows[-1]['timing_events']:rows[-1]['timing_events'][1].synchronize()
        times=[r['timing_events'][0].elapsed_time(r['timing_events'][1]) for r in rows if r['timing_events']]
        result[name]={'updates':len(rows),'gpu_event_ms':sum(times),'last_loss_and_norm':numbers[-1]}
        if stream is not None:
            lines=(json.dumps({'head':name,'update':row['optimizer_updates'],'losses_norm':values})+'\n' for row,values in zip(rows,numbers))
            if batched:stream.write(''.join(lines))
            else:
                for line in lines:stream.write(line)
            stream.flush()
    return result


def input_probe(pairs,profile,options):
    started=time.monotonic();waits=[];n=0;first=None;collate=0.
    loader,policy=packet_loader(CanonicalInputStream(pairs,stop=options['input_probe_samples']),profile)
    last=started
    for packet in loader:
        now=time.monotonic();count=len(packet['metadata'])
        waits.append((now-last,count));last=now;n+=count;collate+=packet['collate_seconds']
        if first is None:first=tree_digest((packet['views'],packet['labels']))
    steady=waits[options['input_warmup_packets']:]
    return {'workers':profile['data_workers'],'prefetch':profile['prefetch_factor'],'policy':policy,
            'samples':n,'cold_first_seconds':waits[0][0],'wall_seconds':time.monotonic()-started,
            'steady_samples_per_second':sum(n for _,n in steady)/max(sum(t for t,_ in steady),1e-9),
            'collate_cpu_seconds':collate,'memory_bytes':policy['host_bound_bytes'],'first_packet_digest':first}


def activity(path,start,end,minimum):
    with Path(path).open() as stream:
        values=sorted(float(row['utilization_percent']) for row in csv.DictReader(stream)
            if start<=float(row['elapsed_seconds'])<=end and row['status']=='ok' and row['utilization_percent'].strip().isdigit())
    n=len(values)
    return {'interval_seconds':end-start,'samples':n,'steady_eligible':end-start>=minimum and n>=60,
            'median_percent':statistics.median(values) if n else None,
            'mean_percent':statistics.mean(values) if n else None,
            'p10_percent':values[int(.1*(n-1))] if n else None,'p90_percent':values[int(.9*(n-1))] if n else None,
            'below_50_fraction':sum(x<50 for x in values)/n if n else None}


def candidate_run(name,profile,options,pairs,backbone,out,initial_rng,initial_sha,reference,reference_inputs):
    restore_rng(initial_rng)
    heads=[]
    for variant in ('U0','UA'):
        model=build_head(backbone).cuda()
        if initialization_identity(model)['initial_head_sha256']!=initial_sha:raise ValueError('Fresh head identity differs')
        optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
        heads.append(IndependentHead(variant,model,optim,capture_rng(),anchor_cqe=variant=='UA'))
    runner=CohortRunner(heads)
    windows=options['warmup_windows']+options['timed_windows']
    steps=windows*profile['backbone_microbatch_views']//2
    batched=name!='v2_serial'
    dataset=CanonicalInputStream(pairs,stop=steps)
    loader,policy=packet_loader(dataset,profile) if batched else (make_loader(dataset,profile),loader_policy(profile))
    pipeline=DoubleBuffer(backbone,profile) if batched else FeaturePipeline(backbone,profile)
    traces=[];records=[];head_ms={'U0':0.,'UA':0.};timed=[];inputs=[];errors=None
    torch.cuda.reset_peak_memory_stats()
    cold=time.monotonic();start=None;save_seconds=0.
    telemetry=out/f'{name}_telemetry.csv'
    with Telemetry(telemetry,2) as monitor, tempfile.TemporaryDirectory(prefix='disposable_',dir=out) as scratch, (out/f'{name}_updates.jsonl').open('w') as log:
        iterator=iter(pipeline(loader))
        try:
            for index in range(windows):
                window=next(iterator)
                if index==0:
                    actual=slice_features(window.features,0,2)
                    # A small CPU reference cannot keep an old GPU slot alive.
                    from causalq.models.dinov3_wrapper import DINOv3Features
                    values=[t.detach().cpu() for t in feature_tensors(actual)]
                    sample=DINOv3Features(values[0],values[1],tuple(values[2:]))
                    errors=feature_error(reference,sample,profile) if reference is not None else None
                    if reference is None:reference=sample
                    del actual
                for sample,original,photo in window.logical():
                    tick=sample['metadata']['step_zero_based']
                    inputs.append(sample['metadata']['id_parameters_sha256'])
                    if tick<len(reference_inputs) and inputs[-1]!=reference_inputs[tick]:raise ValueError('Execution changed the canonical input stream')
                    records.append(runner.update(sample,original,photo,amp=True))
                    if (tick+1)%50==0:
                        for key,value in flush_metrics(records,batched=batched,stream=log).items():head_ms[key]+=value['gpu_event_ms']
                        records.clear()
                del sample,original,photo
                if index>=options['warmup_windows']:
                    timed.append({'index':index,'steps':len(window.samples),'wait':window.data_wait_seconds,
                        'stage':getattr(window,'staging_seconds',0.),'collate':getattr(window,'collate_seconds',0.),
                        'pinned':getattr(window,'pinned',None),'events':window.timing_events,'time':time.monotonic()})
                if index+1==options['warmup_windows']:
                    torch.cuda.current_stream().synchronize()
                    start=time.monotonic()
                if torch.cuda.max_memory_reserved()>torch.cuda.get_device_properties(0).total_memory*.85:
                    raise RuntimeError('Candidate exceeded frozen memory cap')
            for key,value in flush_metrics(records,batched=batched,stream=log).items():head_ms[key]+=value['gpu_event_ms']
            records.clear()
            saved=time.monotonic()
            for head in heads:commit_head(head,Path(scratch),{'engineering_only':True})
            save_seconds=time.monotonic()-saved
            torch.cuda.current_stream().synchronize();stop=time.monotonic()
        finally:
            iterator.close()
            del loader
    elapsed=stop-start
    previous=start
    for row in timed:
        e=row.pop('events');stamp=row.pop('time')
        row.update(h2d_ms=e[0].elapsed_time(e[1]),backbone_ms=e[-2].elapsed_time(e[-1]),wall_seconds=stamp-previous)
        previous=stamp
    result={'candidate':name,'logical_batch':1,'physical_views':profile['backbone_microbatch_views'],
        'workers':profile['data_workers'],'loader':policy,'double_buffer':batched,'peak_live_slots':getattr(pipeline,'peak_live_slots',1),
        'formal_updates':0,'engineering_updates_per_head':{h.name:h.cursor for h in heads},
        'timed_windows':timed,'timed_cohort_samples_per_second':sum(r['steps'] for r in timed)/elapsed,
        'timed_per_head_updates_per_second':{h.name:sum(r['steps'] for r in timed)/elapsed for h in heads},
        'wall_seconds':stop-cold,'timed_seconds':elapsed,'cold_and_warmup_seconds':start-cold,
        'checkpoint_seconds_included':save_seconds,'head_event_ms_including_warmup':head_ms,
        'data_wait_fraction':sum(r['wait'] for r in timed)/elapsed,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'memory_bytes':torch.cuda.max_memory_reserved(),
        'disposable_final_head_digests':{h.name:tree_digest(head_state(h.model)) for h in heads},
        'disposable_final_rng_digests':{h.name:tree_digest(h.rng) for h in heads},
        'same_input_feature_errors':errors,'activity':activity(telemetry,start-monitor.started,stop-monitor.started,options['minimum_steady_seconds'])}
    return result,reference,inputs


def performance_check(out,profile,*,data_root,pretrained_root,git_sha):
    configure();freeze=verify_freeze();options=settings()
    accepted=json.loads((DIRECTORY/'phase34_acceptance.json').read_text())
    if not accepted['accepted'] or accepted['formal_report']['head_updates']!={'U0':40000,'UA':40000}:
        raise ValueError('Need accepted completed v2 baselines before v3 performance check')
    runtime=json.loads((DIRECTORY/'runtime.json').read_text())
    from importlib.metadata import version
    locked={line.split('==')[0].lower():line.split('==')[1] for line in runtime['pip_freeze'] if '==' in line}
    if (torch.__version__!=runtime['torch'] or torch.cuda.get_device_name()!=runtime['gpu'] or
            list(torch.cuda.get_device_capability())!=runtime['compute_capability'] or
            torch.cuda.get_device_properties(0).total_memory!=runtime['visible_memory_bytes'] or
            any(version(name)!=locked[name.lower()] for name in ('numpy','Pillow','PyYAML','torchvision'))):
        raise ValueError('Performance check requires the accepted Phase33 runtime')
    weights=Path(pretrained_root)/'dinov2_vitl14/dinov2_vitl14_hpi_p16.pth'
    expected=json.loads((DIRECTORY/'weight_identity.json').read_text())
    if file_sha(weights)!=expected['converted_file_sha256']:raise ValueError('Frozen converted weight hash mismatch')
    backbone=DINOv2Backbone.from_pretrained(weights,dtype=torch.bfloat16).cuda()
    digest=tree_digest(backbone.state_dict())
    if digest!=json.loads((DIRECTORY/'prepare_report.json').read_text())['runtime_backbone_tensor_digest']:
        raise ValueError('Frozen GPU backbone tensor digest mismatch')
    train,_=build_source_datasets(Path(data_root)/'gta5',TrainTransform((512,512)),SourceEvalTransform())
    base={**profile,'loader_packet_samples':options['loader_packet_samples'],'prefetch_factor':options['prefetch_factor']}
    probes=[];skipped=[]
    for workers in options['worker_candidates']:
        p={**base,'data_workers':workers}
        try:packet_policy(p)
        except ValueError as exc:skipped.append({'workers':workers,'reason':str(exc)});continue
        probes.append(input_probe(train.pairs,p,options))
        print('input_probe',workers,probes[-1]['steady_samples_per_second'],flush=True)
    if len({r['first_packet_digest'] for r in probes})!=1:raise ValueError('Input worker count changed tensors')
    selected=choose(probes,'steady_samples_per_second',options['speed_tie_fraction'])
    # Existing 32-view isolated GPU ceiling is ~86 source pairs/s; probe prefetch4 only if CPU is below it.
    calibration=json.loads((DIRECTORY/'hardware_profile.json').read_text())['batch_candidates']
    old_rate=next(r['end_to_end_views_per_second']/2 for r in calibration if r['batch']==32)
    if selected['steady_samples_per_second']<old_rate:
        p={**base,'data_workers':selected['workers'],'prefetch_factor':options['optional_prefetch_factor']}
        try:packet_policy(p)
        except ValueError as exc:skipped.append({'workers':p['data_workers'],'prefetch':p['prefetch_factor'],'reason':str(exc)})
        else:
            probe=input_probe(train.pairs,p,options)
            if probe['first_packet_digest']!=selected['first_packet_digest']:raise ValueError('Prefetch changed input tensors')
            probes.append(probe);selected=choose(probes,'steady_samples_per_second',options['speed_tie_fraction'])
    selected_profile={**base,'data_workers':selected['workers'],'prefetch_factor':selected['prefetch']}
    larger=None
    for r in sorted((r for r in calibration if r['batch'] in options['larger_batch_candidates'] and r['ok']),key=lambda r:-r['end_to_end_views_per_second']):
        candidate={**selected_profile,'backbone_microbatch_views':r['batch']}
        if r['two_window_conservative_peak_bytes']>=torch.cuda.get_device_properties(0).total_memory*.6:continue
        try:packet_policy(candidate)
        except ValueError:continue
        larger=candidate;break
    plans=[('v2_serial',profile),('v3_batch32',selected_profile)]
    if larger:plans.append((f"v3_batch{larger['backbone_microbatch_views']}",larger))
    budget=update_budget([p for _,p in plans],options)
    write_json(out/'input_probes.json',{'rows':probes,'skipped':skipped,'larger_candidate':larger,'per_head_engineering_budget':budget})
    rng=capture_rng();reference=None;inputs=[];rows=[]
    for name,p in plans:
        gc.collect();torch.cuda.empty_cache()
        row,reference,observed=candidate_run(name,p,options,train.pairs,backbone,out,rng,freeze['initial_head_sha256'],reference,inputs)
        if not inputs:inputs=observed
        rows.append(row);write_json(out/f'{name}.json',row)
        print('performance_candidate',name,row['timed_cohort_samples_per_second'],flush=True)
    if tree_digest(backbone.state_dict())!=digest or any(p.grad is not None for p in backbone.parameters()):raise ValueError('Backbone changed')
    baseline=rows[0]['timed_cohort_samples_per_second']
    paired32={'head_digests_equal':rows[0]['disposable_final_head_digests']==rows[1]['disposable_final_head_digests'],
              'rng_digests_equal':rows[0]['disposable_final_rng_digests']==rows[1]['disposable_final_rng_digests']}
    if not paired32['rng_digests_equal']:raise ValueError('Double buffer changed head RNG at matched batch32 cursor')
    picked=choose([r for r in rows if r['timed_cohort_samples_per_second']>=baseline],
                  'timed_cohort_samples_per_second',options['speed_tie_fraction'])
    speedup=picked['timed_cohort_samples_per_second']/baseline-1
    write_json(out/'performance_v3.json',{'git_sha':git_sha,'study':STUDY,'execution_version':options['execution_version'],
        'formal_optimizer_updates':0,'target_data_accessed':False,'source_miou_computed':False,
        'phase34_completed_baselines_reused':True,'engineering_updates_per_head':budget,'runtime':runtime,
        'weight_identity':expected,'candidates':rows,'recommended':picked['candidate'],
        'matched_batch32_disposable_state':paired32,
        'speedup_fraction':speedup,'speed_target_met':speedup>=options['target_speedup_fraction'],
        'data_wait_target_met':picked['data_wait_fraction']<options['target_data_wait_fraction'],
        'gpu_activity_target_met':picked['activity']['median_percent']>=options['target_gpu_median_percent'] if picked['activity']['steady_eligible'] else None,
        'memory_target_met':picked['memory_bytes']<=runtime['visible_memory_bytes']*.85,
        'disposable_states_deleted':True,'profile_frozen':False})
