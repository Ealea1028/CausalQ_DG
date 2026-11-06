"""Phase34 paired baseline cohort and resumable joint source diagnostics."""
import csv
import json
from math import ceil
import os
from pathlib import Path
import time
import torch
import yaml
from PIL import Image
from causalq.protocols.hpi_dinov2_20261031 import ROOT, STUDY, PHASE35_TRAINING_SHA, build_head, initialization_identity
from causalq.models.dinov2_wrapper import DINOv2Backbone
from causalq.datasets.segmentation import TrainTransform
from causalq.datasets.strict_source import build_source_datasets, SourceEvalTransform
from causalq.interventions.style import StyleInterventionBank
from causalq.analysis.selective_stability import effect_and_correlation_rows, multiview_responsibility_rows
from causalq.analysis.diagnostic_summary import DiagnosticSummary
from causalq.analysis.intervention_examples import export_examples
from causalq.metrics.miou import MeanIoU
from .identity import file_sha, tree_digest, write_json
from .input_stream import CanonicalInputStream, make_loader
from .feature_pipeline import FeaturePipeline, slice_features
from .cohort_runner import CohortRunner, IndependentHead, capture_rng, head_state, commit_head, restore_head, baseline_backward
from .hpi_prepare import configure
from .hpi_source_loader import SourceDiagnosticDataset, diagnostic_seeds, source_loader

DIRECTORY = ROOT/'experiments/HPI_DINOV2_20261031'


def verify_freeze(phase=34):
    if phase in (35, 36):
        verify_freeze(phase - 1)
    freeze=json.loads((DIRECTORY/f'phase{phase}_freeze.json').read_text())
    for name,digest in freeze['files'].items():
        if file_sha(ROOT/name)!=digest:
            raise ValueError(f'Phase{phase} frozen file changed: {name}')
    return freeze


def export_coverage(out,variants,horizon):
    """Merge interrupted attempts by absolute step, checking paired input identity."""
    traces=[*Path(out).glob('attempts/*/training.jsonl'),Path(out)/'training.jsonl']
    rows={}
    for trace in traces:
        if not trace.exists():continue
        lines=trace.read_text().splitlines()
        for index,line in enumerate(lines):
            try:value=json.loads(line)
            except json.JSONDecodeError:
                if index==len(lines)-1:continue # incomplete tail from an interrupted write
                raise
            tick=value['cohort_tick']
            for variant in value['heads']:
                if variant not in variants or not 1<=tick<=horizon:raise ValueError('Unexpected training coverage')
                identity=value['input']
                if identity['step_zero_based']!=tick-1:raise ValueError('Input step differs from cohort tick')
                key=(variant,tick)
                if key in rows and rows[key]!=identity:raise ValueError('Recovered input stream changed')
                rows[key]=identity
    expected={(v,tick) for v in variants for tick in range(1,horizon+1)}
    if set(rows)!=expected:raise ValueError('Formal training coverage incomplete')
    for tick in range(1,horizon+1):
        if any(rows[v,tick]!=rows[variants[0],tick] for v in variants):raise ValueError('Heads received unpaired inputs')
    with (Path(out)/'input_coverage.jsonl').open('w') as stream:
        for tick in range(1,horizon+1):
            stream.write(json.dumps({'cohort_tick':tick,'input':rows[variants[0],tick],'variants':variants})+'\n')
    return {'cohort_ticks':horizon,'per_head_updates':{v:horizon for v in variants},'total_optimizer_updates':len(rows),
        'paired_inputs':True,'input_coverage_sha256':file_sha(Path(out)/'input_coverage.jsonl')}


def train_windows(runner, windows, output, identity, *, amp=True, interval=1000,
                  update_fn=baseline_backward, batched_metrics=False):
    """A head commits its own cursor; a shared tick never replaces update counts."""
    buffered=[]
    log_path=Path(output)/'training.jsonl'
    started=time.monotonic();steady_start=None;checkpoint_seconds=0.;data_wait_seconds=0.
    start_cursors={head.name:head.cursor for head in runner.heads}
    with log_path.open('a') as log, (Path(output)/'throughput.jsonl').open('a') as timing:
        for window_index, window in enumerate(windows):
            data_wait_seconds+=window.data_wait_seconds
            for sample,original,photo in window.logical():
                record=runner.update(sample,original,photo,amp=amp,update_fn=update_fn)
                buffered.append((record,sample['metadata']))
                if record['cohort_tick']%50==0 or record['cohort_tick']==runner.horizon:
                    metrics = {}
                    if batched_metrics:
                        for head in runner.heads:
                            items=[item['heads'][head.name] for item,_ in buffered if head.name in item['heads']]
                            if not items:
                                continue
                            packed=torch.stack([torch.cat((item['losses'],item['gradient_norm'].reshape(1))) for item in items]).cpu().tolist()
                            if items[-1]['timing_events']:
                                items[-1]['timing_events'][1].synchronize()
                            metrics[head.name]=iter(packed)
                    for item,metadata in buffered:
                        row={'cohort_tick':item['cohort_tick'],'input':metadata,'heads':{}}
                        for name,data in item['heads'].items():
                            values=next(metrics[name]) if batched_metrics else data['losses'].cpu().tolist()+[float(data['gradient_norm'].cpu())]
                            row['heads'][name]={'optimizer_updates':data['optimizer_updates'],
                                'losses':values[:-1],'gradient_norm':values[-1]}
                            if data['timing_events']:
                                if not batched_metrics:
                                    data['timing_events'][1].synchronize()
                                row['heads'][name]['update_gpu_ms']=data['timing_events'][0].elapsed_time(data['timing_events'][1])
                        log.write(json.dumps(row)+'\n')
                    log.flush();buffered.clear()
                    write_json(Path(output)/'training_status.json',{'cohort_tick':record['cohort_tick'],
                        'head_updates':{h.name:h.cursor for h in runner.heads},
                        'total_optimizer_updates':sum(h.cursor for h in runner.heads)})
                    print('cohort_tick',record['cohort_tick'],{h.name:h.cursor for h in runner.heads},flush=True)
                for head in runner.heads:
                    if head.name in record['heads'] and (head.cursor%interval==0 or head.cursor==runner.horizon):
                        save_start=time.monotonic()
                        commit_head(head,Path(output)/'checkpoints',identity)
                        checkpoint_seconds+=time.monotonic()-save_start
            del sample,original,photo
            if window.timing_events:
                window.timing_events[-1].synchronize()
                timing.write(json.dumps({'physical_views':window.features.patch_tokens.shape[0],
                        'data_wait_seconds':window.data_wait_seconds,
                        'staging_seconds':getattr(window,'staging_seconds',0.),
                        'h2d_ms':window.timing_events[0].elapsed_time(window.timing_events[1]),
                        'backbone_ms':window.timing_events[-2].elapsed_time(window.timing_events[-1])})+'\n')
                if torch.cuda.max_memory_reserved()>torch.cuda.get_device_properties(0).total_memory*.85:
                    raise RuntimeError('Cohort exceeded frozen85% total GPU memory budget')
            if window_index==1:
                steady_start=time.monotonic()
    stopped=time.monotonic();elapsed=stopped-started
    return {'wall_seconds':elapsed,'steady_start_monotonic':steady_start,'stop_monotonic':stopped,
            'checkpoint_seconds':checkpoint_seconds,'data_wait_seconds':data_wait_seconds,
            'cohort_ticks_per_second':max(head.cursor-start_cursors[head.name] for head in runner.heads)/elapsed,
            'updates_per_head_per_second':{head.name:(head.cursor-start_cursors[head.name])/elapsed for head in runner.heads}}


def class_responsibility(scores,labels):
    probabilities=torch.stack([s[0].float().flatten(0,1).softmax(0).reshape(s.shape[1:]).sum(1).double() for s in scores])
    mask=torch.nn.functional.interpolate(labels[:,None].float(),probabilities.shape[-2:],mode='nearest')[0,0]!=255
    p=probabilities[:,:,mask]
    rows=[]
    for view in range(len(scores)):
        middle=(p[0]+p[view])/2
        js=(.5*(p[0]*(p[0].clamp_min(1e-30)/middle.clamp_min(1e-30)).log()+
                      p[view]*(p[view].clamp_min(1e-30)/middle.clamp_min(1e-30)).log())).sum(0).mean()
        entropy=-(p[view]*p[view].clamp_min(1e-30).log()).sum(0).mean()
        for c in range(p.shape[1]):
            rows.append({'class_id':c,'view_index':view,'scope':'all_valid','present':bool((labels==c).any()),
                'mean_responsibility':float(p[view,c].mean()),'pixel_entropy':float(entropy),
                'js_divergence':float(js),'top1_switch_rate':float((p[0].argmax(0)!=p[view].argmax(0)).double().mean())})
    return rows


def source_record_key(pair, identity, head_digests):
    return {'identity':identity,'heads':head_digests,'id':pair.sample_id,
            'image_size':pair.image.stat().st_size,'label_size':pair.label.stat().st_size}


def batched_source_records(heads, backbone, dataset, out, identity, head_digests, style, examples,
                           profile, intervention_ids, *, device, amp, diagnostic_git_sha):
    from .hpi_source_batch import batched_source_readout, source_batches
    pairs = {pair.sample_id: pair for pair in dataset.pairs}
    indices = []
    for index, pair in enumerate(dataset.pairs):
        path = out/'records'/f'{pair.sample_id}.json'
        if not path.exists():
            indices.append(index)
        elif json.loads(path.read_text())['key'] != source_record_key(pair, identity, head_digests):
            raise ValueError(f'Source diagnostic resume identity differs: {pair.sample_id}')
    reused = len(dataset) - len(indices)
    print('source_records_reused', reused, '/', len(dataset), flush=True)
    if not indices:
        return
    transform = dataset.transform
    maximum_edge = (ceil(transform.long_edge / transform.patch_size) * transform.patch_size
                    if isinstance(transform, SourceEvalTransform) else 1024)
    workers = profile['data_workers'] if torch.device(device).type == 'cuda' else 0
    loader, policy = source_loader(SourceDiagnosticDataset(dataset, indices, style), profile,
                                   workers=workers, maximum_edge=maximum_edge)
    write_json(out.parent/'source_loader.json', policy)
    print('source_loader', json.dumps(policy), flush=True)

    def prepared():
        for packet in loader:
            for sample in packet:
                sample_id = sample['id']
                if sample_id in examples:
                    with Image.open(pairs[sample_id].image) as raw:
                        export_examples(out/'examples'/sample_id, raw, sample['views'].split(1),
                                        sample['traces'], sample['seeds'], style, sample_id, identity=identity)
                yield sample_id, sample['views'], sample['label']

    completed = 0
    batches = source_batches(prepared(), max_images=profile['source_diagnostic_image_batch'],
                             budget_bytes=profile['source_readout_budget_bytes'])
    for batch in batches:
        if completed == 0:
            print('source_first_batch', [sample_id for sample_id, _, _ in batch], flush=True)
        predictions = batched_source_readout(heads, backbone, batch, device=device, amp=amp,
            intervention_ids=intervention_ids, counterfactual_chunk=profile['counterfactual_chunk'])
        for sample_id, models in predictions.items():
            key = source_record_key(pairs[sample_id], identity, head_digests)
            write_json(out/'records'/f'{sample_id}.json', {
                'key': key, 'models': models, 'diagnostic_git_sha': diagnostic_git_sha})
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(indices):
                print('new_source_records', completed, '/', len(indices),
                      'total_source_records', reused + completed, '/', len(dataset), flush=True)


def source_diagnostics(heads,backbone,dataset,out,identity,*,device='cuda',amp=True,profile=None,
                       intervention_ids=(),diagnostic_git_sha=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    style=StyleInterventionBank()
    summaries={h.name:DiagnosticSummary() for h in heads}
    class_summaries={h.name:DiagnosticSummary() for h in heads}
    metrics={h.name:MeanIoU(h.model.num_classes) for h in heads}
    head_digests={h.name:tree_digest(head_state(h.model)) for h in heads}
    for h in heads:h.model.eval()
    examples={'18734','12488','08744'}
    route_rows={h.name:[] for h in heads}
    intervention_rows={h.name:{} for h in heads}
    diagnostic_counts={}
    if profile is not None:
        diagnostic_git_sha = diagnostic_git_sha or identity.get('git_sha')
        batched_source_records(heads,backbone,dataset,out,identity,head_digests,style,examples,profile,intervention_ids,
                               device=device,amp=amp,diagnostic_git_sha=diagnostic_git_sha)
    for index,pair in enumerate(dataset.pairs):
        record_path=out/'records'/f'{pair.sample_id}.json'
        # Binding records to final heads makes resuming analysis read-only.
        key=source_record_key(pair,identity,head_digests)
        if record_path.exists():
            record=json.loads(record_path.read_text())
            if record['key']!=key:raise ValueError('Source diagnostic resume identity differs')
        else:
            if profile is not None:
                raise ValueError(f'Batched source diagnostics did not commit {pair.sample_id}')
            sample=dataset[index];image=sample['image'][None];labels=sample['label'][None]
            views=[image];traces=[];seeds=diagnostic_seeds(pair.sample_id)
            for seed in seeds:
                view,trace=style.photometric_view_with_trace(image,generator=torch.Generator().manual_seed(seed))
                views.append(view);traces.append(trace)
            record={'key':key,'models':{}}
            with torch.inference_mode(),torch.autocast(torch.device(device).type,dtype=torch.bfloat16,enabled=amp):
                features=backbone(torch.cat(views).to(device,non_blocking=True))
                for h in heads:
                    effects=[];scores=[];confusion=MeanIoU(h.model.num_classes)
                    for view in range(5):
                        result=h.model.forward_features(slice_features(features,view,view+1),image.shape[-2:],
                            include_anchor=True,component_protocol=STUDY)
                        if not torch.isfinite(result.logits).all():raise FloatingPointError('Nonfinite source logits')
                        if view==0:confusion.update(result.logits,labels)
                        full=result.scaled_delta_logits.float();anchor=result.anchor_scaled_delta_logits.float()
                        effects.append({'full':full.cpu(),'anchor':anchor.cpu(),'residual':(full-anchor).cpu()})
                        scores.append(result.query_score_maps.float().cpu())
                    er,cr=effect_and_correlation_rows(effects,labels)
                    record['models'][h.name]={'effects':er,'correlations':cr,'gradients':[],
                        'responsibility':multiview_responsibility_rows(scores,labels),
                        'class_responsibility':class_responsibility(scores,labels),
                        'confusion':confusion.confusion.tolist()}
            if pair.sample_id in examples:
                with Image.open(pair.image) as raw:
                    export_examples(out/'examples'/pair.sample_id,raw,views,traces,seeds,style,pair.sample_id,identity=identity)
            write_json(record_path,record)
        if profile is not None:
            producer=record.get('diagnostic_git_sha') or identity.get('git_sha')
            diagnostic_counts[producer]=diagnostic_counts.get(producer,0)+1
        for h in heads:
            value=record['models'][h.name]
            summaries[h.name].add_image(value)
            class_summaries[h.name].add_rows('class_responsibility',value['class_responsibility'])
            class_summaries[h.name].images+=1
            metrics[h.name].confusion+=torch.tensor(value['confusion'],dtype=torch.int64)
            if 'routing' in value:
                route_rows[h.name].append(value['routing'])
            if 'main_path' in value:
                intervention_rows[h.name][pair.sample_id]=value['main_path']
        if (index+1)%25==0:print('source_images',index+1,'/',len(dataset),flush=True)
    if profile is not None:
        write_json(out/'provenance.json',{'training_git_sha':identity.get('git_sha'),
            'records_by_diagnostic_git_sha':diagnostic_counts})
    for h in heads:
        if tree_digest(head_state(h.model))!=head_digests[h.name]:raise ValueError('Source diagnostics changed head')
        directory=out/h.name;directory.mkdir(exist_ok=True)
        if route_rows[h.name]:
            from statistics import fmean
            from .hpi_source_batch import summarize_main_path
            rows = route_rows[h.name]
            if len(rows)!=len(dataset) or set(intervention_rows[h.name])!=set(intervention_ids):
                raise ValueError('Routing diagnostics coverage mismatch')
            write_json(directory/'routing_summary.json',{'image_count':len(rows),'gamma':fmean(r['gamma'] for r in rows),
                **{key:[fmean(r[key][i] for r in rows) for i in range(5)] for key in
                   ('gate_mean','injection_feature_update_rms','route_patch_entropy')},
                'mean_gate_style_variance':fmean(r['gate_mean_style_variance'] for r in rows)})
            write_json(directory/'main_path128.json',{'image_count':len(intervention_ids),'views':5,
                'scope':'GT-present class deletion, route plus selected additive readout',
                'image_macro':summarize_main_path(intervention_rows[h.name]),
                'by_image':intervention_rows[h.name]})
        write_json(directory/'source_metrics.json',{**metrics[h.name].compute(),
            'confusion_matrix':metrics[h.name].confusion.tolist(),'sample_count':len(dataset),
            'source_ids':[p.sample_id for p in dataset.pairs],'target_miou':None,'head_sha256':head_digests[h.name],
            'residual_status':getattr(h.model,'residual_status','trainable'),
            'effective_independent_queries_per_class':getattr(h.model,'effective_independent_queries_per_class',2),
            'slot_status':'duplicated_anchor_slots' if getattr(h.model,'residual_status',None)=='structurally_absent' else 'distinct_slots'})
        write_json(directory/'source_effect_summary.json',summaries[h.name].result())
        write_json(directory/'class_responsibility.json',class_summaries[h.name].result())
        with (directory/'source_class_iou.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['class_id','iou','iou_percent'])
            for c,iou in enumerate(metrics[h.name].compute()['class_iou']):
                writer.writerow([c,iou,None if iou is None else iou*100])
        with (directory/'source_effect_class_statistics.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['kind','population','class_id','component','scope','view_index','metric','mean','median','count','null_count'])
            for table in (summaries[h.name].result(),class_summaries[h.name].result()):
                for row in table['image_macro_statistics']:
                    for metric,stats in row['metrics'].items():
                        writer.writerow([row['kind'],row['population'],row.get('class_id'),row.get('component'),row.get('scope'),row.get('view_index'),metric,stats['mean'],stats.get('median'),stats['count'],stats['null_count']])


def train_baselines(out,profile,*,data_root,pretrained_root,git_sha):
    return train_cohort(out,profile,data_root=data_root,pretrained_root=pretrained_root,git_sha=git_sha,phase=34)


def require_final_source_heads(heads):
    if len(heads) != 3 or any(h.cursor != 40000 for h in heads):
        raise ValueError('Phase35 source recovery requires all three final 40000-update heads')


def train_cohort(out,profile,*,data_root,pretrained_root,git_sha,phase,telemetry_origin=None,source_resume=False):
    configure();freeze=verify_freeze(phase)
    ablations = phase in (35, 36)
    if source_resume and phase != 35:
        raise ValueError('Source recovery is only released for Phase35')
    if ablations and profile['execution_version'] != freeze['execution_version']:
        raise ValueError(f'Phase{phase} requires the frozen v3 profile')
    loss_fields = ('original_ce', 'photometric_ce', 'weighted_anchor_cqe')
    if phase == 36:
        from .hpi_phase36 import TRAINING_LOSS_FIELDS, effect_backward
        loss_fields, update_fn = TRAINING_LOSS_FIELDS, effect_backward
    elif phase == 35:
        from .hpi_phase35 import ablation_backward
        loss_fields, update_fn = ('original_ce', 'photometric_ce', 'weighted_localization'), ablation_backward
    runtime=json.loads((DIRECTORY/'runtime.json').read_text())
    if torch.__version__!=runtime['torch'] or torch.cuda.get_device_name()!=runtime['gpu'] or list(torch.cuda.get_device_capability())!=runtime['compute_capability']:
        raise ValueError('Frozen runtime/GPU differs from accepted Phase33')
    from importlib.metadata import version
    actual_packages={name:version(name) for name in ('numpy','Pillow','PyYAML','torchvision')}
    import platform,sys,subprocess
    write_json(Path(out)/'runtime_actual.json',{'python':sys.version,'platform':platform.platform(),
        'torch':torch.__version__,'packages':actual_packages,'cuda_runtime':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(),'compute_capability':list(torch.cuda.get_device_capability()),
        'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)})
    locked={line.split('==')[0].lower():line.split('==')[1] for line in runtime['pip_freeze'] if '==' in line}
    if any(locked.get(k.lower())!=v for k,v in actual_packages.items()):raise ValueError('Frozen runtime package version differs')
    weights=Path(pretrained_root)/'dinov2_vitl14/dinov2_vitl14_hpi_p16.pth'
    weight_identity=json.loads((DIRECTORY/'weight_identity.json').read_text())
    if file_sha(weights)!=weight_identity['converted_file_sha256']:raise ValueError('Converted weight hash differs')
    backbone=DINOv2Backbone.from_pretrained(weights,dtype=torch.bfloat16).cuda()
    backbone_digest=tree_digest(backbone.state_dict())
    accepted=json.loads((DIRECTORY/'prepare_report.json').read_text())
    if backbone_digest!=accepted['runtime_backbone_tensor_digest']:raise ValueError('Runtime backbone digest differs')
    train,val=build_source_datasets(Path(data_root)/'gta5',TrainTransform((512,512)),SourceEvalTransform())
    training_git_sha = PHASE35_TRAINING_SHA if source_resume else git_sha
    identity={'git_sha':training_git_sha,'freeze_sha256':file_sha(DIRECTORY/f'phase{phase}_freeze.json'),'backbone':backbone_digest}
    out=Path(out);heads=[]
    for name in freeze['variants']:
        (out/f'config_{name}.yaml').write_bytes((ROOT/f'configs/{STUDY}/formal/{name.lower()}_seed0.yaml').read_bytes())
    for name in freeze['variants']:
        if ablations:
            from .hpi_phase35 import build_ablation_head
            architecture = 'B3' if phase == 36 else name
            model=build_ablation_head(backbone,architecture,freeze['initial_head_sha256']).cuda()
        else:
            model=build_head(backbone).cuda()
            if initialization_identity(model)['initial_head_sha256']!=freeze['initial_head_sha256']:
                raise ValueError('Fresh common head initialization differs')
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
        head=IndependentHead(name,model,optimizer,capture_rng(),anchor_cqe=name=='UA')
        restore_head(head,out/'checkpoints',identity)
        if ablations:
            from causalq.models.hpi_anchor_only import verify_frozen_residuals
            verify_frozen_residuals(model)
        if not 0<=head.cursor<=40000 or any(g['lr']!=1e-4*(1-head.cursor/40000)**.9 for g in optimizer.param_groups):
            raise ValueError('Restored cursor/LR differs')
        heads.append(head)
        if source_resume:
            print('source_resume_head', name, head.cursor, flush=True)
    if source_resume:
        require_final_source_heads(heads)
    runner=CohortRunner(heads)
    start=time.monotonic()
    timing=None
    if min(h.cursor for h in heads)<40000:
        stream=CanonicalInputStream(train.pairs,start=runner.aligned_start(profile['backbone_microbatch_views']))
        if ablations:
            from .hpi_double_buffer import DoubleBuffer
            from .hpi_packets import packet_loader
            loader,policy=packet_loader(stream,profile)
            write_json(out/'loader_policy.json',policy)
            timing=train_windows(runner,DoubleBuffer(backbone,profile)(loader),out,identity,
                                 update_fn=update_fn,batched_metrics=True)
            del loader
            if timing['steady_start_monotonic'] is not None and telemetry_origin is not None:
                write_json(out/'training_interval.json',{'start':timing['steady_start_monotonic']-telemetry_origin,
                    'stop':timing['stop_monotonic']-telemetry_origin})
        else:
            loader=make_loader(stream,profile)
            train_windows(runner,FeaturePipeline(backbone,profile)(loader),out,identity)
    if any(h.cursor!=40000 for h in heads):raise ValueError('Incomplete formal updates')
    coverage=export_coverage(out,freeze['variants'],40000)
    if ablations:
        intervention_ids=(DIRECTORY/'source_intervention128.txt').read_text().splitlines()
        if len(set(intervention_ids))!=128 or not set(intervention_ids)<={p.sample_id for p in val.pairs}:
            raise ValueError('Frozen intervention subset differs from source holdout')
        source_diagnostics(heads,backbone,val,out/'source',identity,profile=profile,intervention_ids=intervention_ids,
                           diagnostic_git_sha=git_sha)
    else:
        source_diagnostics(heads,backbone,val,out/'source',identity)
    if tree_digest(backbone.state_dict())!=backbone_digest or any(p.grad is not None for p in backbone.parameters()):
        raise ValueError('Backbone changed during training/diagnostics')
    final_checkpoints={}
    for h in heads:
        markers=list((out/'checkpoints'/h.name).glob('40000_*/complete.json'))
        if len(markers)!=1:raise ValueError('Final checkpoint identity is ambiguous')
        final_checkpoints[h.name]={'path':str((markers[0].parent/'state.pth').relative_to(out)),
            'sha256':json.loads(markers[0].read_text())['state_sha256']}
    import zipfile
    with zipfile.ZipFile(out/'examples.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((out/'source/examples').rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(out/'source/examples'))
    recovery={'diagnostic_git_sha':git_sha,'formal_optimizer_updates_this_invocation':0} if source_resume else {}
    write_json(out/'formal_report.json',{'phase':phase,'git_sha':training_git_sha,'identity':identity,**recovery,
        'input_coverage':coverage,
        'final_checkpoints':final_checkpoints,'weight_identity':weight_identity,
        'head_updates':{h.name:h.cursor for h in heads},'total_optimizer_updates':freeze['total_optimizer_updates'],
        'target_data_accessed':False,'checkpoint_selection':'fixed_final','invocation_elapsed_seconds':time.monotonic()-start,
        'training_invocation_timing':timing,
        'training_loss_fields':loss_fields,
        'runtime_packages':actual_packages,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'per_head_trainable_parameters':{h.name:sum(p.numel() for p in h.model.parameters() if p.requires_grad) for h in heads},
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'final_head_sha256':{h.name:tree_digest(head_state(h.model)) for h in heads}})
