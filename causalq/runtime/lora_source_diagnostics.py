"""Phase45 final-checkpoint inventory and read-only source summaries."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean

import torch

from causalq.analysis.diagnostic_summary import DiagnosticSummary
from causalq.analysis.selective_stability import (
    effect_and_correlation_rows, structural_effect_and_correlation_rows, multiview_responsibility_rows,
)
from causalq.metrics.miou import MeanIoU
from causalq.runtime.hpi_source_batch import main_path_rows
from causalq.runtime.identity import file_sha, write_json
from causalq.runtime.lora_formal import DIRECTORY, ROOT, build_backbone, core_head
from causalq.runtime.lora_l1_formal import core_l1_head
from causalq.runtime.lora_phase44_b1 import b1_head
from causalq.runtime.lora_phase44_b2 import b2_head
from causalq.runtime.lora_training import compact_state, load_state_file

VARIANTS = ('U0', 'UA', 'C0', 'CA', 'B1', 'B2', 'B3')
BACKBONES = {'D3': 'dinov3_vitl16', 'D2': 'dinov2_vitl14_hpi_p16'}
SAMPLE = DIRECTORY/'phase45_source128.txt'
CONTRACT = DIRECTORY/'phase45_freeze.json'


def inventory():
    models = {}
    for phase, module in ((42, ''), (43, ''), (44, '_b1'), (44, '_b2'), (44, '_b3')):
        a = json.loads((DIRECTORY/f'phase{phase}{module}_completion_acceptance.json').read_bytes())
        if not a['accepted'] or a['target_data_accessed'] or a['formal_optimizer_updates_total'] != len(a['models'])*40000:
            raise ValueError('Invalid training acceptance')
        for key, pin in a['models'].items():
            short, variant = key.split('_') if module else ('D3' if phase == 42 else 'D2', key)
            name = f'{short}_{variant}_L1'
            if name in models or pin['step'] != 40000 or pin['source_cursor'] != 320000:
                raise ValueError(f'Conflicting final {name}')
            models[name] = dict(pin, phase=phase, producer=a['producer'], backbone=BACKBONES[short], variant=variant, lora=True)
    expected = {f'{b}_{v}_L1' for b in BACKBONES for v in VARIANTS}
    if set(models) != expected:
        raise ValueError('Expected fourteen trained LoRA finals')
    retirement = json.loads((DIRECTORY/'phase41_retirement_apply_freeze.json').read_bytes())
    pin, = retirement['protected']
    models['D3_U0_L0'] = dict(pin, path=str(Path(retirement['run_root'])/pin['path']), phase=41, lora=False)
    return models


def verify_contract():
    contract = json.loads(CONTRACT.read_bytes())
    for path, sha in contract['files'].items():
        if hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != sha:
            raise ValueError(f'Phase45 frozen file changed: {path}')
    return contract


def verify_final(pin, base):
    from tools.retire_lora_l0 import safe
    path = safe(base, Path(pin['path']))
    marker = safe(base, path.with_name('complete.json'))
    if file_sha(marker) != pin['marker_sha256'] or file_sha(path) != pin['sha256']:
        raise ValueError(f'Final checkpoint hash mismatch: {path}')
    row = json.loads(marker.read_bytes())
    i = row['identity']
    if row['step'] != 40000 or row['sha256'] != pin['sha256'] or i['backbone'] != pin['backbone'] or i['variant'] != pin['variant']:
        raise ValueError(f'Final checkpoint identity mismatch: {path}')
    if pin['lora']:
        digest = hashlib.sha256(json.dumps(i, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if digest != pin['identity_sha256'] or i['training_producer_git_sha'] != pin['producer'] or i['phase'] != pin['phase']:
            raise ValueError(f'Final checkpoint producer mismatch: {path}')
    return path, i


def load_final(pin, base, weights):
    path, identity = verify_final(pin, base)
    backbone = build_backbone(pin['backbone'], weights)
    variant = pin['variant']
    if not pin['lora']:
        model = core_head(backbone, variant)
    elif variant == 'B1':
        model = b1_head(backbone)
    elif variant in ('B2', 'B3'):
        model = b2_head(backbone)
    else:
        model = core_l1_head(backbone, variant)
    state = load_state_file(path)
    expected = compact_state(model)
    payload = state['model']
    if (state['schema'] != 1 or state['identity'] != identity or state['step'] != 40000
            or state['source_cursor'] != 320000 or set(expected) != set(payload)):
        raise ValueError(f'Compact checkpoint coverage mismatch: {path}')
    for key, tensor in payload.items():
        if tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid final tensor {key}')
    model.load_state_dict(payload, strict=False)
    if variant.startswith('B'):
        from causalq.models.hpi_anchor_only import verify_frozen_residuals
        verify_frozen_residuals(model)
    return model.eval()


def write_record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_suffix('.partial')
    with gzip.open(stage, 'wt', encoding='utf-8', compresslevel=1) as stream:
        json.dump(value, stream, allow_nan=False, separators=(',', ':'))
    os.replace(stage, path)


def read_record(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def responsibility(scores, label):
    rows = multiview_responsibility_rows(list(scores.split(1)), label)
    grid_labels = torch.nn.functional.interpolate(label[:,None].float(), scores.shape[-2:], mode='nearest')[0,0]
    variance = scores.softmax(2).double().var(0, unbiased=False)
    per_class = {}
    for row in rows:
        key = row['class_id'], row['query_r'], row['scope']
        if key not in per_class:
            mask = grid_labels == key[0] if key[2] == 'gt_class' else grid_labels != 255
            per_class[key] = float(variance[key[0],key[1]][mask].mean()) if row['pixels'] else None
        row['five_view_population_variance'] = per_class[key]
    return rows


@torch.inference_mode()
def readout(model, batch, *, device='cuda', amp=True, intervention_ids=(), observer=None):
    """Run every view's DINO once and share the features with source and PCA outputs."""
    from causalq.runtime.hpi_training import class_responsibility
    images = torch.cat([x[1] for x in batch]).to(device, non_blocking=True)
    size = images.shape[-2:]
    structural = getattr(model,'residual_status',None) == 'structurally_absent'
    options = {'include_anchor':not structural,'component_protocol':'lora_causalq_1114'}
    route = None
    with torch.autocast(torch.device(device).type,dtype=torch.bfloat16,enabled=amp):
        features = model.backbone(images)
        if hasattr(model,'forward_routed'):
            prediction, route = model.forward_routed(features,size,**options)
        else:
            prediction = model.forward_features(features,size,**options)
    if not torch.isfinite(prediction.logits).all(): raise ValueError('Nonfinite source logits')
    if observer is not None: observer(features,prediction,batch)
    scores = prediction.query_score_maps.float().cpu()
    full = prediction.scaled_delta_logits.float().cpu()
    anchor = None if structural else prediction.anchor_scaled_delta_logits.float().cpu()
    result = {}
    if route is not None:
        attention = route.patch_attention.float()
        route_values = torch.stack((route.injected.float().square().mean((1,2)).sqrt(),
                        route.gate.float().mean((1,2,3)),
                        -(attention*attention.clamp_min(1e-12).log()).sum(-1).mean((1,2))),dim=1).double().cpu()
    original_predictions = prediction.logits[::5].argmax(1).cpu()
    for index, (sample_id, _, label) in enumerate(batch):
        offset = index*5
        sl = slice(offset,offset+5)
        if structural:
            effects, correlations = structural_effect_and_correlation_rows(full[sl],label)
        else:
            maps = [{'full':f[None],'anchor':a[None],'residual':(f-a)[None]} for f,a in zip(full[sl],anchor[sl])]
            effects, correlations = effect_and_correlation_rows(maps,label)
        metric = MeanIoU(model.num_classes)
        metric.update(original_predictions[index:index+1],label)
        r = {'effects':effects,'correlations':correlations,'gradients':[],
             'responsibility':responsibility(scores[sl],label),
             'class_responsibility':class_responsibility(list(scores[sl].split(1)),label),
             'confusion':metric.confusion.tolist()}
        if route is not None:
            r['routing'] = {'gamma':float(route.gamma),'injection_feature_update_rms':route_values[sl,0].tolist(),
                            'gate_mean':route_values[sl,1].tolist(),
                            'gate_mean_style_variance':float(route_values[sl,1].var(unbiased=False)),
                            'route_patch_entropy':route_values[sl,2].tolist()}
            if sample_id in intervention_ids:
                with torch.autocast(torch.device(device).type,dtype=torch.bfloat16,enabled=amp):
                    r['main_path'] = main_path_rows(model,features,prediction.logits[sl],label,offset,4)
        result[sample_id] = r
    return result


def summarize(model_dir, ids, intervention_ids, structural):
    from causalq.runtime.hpi_source_batch import summarize_main_path
    effect, classes = DiagnosticSummary(), DiagnosticSummary()
    metrics = MeanIoU(19)
    routes, interventions = [], {}
    pixel_effects = defaultdict(lambda: [0., 0, 0])
    for sample_id in ids:
        r = read_record(model_dir/'records'/f'{sample_id}.json.gz')['record']
        for row in r['correlations']:
            if row['component'] == 'variance_decomposition':
                scale = max(1., row['full_variance'], row['anchor_variance'], row['residual_variance'])
                if row['variance_identity_abs_error'] > 1e-6*scale:
                    raise ValueError('Variance/covariance identity failed')
                residual = row['residual_variance']
                row['anchor_residual_variance_ratio'] = row['anchor_variance']/residual if residual > 0 else None
        effect.add_image(r)
        classes.add_rows('class_responsibility', r['class_responsibility'])
        classes.images += 1
        for row in r['effects']:
            for field in ('raw_variance','rms_normalized_variance','joint_rms','mean_abs'):
                populations = ('all_classes', 'gt_present') if row['present'] else ('all_classes',)
                for population in populations:
                    entry = pixel_effects[(population, row['component'], field)]
                    value = row[field]
                    if value is None: entry[2] += 1
                    else:
                        entry[0] += value * row['valid_pixels']
                        entry[1] += row['valid_pixels']
        metrics.confusion += torch.tensor(r['confusion'], dtype=torch.int64)
        if 'routing' in r: routes.append(r['routing'])
        if 'main_path' in r: interventions[sample_id] = r['main_path']
    source = metrics.compute()
    source.update(confusion_matrix=metrics.confusion.tolist(), valid_pixels=int(metrics.confusion.sum()),
                  miou_percent=100*source['miou'], sample_count=len(ids), source_ids=ids, target_miou=None,
                  residual_status='structurally_absent' if structural else 'trainable',
                  slot_status='duplicated_anchor_slots' if structural else 'distinct_slots',
                  effective_independent_queries_per_class=1 if structural else 2)
    write_json(model_dir/'source_metrics.json', source)
    summary = effect.result()
    summary['effect_scope'] = 'routed_readout_effect' if routes else 'readout_effect'
    by_slot = defaultdict(dict)
    for row in summary['image_macro_statistics']:
        if row['kind'] != 'responsibility' or row['class_id'] < 0: continue
        key = row['scope'],row['class_id'],row['view_index']
        by_slot[key][row['query_r']] = row['metrics']['mean_responsibility']['mean']
    usage = []
    for (scope, class_id, view), slots in sorted(by_slot.items()):
        values = [slots[k] for k in sorted(slots)]
        entropy = -sum(v*math.log(max(v,1e-30)) for v in values) if all(v is not None for v in values) else None
        usage.append({'scope':scope,'class_id':class_id,'view_index':view,'mean_responsibility':values,
                      'usage_entropy':entropy,'effective_query_count':math.exp(entropy) if entropy is not None else None,
                      'normalized_usage_entropy':entropy/math.log(len(values)) if entropy is not None else None})
    summary['image_weighted_dataset_slot_usage'] = usage
    summary['pixel_weighted_effects'] = [dict(population=p,component=c,metric=k,mean=s/n if n else None,
        pixels=n,null_image_classes=nulls) for (p,c,k),(s,n,nulls) in sorted(pixel_effects.items())]
    write_json(model_dir/'source_effect_summary.json', summary)
    write_json(model_dir/'class_responsibility.json', classes.result())
    with (model_dir/'responsibility_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scope','class_id','slot','view','population','metric','mean','count','null_count'])
        for row in summary['image_macro_statistics']:
            if row['kind'] == 'responsibility':
                for metric, x in row['metrics'].items():
                    writer.writerow([row['scope'],row['class_id'],row['query_r'],row['view_index'],row['population'],metric,x['mean'],x['count'],x['null_count']])
    if routes:
        if len(routes) != len(ids) or set(interventions) != set(intervention_ids):
            raise ValueError('Routed diagnostic coverage mismatch')
        write_json(model_dir/'main_path128.json', {'images':len(interventions),'views':5,
                   'image_macro':summarize_main_path(interventions),'by_image':interventions})
        write_json(model_dir/'routing_summary.json', {'images':len(routes),'gamma':fmean(x['gamma'] for x in routes),
                   **{k:[fmean(x[k][i] for x in routes) for i in range(5)] for k in
                      ('gate_mean','injection_feature_update_rms','route_patch_entropy')},
                   'mean_gate_style_variance':fmean(x['gate_mean_style_variance'] for x in routes)})
    return source
