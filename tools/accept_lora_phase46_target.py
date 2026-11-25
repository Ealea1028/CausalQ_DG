"""Recompute the final LoRA study results from the accepted source and target returns."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np

from causalq.runtime.identity import file_sha, write_json
from causalq.runtime.lora_phase46 import (
    BATCH_SIZE, FREEZE, GEOMETRY, SOURCE, TARGET_SHA, load_contract, validate_record,
)
from tools.accept_lora_phase45_source import audit_bundle, confusion_miou, need, read


PRODUCER = 'd8d45d5bef4b081617bbffc6b03e94cb239574de'
VARIANTS = ('U0', 'UA', 'C0', 'CA', 'B1', 'B2', 'B3')
CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
           'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
           'truck', 'bus', 'train', 'motorcycle', 'bicycle')


def audit_target_model(root, name, ids, identity, index, receipt, source, table):
    folder = root/'models'/name
    records = folder/'records'
    paths = [records/f'{i}.json' for i in ids]
    need(set(paths) == set(records.rglob('*.json')), f'Target record coverage differs: {name}')
    counts = []
    for sample_id, path in zip(ids, paths):
        row = read(path)
        actual = validate_record(name, sample_id, row, identity)
        need(actual == tuple(index[sample_id]), f'Target input differs: {name}/{sample_id}')
        counts.append(row['confusion'][name])
    confusion = np.sum(counts, axis=0, dtype=np.int64).tolist()
    score, classes, pixels = confusion_miou(confusion)
    metrics_path = folder/'target_metrics.json'
    metrics = read(metrics_path)
    need(metrics['ids'] == ids and metrics['sample_count'] == receipt['sample_count'] == len(ids)
         and metrics['identity'] == identity and metrics['geometry'] == identity['geometry']
         and metrics['confusion_matrix'] == confusion and metrics['valid_pixels'] == pixels
         and metrics['class_iou'] == classes and metrics['valid_classes'] == sum(x is not None for x in classes)
         and metrics['state_unchanged'] is True and metrics['rng_unchanged'] is True
         and metrics['records_reused'] == 0 and metrics['new_native_image_forwards'] == len(ids)
         and receipt['metrics_sha256'] == file_sha(metrics_path)
         and receipt['checkpoint_sha256'] == table['checkpoint_sha256'] == identity['checkpoint_sha256']
         and int(table['target_images']) == len(ids)
         and metrics['source_miou'] == float(table['source_miou']) == source
         and all(math.isclose(score, x, rel_tol=0, abs_tol=1e-12) for x in
                 (receipt['miou'], metrics['miou'], metrics['miou_percent']/100,
                  float(table['target_miou']), float(table['target_miou_percent'])/100)),
         f'Target summary differs: {name}')
    utilization = []
    for path in sorted(folder.glob('telemetry*.csv')):
        with path.open(encoding='utf-8', newline='') as stream:
            utilization.extend(float(row['utilization_percent']) for row in csv.DictReader(stream)
                               if row['status'] == 'ok' and row['utilization_percent'])
    need(utilization and all(0 <= x <= 100 for x in utilization), f'Invalid telemetry: {name}')
    return dict(
        target_miou=score, target_class_iou=classes, target_confusion=confusion, target_pixels=pixels,
        target_gpu_median_percent=median(utilization), target_gpu_sample_count=len(utilization),
        target_seconds=metrics['seconds_this_invocation'], data_workers=metrics['data_workers'],
        parameter_count=metrics['parameter_count'], parameters_in_lora=metrics['parameters_in_lora'],
        target_peak_allocated_bytes=metrics['peak_allocated_bytes'],
        target_peak_reserved_bytes=metrics['peak_reserved_bytes'],
        target_metrics_sha256=receipt['metrics_sha256'],
    )


def responsibility(folder):
    effects = read(folder/'source_effect_summary.json')
    classes = read(folder/'class_responsibility.json')
    groups = (
        ('slot', effects['image_macro_statistics'], 'gt_class', 'responsibility'),
        ('class', classes['image_macro_statistics'], 'all_valid', 'class_responsibility'),
    )
    result = {}
    for name, rows, scope, kind in groups:
        views = {x['view_index']: x['metrics'] for x in rows if x['kind'] == kind
                 and x['class_id'] == -1 and x['scope'] == scope and x.get('query_r', 0) == 0}
        need(set(views) == set(range(5)), f'Incomplete responsibility views: {folder}/{scope}')
        result[f'{name}_original_pixel_entropy'] = views[0]['pixel_entropy']['mean']
        for field in ('js_divergence', 'top1_switch_rate'):
            result[f'{name}_photo_mean_{field}'] = mean(views[v][field]['mean'] for v in range(1, 5))
    return result


def contrast_specs():
    for backbone in ('D3', 'D2'):
        def contrast(*terms):
            return {f'{backbone}_{variant}_L{lora}': sign for variant, lora, sign in terms}
        for variant in VARIANTS:
            yield f'{backbone} {variant} L1−L0', contrast((variant, 1, 1), (variant, 0, -1))
        for lora in (0, 1):
            for a, b in (('UA', 'U0'), ('CA', 'C0'), ('CA', 'UA'), ('C0', 'U0'),
                         ('B1', 'U0'), ('B2', 'B1'), ('B3', 'B2')):
                yield f'{backbone} L{lora} {a}−{b}', contrast((a, lora, 1), (b, lora, -1))
            yield f'{backbone} L{lora} (CA−C0)−(UA−U0)', contrast(
                ('CA', lora, 1), ('C0', lora, -1), ('UA', lora, -1), ('U0', lora, 1))
        for a, b in (('UA', 'U0'), ('CA', 'C0')):
            yield f'{backbone} LoRA×({a}−{b})', contrast((a, 1, 1), (b, 1, -1), (a, 0, -1), (b, 0, 1))
    for lora in (0, 1):
        for variant in VARIANTS:
            yield f'L{lora} {variant} D3−D2', {f'D3_{variant}_L{lora}': 1, f'D2_{variant}_L{lora}': -1}


def contrasts(rows):
    metrics = [k for k in next(iter(rows.values()))
               if k != 'residual_status' and (k.endswith('_miou_percent')
                   or k.startswith(('anchor_', 'residual_', 'full_', 'slot_', 'class_')))]
    for name, terms in contrast_specs():
        missing = ','.join(term for term in terms if term not in rows)
        values = {}
        for metric in metrics:
            values[metric] = None
            if not missing and all(rows[term][metric] is not None for term in terms):
                values[metric] = sum(rows[term][metric]*sign for term, sign in terms.items())
        scopes = sorted({rows[term]['effect_scope'] for term in terms if term in rows})
        yield {'contrast':name, 'missing_models':missing, 'effect_scopes':','.join(scopes), **values}


def write_csv(path, rows):
    rows = list(rows)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: 'N/A' if v is None else v for k, v in row.items()})


def accept(target_root, source_root, output, receipt_path):
    source, pins, ids = load_contract()
    source_models = source['models']
    source_digest, source_members = audit_bundle(source_root)
    digest, members = audit_bundle(target_root)
    report_path = target_root/'report.json'
    report = read(report_path)
    need(source_digest == source['archive_sha256'] and source_members == source['archive_members'],
         'Source archive differs from accepted Phase45')
    need(report['phase'] == 46 and report['git_sha'] == PRODUCER and report['status'] == 'complete'
         and report['ok'] is True and report['runtime'] == source['runtime']
         and report['formal_optimizer_updates'] == 0 and report['checkpoints_unchanged'] is True
         and report['target_data_accessed'] is True and report['target_pca'] is False
         and report['target_manifest_sha256'] == TARGET_SHA and report['target_geometry'] == list(GEOMETRY)
         and report['native_batch_size'] == BATCH_SIZE and report['precision'] == 'bf16_autocast'
         and report['source_acceptance_sha256'] == file_sha(SOURCE)
         and report['model_count'] == len(pins) and set(report['models']) == set(pins)
         and report['images_per_model'] == len(ids) and report['logical_image_forwards'] == len(pins)*len(ids)
         and (target_root/'exit_code.txt').read_text().strip() == 'target_exit_code=0',
         'Phase46 report differs from the frozen run')
    index_report = read(target_root/'target_payload_identity.json')
    index = index_report['records']
    index_sha = hashlib.sha256(json.dumps(index, separators=(',', ':'), sort_keys=True).encode()).hexdigest()
    need(index_report['fields'] == ['image_sha256', 'label_sha256', 'valid_pixels']
         and set(index) == set(ids) and index_sha == index_report['sha256'] == report['payload_index_sha256'],
         'Target input index differs')
    with (target_root/'target_miou.csv').open(encoding='utf-8', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    table = {row['model']:row for row in csv_rows}
    need(len(csv_rows) == len(pins) and set(table) == set(pins), 'Target table coverage differs')
    models, rows, class_rows, matrices = {}, {}, [], {}
    for name, pin in pins.items():
        identity = dict(model=name, evaluation_git_sha=PRODUCER, freeze_sha256=file_sha(FREEZE),
                        checkpoint_sha256=pin['sha256'], geometry=list(GEOMETRY), native_batch_size=BATCH_SIZE,
                        target_manifest_sha256=TARGET_SHA, precision='bf16_autocast',
                        target_root='/root/autodl-tmp/datasets/cityscapes')
        src = source_models[name]
        target = audit_target_model(target_root, name, ids, identity, index,
                                    report['models'][name], src['source_miou'], table[name])
        need(target['parameters_in_lora'] == (98304 if pin['lora'] else 0), f'Parameter count differs: {name}')
        src_folder = source_root/'models'/name
        source_matrix = read(src_folder/'source_metrics.json')['confusion_matrix']
        score, per_class, pixels = confusion_miou(source_matrix)
        need(score == src['source_miou'] and pixels == src['valid_pixels'], f'Source recheck differs: {name}')
        matrices[name] = {'source':source_matrix, 'target':target.pop('target_confusion')}
        for i, (source_score, target_score) in enumerate(zip(per_class, target.pop('target_class_iou'))):
            class_rows.append(dict(model=name, train_id=i, class_name=CLASSES[i],
                                   source_iou=source_score, target_iou=target_score))
        models[name] = dict(checkpoint_sha256=pin['sha256'], training_producer=pin['producer'] if pin['lora'] else None,
                            source_miou=score, **target)
        rows[name] = dict(
            model=name, backbone=name.split('_')[0], variant=name.split('_')[1], lora=int(pin['lora']),
            seed=0, source_miou_percent=100*score, target_miou_percent=100*target['target_miou'],
            effect_scope=src['effect_scope'], residual_status=src['residual_status'],
            **{f'{c}_{k}':v for c, values in src['effect_macro'].items() for k, v in values.items()},
            slot0_original_mean_responsibility=src['slot_original_mean'][0],
            slot1_original_mean_responsibility=src['slot_original_mean'][1],
            **responsibility(src_folder), source_gpu_median_percent=src['gpu_median_percent'],
            source_peak_reserved_bytes=src['peak_cuda_reserved_bytes'], **{
                k:v for k,v in target.items() if k not in ('target_miou', 'target_metrics_sha256')},
        )
    pca = read(source_root/'pca/pca_manifest.json')
    pca_rows = []
    for backbone, group in pca['backbones'].items():
        for layer, basis in group['layers'].items():
            for variant, values in basis['matched_drift'].items():
                pca_rows.append(dict(backbone=backbone, layer=layer, variant=variant, original=group['original_group'],
                                     explained_variance_pc123=sum(basis['explained_variance_ratio']),
                                     style_variance_original=basis['style_variance_by_group']['Original'],
                                     style_variance_adapted=basis['style_variance_by_group'][variant],
                                     basis_sha256=basis['sha256'], **values))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'models.csv', rows.values())
    write_csv(output/'contrasts.csv', contrasts(rows))
    write_csv(output/'class_iou.csv', class_rows)
    write_csv(output/'joint_pca.csv', pca_rows)
    write_json(output/'confusion_matrices.json', matrices)
    write_json(receipt_path, dict(
        phase=46, accepted=True, producer=PRODUCER, report_sha256=file_sha(report_path),
        archive_name=target_root.name+'.tar.gz', archive_sha256=digest, archive_members=members,
        freeze_sha256=file_sha(FREEZE), source_acceptance_sha256=file_sha(SOURCE),
        input_index_sha256=index_sha, target_manifest_sha256=TARGET_SHA, model_count=len(models),
        images_per_model=len(ids), logical_image_forwards=report['logical_image_forwards'],
        runtime=report['runtime'], hardware=report['hardware'], seconds=report['seconds'],
        formal_optimizer_updates=0, checkpoints_unchanged=True, models=models, study_complete=True,
        table_sha256={name:file_sha(output/name) for name in
                      ('models.csv', 'contrasts.csv', 'class_iou.csv', 'joint_pca.csv', 'confusion_matrices.json')},
    ))
    print(f'Phase46 accepted: {members} verified files, {len(models)*len(ids)} recomputed image records')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    accept(args.target, args.source, args.output, args.receipt)


if __name__ == '__main__':
    main()
