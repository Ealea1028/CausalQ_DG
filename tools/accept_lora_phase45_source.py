"""Audit the returned Phase45 summaries and freeze their source results."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from statistics import median
import tarfile

import numpy as np
from PIL import Image

from causalq.analysis.lora_joint_pca import LAYERS, SEED
from causalq.runtime.identity import file_sha, write_json
from causalq.runtime.lora_source_diagnostics import ROOT, SAMPLE, VARIANTS, inventory

PRODUCER = 'b3b37a143b824cb0f241e02e76e3cf72e1fdf74f'
PREVIOUS = '1d28c61b6f68d45b55893c288c70d1ecba156ddf'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def audit_bundle(root):
    archive = root.with_name(root.name + '.tar.gz')
    digest = file_sha(archive)
    checksum = archive.with_name(archive.name + '.sha256').read_text().split(maxsplit=1)
    need(digest == checksum[0] and PurePosixPath(checksum[1].strip()).name == archive.name, 'Return archive checksum differs')
    manifest = read(root/'export_manifest.json')
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        names = {m.name for m in members}
        need(len(names) == len(members) and names == {*manifest,'export_manifest.json'}, 'Archive coverage differs from manifest')
        for m in members:
            rel = PurePosixPath(m.name)
            need(m.isfile() and not rel.is_absolute() and '..' not in rel.parts, f'Unexpected member: {m.name}')
            data = (root/m.name).read_bytes()
            need(bundle.extractfile(m).read() == data, f'Extracted member differs: {m.name}')
            if m.name in manifest:
                entry = manifest[m.name]
                need(len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256'], f'Member hash differs: {m.name}')
    return digest, len(members)


def confusion_miou(matrix):
    """Macro over nonempty unions, using dataset-level integer counts."""
    data = np.asarray(matrix)
    need(data.shape == (19,19) and data.dtype.kind in 'iu' and (data >= 0).all(), 'Invalid 19-class confusion matrix')
    diagonal = data.diagonal()
    union = data.sum(axis=0) + data.sum(axis=1) - diagonal
    present = union > 0
    need(present.any(), 'No valid union')
    class_iou = [float(d/u) if u else None for d,u in zip(diagonal,union)]
    return float((diagonal[present]/union[present]).mean()), class_iou, int(data.sum())


def audit_models(root, report, pca_ids):
    ids = (ROOT/'manifests/gta5_source_val.txt').read_text().splitlines()
    pinned = inventory()
    need(len(ids) == len(set(ids)) == 2497 and set(report['models']) == set(pinned), 'Source coverage differs')
    table = {r['model']:r for r in csv.DictReader((root/'source_miou.csv').open(encoding='utf-8'))}
    need(set(table) == set(pinned), 'CSV model matrix differs')
    results = {}
    for name,pin in pinned.items():
        folder = root/'models'/name
        metrics = read(folder/'source_metrics.json')
        score, classes, pixels = confusion_miou(metrics['confusion_matrix'])
        need(metrics['source_ids'] == ids and metrics['sample_count'] == 2497 and pixels == metrics['valid_pixels'], f'Source membership differs: {name}')
        need(metrics['target_miou'] is None and math.isclose(score,metrics['miou'],abs_tol=1e-12)
             and metrics['class_iou'] == classes and math.isclose(score,float(table[name]['source_miou']),abs_tol=1e-12), f'Source IoU differs: {name}')
        reference = read(folder/'effect_reference_check.json')
        receipt = read(folder/'inference_receipt.json')
        provenance = read(folder/'record_provenance.json')
        r = report['models'][name]
        need(r['checkpoint_sha256'] == reference['key']['checkpoint_sha256'] == receipt['key']['checkpoint_sha256']
             == table[name]['checkpoint_sha256'] == pin['sha256'] and r['source_miou'] == metrics['miou'], f'Checkpoint receipt differs: {name}')
        need(receipt['state_unchanged'] and receipt['rng_unchanged'] and receipt['logical_view_count'] == 2497*5
             and receipt['key']['git_sha'] == PRODUCER and receipt['key']['runtime'] == report['runtime'], f'Read-only inference receipt differs: {name}')
        need(reference['nulls_match'] and reference['same_forward_sample_id'] in ids and reference['rtol'] == 1e-7
             and reference['atol'] == 1e-10 and 0 <= reference['max_abs_difference'] <= 1e-10, f'Effect comparison differs: {name}')
        counts = provenance['records_by_git_sha']
        need(set(counts) <= ({PRODUCER,PREVIOUS} if name == 'D3_UA_L1' else {PRODUCER})
             and sum(counts.values()) == provenance['sample_count'] == 2497 and counts == r['records_by_git_sha'], f'Producer coverage differs: {name}')

        effects = read(folder/'source_effect_summary.json')
        cross_class = read(folder/'class_responsibility.json')
        macro = effects['image_macro_statistics']
        rows = {x['component']:x['metrics'] for x in macro if x['kind'] == 'effects'
                and x['class_id'] == -1 and x['population'] == 'gt_present'}
        slots = [x for x in macro if x['kind'] == 'responsibility' and x['scope'] == 'gt_class'
                 and x['class_id'] == -1 and x['view_index'] == 0]
        need(set(rows) == {'anchor','residual','full'} and len(slots) == 2
             and effects['sample_count'] == cross_class['sample_count'] == 2497, f'Variance or responsibility coverage differs: {name}')
        need(abs(sum(x['metrics']['mean_responsibility']['mean'] for x in slots)-1) < 1e-6, f'Slot mass differs: {name}')
        if pin['variant'].startswith('B'):
            residual = rows['residual']
            need(metrics['residual_status'] == 'structurally_absent' and residual['raw_variance']['mean'] == 0
                 and residual['joint_rms']['mean'] == 0 and residual['rms_normalized_variance']['mean'] is None
                 and all(x['metrics']['mean_responsibility']['mean'] == .5 for x in slots), f'Structural residual differs: {name}')
        route = None
        if pin['variant'] in ('B2','B3'):
            path = read(folder/'main_path128.json')
            need(path['images'] == 128 and path['views'] == 5 and set(path['by_image']) == set(pca_ids), f'Route coverage differs: {name}')
            need(all(rows and all(len(x['tau']) == 5 for x in rows) for rows in path['by_image'].values()), f'Route view coverage differs: {name}')
            route = path['image_macro']
        utilization = []
        for file in folder.glob('telemetry*.csv'):
            with file.open() as stream:
                utilization.extend(float(v['utilization_percent']) for v in csv.DictReader(stream)
                                   if v['status'] == 'ok' and v['utilization_percent'])
        results[name] = {'source_miou':score,'checkpoint_sha256':pin['sha256'],'source_images':2497,
            'records_by_git_sha':counts,'valid_pixels':pixels,'effect_reference_max_abs_difference':reference['max_abs_difference'],
            'effect_scope':effects['effect_scope'],'effect_macro':{c:{k:x[k]['mean'] for k in
                ('raw_variance','rms_normalized_variance','joint_rms','mean_abs')} for c,x in rows.items()},
            'slot_original_mean':[x['metrics']['mean_responsibility']['mean'] for x in sorted(slots,key=lambda x:x['query_r'])],
            'residual_status':metrics['residual_status'],'route_image_macro':route,
            'gpu_sample_count':len(utilization),'gpu_median_percent':median(utilization) if utilization else None,
            'peak_cuda_reserved_bytes':receipt.get('peak_cuda_reserved_bytes')}
    return results


def audit_pca(root, report):
    manifest = root/'pca/pca_manifest.json'
    need(file_sha(manifest) == report['pca_manifest_sha256'], 'PCA manifest hash differs')
    pca = read(manifest)
    ids = SAMPLE.read_text().splitlines()
    dense = sorted(np.random.default_rng(SEED).choice(ids,12,replace=False).tolist())
    need(pca['sample_ids'] == ids and pca['dense_ids'] == dense and pca['sample_manifest_sha256'] == file_sha(SAMPLE)
         and pca['dense_manifest_sha256'] == hashlib.sha256(('\n'.join(dense)+'\n').encode()).hexdigest(), 'PCA sampling differs')
    need(pca['seed'] == SEED and pca['views'] == ['original','photo1','photo2','photo3','photo4']
         and pca['dense_views'] == ['original','photo1'] and pca['tool_git_sha'] == PRODUCER, 'PCA provenance differs')
    pinned = inventory()
    need(set(pca['backbones']) == {'D3','D2'}, 'Expected two independent backbones')
    bases, pngs = {}, set()
    for short, group in pca['backbones'].items():
        need(group['groups'] == ['Original',*VARIANTS] and set(group['layers']) == set(LAYERS)
             and set(group['coordinates']) == set(ids), f'PCA group/layer coverage differs: {short}')
        rows = 0
        for coord in group['coordinates'].values():
            h,w = coord['grid']
            selected = coord['flat_indices']
            need(0 < h and 0 < w and selected == sorted(set(selected)) and 0 < len(selected) <= 32
                 and min(selected) >= 0 and max(selected) < h*w, f'Invalid PCA patch coordinates: {short}')
            rows += 5*len(selected)
        expected = {f'{short}_{v}_L1':pinned[f'{short}_{v}_L1']['sha256'] for v in VARIANTS}
        if short == 'D3': expected['D3_U0_L0'] = pinned['D3_U0_L0']['sha256']
        need(group['checkpoint_sha256'] == expected, f'PCA checkpoint identities differ: {short}')
        bases[short] = {}
        for layer, value in group['layers'].items():
            path = root/'pca'/short/f'{layer}_basis.npz'
            need(value['basis'] == path.relative_to(root).as_posix() and file_sha(path) == value['sha256']
                 and value['rows_per_group'] == (640 if layer == 'cls' else rows), f'PCA basis scope differs: {short}/{layer}')
            with np.load(path,allow_pickle=False) as arrays:
                basis, mean, scales = arrays['components'], arrays['mean'], arrays['rgb_ranges']
                need(basis.shape == (1024,3) and mean.shape == (1024,) and scales.shape == (2,3)
                     and all(np.isfinite(arrays[k]).all() for k in arrays.files), f'Invalid PCA arrays: {path}')
                need(np.allclose(basis.T@basis,np.eye(3),atol=1e-10) and (basis[np.abs(basis).argmax(0),np.arange(3)] >= 0).all()
                     and scales.tolist() == value['rgb_ranges'] and (scales[0] <= scales[1]).all(), f'Invalid PCA basis or ranges: {path}')
            bases[short][layer] = {'sha256':value['sha256'],'rows_per_group':value['rows_per_group'],
                                  'explained_variance_ratio':value['explained_variance_ratio']}
        pngs |= {root/'pca'/short/f'{layer}_scatter.png' for layer in ('patch','cls')}
        pngs |= {root/'pca'/short/f'{short}_{v}_L1_{sample}.png' for v in VARIANTS for sample in dense}
    need(set((root/'pca').rglob('*.png')) == pngs, 'PCA figure coverage differs')
    for path in pngs:
        with Image.open(path) as figure: figure.verify()
    return bases, len(pngs), ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('extracted',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root = args.extracted
    digest, members = audit_bundle(root)
    report_path = root/'report.json'
    report = read(report_path)
    need(report['phase'] == 45 and report['git_sha'] == PRODUCER and report['ok'] and report['status'] == 'complete'
         and report['model_count'] == 15 and report['optimizer_updates'] == 0 and report['target_data_accessed'] is False
         and report['reference_checkpoints_unchanged'] is True and report['logical_model_image_views'] == 187275
         and (root/'exit_code.txt').read_text().strip() == 'diagnostic_exit_code=0', 'Phase45 did not finish read-only')
    bases, figures, ids = audit_pca(root,report)
    models = audit_models(root,report,ids)
    need(len({x['valid_pixels'] for x in models.values()}) == 1, 'Source valid-pixel coverage differs between models')
    write_json(args.output,{'phase':45,'accepted':True,'producer':PRODUCER,'archive_sha256':digest,'archive_members':members,
        'report_sha256':file_sha(report_path),'pca_manifest_sha256':report['pca_manifest_sha256'],
        'runtime':report['runtime'],'elapsed_seconds':report['seconds'],'optimizer_updates':0,'target_data_accessed':False,
        'model_count':15,'images_per_model':2497,'views_per_image':5,'models':models,'pca_bases':bases,'pca_png_count':figures,
        'next_phase':46})
    print(f'Phase45 accepted: {len(models)} models, {members} checked members, {figures} PCA figures')


if __name__ == '__main__':
    main()
