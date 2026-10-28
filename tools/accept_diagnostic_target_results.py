"""Accept Phase32 reports and independently reduce integer target confusion matrices."""

import argparse
from datetime import date
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np

from causalq.protocols.diagnostic_20261020 import ROOT, STUDY, VARIANTS
from causalq.protocols.diagnostic_final_source import NEW_FINALS, final_source_report
from causalq.protocols.diagnostic_target import target_contract, verify_target_contract, GEOMETRY, TARGET_MANIFEST_SHA
from tools.evaluate_target import target_ids
from tools.train_strict_source import file_sha

PRODUCER = '40c73bcd6914604368c1006e6c3d7ff0e2ad2a0a'
ARCHIVE_SHA = 'b28c46ac6d0ad862a030c12528bbbd6c1b3a91a7ccd7a11ca66ad6281864560b'


def reduce_confusion(report):
    counts = report['confusion_matrix']
    if (len(counts) != 19 or any(len(row) != 19 for row in counts)
            or any(type(n) is not int or n < 0 for row in counts for n in row)):
        raise ValueError('Invalid19-class integer confusion matrix')
    matrix = np.asarray(counts, dtype=np.int64)
    union = matrix.sum(0) + matrix.sum(1) - matrix.diagonal()
    if np.any(union <= 0):raise ValueError('Target class coverage differs')
    iou = matrix.diagonal() / union
    if (report['valid_classes'] != 19 or int(matrix.sum()) != report['valid_pixel_count']
            or not np.allclose(iou, report['class_iou'], atol=1e-12, rtol=0)
            or not math.isclose(float(iou.mean()), report['target_miou'], abs_tol=1e-12, rel_tol=0)):
        raise ValueError('Class IoU/mIoU differs from independent confusion reduction')
    return matrix, iou


def validate_comparison(job):
    freeze = target_contract();digest = verify_target_contract()
    if (job['ok'] is not True or job['phase'] != 32 or job['study'] != STUDY or job['git_sha'] != PRODUCER
            or job['freeze_sha256'] != digest or set(job['reports']) != set(NEW_FINALS)
            or job['reused_models'] != freeze['reused_models']
            or job['optimizer_updates_this_run'] != 0 or job['target_data_accessed'] is not True
            or job['selection_or_tuning_on_target'] is not False or job['target_feedback_already_seen'] is not True
            or job['additional_training_authorized'] is not False):
        raise ValueError('Final reporting-only protocol differs')
    old = json.loads((ROOT / 'experiments/STRICT_DG_PHASE26/final_comparison.json').read_bytes())
    target = {v: r['target_miou'] for v, r in freeze['reused_models'].items()}
    source = {v: r['source_miou'] for v, r in {**freeze['new_models'], **freeze['reused_models']}.items()}
    matrices = {};class_iou = {}
    gt = np.asarray(old['reports']['M0']['confusion_matrix'], dtype=np.int64).sum(1)
    for v in VARIANTS:
        if v in NEW_FINALS:
            r = job['reports'][v];record = freeze['new_models'][v]
            expected_identity = {'git_sha': PRODUCER, 'variant': v, 'study': STUDY, 'phase': 32,
                'freeze_sha256': digest, 'checkpoint_sha256': record['checkpoint_sha256'],
                'target_manifest_sha256': TARGET_MANIFEST_SHA, 'geometry': GEOMETRY,
                'checkpoint_selection': 'fixed_final_iteration40000'}
            if (r['ok'] is not True or r['phase'] != 32 or r['study'] != STUDY or r['variant'] != v
                    or r['identity'] != expected_identity or r['sample_count'] != 500
                    or r['environment'] != final_source_report(v)['metadata']['environment']
                    or r['source_miou'] != source[v] or r['optimizer_updates_this_run'] != 0
                    or r['target_data_accessed'] is not True or r['selection_or_tuning_on_target'] is not False
                    or r['model_rng_unchanged'] is not True or r['target_labels_optimized'] is not False
                    or r['confusion_counts_exact'] is not True):
                raise ValueError('Target identity/runtime/no-update policy differs: '+v)
            target[v] = r['target_miou']
        else:r = old['reports'][freeze['reused_models'][v]['old_variant']]
        matrix, iou = reduce_confusion(r)
        if not np.array_equal(matrix.sum(1), gt):raise ValueError('GT class totals differ across ten models')
        matrices[v] = matrix;class_iou[v] = iou.tolist()
    if job['source_miou'] != source or job['target_miou'] != target:
        raise ValueError('Combined source/target score mapping differs')
    comparisons = {domain: {f'{a}-{b}': 100*(values[a]-values[b]) for a, b in freeze['primary_comparisons']}
                   for domain, values in (('source', source), ('target', target))}
    interactions = {domain: {f'({u}-U0)-({c}-C0)': 100*((values[u]-values['U0'])-(values[c]-values['C0']))
                            for u, c in (('UA', 'CA'), ('UR', 'CR'))}
                    for domain, values in (('source', source), ('target', target))}
    for key, expected in (('comparisons_percentage_points', comparisons), ('interactions_percentage_points', interactions)):
        if set(job[key]) != set(expected):raise ValueError('Missing declared contrast domain')
        for domain, pairs in expected.items():
            if (set(job[key][domain]) != set(pairs) or any(not math.isclose(job[key][domain][k], val, abs_tol=1e-12, rel_tol=0)
                                                        for k, val in pairs.items())):
                raise ValueError('Predeclared contrast/interaction differs')
    rows = [{'variant': v, 'source_miou': source[v], 'target_miou': target[v], 'target_class_iou': class_iou[v],
             'centered': v.startswith('C'), 'seed': 0, 'training_updates': 40000,
             'source_samples': 2497, 'target_samples': 500} for v in VARIANTS]
    return {'rows': rows, 'source_miou': source, 'target_miou': target, 'target_gt_class_totals': gt.tolist(),
            'valid_target_pixels': int(gt.sum()), 'comparisons_percentage_points': comparisons,
            'interactions_percentage_points': interactions}


def accept(root):
    root = Path(root).resolve();checked = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1);rel = PurePosixPath(name);path = root.joinpath(*rel.parts).resolve()
        if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root) or name in checked:
            raise ValueError('Unsafe/duplicate checksum path')
        if file_sha(path) != digest:raise ValueError('Returned SHA256 differs: '+name)
        checked[name] = digest
    if not {'run_manifest.json', 'final_comparison.json'}.issubset(checked):raise ValueError('Missing checked final reports')
    job = json.loads((root / 'final_comparison.json').read_bytes());result = validate_comparison(job)
    manifest = json.loads((root / 'run_manifest.json').read_bytes())
    if (manifest['git_sha'] != PRODUCER or manifest['phase'] != 32 or manifest['study'] != STUDY
            or manifest['freeze_sha256'] != verify_target_contract() or manifest['contract'] != target_contract()):
        raise ValueError('Target run manifest differs')
    expected = {'run_manifest.json', 'final_comparison.json'};ids = target_ids()
    for v in NEW_FINALS:
        paths = list((root / v).glob('*/metrics.json'))
        if len(paths) != 1:raise ValueError('Need one committed result per variant')
        path = paths[0];trace = path.parent / 'target_eval.jsonl'
        for p in (path, trace):expected.add(p.relative_to(root).as_posix())
        report = json.loads(path.read_bytes())
        if report != job['reports'][v] or file_sha(trace) != report['trace_sha256']:
            raise ValueError('Returned individual result/trace differs')
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        if ([r['id'] for r in records] != ids or any(r['prediction_shape'] != [1, 1024, 2048]
                or type(r['valid_pixels']) is not int or not 0 < r['valid_pixels'] <= 1024*2048 for r in records)
                or sum(r['valid_pixels'] for r in records) != result['valid_target_pixels']):
            raise ValueError('Returned target trace ID/geometry/coverage differs')
    if set(checked) != expected:raise ValueError('Missing/unexpected checked target files')
    return {'schema_version': 1, 'phase': 32, 'accepted': True, 'accepted_date': date.today().isoformat(),
            'producer_git_sha': PRODUCER, 'files_verified': checked, 'provided_archive_sha256': ARCHIVE_SHA,
            'archive_bytes_available_locally': False, 'checkpoint_or_image_bytes_replayed_locally': False,
            'all_ten_confusion_matrices_independently_reduced': True, 'target_coverage_traces_checked': True,
            'trace_limit': 'traces record IDs/shapes/GT counts, not predictions; equal trace hashes do not imply equal predictions',
            'study_status': 'complete_single_seed_exploratory', 'additional_training_authorized': False,
            'formal_report': job, **result}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv);result = accept(args.input)
    args.output.write_bytes((json.dumps(result, indent=2, allow_nan=False)+'\n').encode())
    print(json.dumps({k: result[k] for k in ('accepted', 'valid_target_pixels', 'target_miou',
                                           'comparisons_percentage_points', 'interactions_percentage_points')}, indent=2))
    return 0


if __name__ == '__main__':raise SystemExit(main())
