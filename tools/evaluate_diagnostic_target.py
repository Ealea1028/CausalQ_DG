"""Phase32 six-new-model native Cityscapes500 final evaluation; no training."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import uuid
import zipfile

import numpy as np
import torch

from causalq.protocols.diagnostic_20261020 import ROOT, STUDY
from causalq.protocols.diagnostic_final_source import NEW_FINALS, final_source_report, final_config
from causalq.protocols.diagnostic_target import GEOMETRY, TARGET_MANIFEST_SHA, target_contract, verify_target_contract
from tools.evaluate_target import FinalCityscapes, target_ids, evaluate, validate_result, BACKBONE_STATE_SHA
from tools.run_diagnostic_20261020 import load_model, write_json
from tools.inspect_gta5_source_structure import _git
from tools.strict_source_smoke import tree_digest
from tools.train_strict_source import file_sha


def identity(v):
    return {'git_sha': _git('rev-parse', 'HEAD'), 'variant': v, 'study': STUDY, 'phase': 32,
            'freeze_sha256': verify_target_contract(), 'checkpoint_sha256': target_contract()['new_models'][v]['checkpoint_sha256'],
            'target_manifest_sha256': TARGET_MANIFEST_SHA, 'geometry': GEOMETRY,
            'checkpoint_selection': 'fixed_final_iteration40000'}


def initialize(root):
    expected = {'phase': 32, 'study': STUDY, 'git_sha': _git('rev-parse', 'HEAD'),
                'freeze_sha256': verify_target_contract(), 'contract': target_contract()}
    path = root / 'run_manifest.json'
    if path.exists():
        previous = json.loads(path.read_bytes())
        if any(previous[k] != value for k, value in expected.items()):raise ValueError('Phase32 manifest differs')
    else:write_json(path, {**expected, 'started_utc': datetime.now(timezone.utc).isoformat()})


def completed(root, v):
    paths = sorted((root / v).glob('*/metrics.json'))
    if len(paths) > 1:raise ValueError('Multiple committed target results; no result selection permitted')
    if not paths:return None
    r = validate_result(paths[0].parent, identity(v))
    if (r['phase'] != 32 or r['variant'] != v or r['optimizer_updates_this_run'] != 0
            or r['selection_or_tuning_on_target'] is not False or r['target_data_accessed'] is not True
            or r['environment'] != final_source_report(v)['metadata']['environment']
            or r['source_miou'] != target_contract()['new_models'][v]['source_miou']):
        raise ValueError('Final target runtime/reporting policy differs')
    return r, paths[0]


def run_variant(args):
    v = args.variant
    if completed(args.output_root, v):
        print(f'{v}: committed final target result reused, no GPU work', flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('Phase32 requires AutoDL CUDA/target data')
    if (str(torch.__version__) != '2.7.0+cu126' or version('transformers') != '4.56.1'
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
        raise ValueError('Use frozen runtime/CUBLAS configuration')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark = False;torch.backends.cuda.matmul.allow_tf32 = True
    record = target_contract()['new_models'][v]
    model, style, environment = load_model(args, v, frozen=record, source=final_source_report(v), config=final_config(v))
    del style
    if tree_digest(model.backbone.state_dict()) != BACKBONE_STATE_SHA:raise ValueError('Frozen backbone differs')
    ids = target_ids();dataset = FinalCityscapes(args.data_root / 'cityscapes', ids)
    attempt = args.output_root / v / uuid.uuid4().hex;attempt.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    result = evaluate(model, dataset, ids, attempt)
    if file_sha(args.training_output_root / record['checkpoint_relative_to_output_root']) != record['checkpoint_sha256']:
        raise ValueError('Final checkpoint changed during evaluation')
    report = {'schema_version': 1, 'phase': 32, 'study': STUDY, 'ok': True, 'identity': identity(v), 'variant': v,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'environment': environment,
              'source_miou': record['source_miou'], 'optimizer_updates_this_run': 0, 'target_data_accessed': True,
              'selection_or_tuning_on_target': False, 'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3,
              **result}
    write_json(attempt / 'metrics.json', report);print(json.dumps(report, indent=2), flush=True)


def package(root):
    freeze = target_contract();reports = {};files = [root / 'run_manifest.json']
    for v in NEW_FINALS:
        done = completed(root, v)
        if done is None:raise ValueError('Need every new target result before final report')
        r, path = done;reports[v] = r;files += [path, path.parent / 'target_eval.jsonl']
    totals = [np.asarray(r['confusion_matrix'], dtype=np.int64).sum(1).tolist() for r in reports.values()]
    if any(t != totals[0] for t in totals[1:]):raise ValueError('Target GT totals differ across models')
    old = json.loads((ROOT / 'experiments/STRICT_DG_PHASE26/final_comparison.json').read_bytes())
    old_counts = np.asarray(old['reports']['M0']['confusion_matrix'], dtype=np.int64).sum(1).tolist()
    if totals[0] != old_counts:raise ValueError('New target GT totals differ from reused old evaluation')
    target = {v: r['target_miou'] for v, r in freeze['reused_models'].items()}
    target.update({v: r['target_miou'] for v, r in reports.items()})
    source = {v: r['source_miou'] for v, r in {**freeze['reused_models'], **freeze['new_models']}.items()}
    comparisons = {d: {f'{a}-{b}': 100*(values[a]-values[b]) for a, b in freeze['primary_comparisons']}
                   for d, values in (('source', source), ('target', target))}
    interactions = {d: {f'({u}-U0)-({c}-C0)': 100*((values[u]-values['U0'])-(values[c]-values['C0']))
                       for u, c in (('UA', 'CA'), ('UR', 'CR'))} for d, values in (('source', source), ('target', target))}
    result = {'schema_version': 1, 'phase': 32, 'study': STUDY, 'ok': True, 'git_sha': _git('rev-parse', 'HEAD'),
              'freeze_sha256': verify_target_contract(), 'reports': reports, 'reused_models': freeze['reused_models'],
              'target_miou': target, 'source_miou': source, 'comparisons_percentage_points': comparisons,
              'interactions_percentage_points': interactions, 'optimizer_updates_this_run': 0,
              'target_data_accessed': True, 'selection_or_tuning_on_target': False,
              'target_feedback_already_seen': True, 'additional_training_authorized': False,
              'limits': freeze['uncertainty']}
    report_path = root / 'final_comparison.json';write_json(report_path, result);files.append(report_path)
    sums = root / 'SHA256SUMS'
    sums.write_bytes(''.join(f'{file_sha(p)}  {p.relative_to(root).as_posix()}\n' for p in files).encode())
    archive_path = root / 'phase32_export.zip';temporary = root / 'phase32_export.zip.partial'
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*files, sums]:archive.write(path, path.relative_to(root).as_posix())
    temporary.replace(archive_path)
    print(json.dumps({'ok': True, 'target_miou': target, 'export': str(archive_path), 'sha256': file_sha(archive_path)}, indent=2), flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=(*NEW_FINALS, 'initialize', 'package'), required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--training-output-root', type=Path, default=Path('/root/autodl-tmp/outputs/CausalQ_DG'))
    p.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets'))
    p.add_argument('--pretrained-root', type=Path, default=Path('/root/autodl-tmp/pretrained'))
    args = p.parse_args(argv)
    protected = [ROOT, args.data_root, args.pretrained_root]
    protected += [(args.training_output_root / r['checkpoint_relative_to_output_root']).parents[2]
                  for r in target_contract()['new_models'].values()]
    if any(args.output_root.resolve().is_relative_to(path.resolve()) for path in protected):
        p.error('Output must be outside code/data/weights/formal training runs')
    if _git('status', '--porcelain'):raise ValueError('Use clean exact checkout')
    args.output_root.mkdir(parents=True, exist_ok=True);initialize(args.output_root)
    if args.variant == 'package':package(args.output_root)
    elif args.variant != 'initialize':run_variant(args)
    return 0


if __name__ == '__main__':raise SystemExit(main())
