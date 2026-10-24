"""Accept returned formal metadata and independently reduce source confusion counts."""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

import yaml

from causalq.datasets.strict_source import source_manifest_metadata
from causalq.protocols.diagnostic_20261020 import ROOT, NEW_UNCENTERED, STUDY, baseline_source_report
from causalq.protocols.diagnostic_formal import matched_baseline, release_contract, training_config, training_policy, verify_release
from tools.strict_source_smoke import tree_digest

PRODUCER = '0a0350d025b063cca2668ab0606bd80179f13ea3'


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_miou(evaluation):
    matrix = evaluation['confusion_matrix']
    manifest = source_manifest_metadata()['manifest_sha256']['gta5_source_val.txt']
    if (len(matrix) != 19 or any(len(row) != 19 for row in matrix)
            or any(type(n) is not int or n < 0 for row in matrix for n in row)
            or evaluation['sample_count'] != 2497 or evaluation['valid_classes'] != 19
            or evaluation['seen_ids_sha256'] != manifest
            or sum(map(sum, matrix)) != evaluation['valid_pixel_count']
            or any(evaluation[key] is not True for key in ('confusion_counts_exact', 'evaluation_rng_unchanged',
                'training_state_unchanged', 'model_mode_restored'))):
        raise ValueError('Source confusion coverage/state differs')
    ious = []
    for c in range(19):
        union = sum(matrix[c]) + sum(row[c] for row in matrix) - matrix[c][c]
        if union <= 0:raise ValueError('Missing source class')
        ious.append(matrix[c][c] / union)
    miou = math.fsum(ious) / 19
    if (len(evaluation['class_iou']) != 19
            or any(not math.isclose(a, b, abs_tol=1e-12, rel_tol=0) for a, b in zip(ious, evaluation['class_iou']))
            or any(not math.isclose(miou, evaluation[k], abs_tol=1e-12, rel_tol=0)
                   for k in ('source_miou', 'independent_miou'))):
        raise ValueError('Source mIoU differs from integer confusion')
    return miou, [sum(row) for row in matrix]


def accept(root):
    root = root.resolve()
    checked = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        rel = PurePosixPath(name)
        path = root.joinpath(*rel.parts).resolve()
        if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root) or name in checked:
            raise ValueError('Unsafe/duplicate checksum path')
        if file_sha(path) != digest:raise ValueError('Returned SHA256 differs: ' + name)
        checked[name] = digest
    expected_files = {'run_manifest.json', 'phase29_report.json'} | {
        f'{v}/{name}' for v in NEW_UNCENTERED for name in
        ('formal_report.json', 'metadata.json', 'config.yaml', 'initialization.json')}
    if set(checked) != expected_files:raise ValueError('Formal export coverage differs')
    combined = json.loads((root / 'phase29_report.json').read_bytes())
    run = json.loads((root / 'run_manifest.json').read_bytes())
    contract_sha = verify_release()
    if (combined['ok'] is not True or combined['phase'] != 29 or combined['study'] != STUDY
            or combined['git_sha'] != PRODUCER or combined['total_optimizer_updates'] != 160000
            or combined['optimizer_updates_per_variant'] != 40000
            or combined['target_data_accessed'] is not False or combined['target_miou'] is not None
            or set(combined['reports']) != set(NEW_UNCENTERED)
            or set(combined['source_miou']) != set(NEW_UNCENTERED)
            or combined['phase29_contract_sha256'] != contract_sha
            or run['git_sha'] != PRODUCER or run['phase'] != 29
            or run['phase29_contract_sha256'] != contract_sha or run['contract'] != release_contract()):
        raise ValueError('Phase29 producer/budget/contract differs')
    baseline = matched_baseline()
    ground_counts = None
    for variant in NEW_UNCENTERED:
        directory = root / variant
        report = json.loads((directory / 'formal_report.json').read_bytes())
        metadata = json.loads((directory / 'metadata.json').read_bytes())
        initial = json.loads((directory / 'initialization.json').read_bytes())
        config = yaml.safe_load((directory / 'config.yaml').read_bytes())
        if (config != training_config(variant) or report != combined['reports'][variant]
                or metadata != report['metadata'] or metadata['git_sha'] != PRODUCER
                or metadata['variant'] != variant or metadata['seed'] != 0
                or metadata['phase29_contract_sha256'] != contract_sha
                or metadata['config_sha256'] != checked[variant + '/config.yaml']
                or metadata['config_sha256'] != file_sha(ROOT / f'configs/diagnostic_20261020/formal/{variant.lower()}_seed0.yaml')
                or metadata['config_tree_sha256'] != tree_digest(config)
                or metadata['centered_residuals'] is not False
                or any(metadata[k] != value for k, value in training_policy(variant).items())
                or metadata['backbone_weights_sha256'] != config['model']['weights_sha256']
                or metadata['checkpoint_selection'] != 'fixed_final'
                or metadata['formal_training_authorized'] is not True
                or metadata['target_data_accessed'] is not False
                or metadata['target_evaluation_authorized_this_phase'] is not False
                or metadata['schedule_horizon'] != 40000 or metadata['checkpoint_interval'] != 1000
                or metadata['environment'] != baseline['environment']
                or any(metadata[k] != value for k, value in source_manifest_metadata().items())
                or report['ok'] is not True or report['phase'] != 29 or report['variant'] != variant
                or report['formal_optimizer_updates'] != 40000
                or report['formal_training_authorized'] is not True
                or report['checkpoint_selection'] != 'fixed_final' or report['target_data_accessed'] is not False
                or report['target_miou'] is not None
                or report['initial_head_sha256'] != baseline['initial_head_sha256']
                or initial['head_sha256'] != baseline['initial_head_sha256']
                or initial['fresh_seed'] != 0 or initial['promoted_pilot_checkpoint'] is not False
                or report['final_head_sha256'] == report['initial_head_sha256']
                or not re.fullmatch(r'segments/39000_40000_[0-9a-f]{32}/state.pth', report['checkpoint'])
                or not re.fullmatch(r'[0-9a-f]{64}', report['checkpoint_sha256'])):
            raise ValueError('Formal recipe/initialization/state differs')
        miou, counts = source_miou(report['source_evaluation'])
        if any(not math.isclose(miou, n, abs_tol=1e-12, rel_tol=0)
               for n in (report['source_miou'], combined['source_miou'][variant])):
            raise ValueError('Combined source mIoU differs')
        if ground_counts is not None and ground_counts != counts:
            raise ValueError('Matched source GT class totals differ')
        ground_counts = counts
    original = baseline_source_report('U0')['source_miou']
    return {'schema_version': 1, 'accepted_date': '2026-10-24', 'phase': 29, 'accepted': True,
            'producer_git_sha': PRODUCER, 'file_sha256': checked,
            'output_root_from_console': '/root/autodl-tmp/outputs/CausalQ_DG/diagnostic_20261020/phase29/UNCENTERED_SEED0_0a0350d',
            'provided_archive_sha256': 'e10e0e2f25d98d81de3d0c45d47fd29cf1a2657b977cf2fad4ce7e6cca9e5de1',
            'archive_bytes_available_locally': False, 'checkpoints_and_full_traces_available_locally': False,
            'source_confusion_reduced_independently': True, 'source_gt_class_totals': ground_counts,
            'baseline_U0_source_miou': original,
            'source_delta_vs_U0_percentage_points': {v: (m-original)*100 for v, m in combined['source_miou'].items()},
            'target_data_accessed': False, 'formal_report': combined,
            'next_phase': '30: preplanned CA/CR; no result-dependent selection or new pilot'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    result = accept(args.input)
    args.output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps({'accepted': True, 'source_miou': result['formal_report']['source_miou'],
                      'delta_vs_U0_pp': result['source_delta_vs_U0_percentage_points']}, indent=2))
    return 0


if __name__ == '__main__':raise SystemExit(main())
