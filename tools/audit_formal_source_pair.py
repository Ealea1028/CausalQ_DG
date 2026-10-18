"""Accept returned Phase22 metadata and recompute source IoU from integer counts."""

import hashlib
import json
import math
from pathlib import Path

from causalq.datasets.strict_source import source_manifest_metadata
from causalq.protocols.formal_source import formal_config
from tools.strict_source_smoke import tree_digest

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = '2537653ae9be0d84afbb86e22b5ec2200d280885'
PAIR_SHA = '49b13eed277266542f79d59ff0d7c7fa3edc4eb6d78cee53f1f8292ab79b9b64'
SUMS_SHA = '07fbe0f7b6a34bff33151df50ea50d0d5f8be70c6b7f87dc1222169fa68acf7a'


def validate_pair(report):
    if (report['ok'] is not True or report['phase'] != 22 or report['git_sha'] != PRODUCER
            or report['seed'] != 0 or report['optimizer_updates_per_variant'] != 40000
            or report['formal_training_authorized'] is not True or report['target_data_accessed'] is not False
            or report['target_miou'] is not None or set(report['reports']) != {'M0', 'M1'}):
        raise ValueError('Formal pair protocol/budget differs')
    expected_manifest = source_manifest_metadata()
    for variant, worker in report['reports'].items():
        metadata = worker['metadata']
        config_path = ROOT / f'configs/strict_dg/formal/{variant.lower()}_seed0.yaml'
        if (worker['ok'] is not True or worker['variant'] != variant or worker['phase'] != 22
                or worker['formal_optimizer_updates'] != 40000 or worker['checkpoint_selection'] != 'fixed_final'
                or worker['formal_training_authorized'] is not True or worker['new_loss_enabled'] is not False
                or worker['target_data_accessed'] is not False or worker['target_miou'] is not None
                or metadata['git_sha'] != PRODUCER or metadata['variant'] != variant or metadata['seed'] != 0
                or metadata['schedule_horizon'] != 40000 or metadata['checkpoint_selection'] != 'fixed_final'
                or metadata['centered_residuals'] != (variant == 'M1')
                or metadata['config_sha256'] != hashlib.sha256(config_path.read_bytes()).hexdigest()
                or metadata['config_tree_sha256'] != tree_digest(formal_config(variant))
                or any(metadata[key] != value for key, value in expected_manifest.items())
                or not worker['checkpoint'].startswith('segments/39000_40000_')
                or worker['initial_head_sha256'] == worker['final_head_sha256']):
            raise ValueError('Formal worker recipe/provenance differs')
        evaluation = worker['source_evaluation']
        matrix = evaluation['confusion_matrix']
        if (len(matrix) != 19 or any(len(row) != 19 for row in matrix)
                or any(type(x) is not int or x < 0 for row in matrix for x in row)
                or evaluation['sample_count'] != 2497 or evaluation['valid_classes'] != 19
                or evaluation['seen_ids_sha256'] != expected_manifest['manifest_sha256']['gta5_source_val.txt']
                or sum(map(sum, matrix)) != evaluation['valid_pixel_count']
                or any(evaluation[key] is not True for key in ('confusion_counts_exact', 'evaluation_rng_unchanged',
                    'training_state_unchanged', 'model_mode_restored'))):
            raise ValueError('Full source evaluation coverage/counts differ')
        ious = []
        for c in range(19):
            union = sum(matrix[c]) + sum(row[c] for row in matrix) - matrix[c][c]
            if union <= 0:
                raise ValueError('Missing class in formal source confusion')
            iou = matrix[c][c] / union
            if not math.isclose(iou, evaluation['class_iou'][c], abs_tol=1e-12, rel_tol=0):
                raise ValueError('Per-class source IoU differs from confusion')
            ious.append(iou)
        miou = math.fsum(ious) / 19
        if any(not math.isclose(miou, value, abs_tol=1e-12, rel_tol=0) for value in
               (worker['source_miou'], evaluation['source_miou'], evaluation['independent_miou'], report['source_miou'][variant])):
            raise ValueError('Source mIoU differs from confusion')
    left, right = (report['reports'][v] for v in ('M0', 'M1'))
    for key in ('environment', 'manifest_sha256', 'backbone_sha256', 'backbone_weights_sha256', 'loss_backend'):
        if left['metadata'][key] != right['metadata'][key]:
            raise ValueError('Formal matched-pair identity differs')
    delta = right['source_miou'] - left['source_miou']
    if (left['initial_head_sha256'] != right['initial_head_sha256']
            or not math.isclose(delta, report['source_miou_delta_M1_minus_M0'], abs_tol=1e-15)):
        raise ValueError('Formal initialization/delta differs')
    return {'formal_pair_accepted': True, 'producer_git_sha': PRODUCER, 'report_sha256': PAIR_SHA,
            'source_miou': report['source_miou'], 'delta_percentage_points': delta * 100,
            'initial_head_sha256': left['initial_head_sha256'], 'target_data_accessed': False,
            'environment': left['metadata']['environment'],
            'checkpoint_and_full_traces_locally_available': False,
            'scope': 'returned_hash_pinned_reports_and_independent_confusion_reduction'}


def audit_pair(directory=ROOT / 'experiments/STRICT_DG_PHASE22'):
    raw = (directory / 'pair_report.json').read_bytes()
    sums = (directory / 'REPORT_SHA256SUMS').read_bytes()
    if hashlib.sha256(raw).hexdigest() != PAIR_SHA or hashlib.sha256(sums).hexdigest() != SUMS_SHA:
        raise ValueError('Formal returned evidence SHA256 differs')
    expected = {line.split()[1]: line.split()[0] for line in sums.decode().splitlines()}
    if set(expected) != {'M0/formal_report.json', 'M1/formal_report.json', 'pair_report.json'}:
        raise ValueError('Formal checksum coverage differs')
    report = json.loads(raw)
    for variant, worker in report['reports'].items():
        worker_raw = (directory / f'{variant.lower()}_formal_report.json').read_bytes()
        if (hashlib.sha256(worker_raw).hexdigest() != expected[f'{variant}/formal_report.json']
                or json.loads(worker_raw) != worker):
            raise ValueError('Embedded formal report differs')
    if expected['pair_report.json'] != PAIR_SHA:
        raise ValueError('Pair checksum differs')
    return validate_pair(report)
