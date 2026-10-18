"""Returned M2 formal metadata and independent source-confusion acceptance."""

import hashlib
import json
import math
from pathlib import Path

from causalq.protocols.formal_source import formal_config
from tools.audit_formal_source_pair import audit_pair
from tools.strict_source_smoke import tree_digest

ROOT = Path(__file__).resolve().parents[1]
REPORT_SHA = '355a4b5c2e109a35989675aba4d7bf1e0bb416a1b621339d3e9c06bc0b49f5f7'
SUMS_SHA = 'fcfc2263307242b8da2defa9aca9f380a4fd035944462c7b15f667d884fdd578'
PRODUCER = '72b168ae2a3995a711394418354682ed97cdaafc'


def validate_full_cqe_result(report):
    baseline = audit_pair()
    metadata = report['metadata']
    config = formal_config('M2')
    path = ROOT / 'configs/strict_dg/formal/m2_seed0.yaml'
    if (report['ok'] is not True or report['phase'] != 24 or report['variant'] != 'M2'
            or report['formal_optimizer_updates'] != 40000 or report['formal_training_authorized'] is not True
            or report['checkpoint_selection'] != 'fixed_final' or report['target_data_accessed'] is not False
            or report['target_miou'] is not None or report['cqe_enabled'] is not True
            or report['prediction_consistency_enabled'] is not False
            or report['effect_definition'] != 'alpha_times_full_query_delta_pre_addition'
            or report['causal_query_effect'] != config['causal_query_effect']
            or metadata['causal_query_effect'] != config['causal_query_effect']
            or metadata['variant'] != 'M2' or metadata['git_sha'] != PRODUCER or metadata['seed'] != 0
            or metadata['schedule_horizon'] != 40000 or metadata['centered_residuals'] is not True
            or metadata['formal_training_authorized'] is not True or metadata['target_data_accessed'] is not False
            or metadata['cqe_enabled'] is not True or metadata['baseline_audit'] != baseline
            or metadata['config_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()
            or metadata['config_tree_sha256'] != tree_digest(config)
            or metadata['environment'] != baseline['environment']
            or report['initial_head_sha256'] != baseline['initial_head_sha256']
            or report['initial_head_sha256'] == report['final_head_sha256']
            or not report['checkpoint'].startswith('segments/39000_40000_')):
        raise ValueError('M2 formal recipe/provenance/budget differs')
    evaluation = report['source_evaluation']
    matrix = evaluation['confusion_matrix']
    if (evaluation['sample_count'] != 2497 or evaluation['valid_classes'] != 19
            or evaluation['seen_ids_sha256'] != metadata['manifest_sha256']['gta5_source_val.txt']
            or len(matrix) != 19 or any(len(row) != 19 for row in matrix)
            or any(type(x) is not int or x < 0 for row in matrix for x in row)
            or sum(map(sum, matrix)) != evaluation['valid_pixel_count']
            or any(evaluation[k] is not True for k in ('confusion_counts_exact', 'evaluation_rng_unchanged',
                'training_state_unchanged', 'model_mode_restored'))):
        raise ValueError('M2 source evaluation coverage/counts differ')
    ious = []
    for c in range(19):
        union = sum(matrix[c]) + sum(row[c] for row in matrix) - matrix[c][c]
        if union <= 0: raise ValueError('M2 source class absent')
        iou = matrix[c][c] / union
        if not math.isclose(iou, evaluation['class_iou'][c], abs_tol=1e-12, rel_tol=0):
            raise ValueError('M2 per-class IoU differs')
        ious.append(iou)
    miou = math.fsum(ious) / 19
    if any(not math.isclose(miou, x, abs_tol=1e-12, rel_tol=0) for x in
           (report['source_miou'], evaluation['source_miou'], evaluation['independent_miou'])):
        raise ValueError('M2 source mIoU differs')
    return {'formal_m2_accepted': True, 'report_sha256': REPORT_SHA, 'producer_git_sha': PRODUCER,
            'source_miou': report['source_miou'],
            'delta_vs_m1_percentage_points': (report['source_miou'] - baseline['source_miou']['M1']) * 100,
            'target_data_accessed': False, 'checkpoint_and_full_traces_locally_available': False}


def audit_full_cqe(directory=ROOT / 'experiments/STRICT_DG_PHASE24'):
    raw, sums = [(directory / name).read_bytes() for name in ('formal_report.json', 'REPORT_SHA256SUMS')]
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA or hashlib.sha256(sums).hexdigest() != SUMS_SHA:
        raise ValueError('M2 returned evidence SHA256 differs')
    expected = {line.split()[1]: line.split()[0] for line in sums.decode().splitlines()}
    if set(expected) != {'M2/formal_report.json', 'M2/config.yaml', 'M2/metadata.json'}:
        raise ValueError('M2 checksum coverage differs')
    report = json.loads(raw)
    for name, data in [('formal_report.json', raw), ('config.yaml', (ROOT / 'configs/strict_dg/formal/m2_seed0.yaml').read_bytes()),
                       ('metadata.json', (json.dumps(report['metadata'], indent=2) + '\n').encode())]:
        if hashlib.sha256(data).hexdigest() != expected['M2/' + name]:
            raise ValueError('M2 report/config/embedded metadata hash differs')
    return validate_full_cqe_result(report)
