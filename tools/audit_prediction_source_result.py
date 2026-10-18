"""Hash-pinned P1 report acceptance and independent source-confusion reduction."""

import hashlib
import json
import math
from pathlib import Path

from causalq.protocols.formal_source import formal_config
from tools.audit_formal_source_pair import audit_pair
from tools.strict_source_smoke import tree_digest

ROOT = Path(__file__).resolve().parents[1]
REPORT_SHA = '9e029d792319a823675a233ad850804f754c0700ef1389f46cfae34796e78d1f'
SUMS_SHA = 'fa22fa5a6dcfd15f1c29495872d89e538d89e57f7aed3c2717e303f1e6797ca7'
PRODUCER = '268fda1b340f8bd5ceded83f9d9b9c6bf0009c07'


def validate_prediction_result(report):
    baseline = audit_pair()
    metadata = report['metadata']
    config = formal_config('P1')
    path = ROOT / 'configs/strict_dg/formal/p1_seed0.yaml'
    if (report['ok'] is not True or report['phase'] != 23 or report['variant'] != 'P1'
            or report['formal_optimizer_updates'] != 40000 or report['formal_training_authorized'] is not True
            or report['checkpoint_selection'] != 'fixed_final' or report['target_data_accessed'] is not False
            or report['target_miou'] is not None or report['cqe_enabled'] is not False
            or report['prediction_consistency'] != config['prediction_consistency']
            or metadata['prediction_consistency'] != config['prediction_consistency']
            or metadata['variant'] != 'P1' or metadata['git_sha'] != PRODUCER or metadata['seed'] != 0
            or metadata['schedule_horizon'] != 40000 or metadata['centered_residuals'] is not True
            or metadata['formal_training_authorized'] is not True or metadata['target_data_accessed'] is not False
            or metadata['cqe_enabled'] is not False or metadata['baseline_audit'] != baseline
            or metadata['config_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()
            or metadata['config_tree_sha256'] != tree_digest(config)
            or metadata['environment'] != baseline['environment']
            or report['initial_head_sha256'] != baseline['initial_head_sha256']
            or report['initial_head_sha256'] == report['final_head_sha256']
            or not report['checkpoint'].startswith('segments/39000_40000_')):
        raise ValueError('P1 formal recipe/provenance/budget differs')
    evaluation = report['source_evaluation']
    matrix = evaluation['confusion_matrix']
    if (evaluation['sample_count'] != 2497 or evaluation['valid_classes'] != 19
            or evaluation['seen_ids_sha256'] != metadata['manifest_sha256']['gta5_source_val.txt']
            or len(matrix) != 19 or any(len(row) != 19 for row in matrix)
            or any(type(x) is not int or x < 0 for row in matrix for x in row)
            or sum(map(sum, matrix)) != evaluation['valid_pixel_count']
            or any(evaluation[k] is not True for k in ('confusion_counts_exact', 'evaluation_rng_unchanged',
                'training_state_unchanged', 'model_mode_restored'))):
        raise ValueError('P1 source evaluation coverage/counts differ')
    ious = []
    for c in range(19):
        union = sum(matrix[c]) + sum(row[c] for row in matrix) - matrix[c][c]
        if union <= 0:
            raise ValueError('P1 source class absent')
        iou = matrix[c][c] / union
        if not math.isclose(iou, evaluation['class_iou'][c], abs_tol=1e-12, rel_tol=0):
            raise ValueError('P1 per-class IoU differs')
        ious.append(iou)
    miou = math.fsum(ious) / 19
    if any(not math.isclose(miou, x, abs_tol=1e-12, rel_tol=0) for x in
           (report['source_miou'], evaluation['source_miou'], evaluation['independent_miou'])):
        raise ValueError('P1 source mIoU differs')
    return {'formal_p1_accepted': True, 'report_sha256': REPORT_SHA, 'producer_git_sha': PRODUCER,
            'source_miou': miou, 'delta_vs_m1_percentage_points': (miou - baseline['source_miou']['M1']) * 100,
            'target_data_accessed': False, 'checkpoint_and_full_traces_locally_available': False}


def audit_prediction(directory=ROOT / 'experiments/STRICT_DG_PHASE23'):
    raw = (directory / 'formal_report.json').read_bytes()
    sums = (directory / 'REPORT_SHA256SUMS').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA or hashlib.sha256(sums).hexdigest() != SUMS_SHA:
        raise ValueError('P1 returned evidence SHA256 differs')
    expected = {line.split()[1]: line.split()[0] for line in sums.decode().splitlines()}
    if set(expected) != {'P1/formal_report.json', 'P1/config.yaml', 'P1/metadata.json'}:
        raise ValueError('P1 checksum coverage differs')
    report = json.loads(raw)
    config = (ROOT / 'configs/strict_dg/formal/p1_seed0.yaml').read_bytes()
    metadata = (json.dumps(report['metadata'], indent=2) + '\n').encode()
    for name, data in [('formal_report.json', raw), ('config.yaml', config), ('metadata.json', metadata)]:
        if hashlib.sha256(data).hexdigest() != expected['P1/' + name]:
            raise ValueError('P1 report/config/embedded metadata hash differs')
    return validate_prediction_result(report)
