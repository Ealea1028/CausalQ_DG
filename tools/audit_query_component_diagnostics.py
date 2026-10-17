"""Pinned engineering acceptance for the Phase20 20-image diagnostic gate."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path

from causalq.analysis.query_components import fixed_source_ids, intervention_seed
from causalq.datasets.strict_source import MANIFEST_HASHES, load_membership
from tools.audit_anchor_query_effect import audit_anchor, finite_range, CHECKPOINT_SHA, HEAD_SHA
from tools.audit_centered_query import audit_centered

ROOT = Path(__file__).resolve().parents[1]
REPORT_SHA = "7a158511756df69981fa36989b5e39b8e0af524f1807d58fbf62be4b9c4cc3ce"
SUMS_SHA = "aed313bc9850190f94bafab9537f94c0728bd5c745179fe6028a60b34b26c3f6"
PRODUCER_SHA = "9439771489c539d7e9a13b1576305ccd2e3b85fd"
CSV_HASHES = {
    "effect_statistics.csv": "98ff86544fd8047da30303c4eef92cd26ffa8d9192579f8ae265d0b7c3c0e72a",
    "query_similarity.csv": "0e7b28f1d5aa91b3b2ffa022d684da7bf67653e0cc549a56f9a1e2505c6eb496",
    "responsibility.csv": "dc1f1066814c5fb47be8eccc35aafe13d9b401d8897f82a5d90dfcce8fd9f3c2"}


def validate_component_report(report):
    expected = {"schema_version": 1, "phase": 20, "scope": "source_component_diagnostics20x2",
        "git_sha": PRODUCER_SHA, "git_status": "", "ok": True, "optimizer_updates_this_run": 0,
        "checkpoint_training_updates": 20, "formal_training_authorized": False, "target_data_accessed": False,
        "new_loss_enabled": False, "source_miou": None, "target_miou": None, "sample_count": 20,
        "view_count": 40, "backbone_forward_calls": 40, "responsibility_row_count": 1520, "effect_row_count": 1140,
        "model_rng_unchanged": True, "original_files_unchanged": True, "manifest_sha256": MANIFEST_HASHES,
        "checkpoint_sha256": CHECKPOINT_SHA, "head_sha256": HEAD_SHA, "output_sha256": CSV_HASHES,
        "train_count": 22469, "val_count": 2497, "source_validation": "gta5_holdout"}
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError(f"Component report mismatch: {key}")
    if report['anchor_audit'] != audit_anchor(ROOT / 'experiments/STRICT_DG_PHASE19') or report['centered_audit'] != audit_centered(ROOT / 'experiments/STRICT_DG_PHASE18'):
        raise ValueError('Component upstream acceptance differs')
    for key, value in {'torch': '2.7.0+cu126', 'cuda': '12.6', 'transformers': '4.56.1', 'gpu': 'NVIDIA GeForce RTX 4090 D'}.items():
        if report['environment'][key] != value:
            raise ValueError('Component runtime differs')
    runtime = {'initialization_seed': 0, 'autocast': 'bfloat16', 'workers': 0, 'deterministic_algorithms': True,
               'cublas_workspace_config': ':4096:8', 'matmul_allow_tf32': True, 'cudnn_benchmark': False}
    if report['runtime'] != runtime:
        raise ValueError('Component deterministic flags differ')
    _, ids = load_membership()
    chosen = fixed_source_ids(ids)
    if report['selection']['ids'] != chosen or [s['source_id'] for s in report['samples']] != chosen:
        raise ValueError('Component fixed sample coverage differs')
    for sample in report['samples']:
        if (sample['intervention_seed'] != intervention_seed(sample['source_id']) or
                sample['appearance_replay_exact'] is not True or sample['counterfactual_routing_exact'] is not True or
                not finite_range(sample['decomposition_max_error'], 0, 1e-6)):
            raise ValueError('Component sample replay/algebra differs')
        for key in ('original_input_sha256', 'photometric_input_sha256'):
            digest = sample[key]
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Component input hash invalid')
        if not finite_range(sample['analysis_seconds_including_metrics_and_style_replay'], 0, float('inf')):
            raise ValueError('Component timing invalid')
        for metrics in sample['effect_summary'].values():
            for metric in metrics.values():
                if metric['defined_count'] != 19 or not finite_range(metric['mean'], 0, float('inf')):
                    raise ValueError('Component per-image effect summary invalid')
    for component, metrics in report['effect_summary'].items():
        if component not in ('full', 'anchor', 'residual'):
            raise ValueError('Unknown effect component')
        for key, metric in metrics.items():
            if metric['defined_count'] != 380 or not finite_range(metric['mean'], 0, float('inf')):
                raise ValueError('Component effect summary invalid')
            reconstructed = sum(s['effect_summary'][component][key]['mean'] for s in report['samples']) / 20
            if not math.isclose(metric['mean'], reconstructed, rel_tol=1e-9, abs_tol=1e-15):
                raise ValueError('Component aggregate differs from image summaries')
    for view in ('original', 'photometric'):
        for key, metric in report['responsibility_summary'][view].items():
            limit = math.log(2) + 1e-6 if 'entropy' in key else .25 if key == 'cross_view_population_variance' else float('inf')
            if metric['defined_count'] != 454 or not finite_range(metric['mean'], 0, limit):
                raise ValueError('Responsibility summary invalid')
    pairs = report['parameter_rows']
    if [(p['class_id'], p['query_r'], p['query_s']) for p in pairs] != [(c, 0, 1) for c in range(19)]:
        raise ValueError('Component parameter pair coverage differs')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
    writer.writeheader()
    writer.writerows(pairs)
    if hashlib.sha256(stream.getvalue().encode()).hexdigest() != CSV_HASHES['query_similarity.csv']:
        raise ValueError('Parameter CSV reconstruction differs')
    return {'component_diagnostics_accepted': True, 'report_sha256': REPORT_SHA,
            'checkpoint_sha256': CHECKPOINT_SHA, 'target_data_accessed': False, 'formal_training_authorized': False}


def audit_components(directory):
    raw = (directory / 'component_effect_report.json').read_bytes()
    sums = (directory / 'component_SHA256SUMS').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA or hashlib.sha256(sums).hexdigest() != SUMS_SHA:
        raise ValueError('Component report/checksum SHA256 mismatch')
    entries = dict((line.split('  ', 1)[1], line.split('  ', 1)[0]) for line in sums.decode('ascii').splitlines())
    if entries != {'component_effect_report.json': REPORT_SHA, **CSV_HASHES}:
        raise ValueError('Component original checksum manifest differs')
    return validate_component_report(json.loads(raw))
