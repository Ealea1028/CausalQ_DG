"""Audit pinned full-census metadata; raw CSV/visual review remains separate."""

import hashlib
import json
import math
from pathlib import Path
import zipfile

from causalq.analysis.query_components import fixed_source_ids
from causalq.datasets.strict_source import MANIFEST_HASHES, load_membership
from tools.audit_anchor_query_effect import CHECKPOINT_SHA, HEAD_SHA, finite_range
from tools.audit_query_component_diagnostics import audit_components, CSV_HASHES
from tools.analyze_query_components import check_gate20_replay

ROOT = Path(__file__).resolve().parents[1]
REPORT_SHA = '88dd5291bc163165f7f174c9820f5b2f8924fffb82c230de76d8b9e1dbcd4de5'
SUMS_SHA = '53e6740705c8b3ad88e66cfd98d70a3c2514b9eed00f4e64af77c3a302e2b4b1'
ZIP_SHA = '3010b35769c33408eae2af2b57c491d4e65b70880dffb24da35108e935be8d6b'
PRODUCER_SHA = '79e440b5e78b34e39ef1792e67807ba164d6752e'


def sha_stream(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def census_artifact_hashes(report):
    return {**report['output_sha256'], 'sample_index.jsonl': report['sample_index_sha256'],
            'analysis_export.zip': report['analysis_export_sha256'], **report['figure_sha256']}


def validate_census_report(report):
    expected = {'schema_version': 1, 'phase': 20, 'scope': 'source_component_census2497x2',
        'git_sha': PRODUCER_SHA, 'git_status': '', 'ok': True, 'optimizer_updates_this_run': 0,
        'checkpoint_training_updates': 20, 'formal_training_authorized': False, 'target_data_accessed': False,
        'new_loss_enabled': False, 'source_miou': None, 'target_miou': None,
        'sample_count': 2497, 'view_count': 4994, 'backbone_forward_calls': 4994,
        'responsibility_row_count': 189772, 'effect_row_count': 142329, 'figure_count': 60,
        'gate20_replay_count': 20, 'gate20_replay_accepted': True, 'parameter_replay_exact': True,
        'model_rng_unchanged': True, 'original_files_unchanged': True,
        'appearance_replay_exact': True, 'counterfactual_routing_exact': True,
        'manifest_sha256': MANIFEST_HASHES, 'checkpoint_sha256': CHECKPOINT_SHA, 'head_sha256': HEAD_SHA,
        'analysis_export_sha256': ZIP_SHA}
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError(f'Census report mismatch: {key}')
    if report['diagnostic_audit'] != audit_components(ROOT / 'experiments/STRICT_DG_PHASE20'):
        raise ValueError('Census gate20 lineage mismatch')
    prior = json.loads((ROOT / 'experiments/STRICT_DG_PHASE20/component_effect_report.json').read_bytes())
    for key in ('anchor_audit', 'centered_audit', 'original_artifacts', 'remote_anchor_artifacts', 'runtime',
                'style_config', 'environment', 'parameter_rows', 'checkpoint_sha256', 'head_sha256', 'alpha'):
        if report[key] != prior[key]:
            raise ValueError(f'Census upstream identity/runtime mismatch: {key}')
    if report['output_sha256']['query_similarity.csv'] != CSV_HASHES['query_similarity.csv']:
        raise ValueError('Census parameter hash mismatch')
    _, ids = load_membership()
    selected = fixed_source_ids(ids)
    if report['selection']['ids'] != ids or report['selection']['visualization_ids'] != selected:
        raise ValueError('Census source selection mismatch')
    refs = {s['source_id']: s for s in prior['samples']}
    samples = report['samples']
    if len(samples) != 20 or {s['source_id'] for s in samples} != set(refs):
        raise ValueError('Census replay member coverage mismatch')
    for sample in samples:
        check_gate20_replay(sample, refs[sample['source_id']])
    if not finite_range(report['max_decomposition_error'], 0, 1e-6):
        raise ValueError('Census component identity mismatch')
    if set(report['per_class_effect_summary']) != {str(c) for c in range(19)}:
        raise ValueError('Census effect class coverage mismatch')
    for component, metrics in report['effect_summary'].items():
        if component not in ('full', 'anchor', 'residual'):
            raise ValueError('Census effect component mismatch')
        for key, metric in metrics.items():
            parts = [report['per_class_effect_summary'][str(c)][component][key] for c in range(19)]
            count = sum(p['defined_count'] for p in parts)
            if count != 47443 or metric['defined_count'] != count or any(p['defined_count'] != 2497 for p in parts):
                raise ValueError('Census effect denominator mismatch')
            if not finite_range(metric['mean'], 0, float('inf')) or any(not finite_range(p['mean'], 0, float('inf')) for p in parts):
                raise ValueError('Census nonfinite effect summary')
            rebuilt = sum(p['mean'] * p['defined_count'] for p in parts) / count
            if not math.isclose(metric['mean'], rebuilt, rel_tol=1e-9, abs_tol=1e-16):
                raise ValueError('Census effect aggregate mismatch')
    for view in ('original', 'photometric'):
        for key, total in report['responsibility_summary'][view].items():
            parts = [report['per_class_responsibility_summary'][str(c)][view][str(r)][key] for c in range(19) for r in range(2)]
            count = sum(p['defined_count'] for p in parts)
            if count != 55446 or total['defined_count'] != count:
                raise ValueError('Census responsibility denominator mismatch')
            rebuilt = sum(p['mean'] * p['defined_count'] for p in parts if p['defined_count']) / count
            if not finite_range(total['mean'], 0, math.log(2) + 1e-6 if 'entropy' in key else float('inf')):
                raise ValueError('Census responsibility range mismatch')
            if not math.isclose(total['mean'], rebuilt, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('Census responsibility aggregate mismatch')
    figures = report['figures']
    if len(figures) != 20 or {f['source_id'] for f in figures} != set(selected):
        raise ValueError('Census figure source coverage mismatch')
    expected_paths = {f'figures/{i}/{name}.png' for i in selected for name in ('overview', 'effects', 'responsibility')}
    if set(report['figure_sha256']) != expected_paths:
        raise ValueError('Census figure paths mismatch')
    for figure in figures:
        if figure['files'] != ['overview.png', 'effects.png', 'responsibility.png'] or set(figure['effect_signed_scale_by_class']) != {'0', '2', '8', '13', '11'}:
            raise ValueError('Census figure class/panel coverage mismatch')
    for digest in census_artifact_hashes(report).values():
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Census artifact digest invalid')
    return {'census_report_verified': True, 'report_sha256': REPORT_SHA,
            'archive_sha256': ZIP_SHA, 'raw_csv_independently_reviewed': False, 'actual_figures_visually_reviewed': False}


def audit_census(directory):
    raw = (directory / 'census_report.json').read_bytes()
    sums = (directory / 'census_SHA256SUMS').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA or hashlib.sha256(sums).hexdigest() != SUMS_SHA:
        raise ValueError('Census report/checksum SHA256 mismatch')
    report = json.loads(raw)
    lines = sums.decode('ascii').splitlines()
    entries = dict((line.split('  ', 1)[1], line.split('  ', 1)[0]) for line in lines)
    if len(entries) != len(lines) or entries != {'component_effect_report.json': REPORT_SHA, **census_artifact_hashes(report)}:
        raise ValueError('Census original checksum manifest mismatch')
    return validate_census_report(report)


def verify_archive_bytes(path, expected_zip_sha, expected_files):
    """Stream/check every file without extracting untrusted archive paths."""
    with path.open('rb') as handle:
        if sha_stream(handle) != expected_zip_sha:
            raise ValueError('Census archive SHA256 mismatch')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected_files):
            raise ValueError('Census archive file coverage mismatch')
        for name in names:
            if name.startswith(('/', '\\')) or '..' in Path(name).parts:
                raise ValueError('Census unsafe archive path')
            with archive.open(name) as handle:
                if sha_stream(handle) != expected_files[name]:
                    raise ValueError(f'Census archive member SHA256 mismatch: {name}')
    return {'archive_bytes_verified': True, 'file_count': len(names), 'actual_figures_visually_reviewed': False}


def audit_census_archive(path, directory=ROOT / 'experiments/STRICT_DG_PHASE20'):
    audit_census(directory)
    report = json.loads((directory / 'census_report.json').read_bytes())
    files = census_artifact_hashes(report)
    del files['analysis_export.zip']
    return verify_archive_bytes(path, ZIP_SHA, files)
