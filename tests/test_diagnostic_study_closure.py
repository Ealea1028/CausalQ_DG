"""Read-only final evidence acceptance and the predeclared reporting arithmetic."""

from copy import deepcopy
import json

import pytest

from causalq.protocols.diagnostic_20261020 import ROOT
from causalq.protocols.diagnostic_final_source import ACCEPTANCE, NEW_FINALS
from tools import accept_diagnostic_target_results as acceptance
from tools.report_diagnostic_study import result_tables


def accepted():
    return json.loads((ACCEPTANCE / 'phase32_acceptance.json').read_bytes())


def export_fixture(root):
    value = accepted();job = deepcopy(value['formal_report'])
    ids = acceptance.target_ids();pixels = value['valid_target_pixels'];files = []
    for v in NEW_FINALS:
        directory = root / v / 'synthetic_attempt';directory.mkdir(parents=True)
        trace = directory / 'target_eval.jsonl'
        records = [{'id': identity, 'prediction_shape': [1, 1024, 2048],
                    'valid_pixels': pixels//500 + int(i < pixels % 500)} for i, identity in enumerate(ids)]
        trace.write_text(''.join(json.dumps(r)+'\n' for r in records))
        job['reports'][v]['trace_sha256'] = acceptance.file_sha(trace)
        path = directory / 'metrics.json';path.write_text(json.dumps(job['reports'][v]));files += [path, trace]
    manifest = {'phase': 32, 'study': job['study'], 'git_sha': acceptance.PRODUCER,
                'freeze_sha256': acceptance.verify_target_contract(), 'contract': acceptance.target_contract()}
    for name, value in (('final_comparison.json', job), ('run_manifest.json', manifest)):
        path = root / name;path.write_text(json.dumps(value));files.append(path)
    checksums(root, files)
    return job


def checksums(root, files=None):
    files = files or sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (root / 'SHA256SUMS').write_text(''.join(f'{acceptance.file_sha(p)}  {p.relative_to(root).as_posix()}\n' for p in files))


def test_accepts_all_ten_counts_and_checks_six_target_coverage_traces(tmp_path):
    export_fixture(tmp_path);r = acceptance.accept(tmp_path)
    assert len(r['rows']) == 10 and len(r['files_verified']) == 14
    assert r['valid_target_pixels'] == 917018489
    assert r['target_coverage_traces_checked'] and r['all_ten_confusion_matrices_independently_reduced']
    assert r['checkpoint_or_image_bytes_replayed_locally'] is False
    assert r['additional_training_authorized'] is False
    (tmp_path / 'final_comparison.json').write_text('corrupt')
    with pytest.raises(ValueError, match='SHA256'):acceptance.accept(tmp_path)


@pytest.mark.parametrize('change', ['count', 'class_iou', 'target_score', 'checkpoint', 'geometry',
                                  'coverage', 'updates', 'rng', 'combined_score', 'contrast', 'interaction'])
def test_rejects_wrong_metrics_identity_or_reporting(change):
    job = deepcopy(accepted()['formal_report']);r = job['reports']['UA']
    if change == 'count':r['confusion_matrix'][0][0] = -1
    elif change == 'class_iou':r['class_iou'][0] += .001
    elif change == 'target_score':r['target_miou'] += .001
    elif change == 'checkpoint':r['identity']['checkpoint_sha256'] = '0'*64
    elif change == 'geometry':r['identity']['geometry'] = 'resized'
    elif change == 'coverage':r['sample_count'] = 499
    elif change == 'updates':r['optimizer_updates_this_run'] = 1
    elif change == 'rng':r['model_rng_unchanged'] = False
    elif change == 'combined_score':job['target_miou']['C0'] += .001
    elif change == 'contrast':job['comparisons_percentage_points']['target']['UA-U0'] += .1
    elif change == 'interaction':job['interactions_percentage_points']['target']['(UA-U0)-(CA-C0)'] *= -1
    with pytest.raises(ValueError):acceptance.validate_comparison(job)


def test_rejects_wrong_trace_membership_even_after_hashes_are_updated(tmp_path):
    job = export_fixture(tmp_path)
    trace = tmp_path / 'UA/synthetic_attempt/target_eval.jsonl'
    rows = [json.loads(line) for line in trace.read_text().splitlines()];rows[0]['id'] = rows[1]['id']
    trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    job['reports']['UA']['trace_sha256'] = acceptance.file_sha(trace)
    (tmp_path / 'UA/synthetic_attempt/metrics.json').write_text(json.dumps(job['reports']['UA']))
    (tmp_path / 'final_comparison.json').write_text(json.dumps(job));checksums(tmp_path)
    with pytest.raises(ValueError, match='trace ID/geometry/coverage'):acceptance.accept(tmp_path)


def test_report_keeps_all_models_class_scores_and_correct_interaction_direction():
    rows, contrasts, interactions, classes = result_tables(accepted())
    assert len(rows) == 10 and len(classes) == 190 and len(contrasts) == 7 and len(interactions) == 2
    ca = next(r for r in rows if r['variant'] == 'CA')
    assert ca['target_miou_pct'] == pytest.approx(66.32195195146393)
    assert ca['target_delta_vs_same_parameterization_pp'] == pytest.approx(.6621776775067589)
    ua = next(r for r in contrasts if r['comparison'] == 'UA-U0')
    assert ua['target_delta_pp'] == pytest.approx(.866175796583657)
    assert ua['source_delta_pp'] < 0
    assert interactions[0]['target_delta_pp'] == pytest.approx(.20399811907689802)
    assert interactions[1]['target_delta_pp'] == pytest.approx(.3844584426178965)
