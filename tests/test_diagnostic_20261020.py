"""Synthetic CPU diagnostics; never load real source/target data or DINO weights."""

import gzip
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from causalq.analysis.diagnostic_summary import DiagnosticSummary
from causalq.analysis.intervention_examples import export_examples
from causalq.analysis.query_gradients import query_gradient_rows, parameter_rows
from causalq.analysis.selective_stability import effect_and_correlation_rows, multiview_responsibility_rows
from causalq.interventions import StyleInterventionBank
from causalq.models import QuerySegmentor
from causalq.protocols.diagnostic_20261020 import (STUDY, ROOT, VARIANTS, diagnostic_config, contract,
    baseline_records, preview_ids, style_seeds, stable_seed)
from causalq.protocols.formal_source import formal_config
from tools.run_diagnostic_20261020 import (make_views, diagnose_one, isolated_rng, state_versions,
    validate_cached, write_json, fingerprint)
from tools.strict_source_smoke import tree_digest
from test_strict_source_smoke import TinyFrozenBackbone


def model(centered=False):
    return QuerySegmentor(TinyFrozenBackbone(), decoder_channels=32, num_classes=4, dropout=.1,
        queries_per_class=2, interaction='static', cross_attention_layers=0, num_heads=2,
        alpha_init=.3, centered_residuals=centered).eval()


def sample():
    return {'id': 'synthetic', 'image': torch.randn(3, 8, 8),
            'label': (torch.arange(64).reshape(8, 8) % 4).long()}


def test_protocol_reuses_existing_models_and_does_not_release_training():
    import yaml
    assert list(baseline_records()) == ['U0', 'C0', 'CP', 'CF']
    assert preview_ids() == ['18734', '12488', '08744']
    assert len(style_seeds('18734')) == 4 and len(set(style_seeds('18734'))) == 4
    assert contract()['training_released_this_phase'] is False
    assert json.loads((ROOT / 'experiments/DIAGNOSTIC_20261020/protocol.json').read_bytes()) == contract()
    for v in VARIANTS:
        cfg = diagnostic_config(v)
        assert yaml.safe_load((ROOT / f'configs/diagnostic_20261020/{v.lower()}_seed0.yaml').read_bytes()) == cfg
        assert cfg['query']['centered_residuals'] == v.startswith('C')
        assert cfg['optimizer'] == formal_config('M0')['optimizer']
        assert cfg['style'] == formal_config('M0')['style']
        assert cfg['protocol']['target_access'] == 'forbidden_during_training'


def test_uncentered_component_access_is_explicit_and_preserves_default_guard():
    m = model();x = sample()['image'][None]
    with pytest.raises(ValueError, match='Anchor-only scope'):m.forward_components(x, include_anchor=True)
    with pytest.raises(ValueError, match='Unknown'):m.forward_components(x, include_anchor=True, component_protocol='wrong')
    plain = m(x)
    output = m.forward_components(x, include_anchor=True, component_protocol=STUDY)
    assert torch.equal(plain, output.logits)
    a = m.get_anchor_query_effect(output=output, component_protocol=STUDY)
    r = m.get_residual_contribution(output=output, component_protocol=STUDY)
    torch.testing.assert_close(a + r, m.get_query_effect(output=output))
    with torch.no_grad():m.get_query_residuals().zero_()
    assert torch.count_nonzero(m.get_residual_contribution(x, component_protocol=STUDY)) == 0


def test_uncentered_gauge_shift_preserves_full_queries_but_not_anchor_effect():
    m = model();x = sample()['image'][None]
    q = m.query_bank(1).detach().clone()
    original = m(x);a = m.get_anchor_query_effect(x, component_protocol=STUDY)
    with torch.no_grad():
        shift = torch.randn_like(m.get_anchors()) * .1
        m.get_anchors().add_(shift);m.get_query_residuals().sub_(shift[:, None])
    torch.testing.assert_close(q, m.query_bank(1), atol=2e-8, rtol=1e-5)
    torch.testing.assert_close(original, m(x), atol=2e-5, rtol=1e-5)
    assert not torch.allclose(a, m.get_anchor_query_effect(x, component_protocol=STUDY))


@pytest.mark.parametrize('grayscale', [0., .1, 1.])
def test_traced_photometric_matches_previous_draw_order_and_caller_rng(grayscale):
    style = StyleInterventionBank(grayscale_probability=grayscale);x = sample()['image'][None]
    start = torch.get_rng_state();expected = style.photometric_view(x);after = torch.get_rng_state()
    torch.set_rng_state(start);actual, trace = style.photometric_view_with_trace(x)
    assert torch.equal(expected, actual) and torch.equal(after, torch.get_rng_state())
    assert set(trace) == {'brightness', 'contrast', 'saturation', 'gamma', 'temperature', 'grayscale'}
    start = torch.get_rng_state();views, traces = make_views(x, style, 'synthetic')
    assert torch.equal(start, torch.get_rng_state()) and len(views) == 5 and len(traces) == 4
    assert torch.equal(views[1], make_views(x, style, 'synthetic')[0][1])


def test_amplitude_scaling_normalized_variance_and_covariance_identity():
    x = torch.arange(1, 17).reshape(1, 1, 4, 4).float()
    label = torch.zeros(1, 4, 4, dtype=torch.long);label[0, 0, 0] = 255
    views = [{'anchor': x * s, 'residual': -x * s / 2, 'full': x * s / 2} for s in (1., 2., 3., 4., 5.)]
    effects, correlations = effect_and_correlation_rows(views, label)
    a = next(r for r in effects if r['component'] == 'anchor')
    assert a['raw_variance'] > 0 and a['rms_normalized_variance'] < 1e-25
    assert a['pair_raw_variance'] == pytest.approx(float((x[0, 0][label[0] != 255] ** 2).mean()) / 4)
    joint = correlations[0];assert joint['variance_identity_abs_error'] < 1e-12
    assert correlations[1]['anchor_residual_pearson'] == pytest.approx(-1)
    assert correlations[1]['difference_rms'] == pytest.approx(correlations[1]['full_rms'])
    scaled = [{k: v * .1 for k, v in view.items()} for view in views]
    second = effect_and_correlation_rows(scaled, label)[0][1]
    assert second['raw_variance'] == pytest.approx(a['raw_variance'] * .01, rel=1e-6)


def test_zero_energy_and_constant_correlations_are_null_not_perfect_stability():
    x = torch.zeros(1, 1, 2, 2);label = torch.zeros(1, 2, 2, dtype=torch.long)
    effects, corrs = effect_and_correlation_rows([{k: x for k in ('anchor', 'residual', 'full')}] * 5, label)
    assert all(e['rms_normalized_variance'] is None and e['near_zero_view_count'] == 5 for e in effects)
    assert corrs[1]['anchor_residual_pearson'] is None
    with pytest.raises(ValueError, match='No valid'):
        effect_and_correlation_rows([{'full': x, 'anchor': x, 'residual': x}] * 2, label + 255)


def test_responsibility_ties_missing_classes_and_style_changes():
    scores = torch.zeros(1, 2, 2, 2, 2);photo = scores.clone();photo[:, :, 0] = 3
    label = torch.zeros(1, 4, 4, dtype=torch.long)
    rows = multiview_responsibility_rows([scores, photo], label)
    original = [r for r in rows if r['class_id'] == 0 and r['scope'] == 'gt_class' and r['view_index'] == 0]
    assert sum(r['mean_responsibility'] for r in original) == pytest.approx(1.)
    assert all(r['top1_share'] == .5 and r['effective_query_count'] == pytest.approx(2) for r in original)
    assert any(r['js_divergence'] > 0 for r in rows if r['class_id'] == 0 and r['view_index'] == 1)
    assert all(r['mean_responsibility'] is None for r in rows if r['class_id'] == 1 and r['scope'] == 'gt_class')
    summary = DiagnosticSummary();summary.add_rows('responsibility', rows)
    weighted = summary.result()['pixel_weighted_responsibility']
    assert all(sum(r['mean_responsibility']) == pytest.approx(1) for r in weighted)


@pytest.mark.parametrize('variant', ['U0', 'UP', 'UF', 'UA', 'UR', 'C0', 'CA', 'CR'])
def test_gradient_chain_rules_projections_and_read_only_state(variant):
    m = model(variant.startswith('C'));s = sample();x = s['image'][None];label = s['label'][None]
    before = tree_digest(m.state_dict());rng = torch.get_rng_state();versions = state_versions(m)
    rows = query_gradient_rows(m, x, x * .8 + .1, label, variant)
    assert before == tree_digest(m.state_dict()) and versions == state_versions(m)
    assert torch.equal(rng, torch.get_rng_state()) and all(p.grad is None for p in m.parameters())
    total = [r for r in rows if r['term'] == 'total' and r['class_id'] == 0]
    assert sum(r['projection_via_anchor'] for r in total) == pytest.approx(1, abs=1e-5)
    assert sum(r['projection_total_anchor'] for r in total) + total[0]['direct_projection_total'] == pytest.approx(1, abs=1e-5)
    assert total[0]['chain_rule_max_error'] < 1e-5
    if variant in ('UA', 'UR', 'CA', 'CR'):
        assert any(r['direct_anchor_norm'] > 0 for r in rows if r['term'] == 'consistency')
    else:assert all(r['direct_anchor_norm'] == 0 for r in rows)


def test_alpha_zero_query_gradients_and_projections_are_explicit():
    m = model();m.query_head.alpha.data.zero_();s = sample();x = s['image'][None]
    rows = query_gradient_rows(m, x, x * .9, s['label'][None], 'U0')
    assert all(r['query_gradient_norm'] == 0 and r['projection_via_anchor'] is None for r in rows)


def test_end_to_end_one_synthetic_image_and_examples(tmp_path):
    m = model();style = StyleInterventionBank();s = sample()
    before = tree_digest((m.state_dict(), torch.get_rng_state()))
    record, views, traces = diagnose_one(m, style, s, s, 'synthetic', 'U0')
    assert before == tree_digest((m.state_dict(), torch.get_rng_state()))
    summary = DiagnosticSummary();summary.add_image(record)
    assert summary.result()['sample_count'] == 1 and len(record['effects']) == 12
    export_examples(tmp_path, Image.new('RGB', (8, 8)), views, traces, style_seeds('synthetic'), style, 'synthetic')
    assert (tmp_path / 'contact_sheet.png').exists() and len(list(tmp_path.glob('diagnostic_only_*.png'))) == 6
    assert json.loads((tmp_path / 'interventions.json').read_bytes())['source_id'] == 'synthetic'
    assert parameter_rows(m)[0]['residual_mean_norm'] > 0


def test_summary_pixel_weighting_does_not_average_image_means_equally():
    summary = DiagnosticSummary()
    for n, p in ((1, .9), (9, .1)):
        summary.add_rows('responsibility', [{'class_id': 0, 'scope': 'gt_class', 'view_index': 0,
            'query_r': slot, 'pixels': n, 'present': True, 'mean_responsibility': rho, 'pixel_entropy': .2}
            for slot, rho in enumerate((p, 1-p))])
    result = summary.result()['pixel_weighted_responsibility'][0]
    assert result['mean_responsibility'][0] == pytest.approx(.18)


def test_global_present_class_macro_weights_images_equally_and_keeps_class_nulls():
    summary = DiagnosticSummary()
    summary.add_rows('effects', [{'class_id': 0, 'component': 'anchor', 'present': True, 'mean_abs': 10.},
                                 {'class_id': 1, 'component': 'anchor', 'present': True, 'mean_abs': 20.}])
    summary.add_rows('effects', [{'class_id': 0, 'component': 'anchor', 'present': False, 'mean_abs': 0.},
                                 {'class_id': 1, 'component': 'anchor', 'present': True, 'mean_abs': 20.}])
    global_row = next(r for r in summary.result()['image_macro_statistics']
                      if r['class_id'] == -1 and r['population'] == 'gt_present')
    assert global_row['metrics']['mean_abs']['mean'] == pytest.approx(17.5)
    assert global_row['metrics']['mean_abs']['count'] == 2


def test_correlation_fisher_aggregation_respects_image_macro_class_weighting():
    import math
    summary = DiagnosticSummary()
    summary.add_rows('correlations', [{'class_id': 0, 'value_pearson': .9}, {'class_id': 1, 'value_pearson': .1}])
    summary.add_rows('correlations', [{'class_id': 0, 'value_pearson': .5}])
    row = next(r for r in summary.result()['image_macro_statistics'] if r['class_id'] == -1)
    expected = math.tanh(((math.atanh(.9)+math.atanh(.1))/2 + math.atanh(.5))/2)
    assert row['metrics']['value_pearson']['fisher_mean'] == pytest.approx(expected)


def test_atomic_report_is_strict_json(tmp_path):
    path = tmp_path / 'report.json';write_json(path, {'ok': True})
    assert json.loads(path.read_bytes())['ok'] is True
    with pytest.raises(ValueError):write_json(path, {'bad': float('nan')})


def test_resume_cached_image_contract_and_source_file_changes_rejected(tmp_path):
    image = tmp_path / 'image.png';label = tmp_path / 'label.png'
    image.write_bytes(b'a');label.write_bytes(b'b')
    pair = SimpleNamespace(image=image, label=label)
    record = {'identity': {'sha': 'first'}, 'source_id': 'synthetic', 'source_files': fingerprint(pair)}
    path = tmp_path / 'sample.json.gz'
    with gzip.open(path, 'wt') as stream:json.dump(record, stream)
    assert validate_cached(path, {'sha': 'first'}, 'synthetic', pair) == record
    with pytest.raises(ValueError):validate_cached(path, {'sha': 'second'}, 'synthetic', pair)
    image.write_bytes(b'changed')
    with pytest.raises(ValueError):validate_cached(path, {'sha': 'first'}, 'synthetic', pair)


def test_disk_report_is_read_only_and_excludes_all_four_final_checkpoints(tmp_path, monkeypatch):
    from tools import report_diagnostic_disk as disk
    for frozen in baseline_records().values():
        final = tmp_path / frozen['checkpoint_relative_to_output_root']
        final.parent.mkdir(parents=True, exist_ok=True);final.write_bytes(b'final')
        middle = final.parents[1] / '1000_2000_test' / 'state.pth'
        middle.parent.mkdir();middle.write_bytes(b'middle')
    monkeypatch.setattr(disk.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='123\tpath', stderr=''))
    result = disk.disk_report(tmp_path, tmp_path, tmp_path/'weights', tmp_path/'data')
    assert result['read_only'] and result['deletion_performed'] is False
    assert len(result['protected_final_checkpoints']) == 4
    assert len(result['closed_study_intermediate_candidates']) == 4 and result['candidate_bytes'] == 24
    assert len(list(tmp_path.rglob('state.pth'))) == 8


@pytest.mark.parametrize('command', [
    'bash scripts/run_diagnostic_20261020.sh invalid',
    'source scripts/run_diagnostic_20261020.sh invalid',
    'bash scripts/run_diagnostic_20261020.sh 0000000000000000000000000000000000000000 train-uncentered',
])
def test_launcher_errors_preserve_interactive_parent_and_do_not_launch_training(command):
    bash = 'D:/Git/Git/bin/bash.exe' if Path('D:/Git/Git/bin/bash.exe').is_file() else shutil.which('bash')
    if not bash:pytest.skip('bash is not installed')
    command = command.replace('bash scripts/', '"$BASH" scripts/')
    result = subprocess.run([bash, '-c', command+'; echo parent_alive'],
                            cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    assert result.returncode == 0 and 'parent_alive' in result.stdout
