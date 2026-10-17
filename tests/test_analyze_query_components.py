import json
from pathlib import Path
import sys

import pytest
import torch

from causalq.interventions import StyleInterventionBank
from tools import analyze_query_components as analysis
from tools.strict_source_resume import capture_rng
from tools.strict_source_smoke import tree_digest
from test_anchor_query_effect import make_model

ROOT = Path(__file__).resolve().parents[1]


def test_pair_analysis_replay_two_forwards_and_unchanged_state_rng():
    model = make_model()
    with torch.no_grad(): model.query_head.alpha.fill_(.2)
    image = torch.randn(1, 3, 8, 8)
    label = torch.randint(0, 4, (1, 8, 8))
    style = StyleInterventionBank()
    before = tree_digest((model.state_dict(), capture_rng()))
    rr, er, summary = analysis.analyze_pair(model, image, label, style, 1016)
    assert model.backbone.calls == 2
    assert len(rr) == 4 * 2 * 2 and len(er) == 4 * 3
    assert summary['appearance_replay_exact'] and summary['counterfactual_routing_exact']
    assert before == tree_digest((model.state_dict(), capture_rng()))
    repeated = analysis.analyze_pair(model, image, label, style, 1016)
    assert repeated == (rr, er, summary)
    with pytest.raises(ValueError): analysis.analyze_pair(model.train(), image, label, style, 1016)


def test_csv_missing_values_and_summary_denominators(tmp_path):
    rows = [{'class_id': 0, 'value': None}, {'class_id': 1, 'value': 2.}]
    analysis.csv_write(tmp_path / 'result.csv', rows)
    assert '0,' in (tmp_path / 'result.csv').read_text()
    assert analysis.mean_present(rows, 'value') == {'mean': 2., 'defined_count': 1}
    assert analysis.mean_present(rows[:1], 'value') == {'mean': None, 'defined_count': 0}


def test_cli_audits_remote_evidence_before_cuda_or_dataset(tmp_path, monkeypatch):
    checkout = tmp_path / 'checkout'
    for directory, names in [('STRICT_DG_PHASE19', ('anchor_report.json', 'anchor_SHA256SUMS')),
                             ('STRICT_DG_PHASE18', ('centered_report.json', 'centered_SHA256SUMS'))]:
        destination = checkout / 'experiments' / directory
        destination.mkdir(parents=True)
        for name in names:
            (destination / name).write_bytes((ROOT / 'experiments' / directory / name).read_bytes())
    monkeypatch.setattr(analysis, 'ROOT', checkout)
    monkeypatch.setattr(analysis.gate, '_git', lambda *a: '' if a[0] == 'status' else 'a' * 40)
    monkeypatch.delitem(sys.modules, 'tools.train', raising=False)
    monkeypatch.delitem(sys.modules, 'causalq.datasets.cityscapes', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA queried before remote audit'))
    monkeypatch.setattr(analysis.gate, 'build_source_datasets', lambda *a: pytest.fail('Real source accessed locally'))
    output = tmp_path / 'out'
    assert analysis.main(['--data-root', str(tmp_path / 'data'), '--pretrained-root', str(tmp_path / 'weights'),
        '--checkpoint', str(tmp_path / 'centered/state.pth'), '--source-manifest', str(ROOT / 'manifests/gta5_source_val.txt'),
        '--anchor-dir', str(tmp_path / 'missing'), '--output-dir', str(output)]) == 1
    report = json.loads((output / 'component_effect_report.json').read_bytes())
    assert report['anchor_audit']['anchor_components_accepted']
    assert 'FileNotFoundError' in report['error'] and report['optimizer_updates_this_run'] == 0
