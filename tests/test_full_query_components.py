from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

from causalq.analysis.component_figures import render_component_figures
from causalq.interventions import StyleInterventionBank
from tools import analyze_query_components as analysis
from tools.strict_source_resume import capture_rng
from tools.strict_source_smoke import tree_digest
from test_anchor_query_effect import make_model

ROOT = Path(__file__).resolve().parents[1]


def test_full_census_replays_original_gate20_inputs_and_metrics():
    report = json.loads((ROOT / 'experiments/STRICT_DG_PHASE20/component_effect_report.json').read_bytes())
    original = report['samples'][0]
    analysis.check_gate20_replay(deepcopy(original), original)
    changed = deepcopy(original)
    changed['photometric_input_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='replay'): analysis.check_gate20_replay(changed, original)
    changed = deepcopy(original)
    changed['effect_summary']['anchor']['original_mean_abs']['mean'] *= 2
    with pytest.raises(ValueError, match='replay'): analysis.check_gate20_replay(changed, original)


def test_descriptive_census_comparison_preserves_negative_results():
    samples = [{'effect_summary': {k: {'cross_view_population_variance': {'mean': v}} for k, v in pair.items()}}
               for pair in ({'anchor': 1., 'residual': 4.}, {'anchor': 5., 'residual': 2.}, {'anchor': 0., 'residual': 0.},
                            {'anchor': None, 'residual': None})]
    result = analysis.census_summary(samples)
    assert result['image_count'] == 4 and result['defined_image_count'] == 3 and result['images_raw_anchor_variance_lower'] == 1
    assert result['images_raw_anchor_variance_equal'] == 1
    assert result['raw_anchor_minus_residual_variance_mean'] == 0


def test_cached_scientific_figures_require_no_extra_backbone_or_rng_changes(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'mpl'))
    model = make_model()
    with torch.no_grad(): model.query_head.alpha.fill_(.2)
    image = torch.randn(1, 3, 8, 8)
    label = torch.zeros(1, 8, 8, dtype=torch.long)
    before = tree_digest((model.state_dict(), capture_rng()))
    rendered = []
    def render(image, photo, labels, outputs, effects):
        rendered.append(render_component_figures(tmp_path / 'source0', 'source0', image, photo, labels,
            outputs, effects, classes=((0, 'road'), (2, 'building'))))
    analysis.analyze_pair(model, image, label, StyleInterventionBank(), 1016, renderer=render)
    assert model.backbone.calls == 2
    assert before == tree_digest((model.state_dict(), capture_rng()))
    assert rendered[0]['gt_class_pixel_counts'] == {'0': 64, '2': 0}
    assert len(rendered[0]['files']) == 3
    from PIL import Image
    for name in rendered[0]['files']:
        with Image.open(tmp_path / 'source0' / name) as figure:
            assert figure.width >= 1000 and figure.height >= 300
            figure.verify()


def test_render_failure_restores_all_rng():
    model = make_model()
    image = torch.randn(1, 3, 8, 8)
    label = torch.zeros(1, 8, 8, dtype=torch.long)
    before = tree_digest(capture_rng())
    def broken(*unused):
        random.random()
        np.random.rand()
        torch.rand(4)
        raise RuntimeError('synthetic render failure')
    with pytest.raises(RuntimeError, match='render failure'):
        analysis.analyze_pair(model, image, label, StyleInterventionBank(), 1016, renderer=broken)
    assert before == tree_digest(capture_rng())


def test_full_source_audits_saved_gate_before_cuda(tmp_path, monkeypatch):
    checkout = tmp_path / 'checkout'
    target = checkout / 'experiments/STRICT_DG_PHASE20'
    target.mkdir(parents=True)
    for name in ('component_effect_report.json', 'component_SHA256SUMS'):
        (target / name).write_bytes((ROOT / 'experiments/STRICT_DG_PHASE20' / name).read_bytes())
    monkeypatch.setattr(analysis, 'ROOT', checkout)
    monkeypatch.setattr(analysis.gate, '_git', lambda *a: '' if a[0] == 'status' else 'a' * 40)
    monkeypatch.delitem(sys.modules, 'tools.train', raising=False)
    monkeypatch.delitem(sys.modules, 'causalq.datasets.cityscapes', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA queried before diagnostic artifact audit'))
    output = tmp_path / 'out'
    assert analysis.main(['--full-source', '--diagnostic-dir', str(tmp_path / 'missing'),
        '--data-root', str(tmp_path / 'data'), '--pretrained-root', str(tmp_path / 'weights'),
        '--checkpoint', str(tmp_path / 'centered/state.pth'), '--source-manifest', str(checkout / 'manifests/gta5_source_val.txt'),
        '--anchor-dir', str(tmp_path / 'anchor'), '--output-dir', str(output)]) == 1
    report = json.loads((output / 'component_effect_report.json').read_bytes())
    assert report['diagnostic_audit']['component_diagnostics_accepted'] and 'FileNotFoundError' in report['error']
