"""Synthetic matching, continuation and export tests for the preplanned CA/CR."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import shutil
import sys
import zipfile

import pytest
import torch
import yaml

from causalq.interventions import StyleInterventionBank
from causalq.losses.causal_query_effect import causal_query_effect_loss
from causalq.protocols.diagnostic_20261020 import ROOT, STUDY
from causalq.protocols.diagnostic_formal import training_config, verify_release
from causalq.protocols.diagnostic_centered_formal import (
    NEW_CENTERED, centered_training_config, validate_centered_training_config, verify_centered_release,
)
from causalq.utils.checkpoint import load_training_checkpoint
from tools.deterministic_source_ce import deterministic_source_ce
from tools.selective_cqe_source_update import selective_cqe_source_update
from tools.strict_source_resume import capture_rng, restore_rng
from tools.strict_source_runner import train_segment
from tools.strict_source_smoke import tree_digest
from tools import train_formal_source as formal, package_diagnostic_formal as pack
from test_strict_source_runner import make_state


def test_centered_controls_change_only_centering_and_experiment_identity():
    assert len(verify_centered_release()) == len(verify_release()) == 64
    for centered, uncentered in [('CA', 'UA'), ('CR', 'UR')]:
        config = yaml.safe_load((ROOT / f'configs/diagnostic_20261020/formal/{centered.lower()}_seed0.yaml').read_bytes())
        validate_centered_training_config(config, centered)
        assert config['query']['centered_residuals'] is True
        config['experiment'] = training_config(uncentered)['experiment']
        config['query']['centered_residuals'] = False
        assert config == training_config(uncentered)
    changed = centered_training_config('CA')
    changed['query']['alpha_init'] = .1
    with pytest.raises(ValueError, match='drift'):validate_centered_training_config(changed, 'CA')
    with pytest.raises(ValueError, match='Only Phase30'):centered_training_config('M3')


@pytest.mark.parametrize('variant', NEW_CENTERED)
def test_centered_selective_objective_and_gradient_match_explicit_formula(variant):
    model, optimizer, dataset, _ = make_state()
    model.query_bank.centered_residuals = True
    with torch.no_grad():model.query_head.alpha.fill_(.1)
    reference = deepcopy(model)
    reference_opt = torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    config = centered_training_config(variant)
    sample = dataset[0]
    image, label = sample['image'].unsqueeze(0), sample['label'].unsqueeze(0)
    label[:, 0, 0] = 255
    photo = StyleInterventionBank().photometric_view(image)
    rng = capture_rng()
    calls = []
    hook = model.backbone.register_forward_hook(lambda *a: calls.append(True))
    record = selective_cqe_source_update(model, optimizer, image, photo, label, 1, config, device='cpu', amp=False)
    hook.remove()
    restore_rng(rng)
    original = reference.forward_components(image, include_anchor=True, component_protocol=STUDY)
    styled = reference.forward_components(photo, include_anchor=True, component_protocol=STUDY)
    def effect(output):
        anchor = output.anchor_scaled_delta_logits.float()
        return anchor if variant == 'CA' else output.scaled_delta_logits.float() - anchor
    cqe = causal_query_effect_loss(effect(styled), effect(original).detach(), label)
    loss = deterministic_source_ce(original.logits, label) + deterministic_source_ce(styled.logits, label) + cqe
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in reference.parameters() if p.requires_grad], 1.)
    reference_opt.step()
    assert len(calls) == 2 and record['objective'] == pytest.approx(float(loss.detach()))
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-5)
    assert model.query_bank.centered_residuals
    torch.testing.assert_close(model.get_centered_residuals().mean(1), torch.zeros_like(model.get_anchors()), atol=1e-8, rtol=0)
    assert all(p.grad is None for p in model.backbone.parameters())


@pytest.mark.parametrize('variant', NEW_CENTERED)
def test_centered_exact_resume_preserves_dropout_style_sampler_and_optimizer(tmp_path, variant):
    for name, start, stop in [('full', 0, 8), ('prefix', 0, 4), ('resumed', 4, 8)]:
        output = tmp_path / name
        output.mkdir()
        model, optimizer, dataset, _ = make_state()
        model.query_bank.centered_residuals = True
        train_segment(model, optimizer, dataset, StyleInterventionBank(), centered_training_config(variant), output,
                      start=start, stop=stop, metadata={'seed': 0, 'variant': variant},
                      resume=tmp_path / 'prefix/state.pth' if start else None,
                      device='cpu', amp=False, update_fn=selective_cqe_source_update)
    full = load_training_checkpoint(tmp_path / 'full/state.pth')
    resumed = load_training_checkpoint(tmp_path / 'resumed/state.pth')
    for key in ('trainable_model', 'optimizer', 'rng', 'sampler'):
        assert tree_digest(full[key]) == tree_digest(resumed[key])
    assert (tmp_path / 'full/train.jsonl').read_bytes() == (
        (tmp_path / 'prefix/train.jsonl').read_bytes() + (tmp_path / 'resumed/train.jsonl').read_bytes())


@pytest.mark.parametrize('variant', NEW_CENTERED)
def test_centered_entry_requires_autodl_before_real_weights_or_data(tmp_path, monkeypatch, variant):
    checkout = tmp_path / 'checkout'
    rel = f'configs/diagnostic_20261020/formal/{variant.lower()}_seed0.yaml'
    path = checkout / rel
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / rel).read_bytes())
    monkeypatch.setattr(formal, 'ROOT', checkout)
    monkeypatch.setattr(formal, '_git', lambda *a: '' if a[0] == 'status' else 'a'*40)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(formal, 'build_source_datasets', lambda *a: pytest.fail('Real source access locally'))
    monkeypatch.setattr(formal.DINOv3Backbone, 'from_pretrained', lambda *a, **k: pytest.fail('Real weights locally'))
    monkeypatch.delitem(sys.modules, 'tools.train', raising=False)
    monkeypatch.delitem(sys.modules, 'causalq.datasets.cityscapes', raising=False)
    with pytest.raises(RuntimeError, match='AutoDL CUDA'):
        formal.main(['--variant', variant, '--output-root', str(tmp_path / 'out')])


def test_centered_package_excludes_states_and_stops_before_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(pack, '_git', lambda *a: 'a'*40)
    for variant in NEW_CENTERED:
        directory = tmp_path / variant
        directory.mkdir()
        config = ROOT / f'configs/diagnostic_20261020/formal/{variant.lower()}_seed0.yaml'
        (directory / 'config.yaml').write_bytes(config.read_bytes())
        metadata = {'git_sha': 'a'*40, 'variant': variant, 'centered_residuals': True,
                    'phase30_contract_sha256': verify_centered_release(), 'config_sha256': pack.file_sha(config)}
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        initial = pack.centered_baseline()['initial_head_sha256']
        (directory / 'initialization.json').write_text(json.dumps({'head_sha256': initial}))
        (directory / 'formal_report.json').write_text(json.dumps({'phase': 30, 'initial_head_sha256': initial,
            'source_miou': .5, 'source_evaluation': {'training_state_unchanged': True}}))
        (directory / 'state.pth').write_bytes(b'synthetic state excluded')
    monkeypatch.setattr(pack, 'validate_final_report', lambda root, metadata:
                        json.loads((root / 'formal_report.json').read_bytes()))
    report = pack.package(tmp_path, phase=30)
    assert report['total_optimizer_updates'] == 80000 and set(report['reports']) == set(NEW_CENTERED)
    assert 'Phase31' in report['next_phase'] and report['target_data_accessed'] is False
    with zipfile.ZipFile(tmp_path / 'phase30_export.zip') as archive:
        assert 'phase30_report.json' in archive.namelist()
        assert not any(x.endswith(('.pth', '.jsonl')) for x in archive.namelist())
    metadata['centered_residuals'] = False
    (tmp_path / 'CR/metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='identity'):pack.package(tmp_path, phase=30)


def test_centered_launcher_failures_keep_parent_terminal_alive():
    bash = 'D:/Git/Git/bin/bash.exe' if Path('D:/Git/Git/bin/bash.exe').is_file() else shutil.which('bash')
    if not bash:pytest.skip('Bash unavailable')
    script = (ROOT / 'scripts/run_diagnostic_20261020.sh').as_posix()
    result = subprocess.run([bash, '-n', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for command in ['source "$1"; echo parent_alive:$?', '"$2" "$1" invalid train-centered-controls; echo parent_alive:$?']:
        result = subprocess.run([bash, '-c', command, 'test', script, bash], capture_output=True, text=True)
        assert result.returncode == 0 and 'parent_alive:2' in result.stdout
