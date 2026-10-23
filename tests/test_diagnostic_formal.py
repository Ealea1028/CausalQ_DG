"""Synthetic tests of the Phase29 objective, initialization, recovery and handoff."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
import yaml

from causalq.interventions import StyleInterventionBank
from causalq.losses.causal_query_effect import causal_query_effect_loss
from causalq.protocols.diagnostic_20261020 import STUDY, diagnostic_config, NEW_UNCENTERED
from causalq.protocols.diagnostic_formal import training_config, validate_training_config, verify_release
from causalq.utils.checkpoint import load_training_checkpoint
from tools.deterministic_source_ce import deterministic_source_ce
from tools.full_cqe_source_update import full_cqe_source_update
from tools.prediction_source_update import prediction_source_update
from tools.selective_cqe_source_update import selective_cqe_source_update, selective_effect
from tools.strict_source_resume import capture_rng, restore_rng
from tools.strict_source_runner import train_segment
from tools.strict_source_smoke import tree_digest
from tools import train_formal_source as formal
from tools import package_diagnostic_formal as pack
from test_strict_source_runner import make_state

ROOT = Path(__file__).resolve().parents[1]


def test_phase29_recipes_freeze_only_authorized_differences():
    assert len(verify_release()) == 64
    for variant in NEW_UNCENTERED:
        config = yaml.safe_load((ROOT / f'configs/diagnostic_20261020/formal/{variant.lower()}_seed0.yaml').read_bytes())
        validate_training_config(config, variant)
        assert config['query']['centered_residuals'] is False
        assert config['data']['targets'] == []
        original = diagnostic_config(variant)
        config['experiment'] = original['experiment']
        config['protocol']['training_authorized'] = False
        assert config == original
        changed = training_config(variant)
        changed['optimizer']['lr'] *= 2
        with pytest.raises(ValueError, match='drift'):
            validate_training_config(changed, variant)
    for variant in ('U0', 'CA', 'CR', 'M3'):
        with pytest.raises(ValueError, match='Only Phase29'):
            training_config(variant)


@pytest.mark.parametrize('variant', ['UA', 'UR'])
@pytest.mark.parametrize('alpha', [0., .1])
def test_selective_update_matches_explicit_gradient_two_forwards(variant, alpha):
    model, optimizer, dataset, _ = make_state()
    with torch.no_grad():model.query_head.alpha.fill_(alpha)
    config = training_config(variant)
    sample = dataset[0]
    image = sample['image'].unsqueeze(0)
    label = sample['label'].unsqueeze(0)
    label[:, 0, 0] = 255
    photo = StyleInterventionBank().photometric_view(image)
    reference = deepcopy(model)
    reference_opt = torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    rng = capture_rng()
    calls = []
    hook = model.backbone.register_forward_hook(lambda *a: calls.append(True))
    record = selective_cqe_source_update(model, optimizer, image, photo, label, 1, config, device='cpu', amp=False)
    hook.remove()
    restore_rng(rng)
    original = reference.forward_components(image, include_anchor=True, component_protocol=STUDY)
    styled = reference.forward_components(photo, include_anchor=True, component_protocol=STUDY)
    # Spell out Full-Anchor independently of the update's effect selector.
    def effect(output):
        anchor = output.anchor_scaled_delta_logits.float()
        return anchor if variant == 'UA' else output.scaled_delta_logits.float() - anchor
    cqe = causal_query_effect_loss(effect(styled), effect(original).detach(), label)
    objective = deterministic_source_ce(original.logits, label) + deterministic_source_ce(styled.logits, label) + cqe
    objective.backward()
    torch.nn.utils.clip_grad_norm_([p for p in reference.parameters() if p.requires_grad], 1.)
    reference_opt.step()
    assert len(calls) == 2
    assert record['objective'] == pytest.approx(float(objective.detach()))
    key = 'loss_anchor_cqe' if variant == 'UA' else 'loss_residual_cqe'
    assert record[key] == pytest.approx(float(cqe.detach()))
    assert record[key] == 0 if alpha == 0 else record[key] > 0
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-5)
    assert not model.query_bank.centered_residuals
    assert model.get_query_residuals().mean(1).norm() > 0
    assert all(p.grad is None for p in model.backbone.parameters())


def test_residual_style_anchor_is_not_detached():
    model, _, dataset, _ = make_state()
    with torch.no_grad():model.query_head.alpha.fill_(.1)
    output = model.forward_components(dataset[0]['image'].unsqueeze(0), include_anchor=True, component_protocol=STUDY)
    loss = selective_effect(model, output, 'residual').square().sum()
    a, full = torch.autograd.grad(loss, (output.anchor_scaled_delta_logits, output.scaled_delta_logits))
    torch.testing.assert_close(a, -full)
    assert a.norm() > 0


@pytest.mark.parametrize('variant', NEW_UNCENTERED)
def test_phase29_resume_matches_continuous_stochastic_training(tmp_path, variant):
    update = {'UP': prediction_source_update, 'UF': full_cqe_source_update}.get(variant, selective_cqe_source_update)
    for name, start, stop in [('continuous', 0, 8), ('prefix', 0, 4), ('resumed', 4, 8)]:
        output = tmp_path / name
        output.mkdir()
        model, optimizer, dataset, _ = make_state()
        train_segment(model, optimizer, dataset, StyleInterventionBank(), training_config(variant), output,
                      start=start, stop=stop, metadata={'seed': 0, 'variant': variant},
                      resume=tmp_path / 'prefix/state.pth' if start else None,
                      device='cpu', amp=False, update_fn=update)
    full = load_training_checkpoint(tmp_path / 'continuous/state.pth')
    resumed = load_training_checkpoint(tmp_path / 'resumed/state.pth')
    for key in ('trainable_model', 'optimizer', 'sampler', 'rng'):
        assert tree_digest(full[key]) == tree_digest(resumed[key])
    assert (tmp_path / 'continuous/train.jsonl').read_bytes() == (
        (tmp_path / 'prefix/train.jsonl').read_bytes() + (tmp_path / 'resumed/train.jsonl').read_bytes())


@pytest.mark.parametrize('variant', NEW_UNCENTERED)
def test_phase29_local_entry_refuses_real_data_without_cuda(tmp_path, monkeypatch, variant):
    checkout = tmp_path / 'checkout'
    relative = f'configs/diagnostic_20261020/formal/{variant.lower()}_seed0.yaml'
    destination = checkout / relative
    destination.parent.mkdir(parents=True)
    destination.write_bytes((ROOT / relative).read_bytes())
    monkeypatch.setattr(formal, 'ROOT', checkout)
    monkeypatch.setattr(formal, '_git', lambda *args: '' if args[0] == 'status' else 'a'*40)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(formal, 'build_source_datasets', lambda *a: pytest.fail('Real data accessed locally'))
    monkeypatch.setattr(formal.DINOv3Backbone, 'from_pretrained', lambda *a, **k: pytest.fail('Real weights loaded locally'))
    monkeypatch.delitem(sys.modules, 'tools.train', raising=False)
    monkeypatch.delitem(sys.modules, 'causalq.datasets.cityscapes', raising=False)
    with pytest.raises(RuntimeError, match='AutoDL CUDA'):
        formal.main(['--variant', variant, '--output-root', str(tmp_path / 'out')])


def test_phase29_import_excludes_legacy_and_target_modules():
    script = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname in ('tools.train','causalq.datasets.cityscapes'):raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import tools.train_formal_source,tools.selective_cqe_source_update,tools.package_diagnostic_formal
"""
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_phase29_launcher_parent_survives_errors():
    bash = 'D:/Git/Git/bin/bash.exe' if Path('D:/Git/Git/bin/bash.exe').exists() else shutil.which('bash')
    if not bash:pytest.skip('Bash unavailable')
    script = (ROOT / 'scripts/run_diagnostic_20261020.sh').as_posix()
    result = subprocess.run([bash, '-n', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for command in ['source "$1"; echo parent_alive:$?', '"$2" "$1" invalid train-uncentered; echo parent_alive:$?']:
        result = subprocess.run([bash, '-c', command, 'test', script, bash], capture_output=True, text=True)
        assert result.returncode == 0 and 'parent_alive:2' in result.stdout


def test_phase29_package_contains_only_reports_and_preserves_manifest(tmp_path, monkeypatch):
    import zipfile
    monkeypatch.setattr(pack, '_git', lambda *args: 'a'*40)
    expected_initial = pack.matched_baseline()['initial_head_sha256']
    for variant in NEW_UNCENTERED:
        directory = tmp_path / variant
        directory.mkdir()
        config = ROOT / f'configs/diagnostic_20261020/formal/{variant.lower()}_seed0.yaml'
        (directory / 'config.yaml').write_bytes(config.read_bytes())
        metadata = {'git_sha': 'a'*40, 'variant': variant, 'centered_residuals': False,
                    'phase29_contract_sha256': verify_release(), 'config_sha256': pack.file_sha(config)}
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        (directory / 'initialization.json').write_text(json.dumps({'head_sha256': expected_initial}))
        (directory / 'formal_report.json').write_text(json.dumps({
            'phase': 29, 'initial_head_sha256': expected_initial, 'source_miou': .5,
            'source_evaluation': {'training_state_unchanged': True}}))
        (directory / 'state.pth').write_bytes(b'local synthetic checkpoint excluded')
    monkeypatch.setattr(pack, 'validate_final_report', lambda root, metadata:
                        json.loads((root / 'formal_report.json').read_bytes()))
    pack.initialize(tmp_path)
    first = (tmp_path / 'run_manifest.json').read_bytes()
    report = pack.package(tmp_path)
    assert (tmp_path / 'run_manifest.json').read_bytes() == first
    assert report['total_optimizer_updates'] == 160000
    assert set(report['source_miou']) == set(NEW_UNCENTERED)
    assert report['target_data_accessed'] is False
    with zipfile.ZipFile(tmp_path / 'phase29_export.zip') as archive:
        names = archive.namelist()
        assert not any(name.endswith(('.pth', '.jsonl')) for name in names)
        assert 'run_manifest.json' in names and 'phase29_report.json' in names
    metadata = json.loads((tmp_path / 'UF/metadata.json').read_bytes())
    metadata['centered_residuals'] = True
    (tmp_path / 'UF/metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='identity'):
        pack.package(tmp_path)
