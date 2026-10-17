from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from causalq.interventions import StyleInterventionBank
from causalq.protocols.formal_source import ROOT, formal_config, validate_formal_config
from causalq.utils.checkpoint import load_training_checkpoint
from tools import train_formal_source as formal
from tools.strict_source_runner import train_segment
from tools.strict_source_smoke import tree_digest
from test_strict_source_runner import make_state


def test_formal_recipes_only_change_centering_and_identity():
    configs = []
    for variant in ('M0', 'M1'):
        config = yaml.safe_load((ROOT / f'configs/strict_dg/formal/{variant.lower()}_seed0.yaml').read_bytes())
        validate_formal_config(config, variant)
        assert config['protocol']['training_authorized'] is True
        assert config['validation']['interval'] == config['train']['max_iterations'] == 40000
        assert config['data']['targets'] == []
        config.pop('experiment')
        config['query'].pop('centered_residuals')
        configs.append(config)
    assert configs[0] == configs[1]
    changed = formal_config('M1')
    changed['style']['lambda_cf'] = 0
    with pytest.raises(ValueError, match='drift'):
        validate_formal_config(changed, 'M1')
    # Historical evidence contracts are not rewritten as authorized training.
    assert yaml.safe_load((ROOT / 'configs/strict_dg/gta_dinov3l_m0.yaml').read_bytes())['protocol']['training_authorized'] is False


@pytest.mark.parametrize('centered', [False, True])
def test_recovery_ignores_incomplete_attempt_and_replays_exact_state(tmp_path, monkeypatch, centered):
    root = tmp_path / 'formal'
    root.mkdir()
    metadata = {'seed': 0, 'centered_residuals': centered}
    model, optimizer, dataset, config = make_state()
    model.query_bank.centered_residuals = centered
    full = tmp_path / 'continuous'
    full.mkdir()
    train_segment(model, optimizer, dataset, StyleInterventionBank(), config, full,
        start=0, stop=8, metadata=metadata, device='cpu', amp=False)
    actual = formal.train_segment
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(kwargs['start'])
        if kwargs['start'] == 4:
            (args[5] / 'train.jsonl').write_text('partial run preserved\n')
            raise RuntimeError('simulated GPU interruption')
        return actual(*args, **kwargs)

    monkeypatch.setattr(formal, 'train_segment', interrupted)
    model, optimizer, dataset, config = make_state()
    model.query_bank.centered_residuals = centered
    with pytest.raises(RuntimeError, match='interruption'):
        formal.run_segments(model, optimizer, dataset, StyleInterventionBank(), config, root, metadata,
            horizon=8, interval=4, device='cpu', amp=False)
    assert calls == [0, 4]
    assert len(formal.completed_segments(root, metadata, horizon=8, interval=4)) == 1
    monkeypatch.setattr(formal, 'train_segment', actual)
    model, optimizer, dataset, config = make_state()
    model.query_bank.centered_residuals = centered
    segments = formal.run_segments(model, optimizer, dataset, StyleInterventionBank(), config, root, metadata,
        horizon=8, interval=4, device='cpu', amp=False)
    expected = load_training_checkpoint(full / 'state.pth')
    resumed = load_training_checkpoint(segments[-1][1] / 'state.pth')
    for key in ('trainable_model', 'optimizer', 'rng', 'sampler'):
        assert tree_digest(expected[key]) == tree_digest(resumed[key])
    assert (full / 'train.jsonl').read_text() == ''.join((p / 'train.jsonl').read_text() for _, p in segments)
    # No optimizer work is repeated once all checkpoints are committed.
    monkeypatch.setattr(formal, 'train_segment', lambda *a, **k: pytest.fail('Completed segment repeated'))
    formal.run_segments(model, optimizer, dataset, StyleInterventionBank(), config, root, metadata,
        horizon=8, interval=4, device='cpu', amp=False)
    changed = deepcopy(metadata)
    changed['seed'] = 1
    with pytest.raises(ValueError, match='metadata'):
        formal.completed_segments(root, changed, horizon=8, interval=4)
    (segments[0][1] / 'train.jsonl').write_text('tampered')
    with pytest.raises(ValueError, match='hash'):
        formal.completed_segments(root, metadata, horizon=8, interval=4)


def test_formal_cli_never_loads_real_data_locally(tmp_path, monkeypatch):
    checkout = tmp_path / 'checkout'
    destination = checkout / 'configs/strict_dg/formal/m0_seed0.yaml'
    destination.parent.mkdir(parents=True)
    destination.write_bytes((ROOT / 'configs/strict_dg/formal/m0_seed0.yaml').read_bytes())
    monkeypatch.setattr(formal, 'ROOT', checkout)
    monkeypatch.setattr(formal, '_git', lambda *a: '' if a[0] == 'status' else 'a' * 40)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(formal, 'build_source_datasets', lambda *a: pytest.fail('Real data attempted locally'))
    monkeypatch.setattr(formal.DINOv3Backbone, 'from_pretrained', lambda *a, **k: pytest.fail('Real weights attempted locally'))
    monkeypatch.delitem(sys.modules, 'tools.train', raising=False)
    monkeypatch.delitem(sys.modules, 'causalq.datasets.cityscapes', raising=False)
    with pytest.raises(RuntimeError, match='AutoDL CUDA'):
        formal.main(['--variant', 'M0', '--output-root', str(tmp_path / 'out')])


def test_formal_source_import_blocks_target_and_legacy():
    script = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname in ('tools.train','causalq.datasets.cityscapes'):
            raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import tools.train_formal_source
"""
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
