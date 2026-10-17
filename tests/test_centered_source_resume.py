import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from causalq.interventions import StyleInterventionBank
from tools import check_centered_source_resume as gate
from tools.strict_source_runner import train_segment
from test_strict_source_runner import make_state

ROOT = Path(__file__).resolve().parents[1]


def test_centered_schema2_runner_replays_crop_style_dropout_rng_across_epoch(tmp_path):
    metadata = {'seed': 0, 'centered_residuals': True, 'schedule_horizon': 40000}
    for index, (mode, (start, stop)) in enumerate(gate.BOUNDS.items()):
        output = tmp_path / mode
        output.mkdir()
        model, optimizer, dataset, config = make_state()
        model.query_bank.centered_residuals = True
        gate.require_centered_model(model, metadata)
        result = train_segment(model, optimizer, dataset, StyleInterventionBank(), config, output,
            start=start, stop=stop, metadata=metadata, resume=tmp_path / 'prefix/state.pth' if start else None,
            device='cpu', amp=False)
        worker = {'ok': True, 'git_sha': 'a' * 40, 'git_status': '', 'mode': mode, 'process_id': index + 100,
            'formal_training_authorized': False, 'target_data_accessed': False,
            'centered_residuals': True, 'checkpoint_schema_version': 2, 'metadata': metadata, **result,
            'files_sha256': {name: gate.file_sha(output / name) for name in ('train.jsonl', 'state.pth')}}
        (output / 'worker_report.json').write_text(json.dumps(worker))
    # Synthetic worker IDs exercise verifier logic; actual independent CUDA
    # processes remain the AutoDL requirement. Dataset length5 crosses epoch.
    result = gate.verify_workers(tmp_path, 'a' * 40)
    assert result['exact_resume_verified'] and result['checkpoint_schema_version'] == 2
    assert result['total_optimizer_updates_across_processes'] == 16
    changed = json.loads((tmp_path / 'resume/worker_report.json').read_text())
    changed['process_id'] = 100
    (tmp_path / 'resume/worker_report.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='independent'): gate.verify_workers(tmp_path, 'a' * 40)
    changed['process_id'] = 102
    (tmp_path / 'resume/worker_report.json').write_text(json.dumps(changed))
    path = tmp_path / 'resume/train.jsonl'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='artifact hash'): gate.verify_workers(tmp_path, 'a' * 40)


def test_semantic_guard_rejects_legacy_mode_or_changed_horizon():
    model, _, _, _ = make_state()
    metadata = {'centered_residuals': True, 'schedule_horizon': 40000}
    with pytest.raises(ValueError, match='centered'): gate.require_centered_model(model, metadata)
    model.query_bank.centered_residuals = True
    with pytest.raises(ValueError, match='40k'): gate.require_centered_model(model, {**metadata, 'schedule_horizon': 8})


def test_cli_checks_completed_diagnostics_before_real_model_or_data(tmp_path, monkeypatch):
    checkout = tmp_path / 'checkout'
    destination = checkout / 'configs/strict_dg/gta_dinov3l_m1_centered.yaml'
    destination.parent.mkdir(parents=True)
    destination.write_bytes((ROOT / 'configs/strict_dg/gta_dinov3l_m1_centered.yaml').read_bytes())
    monkeypatch.setattr(gate, 'ROOT', checkout)
    monkeypatch.setattr(gate, '_git', lambda *a: '' if a[0] == 'status' else 'a' * 40)
    monkeypatch.setattr(gate, 'audit_pilot', lambda *a: {'fixture_prior_audit': True})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(gate, 'build_source_datasets', lambda *a: pytest.fail('Real source attempted locally'))
    monkeypatch.setattr(gate.DINOv3Backbone, 'from_pretrained', lambda *a, **k: pytest.fail('Real weights attempted locally'))
    monkeypatch.delitem(sys.modules, 'tools.train', raising=False)
    monkeypatch.delitem(sys.modules, 'causalq.datasets.cityscapes', raising=False)
    output = tmp_path / 'out'
    assert gate.main(['--mode', 'continuous', '--data-root', str(tmp_path / 'data'),
        '--pretrained-root', str(tmp_path / 'weights'), '--output-root', str(output)]) == 1
    report = json.loads((output / 'continuous/worker_report.json').read_bytes())
    assert report['diagnostics_audit']['priority3_engineering_complete']
    assert 'requires AutoDL CUDA' in report['error'] and report['formal_training_authorized'] is False


def test_fresh_import_has_no_legacy_or_target_module():
    script = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname in ('tools.train','causalq.datasets.cityscapes'):
            raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import tools.check_centered_source_resume
"""
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
