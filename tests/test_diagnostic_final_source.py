"""Phase31 registry, restoration and source-only export tests with synthetic data."""

from copy import deepcopy
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest
import torch

from causalq.protocols.diagnostic_20261020 import ROOT, STUDY, style_seeds
from causalq.protocols.diagnostic_final_source import (
    ACCEPTANCE, NEW_FINALS, final_records, final_source_report, final_config, final_contract, verify_final_contract,
)
from causalq.utils.checkpoint import trainable_state_dict
from tools import run_diagnostic_20261020 as shared
from tools import run_final_source_diagnostics as final
from tools.accept_diagnostic_formal import accept
from tools.strict_source_smoke import tree_digest
from test_diagnostic_20261020 import model


def test_six_final_registry_preserves_exact_recipes_and_baseline_reuse():
    assert NEW_FINALS == ('UP', 'UF', 'UA', 'UR', 'CA', 'CR')
    assert len(verify_final_contract()) == 64
    frozen = final_contract()
    assert frozen['optimizer_updates_this_phase'] == 0 and frozen['target_data_access_authorized'] is False
    assert frozen['baseline_diagnostics_reused']['repeat_inference'] is False
    assert frozen['intervention_examples']['export_again'] is False
    for v, record in final_records().items():
        report = final_source_report(v)
        assert record['checkpoint_sha256'] == report['checkpoint_sha256']
        assert record['head_sha256'] == report['final_head_sha256']
        assert record['checkpoint_relative_to_output_root'].endswith('/'+v+'/'+report['checkpoint'])
        assert record['config_sha256'] == final.file_sha(ROOT / f'configs/diagnostic_20261020/formal/{v.lower()}_seed0.yaml')
        assert final_config(v)['query']['centered_residuals'] == v.startswith('C')
    with pytest.raises(ValueError, match='reuse old'):final_config('U0')
    assert len(style_seeds('18734')) == 4


def make_export(root):
    accepted = json.loads((ACCEPTANCE / 'phase30_acceptance.json').read_bytes())
    job = accepted['formal_report']
    manifest = json.loads((ACCEPTANCE / 'phase30_freeze.json').read_bytes())
    manifest = {'phase': 30, 'git_sha': job['git_sha'], 'phase30_contract_sha256': job['phase30_contract_sha256'],
                'contract': manifest}
    (root / 'run_manifest.json').write_text(json.dumps(manifest))
    (root / 'phase30_report.json').write_text(json.dumps(job))
    for v, r in job['reports'].items():
        directory = root / v;directory.mkdir()
        (directory / 'formal_report.json').write_text(json.dumps(r))
        (directory / 'metadata.json').write_text(json.dumps(r['metadata']))
        (directory / 'initialization.json').write_text(json.dumps({'head_sha256': r['initial_head_sha256']}))
        (directory / 'config.yaml').write_bytes((ROOT / f'configs/diagnostic_20261020/formal/{v.lower()}_seed0.yaml').read_bytes())
    return job


def checksum(root):
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (root / 'SHA256SUMS').write_text(''.join(f'{final.file_sha(p)}  {p.relative_to(root).as_posix()}\n' for p in files))


def test_formal_acceptance_recomputes_counts_and_rejects_corruption(tmp_path):
    job = make_export(tmp_path);checksum(tmp_path)
    accepted = accept(tmp_path, 30)
    assert accepted['independent_source_miou']['CA'] == pytest.approx(.7383646467188167)
    assert accepted['checkpoints_and_full_traces_available_locally'] is False
    path = tmp_path / 'CA/formal_report.json'
    report = deepcopy(job['reports']['CA']);report['source_evaluation']['confusion_matrix'][0][0] += 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='SHA256'):accept(tmp_path, 30)
    job['reports']['CA'] = report
    (tmp_path / 'phase30_report.json').write_text(json.dumps(job));checksum(tmp_path)
    with pytest.raises(ValueError, match='coverage/confusion'):accept(tmp_path, 30)


@pytest.mark.parametrize('variant', ['UP', 'CA'])
def test_shared_loader_uses_frozen_centering_flag_not_baseline_name(tmp_path, monkeypatch, variant):
    m = model(variant.startswith('C'));backbone = m.backbone
    monkeypatch.setattr(backbone, 'to', lambda *a, **k: backbone)
    monkeypatch.setattr(m, 'to', lambda *a, **k: m)
    captured = {}
    def constructor(*args, **kwargs):
        captured.update(kwargs)
        return m
    monkeypatch.setattr(shared, 'QuerySegmentor', constructor)
    monkeypatch.setattr(shared.DINOv3Backbone, 'from_pretrained', lambda *a, **k: backbone)
    monkeypatch.setattr(shared.torch.cuda, 'get_device_name', lambda *a: 'synthetic')
    monkeypatch.setattr(shared, 'version', lambda *a: 'synthetic')
    environment = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                   'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                   'transformers': 'synthetic', 'gpu': 'synthetic'}
    meta = {'environment': environment}
    payload = {'schema_version': 2, 'iteration': 40000, 'metadata': meta, 'trainable_model': trainable_state_dict(m)}
    monkeypatch.setattr(shared, 'load_training_checkpoint', lambda *a: payload)
    monkeypatch.setattr(shared, 'file_sha', lambda p: 'synthetic-sha')
    frozen = {'checkpoint_relative_to_output_root': 'synthetic/state.pth', 'checkpoint_sha256': 'synthetic-sha',
              'head_sha256': tree_digest(trainable_state_dict(m))}
    config = final_config(variant);config['model']['weights_sha256'] = 'synthetic-sha'
    args = SimpleNamespace(training_output_root=tmp_path, pretrained_root=tmp_path)
    result, _, actual_env = shared.load_model(args, variant, frozen=frozen, source={'metadata': meta}, config=config)
    assert result is m and actual_env == environment
    assert captured['centered_residuals'] == variant.startswith('C')


def test_final_entry_refuses_local_real_data_or_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(final, 'ROOT', tmp_path / 'checkout')
    relative = 'configs/diagnostic_20261020/formal/up_seed0.yaml'
    destination = final.ROOT / relative
    destination.parent.mkdir(parents=True)
    destination.write_bytes((ROOT / relative).read_bytes())
    monkeypatch.setattr(final, '_git', lambda *a: '' if a[0] == 'status' else 'a'*40)
    monkeypatch.setattr(shared, '_git', lambda *a: 'a'*40)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(shared, 'build_source_datasets', lambda *a: pytest.fail('Real data locally'))
    monkeypatch.setattr(shared, 'load_model', lambda *a, **k: pytest.fail('Real weights locally'))
    with pytest.raises(RuntimeError, match='Phase31 requires AutoDL CUDA'):
        final.main(['--variant', 'UP', '--output-root', str(tmp_path / 'out')])


def test_final_package_is_reports_only_and_rejects_wrong_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(final, '_git', lambda *a: 'a'*40)
    records = final_records()
    for v in NEW_FINALS:
        directory = tmp_path / v;directory.mkdir()
        summary = {'ok': True, 'sample_count': 2497, 'views': 5, 'optimizer_updates_this_run': 0,
                   'model_rng_unchanged': True, 'target_data_accessed': False, 'checkpoint_training_updates': 40000,
                   'training_producer_git_sha': records[v]['producer_git_sha'],
                   'effect_geometry': 'longedge1024_aspect_pad16', 'gradient_geometry': 'deterministic_512_crop',
                   'image_macro_statistics': [], 'pixel_weighted_responsibility': [],
                   'identity': {'study': STUDY, 'phase': 31, 'variant': v, 'git_sha': 'a'*40,
                                'contract_sha256': verify_final_contract(),
                                'checkpoint_sha256': records[v]['checkpoint_sha256'],
                                'original_config_sha256': records[v]['config_sha256'],
                                'diagnostic_config_sha256': records[v]['config_sha256']}}
        (directory / 'summary.json').write_text(json.dumps(summary))
        (directory / 'class_summary.csv').write_text('synthetic-summary')
        (directory / 'state.pth').write_bytes(b'excluded')
        (directory / 'samples').mkdir();(directory / 'samples/synthetic.json.gz').write_bytes(b'excluded')
    result = final.package(tmp_path)
    assert set(result['reports']) == set(NEW_FINALS)
    with zipfile.ZipFile(tmp_path / 'phase31_export.zip') as archive:
        assert len(archive.namelist()) == 15
        assert not any('samples/' in p or p.endswith('.pth') or 'intervention_examples/' in p for p in archive.namelist())
    summary['identity']['checkpoint_sha256'] = 'b'*64
    (tmp_path / 'CR/summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='matching final'):final.package(tmp_path)


def test_final_manifest_rejects_commit_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(final, '_git', lambda *a: 'a'*40)
    final.initialize(tmp_path)
    before = (tmp_path / 'run_manifest.json').read_bytes()
    final.initialize(tmp_path)
    assert before == (tmp_path / 'run_manifest.json').read_bytes()
    monkeypatch.setattr(final, '_git', lambda *a: 'b'*40)
    with pytest.raises(ValueError, match='manifest differs'):final.initialize(tmp_path)


def test_final_source_launcher_has_no_training_or_target_command_and_parent_survives():
    script = ROOT / 'scripts/run_diagnostic_20261020.sh'
    text = script.read_text()
    block = text.split('if [[ "$PHASE" == 31 ]]; then', 1)[1].split('\nfi', 1)[0]
    assert 'initialize UP UF UA UR CA CR package' in block
    assert 'train_formal_source' not in block and 'target-evaluation' not in block
    bash = 'D:/Git/Git/bin/bash.exe' if Path('D:/Git/Git/bin/bash.exe').is_file() else shutil.which('bash')
    if not bash:pytest.skip('Bash unavailable')
    check = subprocess.run([bash, '-n', script.as_posix()], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    check = subprocess.run([bash, '-c', '"$1" "$2" invalid final-source-diagnostics; echo parent_alive:$?',
                            'test', bash, script.as_posix()], capture_output=True, text=True)
    assert check.returncode == 0 and 'parent_alive:2' in check.stdout
