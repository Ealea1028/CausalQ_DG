"""Phase31 acceptance and Phase32 final target release use metadata/synthetic data only."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest
import torch

from causalq.protocols.diagnostic_20261020 import ROOT, STUDY
from causalq.protocols.diagnostic_final_source import ACCEPTANCE, NEW_FINALS, final_contract, verify_final_contract
from causalq.protocols.diagnostic_target import target_contract, verify_target_contract
from tools import accept_final_source_diagnostics as acceptor
from tools import evaluate_diagnostic_target as target
from tools.analyze_final_source_diagnostics import comparison_rows


def export_source(root, change=None):
    accepted = json.loads((ACCEPTANCE / 'phase31_acceptance.json').read_bytes())
    reports = deepcopy(accepted['reports']);compact = {};files = []
    for v, r in reports.items():
        r.pop('summary_sha256')
        if change and v == 'UA':change(r)
        directory = root / v;directory.mkdir()
        path = directory / 'summary.json';path.write_text(json.dumps(r));files.append(path)
        path = directory / 'class_summary.csv';path.write_text('synthetic summary');files.append(path)
        compact[v] = {k: val for k, val in r.items() if k not in ('image_macro_statistics', 'pixel_weighted_responsibility')}
    manifest = {'phase': 31, 'study': STUDY, 'git_sha': acceptor.PRODUCER,
                'contract_sha256': verify_final_contract(), 'contract': final_contract()}
    job = {**manifest, 'ok': True, 'reports': compact, 'target_data_accessed': False, 'optimizer_updates_this_run': 0}
    for name, value in (('job_report.json', job), ('run_manifest.json', manifest)):
        path = root / name;path.write_text(json.dumps(value));files.append(path)
    (root / 'SHA256SUMS').write_text(''.join(f'{target.file_sha(p)}  {p.relative_to(root).as_posix()}\n' for p in files))


def test_phase31_acceptance_checks_scope_and_preserves_remote_limit(tmp_path):
    export_source(tmp_path);result = acceptor.accept(tmp_path)
    assert len(result['files_verified']) == 14 and result['accepted'] is True
    assert result['checkpoint_bytes_available_locally'] is False
    assert result['raw_per_image_records_available_locally'] is False
    (tmp_path / 'UA/summary.json').write_text('corrupt')
    with pytest.raises(ValueError, match='SHA256'):acceptor.accept(tmp_path)


@pytest.mark.parametrize('field,value', [('sample_count', 2496), ('model_rng_unchanged', False),
                                        ('target_data_accessed', True), ('optimizer_updates_this_run', 1)])
def test_phase31_rejects_wrong_coverage_or_updates(tmp_path, field, value):
    export_source(tmp_path, lambda r: r.update({field: value}))
    with pytest.raises(ValueError, match='coverage/state'):acceptor.accept(tmp_path)


def test_target_freeze_reports_every_model_and_preserves_parameterization():
    freeze = target_contract();assert len(verify_target_contract()) == 64
    assert set(freeze['new_models']) == set(NEW_FINALS) and len(freeze['reused_models']) == 4
    assert freeze['total_new_target_images'] == 3000 and freeze['optimizer_updates_this_phase'] == 0
    assert freeze['additional_training_authorized'] is False and freeze['target_results_may_change_recipe'] is False
    assert freeze['new_models']['UP']['consistency_mechanism'] == 'prediction'
    assert all(r['centered_residuals'] == v.startswith('C') for v, r in freeze['new_models'].items())
    assert all(r['evaluate_again'] is False for r in freeze['reused_models'].values())
    assert ['UA', 'UR'] in freeze['primary_comparisons']
    assert freeze['interaction_comparisons'] == ['(UA-U0)-(CA-C0)', '(UR-U0)-(CR-C0)']


def test_target_run_refuses_local_real_data_and_weights(tmp_path, monkeypatch):
    args = SimpleNamespace(output_root=tmp_path, variant='UA')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(target, 'FinalCityscapes', lambda *a: pytest.fail('Real target data locally'))
    monkeypatch.setattr(target, 'load_model', lambda *a, **k: pytest.fail('Real weights locally'))
    with pytest.raises(RuntimeError, match='Phase32 requires AutoDL CUDA'):target.run_variant(args)


def test_target_package_reuses_four_old_results_without_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(target, '_git', lambda *a: 'a'*40)
    target.initialize(tmp_path)
    old = json.loads((ROOT / 'experiments/STRICT_DG_PHASE26/final_comparison.json').read_bytes())
    freeze = target_contract()
    for v in NEW_FINALS:
        directory = tmp_path / v / 'synthetic_attempt';directory.mkdir(parents=True)
        trace = directory / 'target_eval.jsonl';trace.write_text('synthetic coverage trace\n')
        r = deepcopy(old['reports']['M0'])
        r.update(phase=32, study=STUDY, variant=v, identity=target.identity(v),
                 source_miou=freeze['new_models'][v]['source_miou'], trace_sha256=target.file_sha(trace))
        (directory / 'metrics.json').write_text(json.dumps(r))
    result = target.package(tmp_path)
    assert len(result['target_miou']) == 10 and len(result['reports']) == 6
    assert result['target_miou']['C0'] == old['target_miou']['M1']
    assert result['comparisons_percentage_points']['target']['UA-UR'] == 0
    assert result['interactions_percentage_points']['target']['(UA-U0)-(CA-C0)'] == pytest.approx(
        100*(old['target_miou']['M1']-old['target_miou']['M0']))
    with zipfile.ZipFile(tmp_path / 'phase32_export.zip') as z:
        assert len(z.namelist()) == 15
        assert not any(n.endswith('.pth') or n.startswith('U0/') for n in z.namelist())
    (tmp_path / 'UA/duplicate').mkdir()
    (tmp_path / 'UA/duplicate/metrics.json').write_bytes((tmp_path / 'UA/synthetic_attempt/metrics.json').read_bytes())
    with pytest.raises(ValueError, match='Multiple committed'):target.completed(tmp_path, 'UA')


def test_target_manifest_rejects_new_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(target, '_git', lambda *a: 'a'*40);target.initialize(tmp_path)
    monkeypatch.setattr(target, '_git', lambda *a: 'b'*40)
    with pytest.raises(ValueError, match='manifest differs'):target.initialize(tmp_path)


def test_source_plot_table_preserves_raw_normalized_and_pair_definitions():
    accepted = [json.loads((ACCEPTANCE / f'phase{p}_acceptance.json').read_bytes()) for p in (28, 31)]
    reports = {v: r for a in accepted for v, r in a['reports'].items()}
    rows = comparison_rows(reports)
    assert len(rows) == 30
    c0 = next(r for r in rows if r['variant'] == 'C0' and r['component'] == 'anchor')
    assert c0['pair_raw_variance'] == pytest.approx(.000938070722692208)
    assert c0['raw_variance'] == pytest.approx(.0019897126404543634)
    assert c0['rms_normalized_variance'] == pytest.approx(.003398684360963031)
    assert c0['raw_variance_definition'] == 'five-view_population_S5'


def test_target_import_has_no_legacy_trainer_or_optimizer_construction():
    code = '''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*a):
  if fullname=='tools.train':raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import tools.evaluate_diagnostic_target
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_target_launcher_uses_six_finals_and_keeps_terminal_open():
    script = ROOT / 'scripts/run_diagnostic_20261020.sh'
    block = script.read_text().split('if [[ "$PHASE" == 32 ]]; then', 1)[1].split('\nfi', 1)[0]
    assert 'initialize UP UF UA UR CA CR package' in block and 'train_formal_source' not in block
    bash = 'D:/Git/Git/bin/bash.exe' if Path('D:/Git/Git/bin/bash.exe').is_file() else shutil.which('bash')
    if not bash:pytest.skip('Bash unavailable')
    result = subprocess.run([bash, '-n', script.as_posix()], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([bash, '-c', '"$1" "$2" invalid target-evaluation; echo parent_alive:$?',
                             'test', bash, script.as_posix()], capture_output=True, text=True)
    assert result.returncode == 0 and 'parent_alive:2' in result.stdout
