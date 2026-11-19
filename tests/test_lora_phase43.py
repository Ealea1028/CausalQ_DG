from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from causalq.runtime.identity import file_sha
from causalq.runtime.lora_l1_formal import DIRECTORY, VARIANTS
from causalq.runtime.lora_phase43 import (CONFIG, POLICY, NAME, PHASE, check_runtime,
                                         completion, load_contract, verify_references)
from causalq.runtime.lora_training import learning_rate, load_state_file
from test_lora_l1_formal import build
from test_lora_phase40 import packet
from tools.resume_lora_phase42_vgpu48 import baseline_record, load_baseline
from tools.train_lora_core_d3_l1 import hints, identity_for
from tools.train_lora_phase43_parallel import execution_plan, validate_execution


def test_phase43_only_changes_backbone_and_keeps_hpi_causalq_and_fresh_lora_scope():
    config, freeze = load_contract()
    assert config['phase'] == 43 and config['backbones'] == ['dinov2_vitl14']
    assert config['variants'] == list(VARIANTS) and config['formal_steps']*len(VARIANTS) == 160000
    assert config['lora']['rank'] == 4 and config['lora']['parameters'] == 98304
    assert config['physical_source_batch'] == 4 and config['effective_source_batch'] == 8
    assert config['target_access_authorized'] is False and freeze['fresh_initialization'] is True
    assert completion()['accepted'] and completion()['formal_optimizer_updates_total'] == 160000


def test_runtime_uses_completed_phase42_environment_without_relaxing_software():
    environment = completion()['runtime']; check_runtime(environment)
    for key in ('gpu', 'driver', 'python', 'torch', 'pip_freeze', 'visible_memory_bytes'):
        with pytest.raises(ValueError, match='Phase43 runtime changed'):
            check_runtime(dict(environment, **{key: None}))


def test_d2_native_checkpoint_is_fresh_and_cannot_be_mistaken_for_a_d3_run(tmp_path):
    trainer = build('CA', kind='d2')
    assert trainer.step == 0 and not trainer.optimizer.state
    args = SimpleNamespace(variant='CA', expected_sha='a'*40, data_root=tmp_path, pretrained_root=tmp_path)
    identity = identity_for(trainer, args, {'test': 'CPU synthetic'}, phase=43, backbone_name=NAME)
    assert identity['phase'] == 43 and identity['backbone'] == NAME
    assert identity['config_sha256'] == file_sha(DIRECTORY.parents[1]/CONFIG)
    assert 'continuation_from' not in identity
    trainer.advance(packet()); trainer.commit(tmp_path/'CA/checkpoints', identity)
    rows = hints(tmp_path, args.expected_sha, phase=43, backbone=NAME)
    assert rows == {'U0': 0, 'UA': 0, 'C0': 0, 'CA': 1}
    with pytest.raises(ValueError): hints(tmp_path, args.expected_sha)
    resumed = build('CA', kind='d2')
    assert resumed.restore(tmp_path/'CA/checkpoints', identity) == 1
    assert resumed.initial_head_sha256 == trainer.initial_head_sha256
    resumed.advance(packet(8)); resumed.commit(tmp_path/'CA/checkpoints', identity)
    state = load_state_file(tmp_path/'CA/checkpoints/00000002/state.pt')
    assert state['source_cursor'] == 16 and state['next_lr'] == learning_rate(2)
    assert all(int(slot['step']) == 2 for slot in state['optimizer']['state'].values())


@pytest.mark.parametrize('corruption', ['payload', 'marker'])
def test_d3_reference_check_is_read_only_and_rejects_changed_finals(tmp_path, corruption):
    accepted = {'producer': 'd3-producer', 'models': {}}
    files = []
    for variant in VARIANTS:
        folder = tmp_path/variant/'checkpoints/00040000'; folder.mkdir(parents=True)
        payload = folder/'state.pt'; payload.write_bytes(b'synthetic-d3-reference-'+variant.encode())
        identity = {'phase': 42, 'backbone': 'dinov3_vitl16', 'variant': variant,
                    'training_producer_git_sha': 'd3-producer'}
        marker = folder/'complete.json'
        marker.write_text(json.dumps({'step': 40000, 'sha256': file_sha(payload), 'identity': identity}))
        accepted['models'][variant] = {'sha256': file_sha(payload), 'marker_sha256': file_sha(marker),
            'identity_sha256': hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
        files.extend([marker, payload])
    before = {p: p.read_bytes() for p in files}
    verify_references(tmp_path, accepted=accepted)
    assert before == {p: p.read_bytes() for p in files}
    folder = tmp_path/'CA/checkpoints/00040000'
    target = folder/('state.pt' if corruption == 'payload' else 'complete.json')
    target.write_bytes(b'changed' if corruption == 'payload' else b'{"step":40000,"identity":{}}')
    with pytest.raises((ValueError, KeyError)):
        verify_references(tmp_path, accepted=accepted)


def test_d2_baseline_has_its_own_namespace_and_rejects_the_d3_measurement(tmp_path):
    args = SimpleNamespace(expected_sha='a'*40, phase=PHASE, backbone_name=NAME, policy_path=POLICY)
    result = {'status': 'baseline_complete', 'windows': [{'calibration_valid': True,
        'aggregate_source_images_per_second': 20.}], 'models': {'UA': {'status': 'interrupted',
        'base_unchanged': True, 'tainted_update': False, 'committed_step': 320, 'restored_step': 0,
        'runtime': completion()['runtime'], 'checkpoint': {'step': 320}}}}
    record = baseline_record(args, result, True)
    assert record['phase'] == 43 and record['backbone'] == NAME and record['initial_step'] == 0
    path = tmp_path/'baseline.json'; path.write_text(json.dumps(record))
    assert load_baseline(path, args, True)['source_images_per_second'] == 20.
    record.update(phase=42, backbone='dinov3_vitl16'); path.write_text(json.dumps(record))
    with pytest.raises(ValueError): load_baseline(path, args, True)


def test_phase43_execution_rejects_foreign_backbone_or_donor_migration():
    base = json.loads((DIRECTORY/'execution_profile.json').read_text())
    plan = execution_plan(4, {'cpu_budget': 12, 'available_host_bytes': 90*(1 << 30),
                             'dev_shm_free_bytes': 45*(1 << 30)}, base, True)
    validate_execution(plan)
    assert plan['jobs'] == 4 and plan['total_workers'] == 8 and plan['config_path'] == CONFIG
    for key, value in [('phase', 42), ('backbone', 'dinov3_vitl16'), ('original_donor_migration', True)]:
        wrong = deepcopy(plan); wrong[key] = value
        with pytest.raises(ValueError): validate_execution(wrong)


def test_parallel_pool_dispatches_d2_workers_and_counts_only_phase43_commits(tmp_path, monkeypatch):
    import tools.train_lora_phase42_parallel as cli
    args = SimpleNamespace(output=tmp_path, expected_sha='a'*40, phase=43, backbone_name=NAME,
                           worker_module='tools.train_lora_phase43_parallel', data_root=tmp_path,
                           pretrained_root=tmp_path, lock_fds=(1, 2, 3))
    monkeypatch.setattr(cli, 'verified_final', lambda *a: None)
    def progress(*a, **kw):
        assert kw == {'phase': 43, 'backbone': NAME}
        return dict.fromkeys(VARIANTS, 40000)
    monkeypatch.setattr(cli, 'hints', progress)
    monkeypatch.setattr(cli.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(cli, 'Telemetry', lambda *a: nullcontext(SimpleNamespace(started=0.)))
    monkeypatch.setattr(cli.time, 'sleep', lambda *a: None)
    children = []
    class Child:
        returncode = 0
        def __init__(self, variant): self.variant = variant; self.pid = 100+len(children)
        def poll(self):
            assert len(children) == 4
            (tmp_path/self.variant/'training_report.json').write_text(json.dumps({'ok': True, 'status': 'complete'}))
            return 0
    def launch(command, **kw):
        assert command[3] == 'tools.train_lora_phase43_parallel' and kw['pass_fds'] == (1, 2, 3)
        child = Child(command[command.index('--variant')+1]); children.append(child); return child
    monkeypatch.setattr(cli.subprocess, 'Popen', launch)
    leg = tmp_path/'leg'; leg.mkdir()
    result = cli.pool(args, {'jobs': 4}, {}, leg, {'single_ua_source_images_per_second': 20.})
    assert result['status'] == 'complete' and result['committed_steps'] == dict.fromkeys(VARIANTS, 40000)
