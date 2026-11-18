from copy import deepcopy
from contextlib import contextmanager
import itertools
import json
from types import SimpleNamespace

import pytest

from causalq.runtime.cohort_runner import capture_rng
from causalq.runtime.identity import file_sha, tree_digest
from causalq.runtime import lora_migration as migration
from causalq.runtime.lora_l1_formal import DIRECTORY, ROOT, VARIANTS
from causalq.runtime.lora_training import compact_state, load_state_file, rng_digest
from test_lora_l1_formal import build
from test_lora_phase40 import packet
from tools.resume_lora_phase42_vgpu48 import (baseline_record, contract, execution_plan,
                                              load_baseline, validate_execution)
from tools.train_lora_phase42_parallel import measured_window


def test_release_keeps_the_exact_scientific_recipe_and_only_accepts_original_donors():
    policy = contract()
    original = json.loads((ROOT/'configs/lora_causalq_1114/phase42_parallel.json').read_text())
    current = json.loads((ROOT/'configs/lora_causalq_1114/phase42_vgpu48.json').read_text())
    changed = {k for k in original if original[k] != current[k]}
    assert changed == {'execution', 'performance_policy', 'initialization'}
    assert policy['preserved_updates']+policy['remaining_updates_at_release'] == 160000
    audit, recovery = migration.accepted()
    assert recovery['accepted'] and recovery['engineering_states_eligible_for_formal'] is False
    assert len(audit['donors']) == 4


def test_formal_runtime_requires_the_exact_new_gpu_software_and_driver():
    _, recovery = migration.accepted()
    actual = recovery['runtime']
    migration.check_runtime(actual)
    for key in ('gpu', 'visible_memory_bytes', 'driver', 'torch', 'pip_freeze'):
        bad = dict(actual, **{key: None})
        with pytest.raises(ValueError, match='runtime changed'): migration.check_runtime(bad)


def test_progress_counts_donors_once_without_counting_engineering_updates():
    audit, _ = migration.accepted()
    initial = migration.effective_progress(dict.fromkeys(VARIANTS, 0))
    assert sum(initial.values()) == 3238
    current = dict(initial, UA=1000)
    assert sum(migration.effective_progress(current).values())-sum(initial.values()) == 247
    assert migration.effective_progress(dict.fromkeys(VARIANTS, 40000)) == dict.fromkeys(VARIANTS, 40000)
    with pytest.raises(ValueError, match='regressed'):
        migration.effective_progress(dict(initial, CA=audit['donors']['CA']['step']-1))


def test_native_checkpoint_migration_changes_only_provenance_and_resumes_without_reset(tmp_path, monkeypatch):
    donor = build(); donor.advance(packet())
    source = tmp_path/'original'; old_root = source/'UA/checkpoints'
    old_identity = {'producer': 'original', 'initial_head_sha256': donor.initial_head_sha256,
                    'initial_adapter_sha256': donor.initial_adapter_sha256}
    marker = donor.commit(old_root, old_identity)
    folder = old_root/'00000001'; payload = folder/'state.pt'
    row = {'step': 1, 'identity': old_identity, 'expected_sha256': file_sha(payload), 'bytes': payload.stat().st_size}
    audit = {'source_root': str(source), 'source_producer': 'original', 'donors': {'UA': row}}
    monkeypatch.setattr(migration, 'accepted', lambda: (audit, {'gate_producer': 'synthetic-gate'}))
    identity = {'producer': 'new', 'continuation_from': migration.provenance('UA')}
    before = {p: p.read_bytes() for p in folder.iterdir()}
    destination = tmp_path/'new/UA/checkpoints'
    restored = build()
    step, receipt = migration.restore_or_migrate(restored, destination, identity, 'UA')
    assert step == 1 and receipt['optimizer_updates'] == 0
    assert all(receipt[k] for k in ('model_equal', 'optimizer_equal', 'rng_equal'))
    original = load_state_file(payload)
    assert tree_digest(compact_state(restored.model)) == tree_digest(original['model'])
    assert tree_digest(restored.optimizer.state_dict()) == tree_digest(original['optimizer'])
    assert rng_digest(capture_rng()) == rng_digest(original['rng'])
    restored.commit(destination, identity)
    migrated = load_state_file(destination/'00000001/state.pt')
    assert tree_digest({k: v for k, v in migrated.items() if k != 'identity'}) == tree_digest(
        {k: v for k, v in original.items() if k != 'identity'})
    restored.advance(packet(8)); restored.commit(destination, identity)
    resumed = build()
    step, receipt = migration.restore_or_migrate(resumed, destination, identity, 'UA')
    assert step == 2 and receipt['action'] == 'same_sha_restore'
    assert before == {p: p.read_bytes() for p in before}
    bad = deepcopy(identity); bad['continuation_from']['engineering_states_used'] = True
    with pytest.raises(ValueError, match='provenance'):
        migration.restore_or_migrate(build(), tmp_path/'rejected', bad, 'UA')
    assert not (tmp_path/'rejected').exists()


def test_twelve_cpu_host_gets_four_independent_jobs_with_bounded_loaders():
    base = json.loads((DIRECTORY/'execution_profile.json').read_text())
    machine = {'cpu_budget': 12, 'available_host_bytes': 96*(1 << 30), 'dev_shm_free_bytes': 45*(1 << 30)}
    plan = execution_plan(4, machine, base, True)
    validate_execution(plan)
    assert plan['jobs'] == 4 and plan['workers_per_job'] == 2 and plan['total_workers'] == 8
    bad = deepcopy(plan); bad['profile']['effective_source_batch'] = 16
    with pytest.raises(ValueError): validate_execution(bad)


def test_same_gpu_calibration_is_reused_and_parallel_speed_uses_its_value(tmp_path):
    _, recovery = migration.accepted()
    args = SimpleNamespace(expected_sha='a'*40)
    result = {'status': 'baseline_complete', 'windows': [{'calibration_valid': True,
        'aggregate_source_images_per_second': 16.}], 'models': {'UA': {'status': 'interrupted',
        'base_unchanged': True, 'tainted_update': False, 'committed_step': 1053,
        'restored_step': 753, 'runtime': recovery['runtime'], 'checkpoint': {'step': 1053}}}}
    record = baseline_record(args, result, True)
    path = tmp_path/'baseline.json'; path.write_text(json.dumps(record))
    assert load_baseline(path, args, True)['source_images_per_second'] == 16.
    with pytest.raises(ValueError): load_baseline(path, args, False)
    start = {v: {'step': 0, 'data_wait_seconds': 0} for v in VARIANTS}
    end = {v: {'step': 60, 'data_wait_seconds': 1} for v in VARIANTS}
    window = measured_window(end, start, 120., {'samples': 120, 'median': 95, 'p10': 90},
                             .4, record['source_images_per_second']/8, {'minimum_baseline_fraction': .95})
    assert window['target_met'] and window['aggregate_source_images_per_second'] == 16.


def test_calibration_saves_ua_after_one_window_even_when_one_job_leaves_gpu_capacity(tmp_path, monkeypatch):
    import tools.train_lora_phase42_parallel as cli
    args = SimpleNamespace(output=tmp_path, expected_sha='a'*40, data_root=tmp_path,
                           pretrained_root=tmp_path, worker_module='formal48', lock_fds=(7, 8, 9))
    monkeypatch.setattr(cli, 'verified_final', lambda *a: None)
    monkeypatch.setattr(cli, 'hints', lambda *a: dict.fromkeys(VARIANTS, 0))
    monkeypatch.setattr(cli.signal, 'signal', lambda *a: None)
    ticks = itertools.count()
    monkeypatch.setattr(cli.time, 'monotonic', lambda: float(next(ticks)))
    monkeypatch.setattr(cli.time, 'sleep', lambda *a: None)
    @contextmanager
    def monitor(path, _):
        path.write_text('elapsed_seconds,memory_used_mib\n'+''.join(f'{i},100\n' for i in range(2000)))
        yield SimpleNamespace(started=0.)
    monkeypatch.setattr(cli, 'Telemetry', monitor)
    monkeypatch.setattr(cli, 'telemetry', lambda *a, **kw: {'samples': 120, 'median': 45, 'p10': 40})
    steps = itertools.count(753)
    monkeypatch.setattr(cli, 'read_progress', lambda *a: {'step': next(steps), 'ready': True, 'data_wait_seconds': 0})
    class Child:
        pid = 123; returncode = None
        def poll(self): return self.returncode
        def send_signal(self, sig):
            self.returncode = 130
            (tmp_path/'UA/training_report.json').write_text(json.dumps({'ok': False, 'status': 'interrupted'}))
    def launch(command, **kwargs):
        assert command[3] == 'formal48' and command[command.index('--variant')+1] == 'UA'
        assert kwargs['pass_fds'] == (7, 8, 9)
        return Child()
    monkeypatch.setattr(cli.subprocess, 'Popen', launch)
    leg = tmp_path/'leg'; leg.mkdir()
    result = cli.pool(args, {'jobs': 1, 'variants': ['UA'], 'baseline_calibration': True}, {}, leg,
                      {'single_ua_source_images_per_second': 0., 'minimum_baseline_fraction': .95,
                       'visible_memory_bytes': 48*(1 << 30)})
    assert result['status'] == 'baseline_complete' and set(result['models']) == {'UA'}
    assert len(result['windows']) == 1 and result['windows'][0]['calibration_valid']
    assert result['windows'][0]['target_met'] is False
