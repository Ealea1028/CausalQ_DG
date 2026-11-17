"""Cancellation/cleanup uses tiny synthetic files; never real model payloads."""
from copy import deepcopy
import json

import pytest

from tools import retire_lora_l0 as cleanup


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, 'active_l0_processes', lambda: [])
    root = tmp_path/'phase41'; root.mkdir()
    acceptance = {'latest': {}}
    for backbone in ('dinov3_vitl16', 'dinov2_vitl14_hpi_p16'):
        acceptance['latest'][backbone] = {}
        for variant in ('U0', 'UA', 'C0', 'CA'):
            identity = {'backbone': backbone, 'variant': variant, 'lora_enabled': False, 'producer': 'synthetic'}
            for step in (1000, 40000):
                folder = root/backbone/'checkpoints'/variant/f'{step:08d}'; folder.mkdir(parents=True)
                payload = folder/'state.pt'; payload.write_bytes(f'{backbone}-{variant}-{step}'.encode())
                marker = {'identity': identity, 'step': step, 'sha256': cleanup.sha(payload)}
                cleanup.exclusive_json(folder/'complete.json', marker)
            acceptance['latest'][backbone][variant] = {**marker, 'bytes': payload.stat().st_size}
    metadata = root/'training_report.json'; metadata.write_text('{"keep":true}')
    unrelated = root/'other.pth'; unrelated.write_bytes(b'preserve')
    outside = tmp_path/'weights.pth'; outside.write_bytes(b'preserve')
    return root, acceptance, metadata, unrelated, outside


def write_plan(root, acceptance, path):
    plan = cleanup.preview(root, acceptance)
    cleanup.exclusive_json(path, plan)
    return plan, cleanup.sha(path)


def test_preview_keeps_every_byte_and_only_pins_accepted_final(fixture, tmp_path):
    root, accepted, *_ = fixture
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    plan = cleanup.preview(root, accepted)
    assert plan['candidate_count'] == 15 and len(plan['protected']) == 1
    assert plan['protected'][0]['path'] == cleanup.PIN
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert not (root/'L0_RETIRED.json').exists()


def test_apply_exact_payloads_preserves_all_metadata_and_is_retryable(fixture, tmp_path):
    root, accepted, metadata, unrelated, outside = fixture
    plan_file = tmp_path/'preview.json'
    plan, digest = write_plan(root, accepted, plan_file)
    markers = {p: p.read_bytes() for p in root.rglob('complete.json')}
    report = cleanup.apply(root, accepted, plan_file, digest, tmp_path/'result.json')
    assert report['deleted_count'] == 15 and report['remaining_candidate_bytes'] == 0
    assert list(root.rglob('state.pt')) == [root/cleanup.PIN]
    assert (root/'L0_RETIRED.json').is_file()
    assert all(p.read_bytes() == data for p, data in markers.items())
    assert all(p.is_file() for p in (metadata, unrelated, outside, plan_file))
    assert len(list(root.rglob('retired.json'))) == 15
    retry = cleanup.apply(root, accepted, plan_file, digest, tmp_path/'retry.json')
    assert retry['deleted_count'] == 0
    journal = [json.loads(line) for line in (tmp_path/'result.jsonl').read_text().splitlines()]
    assert sum('intent' in line for line in journal) == 15 and journal[-1]['complete'] is True


@pytest.mark.parametrize('fault', ['pin', 'hash', 'identity', 'orphan', 'partial', 'missing', 'step', 'marker'])
def test_changed_or_uncommitted_artifacts_fail_before_any_deletion(fixture, tmp_path, fault):
    root, accepted, *_ = fixture
    plan_file = tmp_path/'preview.json'; plan, digest = write_plan(root, accepted, plan_file)
    candidate = root/plan['candidates'][-1]['path']
    marker = candidate.parent/'complete.json'
    if fault == 'pin': (root/cleanup.PIN).write_bytes(b'corrupt')
    if fault == 'hash': candidate.write_bytes(b'corrupt')
    if fault == 'identity':
        value = json.loads(marker.read_text()); value['identity']['lora_enabled'] = True
        marker.write_text(json.dumps(value))
    if fault == 'marker': marker.write_text(marker.read_text()+'\n')
    if fault == 'orphan': marker.unlink()
    if fault == 'partial': (candidate.parent/'state.pt.partial').write_bytes(b'partial')
    if fault == 'missing': candidate.unlink()
    if fault == 'step':
        value = json.loads(marker.read_text()); value['step'] = 41000; marker.write_text(json.dumps(value))
    present = list(root.rglob('state.pt'))
    with pytest.raises(ValueError): cleanup.apply(root, accepted, plan_file, digest, tmp_path/'result.json')
    assert all(p.exists() for p in present)
    assert not (root/'L0_RETIRED.json').exists()


@pytest.mark.parametrize('fault', ['hash', 'traversal', 'pin', 'duplicate', 'new_checkpoint'])
def test_preview_tampering_or_new_checkpoint_requires_review(fixture, tmp_path, fault):
    root, accepted, *_ = fixture
    plan = cleanup.preview(root, accepted)
    if fault == 'traversal': plan['candidates'][0]['path'] = '../weights.pth'
    if fault == 'pin': plan['candidates'][0] = deepcopy(plan['protected'][0])
    if fault == 'duplicate': plan['candidates'].append(plan['candidates'][0])
    if fault == 'new_checkpoint':
        old = root/plan['candidates'][0]['path']; folder = old.parent.parent/'00002000'; folder.mkdir()
        (folder/'state.pt').write_bytes(b'new')
        marker = deepcopy(plan['candidates'][0]['original_complete_marker'])
        marker.update(step=2000, sha256=cleanup.sha(folder/'state.pt'))
        cleanup.exclusive_json(folder/'complete.json', marker)
    path = tmp_path/'preview.json'; cleanup.exclusive_json(path, plan)
    digest = '0'*64 if fault == 'hash' else cleanup.sha(path)
    with pytest.raises(ValueError): cleanup.apply(root, accepted, path, digest, tmp_path/'result.json')
    assert len(list(root.rglob('state.pt'))) >= 16


def test_active_controller_and_guard_prevent_restarting_cancelled_training(fixture, tmp_path, monkeypatch):
    root, accepted, *_ = fixture
    monkeypatch.setattr(cleanup, 'active_l0_processes', lambda: [123])
    with pytest.raises(ValueError, match='active'): cleanup.preview(root, accepted)
    from tools import resume_lora_baseline as resume
    monkeypatch.setattr(resume, 'RUN', root)
    (root/'L0_RETIRED.json').write_text('{}')
    with pytest.raises(ValueError, match='retired'): resume.preflight(tmp_path/'nonexistent_code')


def test_proc_detection_ignores_unrelated_jobs(tmp_path):
    for pid, command in [(123, 'python -m tools.resume_lora_baseline --output /tmp/session'),
                         (124, 'python -m tools.train_lora_core_l0 --worker'),
                         (125, 'python unrelated.py')]:
        path = tmp_path/str(pid); path.mkdir(); (path/'cmdline').write_bytes(command.replace(' ', '\0').encode())
    assert cleanup.active_l0_processes(tmp_path) == [123, 124]


def test_redirected_and_hardlinked_paths_fail_closed(fixture, tmp_path, monkeypatch):
    root, accepted, *_ = fixture
    pin = root/cleanup.PIN
    original = type(pin).is_symlink
    monkeypatch.setattr(type(pin), 'is_symlink', lambda p: p == pin or original(p))
    with pytest.raises(ValueError): cleanup.preview(root, accepted)
    monkeypatch.undo()
    import os
    try: os.link(pin, tmp_path/'linked.pt')
    except OSError: pytest.skip('Host does not permit hard links')
    with pytest.raises(ValueError, match='Hard-linked'): cleanup.preview(root, accepted)


def test_partial_apply_retries_only_remaining_payloads(fixture, tmp_path, monkeypatch):
    root, accepted, *_ = fixture
    path = tmp_path/'preview.json'; plan, digest = write_plan(root, accepted, path)
    original = type(root).unlink
    failed = root/plan['candidates'][1]['path']
    def interrupted(p, *args, **kwargs):
        if p == failed: raise OSError('simulated disk error')
        return original(p, *args, **kwargs)
    monkeypatch.setattr(type(root), 'unlink', interrupted)
    with pytest.raises(OSError): cleanup.apply(root, accepted, path, digest, tmp_path/'failed.json')
    monkeypatch.setattr(type(root), 'unlink', original)
    result = cleanup.apply(root, accepted, path, digest, tmp_path/'retry.json')
    assert result['deleted_count'] == 14 and result['remaining_candidate_bytes'] == 0


def test_revised_scope_is_fourteen_fresh_l1_models_with_one_l0_reference():
    scope = json.loads((cleanup.ROOT/'experiments/LORA_CAUSALQ_1114/lora_only_scope_20261117.json').read_text())
    assert len(scope['backbones'])*len(scope['variants']) == scope['fresh_l1_models'] == 14
    assert scope['train_lora_states'] == [1]
    assert scope['additional_formal_updates'] == 14*40000
    assert scope['retained_l0_reference']['relative_payload'] == cleanup.PIN
    for phase in ('42', '43', '44'):
        assert scope['phases'][phase]['lora'] == 1


def test_reviewed_deletion_allowlist_matches_exact_cancelled_models_and_steps():
    folder = cleanup.ROOT/'experiments/LORA_CAUSALQ_1114'
    freeze = json.loads((folder/'phase41_retirement_apply_freeze.json').read_text())
    accepted = json.loads((folder/'phase41_retirement_preview_acceptance.json').read_text())
    expected = set()
    for backbone, limit in [('dinov3_vitl16', 40000), ('dinov2_vitl14_hpi_p16', 27000)]:
        for variant in ('U0', 'UA', 'C0', 'CA'):
            for step in range(1000, limit+1, 1000):
                relative = f'{backbone}/checkpoints/{variant}/{step:08d}/state.pt'
                if relative != cleanup.PIN:
                    expected.add(relative)
    rows = freeze['candidates']
    assert len(rows) == len({row['path'] for row in rows}) == freeze['candidate_count'] == 267
    assert {row['path'] for row in rows} == expected
    assert sum(row['bytes'] for row in rows) == freeze['candidate_apparent_bytes'] == 12183196383
    assert len(freeze['protected']) == 1 and freeze['protected'][0]['path'] == cleanup.PIN
    assert freeze['protected'][0]['sha256'] == accepted['protected_final_sha256']
    assert freeze['approved_preview_sha256'] == accepted['preview_sha256']
    assert freeze['policy'] == cleanup.POLICY and freeze['run_root'] == cleanup.RUN.as_posix()
    for row in rows:
        assert row['path'] != cleanup.PIN and row['bytes'] == 45629949
        assert all(len(row[key]) == 64 and set(row[key]) <= set('0123456789abcdef')
                   for key in ('sha256', 'marker_sha256'))
