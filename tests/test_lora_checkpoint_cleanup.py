import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from causalq.runtime.identity import file_sha, write_json
from causalq.runtime.lora_l1_formal import checkpoint_rows
from causalq.runtime.lora_training import learning_rate, restore_checkpoint, save_checkpoint
from tools import apply_lora_checkpoint_cleanup as cleanup


@pytest.fixture
def history(tmp_path, monkeypatch):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate(0))
    identity = {'backbone':'synthetic', 'variant':'U0'}
    rows = []
    for step in (1, 2, 3, 4):
        optimizer.zero_grad()
        optimizer.param_groups[0]['lr'] = learning_rate(step-1)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        folder = tmp_path/f'{step:08d}'
        save_checkpoint(folder, model, optimizer, step, identity)
        payload = folder/'state.pt'
        stat = payload.stat()
        rows.append({
            'path':str(payload), 'step':step, 'decision':'KEEP' if step > 2 else 'CANDIDATE',
            'bytes':stat.st_size, 'allocated_bytes':getattr(stat, 'st_blocks', (stat.st_size+511)//512)*512,
            'inode':stat.st_ino, 'device':stat.st_dev, 'payload_sha256':file_sha(payload),
            'marker_sha256':file_sha(folder/'complete.json'),
        })
    if os.name == 'nt':
        # NTFS stat does not expose the POSIX st_blocks field used on AutoDL.
        stat_method = type(tmp_path).stat

        def stat_posix(path, **kwargs):
            stat = stat_method(path, **kwargs)
            fields = {k:getattr(stat, k) for k in dir(stat) if k.startswith('st_')}
            return SimpleNamespace(**fields, st_blocks=(stat.st_size+511)//512)

        monkeypatch.setattr(type(tmp_path), 'stat', stat_posix)
    return rows, identity, model, optimizer


def test_retirement_preserves_markers_and_normal_recovery(tmp_path, history):
    rows, identity, model, optimizer = history
    for row in rows[:2]:
        assert cleanup.inspect_step(tmp_path, row, identity)[0] == 'live'
        cleanup.retire_step(tmp_path, row, 'live')
        assert cleanup.inspect_step(tmp_path, row, identity)[0] == 'retired'
    with pytest.raises(ValueError, match='protected checkpoint'):
        cleanup.retire_step(tmp_path, rows[2], 'live')
    recovered = checkpoint_rows(tmp_path, identity)
    assert [step for step, _, _ in recovered] == [4, 3]
    fresh_model = nn.Linear(2, 1)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters())
    assert restore_checkpoint(recovered[0][1], fresh_model, fresh_optimizer, identity) == 4
    assert torch.equal(fresh_model.weight, model.weight)
    assert fresh_optimizer.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']


def test_retry_finishes_after_marker_move(tmp_path, history, monkeypatch):
    rows, identity, _, _ = history
    row = rows[0]
    payload = tmp_path/'00000001'/'state.pt'
    unlink = type(payload).unlink

    def fail_once(path, *args, **kwargs):
        if path == payload:
            raise OSError('interrupted unlink')
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(type(payload), 'unlink', fail_once)
        with pytest.raises(OSError, match='interrupted'):
            cleanup.retire_step(tmp_path, row, 'live')
    assert cleanup.inspect_step(tmp_path, row, identity)[0] == 'pending'
    cleanup.retire_step(tmp_path, row, 'pending')
    assert cleanup.inspect_step(tmp_path, row, identity)[0] == 'retired'


def test_changed_or_outside_payload_stops_before_action(tmp_path, history):
    rows, identity, _, _ = history
    row = rows[0]
    keep = tmp_path/'00000004'/'state.pt'
    with keep.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Payload changed'):
        cleanup.inspect_step(tmp_path, rows[-1], identity)
    with pytest.raises(ValueError, match='outside authorized root'):
        cleanup.inspect_step(tmp_path, dict(row, path=str(tmp_path.parent/'state.pt')), identity)
    assert (tmp_path/'00000001'/'state.pt').exists()


def test_retained_snapshot_checks_optimizer_and_cursor(tmp_path, history):
    _, identity, _, _ = history
    folder = tmp_path/'00000004'
    marker = cleanup.read(folder/'complete.json')
    assert marker['identity'] == identity
    state = cleanup.load_state_file(folder/'state.pt')
    state['rng']['cuda'] = [torch.zeros(4, dtype=torch.uint8)]
    payload = tmp_path/'retained.pt'
    torch.save(state, payload)
    cleanup.verify_snapshot(payload, marker, lora=False)
    state['source_cursor'] -= 8
    torch.save(state, payload)
    with pytest.raises(ValueError, match='Invalid recovery cursor'):
        cleanup.verify_snapshot(payload, marker, lora=False)


def test_catalog_and_resume_after_a_failed_apply(tmp_path, history, monkeypatch):
    rows, identity, _, _ = history
    run_root = tmp_path/'run'
    root = run_root/'U0'/'checkpoints'
    root.mkdir(parents=True)
    for row in rows:
        state = Path(row['path'])
        destination = root/state.parent.name
        state.parent.rename(destination)
        row['path'] = str(destination/state.name)
    audit = tmp_path/'disk_audits'
    audit.mkdir()
    for lock in (tmp_path/'gpu0.lock', run_root/'run.lock'):
        lock.touch()
    monkeypatch.setattr(cleanup, 'verify_final', lambda pin, base: (Path(pin['path']), identity))
    monkeypatch.setattr(cleanup, 'verify_snapshot', lambda *a, **k: None)
    finals = {'D3_U0_L1':dict(path=rows[-1]['path'], lora=True, sha256=rows[-1]['payload_sha256'],
                             marker_sha256=rows[-1]['marker_sha256'])}
    release = {'verified_report_sha256':'a'*64, 'policy':'test'}
    catalog = audit/'cleanup_manifest_aaaaaaaaaaaa.json'
    groups = {'D3_U0_L1':rows}
    lock_module = SimpleNamespace(LOCK_EX=1, LOCK_NB=2)
    monkeypatch.setitem(sys.modules, 'fcntl', lock_module)

    def blocked(fd, operation):
        raise BlockingIOError('run busy')

    lock_module.flock = blocked
    with pytest.raises(BlockingIOError, match='run busy'):
        cleanup.run(tmp_path, audit, audit/'busy.json', release, groups, finals, 'test', apply=True)
    assert not catalog.exists() and Path(rows[0]['path']).is_file()
    lock_module.flock = lambda *a: None
    original_retire = cleanup.retire_step

    def interrupted(base, row, kind):
        assert catalog.is_file()
        if row['step'] == 2:
            raise OSError('disk interrupted')
        original_retire(base, row, kind)

    monkeypatch.setattr(cleanup, 'retire_step', interrupted)
    with pytest.raises(OSError, match='disk interrupted'):
        cleanup.run(tmp_path, audit, audit/'failed.json', release, groups, finals, 'test', apply=True)
    assert cleanup.read(audit/'failed.json')['deleted'] == [rows[0]['path']]
    before = file_sha(catalog)
    monkeypatch.setattr(cleanup, 'retire_step', original_retire)
    result = cleanup.run(tmp_path, audit, audit/'done.json', release, groups, finals, 'test', apply=True)
    assert result['ok'] and result['candidates_present_after'] == 0
    assert result['already_retired'] == [rows[0]['path']]
    assert result['deleted'] == [rows[1]['path']]
    assert file_sha(catalog) == before
