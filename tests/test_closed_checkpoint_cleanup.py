"""Deletion tests use only tiny synthetic files inside pytest's temporary root."""

from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath

import pytest

from tools import prune_closed_intermediates as cleanup


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, 'active_legacy_trainers', lambda: [])
    base = tmp_path / 'outputs';base.mkdir()
    original = json.loads(cleanup.FREEZE.read_bytes())
    frozen = {'variants': {}};candidates = [];finals = []
    for name, record in original['variants'].items():
        relative = PurePosixPath(record['checkpoint_relative_to_output_root'])
        final = base.joinpath(*relative.parts);final.parent.mkdir(parents=True)
        data = (name+'-final').encode();final.write_bytes(data);finals.append(final)
        frozen['variants'][name] = {'checkpoint_relative_to_output_root': relative.as_posix(),
                                   'checkpoint_sha256': hashlib.sha256(data).hexdigest()}
        rel = relative.parents[2] / 'segments' / ('00000_01000_'+'a'*32) / 'state.pth'
        path = base.joinpath(*rel.parts);path.parent.mkdir();path.write_bytes(b'intermediate')
        candidates.append({'relative_path': rel.as_posix(), 'expected_bytes': 12})
    extra = base / 'unlisted.pth';extra.write_bytes(b'keep')
    metadata = base / 'metadata.json';metadata.write_bytes(b'keep')
    current = base / 'diagnostic_20261020/samples/source.json.gz'
    current.parent.mkdir(parents=True);current.write_bytes(b'keep')
    plan = {'candidate_count': 4, 'candidate_apparent_bytes': 48, 'source_disk_report_sha256': 'f'*64,
            'deletion_effect': 'legacy recovery disabled', 'candidates': candidates}
    return base, plan, frozen, finals, (extra, metadata, current)


def paths(base, plan):
    return [base.joinpath(*PurePosixPath(x['relative_path']).parts) for x in plan['candidates']]


def test_preview_is_default_and_creates_no_journal_or_deletions(fixture):
    base, plan, frozen, _, protected = fixture
    result = cleanup.prune(base, plan, frozen)
    assert result['apply'] is False and result['present_count'] == 4 and result['deleted_count'] == 0
    assert result['present_apparent_bytes'] == 48
    assert not (base/'cleanup_20261021').exists()
    assert all(p.exists() for p in [*paths(base, plan), *protected])


def test_apply_is_exact_allowlist_journaled_and_idempotent(fixture):
    base, plan, frozen, finals, protected = fixture
    result = cleanup.prune(base, plan, frozen, apply=True)
    assert result['deleted_count'] == 4 and result['deleted_apparent_bytes'] == 48
    assert all(p.exists() for p in [*finals, *protected])
    assert all(not p.exists() for p in paths(base, plan))
    journal = next((base/'cleanup_20261021').glob('*.jsonl'))
    entries = [json.loads(x) for x in journal.read_text().splitlines()]
    assert sum('intent' in x for x in entries) == 4 and entries[-1]['complete'] is True
    second = cleanup.prune(base, plan, frozen, apply=True)
    assert second['deleted_count'] == 0 and second['already_absent_count'] == 4


def test_all_finals_verified_before_any_deletion(fixture):
    base, plan, frozen, finals, _ = fixture
    finals[-1].write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Protected final'):cleanup.prune(base, plan, frozen, apply=True)
    assert all(p.exists() for p in paths(base, plan))


@pytest.mark.parametrize('change', ['size', 'duplicate', 'traversal', 'final', 'outside', 'future', 'directory', 'link'])
def test_unsafe_or_changed_candidates_fail_before_any_deletion(fixture, monkeypatch, change):
    base, plan, frozen, finals, _ = fixture
    original_paths = paths(base, plan);bad = deepcopy(plan)
    if change == 'size':original_paths[-1].write_bytes(b'changed')
    if change == 'duplicate':bad['candidates'][-1] = bad['candidates'][0]
    if change == 'traversal':bad['candidates'][0]['relative_path'] = '../outside/state.pth'
    if change == 'final':bad['candidates'][0]['relative_path'] = finals[0].relative_to(base).as_posix()
    if change == 'outside':bad['candidates'][0]['relative_path'] = 'diagnostic_20261020/segments/'+'0_1000_'+'a'*32+'/state.pth'
    if change == 'future':bad['candidates'][0]['relative_path'] = bad['candidates'][0]['relative_path'].replace('00000_01000_', '40000_41000_')
    if change == 'directory':original_paths[0].unlink();original_paths[0].mkdir()
    if change == 'link':
        original = type(base).is_symlink
        monkeypatch.setattr(type(base), 'is_symlink', lambda p: p == original_paths[0] or original(p))
    with pytest.raises(ValueError):cleanup.prune(base, bad, frozen, apply=True)
    assert all(p.exists() for p in original_paths)
    assert all(p.exists() for p in finals)


def test_active_old_formal_training_blocks_pruning(fixture, monkeypatch):
    base, plan, frozen, _, _ = fixture
    monkeypatch.setattr(cleanup, 'active_legacy_trainers', lambda: [123])
    with pytest.raises(ValueError, match='trainer is active'):cleanup.prune(base, plan, frozen, apply=True)
    assert all(p.exists() for p in paths(base, plan))


def test_frozen_disk_allowlist_matches_exact_closed_nonfinal_scope():
    plan = json.loads(cleanup.PLAN.read_bytes())
    frozen = json.loads(cleanup.FREEZE.read_bytes())
    roots = {PurePosixPath(x['checkpoint_relative_to_output_root']).parents[2] for x in frozen['variants'].values()}
    assert len(plan['candidates']) == plan['candidate_count'] == 156
    assert sum(x['expected_bytes'] for x in plan['candidates']) == plan['candidate_apparent_bytes']
    for row in plan['candidates']:
        rel = PurePosixPath(row['relative_path'])
        assert rel.parents[2] in roots and rel.name == 'state.pth'
        assert int(rel.parent.name.split('_')[1]) <= 39000
