"""Retire only the reviewed intermediate states of the closed Phase42–44 runs."""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import signal
import subprocess
import sys

import numpy as np
import torch

from causalq.runtime.identity import file_sha, write_json
from causalq.runtime.lora_l1_formal import checkpoint_rows
from causalq.runtime.lora_phase46 import load_contract
from causalq.runtime.lora_source_diagnostics import ROOT, verify_final
from causalq.runtime.lora_training import learning_rate, load_state_file
from tools.retire_lora_l0 import exclusive_json, safe
from tools.verify_lora_checkpoint_preview import BASE, PREVIEW, load_preview

RELEASE = PREVIEW.with_name('phase46_checkpoint_cleanup_release.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def verified_rows(report_path, release, finals):
    require(file_sha(report_path) == release['verified_report_sha256'], 'Read-only report SHA differs')
    report = read(report_path)
    expected = load_preview(PREVIEW, release['preview_sha256'], finals)
    require(report['ok'] is True and report['kind'] == 'checkpoint_cleanup_read_only'
            and report['git_sha'] == release['verified_tool_git_sha']
            and report['preview_sha256'] == release['preview_sha256']
            and report['checkpoint_payloads_deleted'] == 0
            and report['protected_finals'] == {
                name: {'path':Path(pin['path']).as_posix(), 'sha256':pin['sha256'],
                       'marker_sha256':pin['marker_sha256']}
                for name, pin in finals.items()
            }, 'Read-only report or final pins differ')
    by_model = {}
    seen = set()
    for row in report['rows']:
        key = PurePosixPath(row['path'])
        require(key not in seen and key in expected, f'Unexpected verified row: {key}')
        prior = expected[key]
        require(all(row[k] == v for k, v in prior.items())
                and row['payload_sha256'] == prior['marker_sha256_of_payload']
                and re.fullmatch('[0-9a-f]{64}', row['marker_sha256'])
                and 0 < row['bytes'] <= row['allocated_bytes']
                and row['inode'] > 0 and row['device'] > 0, f'Invalid verified row: {key}')
        seen.add(key)
        by_model.setdefault(row['model'], []).append(row)
    counts = Counter(row['decision'] for row in report['rows'])
    total = sum(row['allocated_bytes'] for row in report['rows'] if row['decision'] == 'CANDIDATE')
    require(seen == set(expected) and counts == {'CANDIDATE':release['candidate_count'], 'KEEP':release['l1_keep_count']}
            and total == report['candidate_allocated_bytes'] == release['candidate_allocated_bytes']
            and counts['CANDIDATE'] == report['candidate_count'] and counts['KEEP'] == report['l1_keep_count'],
            'Verified candidate totals differ')
    return by_model


def verify_snapshot(path, marker, *, lora):
    """Verify a retained compact state on CPU after its pinned file hash matched."""
    state = load_state_file(path)
    step = marker['step']
    require(state['schema'] == 1 and state['identity'] == marker['identity']
            and state['step'] == step and state['source_cursor'] == 8*step
            and state['next_lr'] == learning_rate(step), f'Invalid recovery cursor: {path}')
    tensors = state['model']
    require(tensors and all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in tensors.values())
            and sum(v.numel() for k, v in tensors.items() if k.startswith('backbone.')) == (98304 if lora else 0),
            f'Invalid retained compact model: {path}')
    optimizer = state['optimizer']
    ids = [i for group in optimizer['param_groups'] for i in group['params']]
    require(ids and len(ids) == len(set(ids)) and set(ids) == set(optimizer['state'])
            and all(g['lr'] == learning_rate(step-1) for g in optimizer['param_groups']),
            f'Invalid optimizer coverage or schedule: {path}')
    for slot in optimizer['state'].values():
        a, b = slot['exp_avg'], slot['exp_avg_sq']
        require(int(slot['step']) == step and a.shape == b.shape
                and torch.isfinite(a).all() and torch.isfinite(b).all(), f'Invalid AdamW state: {path}')
    rng = state['rng']
    require(set(rng) == {'torch', 'cuda', 'python', 'numpy'} and rng['cuda']
            and all(t.dtype == torch.uint8 and t.ndim == 1 and t.numel() for t in rng['cuda']),
            f'Invalid RNG providers: {path}')
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    torch.Generator().set_state(rng['torch'])


def inspect_step(base, row, identity):
    payload = safe(base, Path(row['path']))
    live = safe(base, payload.with_name('complete.json'))
    retired = safe(base, payload.with_name('retired_complete.json'))
    names = {p.name for p in payload.parent.iterdir()}
    if names == {'complete.json', 'state.pt'}:
        kind, marker_path = 'live', live
    elif row['decision'] == 'CANDIDATE' and names in ({retired.name, 'state.pt'}, {retired.name}):
        kind, marker_path = ('pending' if 'state.pt' in names else 'retired'), retired
    else:
        raise ValueError(f'Unexpected checkpoint directory contents: {payload.parent}')
    require(marker_path.is_file() and file_sha(marker_path) == row['marker_sha256'],
            f'Completion marker changed: {marker_path}')
    record = read(marker_path)
    require(record['identity'] == identity and record['step'] == row['step']
            and record['sha256'] == row['payload_sha256'], f'Checkpoint identity changed: {payload}')
    if kind != 'retired':
        require(payload.is_file(), f'Invalid payload path: {payload}')
        stat = payload.stat()
        require((stat.st_size, stat.st_blocks*512, stat.st_ino, stat.st_dev) ==
                (row['bytes'], row['allocated_bytes'], row['inode'], row['device'])
                and file_sha(payload) == row['payload_sha256'], f'Payload changed: {payload}')
    return kind, record


def sync_directory(path):
    if sys.platform == 'linux':
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def retire_step(base, row, state):
    require(row['decision'] == 'CANDIDATE' and state in ('live','pending'), 'Cannot retire a protected checkpoint')
    payload = safe(base, Path(row['path']))
    if state == 'live':
        live = safe(base, payload.with_name('complete.json'))
        retired = safe(base, payload.with_name('retired_complete.json'))
        require(not retired.exists() and file_sha(live) == row['marker_sha256'],
                f'Completion marker changed before retirement: {live}')
        live.rename(retired)
        sync_directory(payload.parent)
    payload.unlink()
    sync_directory(payload.parent)


def scan(base, groups, finals):
    states, markers = {}, {}
    for name, pin in finals.items():
        final, identity = verify_final(pin, base)
        if not pin['lora']:
            verify_snapshot(final, read(final.with_name('complete.json')), lora=False)
            continue
        rows = groups[name]
        root = final.parent.parent
        require({safe(base, p).name for p in root.iterdir()} == {f"{r['step']:08d}" for r in rows},
                f'Unexpected checkpoint steps: {name}')
        for row in rows:
            path = Path(row['path'])
            state, marker = inspect_step(base, row, identity)
            states[row['path']] = state
            markers[row['path']] = marker
            if row['decision'] == 'KEEP':
                verify_snapshot(path, marker, lora=True)
        print(f'{name}: verified {len(rows)} entries and retained states', flush=True)
    return states, markers


def validate_after(base, groups, finals, catalog):
    for name, pin in finals.items():
        final, identity = verify_final(pin, base)
        if not pin['lora']:
            continue
        rows = groups[name]
        steps = {step for step, _, _ in checkpoint_rows(final.parent.parent, identity)}
        expected = {row['step'] for row in rows if row['decision'] == 'KEEP'}
        require(steps == expected, f'Recovery inventory differs after retirement: {name}')
        for row in rows:
            kind, marker = inspect_step(base, row, identity)
            require(kind == ('live' if row['decision'] == 'KEEP' else 'retired')
                    and marker == catalog['original_markers'][row['path']], f'Unexpected final state: {row["path"]}')


def run(base, audit_root, output, release, groups, finals, git_sha, *, apply):
    import fcntl
    entries = [row for rows in groups.values() for row in rows]
    candidates = [row for row in entries if row['decision'] == 'CANDIDATE']
    catalog_path = safe(audit_root, audit_root/f"cleanup_manifest_{release['verified_report_sha256'][:12]}.json")
    roots = {base/Path(pin['path']).relative_to(base).parts[0] for pin in finals.values()}
    with ExitStack() as stack:
        for path in (base/'gpu0.lock', *(root/'run.lock' for root in sorted(roots))):
            handle = stack.enter_context(safe(base, path).open('rb'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = datetime.now(timezone.utc).isoformat()
        free_before = shutil.disk_usage(base).free
        states, markers = scan(base, groups, finals)
        catalog = {
            'schema':1, 'release':release, 'entries':entries, 'original_markers':markers,
            'protected_finals':{name: {k:pin[k] for k in ('path', 'sha256', 'marker_sha256')}
                                for name, pin in finals.items()},
        }
        if catalog_path.exists():
            require(read(catalog_path) == catalog, 'Existing immutable catalog differs')
        else:
            require(set(states.values()) == {'live'}, 'Interrupted cleanup has no immutable catalog')
            if apply:
                write_json(catalog_path, catalog)
                with catalog_path.open('r+b') as file:
                    os.fsync(file.fileno())
                sync_directory(audit_root)
        result = {
            'ok':False, 'mode':'apply' if apply else 'check', 'policy':release['policy'],
            'git_sha':git_sha, 'environment':{'python':sys.version, 'torch':torch.__version__,
                                             'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')},
            'started_utc':started, 'verified_report_sha256':release['verified_report_sha256'],
            'candidate_count':len(candidates), 'l1_keep_count':len(entries)-len(candidates),
            'protected_finals':{name:pin['sha256'] for name, pin in finals.items()},
            'free_bytes_before':free_before, 'deleted':[], 'already_retired':[],
        }
        try:
            for row in candidates:
                state = states[row['path']]
                if state == 'retired':
                    result['already_retired'].append(row['path'])
                elif apply:
                    retire_step(base, row, state)
                    result['deleted'].append(row['path'])
            if apply:
                validate_after(base, groups, finals, catalog)
            result['ok'] = True
        except BaseException as error:
            result['error'] = str(error)
            raise
        finally:
            free_after = shutil.disk_usage(base).free
            result.update(
                free_bytes_after=free_after, free_bytes_delta=free_after-free_before,
                ended_utc=datetime.now(timezone.utc).isoformat(),
                catalog_sha256=file_sha(catalog_path) if catalog_path.exists() else None,
                deleted_allocated_bytes=sum(row['allocated_bytes'] for row in candidates
                                            if row['path'] in result['deleted']),
                candidates_present_after=sum(Path(row['path']).is_file() for row in candidates),
            )
            exclusive_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    require(sys.platform == 'linux' and Path.cwd().resolve() == ROOT.resolve()
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use the pinned AutoDL checkout with CUDA hidden')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    require(head == args.expected_sha and not status, 'Checkout commit or tracked changes differ')
    release = read(RELEASE)
    audit_root = safe(BASE, BASE/'disk_audits')
    output = safe(audit_root, args.output)
    require(output.parent == audit_root and output.suffix == '.json' and not output.exists(), 'Choose a fresh audit output')
    _, finals, _ = load_contract()
    report_path = safe(audit_root, audit_root/release['verified_report'])
    groups = verified_rows(report_path, release, finals)
    if args.apply:
        print('532 reviewed candidates, at most 25.05 GiB. Protected states are checked before deletion.')
        require(input('Type DELETE 532 to apply: ') == 'DELETE 532', 'Cancelled by operator')

    def stop(signum, frame):
        raise KeyboardInterrupt('Cleanup stop requested')

    signal.signal(signal.SIGTERM, stop)
    result = run(BASE, audit_root, output, release, groups, finals, head, apply=args.apply)
    print(json.dumps({k:result[k] for k in ('ok', 'mode', 'candidates_present_after', 'free_bytes_delta')}))


if __name__ == '__main__':
    main()
