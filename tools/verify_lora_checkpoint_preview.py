"""Read-only AutoDL audit of the previously reviewed Phase42–44 checkpoint list."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

from causalq.runtime.identity import file_sha
from causalq.runtime.lora_l1_formal import checkpoint_rows
from causalq.runtime.lora_phase46 import load_contract
from causalq.runtime.lora_source_diagnostics import ROOT, verify_final
from tools.retire_lora_l0 import exclusive_json, safe


BASE = Path('/root/autodl-tmp/outputs/CausalQ_DG/lora_causalq_1114')
PREVIEW = ROOT/'experiments/LORA_CAUSALQ_1114/phase46_cleanup_preview.tsv'
PREVIEW_SHA = 'd44b256ff2c8dc627ad5dc6f582c0833301e46a14733f50fc335f94a55f7c056'


def load_preview(path, expected_sha, finals):
    if file_sha(path) != expected_sha:
        raise ValueError('Reviewed preview SHA differs')
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if reader.fieldnames != ['decision', 'model', 'step', 'allocated_bytes', 'marker_sha256_of_payload', 'path']:
            raise ValueError('Unexpected preview columns')
        rows = list(reader)
    inspected = {}
    for row in rows:
        name = row['model']
        step = int(row['step'])
        row['step'] = step
        row['allocated_bytes'] = int(row['allocated_bytes'])
        final = PurePosixPath(finals[name]['path'])
        expected = final.parent.parent/f'{step:08d}'/'state.pt'
        if (name == 'D3_U0_L0' or not 0 < step <= 40000 or row['decision'] not in ('KEEP', 'CANDIDATE')
                or str(expected) != row['path'] or row['allocated_bytes'] <= 0 or expected in inspected):
            raise ValueError(f'Unexpected preview row: {name}/{step}')
        inspected[expected] = row
    if Counter(row['decision'] for row in rows) != {'KEEP': 36, 'CANDIDATE': 532}:
        raise ValueError('Preview counts differ')
    for name, pin in finals.items():
        if not pin['lora']:
            continue
        group = {row['step']: row for row in rows if row['model'] == name}
        keep = {39000, 40000, *sorted(group)[-2:], *(s for s in group if s % 1000)}
        if ({s for s, row in group.items() if row['decision'] == 'KEEP'} != keep
                or group[40000]['marker_sha256_of_payload'] != pin['sha256']):
            raise ValueError(f'Protected checkpoint set differs: {name}')
    return inspected


def audit_payload(base, path, row, marker):
    payload = safe(base, path)
    marker_path = safe(base, payload.with_name('complete.json'))
    stat = payload.stat()
    digest = file_sha(payload)
    if (stat.st_size <= 0 or stat.st_blocks*512 != row['allocated_bytes']
            or marker['sha256'] != digest or digest != row['marker_sha256_of_payload']
            or marker['step'] != row['step']):
        raise ValueError(f'Payload SHA, step or allocated bytes differ: {payload}')
    return dict(row, payload_sha256=digest, marker_sha256=file_sha(marker_path),
                bytes=stat.st_size, inode=stat.st_ino, device=stat.st_dev)


def run(output, release):
    import fcntl
    _, finals, _ = load_contract()
    receipt = json.loads((PREVIEW.parent/'phase46_completion_acceptance.json').read_bytes())
    if (not receipt['accepted'] or not receipt['checkpoints_unchanged']
            or {k:v['checkpoint_sha256'] for k,v in receipt['models'].items()} !=
               {k:v['sha256'] for k,v in finals.items()}):
        raise ValueError('Phase46 acceptance differs from the final model pins')
    preview = load_preview(PREVIEW, PREVIEW_SHA, finals)
    audit_root = safe(BASE, BASE/'disk_audits')
    output = safe(audit_root, output)
    if output.parent != audit_root or output.suffix != '.json' or output.exists():
        raise ValueError('Use a new JSON report path directly under disk_audits')
    roots = {BASE/Path(pin['path']).relative_to(BASE).parts[0] for pin in finals.values()}
    verified, protected = [], {}
    with ExitStack() as stack:
        for lock in [BASE/'gpu0.lock', *(root/'run.lock' for root in sorted(roots))]:
            handle = stack.enter_context(safe(BASE, lock).open('rb'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, pin in finals.items():
            final, identity = verify_final(pin, BASE)
            protected[name] = {k:pin[k] for k in ('path', 'sha256', 'marker_sha256')}
            if not pin['lora']:
                continue
            entries = checkpoint_rows(final.parent.parent, identity)
            expected = {p for p, row in preview.items() if row['model'] == name}
            if {folder/'state.pt' for _, folder, _ in entries} != expected:
                raise ValueError(f'Live checkpoint coverage differs: {name}')
            for _, folder, marker in entries:
                path = folder/'state.pt'
                verified.append(audit_payload(BASE, path, preview[path], marker))
            print(f'{name}: verified {len(entries)} payloads', flush=True)
        exclusive_json(output, {
            'kind':'checkpoint_cleanup_read_only', 'git_sha':release, 'ok':True,
            'preview_sha256':PREVIEW_SHA, 'checkpoint_payloads_deleted':0, 'candidate_count':532,
            'candidate_allocated_bytes':sum(r['allocated_bytes'] for r in verified if r['decision'] == 'CANDIDATE'),
            'l1_keep_count':36, 'protected_finals':protected, 'rows':verified,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != 'linux' or Path.cwd().resolve() != ROOT.resolve():
        parser.error('Use the pinned AutoDL checkout')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if head != args.expected_sha or status:
        parser.error('Checkout commit or tracked changes differ')
    run(args.output, head)


if __name__ == '__main__':
    main()
