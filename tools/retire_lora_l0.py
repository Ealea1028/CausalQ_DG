"""Retire the cancelled Phase41 L0 study, preserving the accepted D3 U0 final.

Preview is read-only. Apply requires a separately reviewed preview SHA and keeps
all metadata; it never removes directories, datasets, weights or environments.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('/root/autodl-tmp/outputs/CausalQ_DG/lora_causalq_1114/phase41_core_1af6b2e')
ACCEPTANCE = ROOT / 'experiments/LORA_CAUSALQ_1114/phase41_stop_acceptance.json'
PIN = 'dinov3_vitl16/checkpoints/U0/00040000/state.pt'
POLICY = 'cancel-l0-keep-accepted-d3-u0-final-v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def safe(root, path):
    root, path = Path(root).absolute(), Path(path).absolute()
    if root.resolve() != root or path.resolve() != path:
        raise ValueError('Redirected/noncanonical path')
    if path != root and root not in path.parents:
        raise ValueError('Path outside authorized root')
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink() or getattr(ancestor, 'is_junction', lambda: False)():
            raise ValueError('Symbolic link/junction refused')
    if path.exists() and path.is_file() and path.stat().st_nlink != 1:
        raise ValueError('Hard-linked file refused')
    return path


def exclusive_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def active_l0_processes(proc_root=Path('/proc')):
    active = []
    for path in proc_root.glob('[0-9]*/cmdline'):
        try:
            args = [x.decode(errors='replace') for x in path.read_bytes().split(b'\0') if x]
        except OSError:
            continue
        names = ('tools.train_lora_core_l0', 'tools.resume_lora_baseline')
        if any(a in names or any(a.endswith('/'+n.replace('.', '/')+'.py') for n in names) for a in args):
            active.append(int(path.parent.name))
    return active


def preview(root, acceptance):
    root = safe(root, root)
    if not root.is_dir():
        raise ValueError('Original run missing')
    if active_l0_processes():
        raise ValueError('L0 controller/trainer still active; stop it first')
    pin_record = acceptance['latest']['dinov3_vitl16']['U0']
    pin = safe(root, root / PIN)
    if not pin.is_file() or sha(pin) != pin_record['sha256'] or pin.stat().st_size != pin_record['bytes']:
        raise ValueError('Protected D3 U0 final missing or changed')
    rows, evidence, other = [], [], []
    for path in sorted(root.rglob('*')):
        safe(root, path)
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if path.name.endswith('.partial'):
            raise ValueError('Uncommitted partial file requires review: '+rel)
        if path.name == 'state.pt' and not (path.parent/'complete.json').is_file():
            raise ValueError('Orphan checkpoint requires review: '+rel)
        if path.name == 'complete.json':
            parts = PurePosixPath(rel).parts
            if len(parts) != 5 or parts[1] != 'checkpoints' or not re.fullmatch(r'\d{8}', parts[3]):
                raise ValueError('Unknown checkpoint layout: '+rel)
            backbone, _, variant, step_text, _ = parts
            marker = json.loads(path.read_text())
            expected = acceptance['latest'].get(backbone, {}).get(variant)
            step = marker['step']
            if (expected is None or marker['identity'] != expected['identity'] or type(step) is not int
                    or not 1000 <= step <= 40000 or step % 1000 or step_text != f'{step:08d}'):
                raise ValueError('Checkpoint producer/identity/step changed: '+rel)
            payload = safe(root, path.parent/'state.pt')
            tombstone = safe(root, path.parent/'retired.json')
            present = payload.exists()
            if present:
                if not stat.S_ISREG(payload.stat().st_mode) or sha(payload) != marker['sha256']:
                    raise ValueError('Checkpoint hash/type mismatch: '+rel)
                size = payload.stat().st_size
            else:
                if not tombstone.is_file():
                    raise ValueError('Missing checkpoint without retirement record: '+rel)
                retired = json.loads(tombstone.read_text())
                if (retired.get('policy') != POLICY or retired.get('sha256') != marker['sha256']
                        or retired.get('marker_sha256') != sha(path)):
                    raise ValueError('Retirement record mismatch: '+rel)
                size = retired['bytes']
            row = {'path': payload.relative_to(root).as_posix(), 'backbone': backbone,
                   'variant': variant, 'step': step, 'bytes': size, 'sha256': marker['sha256'],
                   'marker_sha256': sha(path), 'original_complete_marker': marker, 'present': present}
            rows.append(row)
        elif path.suffix in ('.json', '.jsonl', '.csv', '.log', '.txt'):
            evidence.append({'path': rel, 'bytes': path.stat().st_size, 'sha256': sha(path)})
        elif path.name != 'state.pt':
            other.append({'path': rel, 'bytes': path.stat().st_size, 'action': 'preserve'})
    pins = [r for r in rows if r['path'] == PIN]
    if len(pins) != 1 or pins[0]['original_complete_marker']['identity'] != pin_record['identity']:
        raise ValueError('Protected final marker missing/changed')
    candidates = [r for r in rows if r['path'] != PIN]
    return {'schema': 1, 'policy': POLICY, 'ok': True, 'run_root': str(root),
            'protected': pins, 'candidates': candidates, 'metadata_preserved': evidence,
            'other_files_preserved': other, 'candidate_count': len(candidates),
            'present_candidate_bytes': sum(r['bytes'] for r in candidates if r['present']),
            'formal_updates_performed': 0, 'payloads_deleted': 0,
            'free_bytes': shutil.disk_usage(root).free,
            'effect': 'All L0 training permanently cancelled. Only D3 U0 L0 final remains evaluable. '
                      'All 14 future training models are fresh L1; no L0 head is their initialization.'}


def apply(root, acceptance, plan_path, approved_sha256, output, *, tool_git_sha=None):
    if not re.fullmatch('[0-9a-f]{64}', approved_sha256) or sha(plan_path) != approved_sha256:
        raise ValueError('Reviewed preview SHA mismatch')
    plan = json.loads(Path(plan_path).read_text())
    current = preview(root, acceptance)  # Verify ALL remaining hashes before deleting anything.
    if plan.get('schema') != 1 or plan.get('policy') != POLICY or plan.get('run_root') != str(Path(root).absolute()):
        raise ValueError('Preview policy/root mismatch')
    keys = ('path', 'backbone', 'variant', 'step', 'bytes', 'sha256', 'marker_sha256', 'original_complete_marker')
    def records(rows):
        return [{key: row[key] for key in keys} for row in rows]
    if (records(plan['candidates']) != records(current['candidates'])
            or records(plan['protected']) != records(current['protected'])):
        raise ValueError('Preview checkpoint set changed; produce a new preview')
    root = Path(root).absolute()
    output = safe(root.parent, output)
    if output == root or root in output.parents or not output.parent.is_dir():
        raise ValueError('Fresh result must be outside original run')
    journal = safe(root.parent, output.with_suffix('.jsonl'))
    if output.exists() or journal.exists():
        raise ValueError('Fresh output/journal required')
    cancellation = safe(root, root/'L0_RETIRED.json')
    if not cancellation.exists():
        exclusive_json(cancellation, {'policy': POLICY, 'protected': PIN, 'preview_sha256': approved_sha256,
                                     'effect': current['effect']})
    elif json.loads(cancellation.read_text()).get('policy') != POLICY:
        raise ValueError('Unexpected retirement policy')
    deleted, deleted_bytes = 0, 0
    with journal.open('x', encoding='utf-8') as stream:
        def record(value):
            stream.write(json.dumps(value)+'\n'); stream.flush(); os.fsync(stream.fileno())
        record({'started_utc': datetime.now(timezone.utc).isoformat(), 'preview_sha256': approved_sha256,
                'policy': POLICY, 'protected': current['protected'], 'tool_git_sha': tool_git_sha})
        for row in current['candidates']:
            if not row['present']:
                continue
            payload = safe(root, root/row['path'])
            marker = safe(root, payload.parent/'complete.json')
            if sha(payload) != row['sha256'] or sha(marker) != row['marker_sha256']:
                raise ValueError('Payload/marker changed after preflight; preserve remaining files')
            tombstone = safe(root, payload.parent/'retired.json')
            item = {'policy': POLICY, 'sha256': row['sha256'], 'bytes': row['bytes'],
                    'marker_sha256': row['marker_sha256'], 'preview_sha256': approved_sha256}
            if not tombstone.exists():
                exclusive_json(tombstone, item)
            elif json.loads(tombstone.read_text()) != item:
                raise ValueError('Unexpected tombstone; preserve remaining files')
            record({'intent': row})
            payload.unlink()
            deleted += 1; deleted_bytes += row['bytes']
            record({'deleted': row['path']})
        verified = preview(root, acceptance)
        report = {'ok': True, 'policy': POLICY, 'preview_sha256': approved_sha256,
                  'tool_git_sha': tool_git_sha,
                  'deleted_count': deleted, 'deleted_apparent_bytes': deleted_bytes,
                  'remaining_candidate_bytes': verified['present_candidate_bytes'],
                  'protected': verified['protected'], 'free_bytes_before': current['free_bytes'],
                  'free_bytes_after': verified['free_bytes'], 'formal_updates_performed': 0}
        record({'complete': True, **report})
    exclusive_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply-plan', type=Path)
    parser.add_argument('--approved-plan-sha256')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run on AutoDL Linux with CUDA hidden')
    if bool(args.apply_plan) != bool(args.approved_plan_sha256):
        parser.error('Apply needs both reviewed plan and SHA')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error("Set CUDA_VISIBLE_DEVICES='' for this CPU-only operation")
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        parser.error('Tracked source modified')
    output = safe(RUN.parent, args.output)
    if output == RUN or RUN in output.parents or not output.parent.is_dir() or output.exists():
        parser.error('Fresh output outside original run required')
    import fcntl
    with ExitStack() as stack:
        for path in (RUN/'run.lock', RUN.parent/'gpu0.lock'):
            handle = stack.enter_context(safe(RUN.parent, path).open('r+'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acceptance = json.loads(ACCEPTANCE.read_text())
        if args.apply_plan:
            report = apply(RUN, acceptance, args.apply_plan, args.approved_plan_sha256, output,
                           tool_git_sha=git_sha)
        else:
            report = preview(RUN, acceptance)
            report['tool_git_sha'] = git_sha
            report['environment'] = {'python': sys.version, 'executable': sys.executable,
                                     'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}
            exclusive_json(output, report)
    print(json.dumps({k: report[k] for k in ('ok', 'candidate_count', 'present_candidate_bytes',
                     'deleted_count', 'deleted_apparent_bytes') if k in report}, indent=2))
    print('Output:', output, 'SHA256:', sha(output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
