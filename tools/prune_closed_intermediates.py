"""Preview an exact closed-study allowlist; --apply removes only those state files."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import uuid

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'experiments/DIAGNOSTIC_20261020/cleanup_plan_20261021.json'
FREEZE = ROOT / 'experiments/STRICT_DG_PHASE26/final_evaluation_freeze.json'
FREEZE_SHA = '959b031dc5112ee021b1c14dfd8af15d80c998cf7ff510d0065893f69fda95d5'


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):digest.update(block)
    return digest.hexdigest()


def active_legacy_trainers():
    result = []
    proc = Path('/proc')
    if not proc.is_dir():return result
    for path in proc.glob('[0-9]*/cmdline'):
        try:args = [x.decode(errors='replace') for x in path.read_bytes().split(b'\0') if x]
        except (OSError, PermissionError):continue
        if any(x == 'tools.train_formal_source' or x.endswith('/tools/train_formal_source.py') for x in args):
            result.append(int(path.parent.name))
    return result


def check_path(base, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts or '\\' in relative:
        raise ValueError('Unsafe cleanup path')
    path = base.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(base):raise ValueError('Cleanup path escapes output base')
    for ancestor in (path, *path.parents):
        if ancestor == base:break
        if ancestor.is_symlink() or getattr(ancestor, 'is_junction', lambda: False)():
            raise ValueError('Cleanup refuses symbolic links/junctions')
    return path


def preflight(base, plan, frozen):
    base = base.resolve()
    if not base.is_dir():raise ValueError('Original output base does not exist')
    trainers = active_legacy_trainers()
    if trainers:raise ValueError(f'Legacy formal trainer is active: {trainers}; no pruning')
    roots = set();protected = {}
    for record in frozen['variants'].values():
        rel = PurePosixPath(record['checkpoint_relative_to_output_root'])
        path = check_path(base, rel.as_posix())
        if not path.is_file() or file_sha(path) != record['checkpoint_sha256']:
            raise ValueError('Protected final checkpoint is missing or its SHA differs')
        roots.add(rel.parents[2]);protected[path] = record['checkpoint_sha256']
    rows = [];seen = set()
    for item in plan['candidates']:
        rel = PurePosixPath(item['relative_path'])
        if len(rel.parts) < 4 or rel.name != 'state.pth' or rel.parents[2] not in roots or rel.parent.parent.name != 'segments':
            raise ValueError('Candidate outside the four closed-study segment directories')
        match = re.fullmatch(r'(\d+)_(\d+)_[0-9a-f]{32}', rel.parent.name)
        if not match:raise ValueError('Candidate segment name differs')
        start, end = map(int, match.groups())
        if start < 0 or end-start != 1000 or start % 1000 or end > 39000:
            raise ValueError('Only intermediate steps through39000 are allowed')
        path = check_path(base, rel.as_posix())
        if path in protected or path in seen:raise ValueError('Protected or duplicate candidate')
        seen.add(path)
        size = item['expected_bytes']
        if type(size) is not int or size <= 0:raise ValueError('Invalid candidate size')
        state = path.lstat() if path.exists() else None
        if state is not None and (not stat.S_ISREG(state.st_mode) or state.st_nlink != 1 or state.st_size != size):
            raise ValueError('Candidate type/link-count/size differs; nothing deleted')
        rows.append((path, state))
    if len(rows) != plan['candidate_count'] or sum(x['expected_bytes'] for x in plan['candidates']) != plan['candidate_apparent_bytes']:
        raise ValueError('Candidate manifest count/size differs')
    return base, protected, rows


def prune(base, plan, frozen, *, apply=False):
    base, protected, rows = preflight(base, plan, frozen)
    existing = [(p, s) for p, s in rows if s is not None]
    before = shutil.disk_usage(base).free
    report = {'apply': apply, 'protected_final_count': len(protected), 'candidate_count': len(rows),
              'present_count': len(existing), 'already_absent_count': len(rows)-len(existing),
              'present_apparent_bytes': sum(s.st_size for _, s in existing),
              'deleted_count': 0, 'deleted_apparent_bytes': 0, 'free_bytes_before': before,
              'effect': plan['deletion_effect'], 'journal': None}
    if not apply:
        report['preview_paths'] = [str(p) for p, _ in existing]
        return report
    journal_dir = base / 'cleanup_20261021';journal_dir.mkdir(exist_ok=True)
    if journal_dir.is_symlink() or journal_dir.resolve().parent != base:
        raise ValueError('Unsafe cleanup journal directory')
    journal = journal_dir / (uuid.uuid4().hex + '.jsonl')
    report['journal'] = str(journal)
    with journal.open('x', encoding='utf-8') as stream:
        def record(value):
            stream.write(json.dumps(value) + '\n');stream.flush();os.fsync(stream.fileno())
        record({'started_utc': datetime.now(timezone.utc).isoformat(), 'output_base': str(base),
                'candidate_count': len(rows), 'source_disk_report_sha256': plan['source_disk_report_sha256']})
        for path, previous in existing:
            check_path(base, path.relative_to(base).as_posix())
            current = path.lstat()
            if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_nlink) != (
                    previous.st_dev, previous.st_ino, previous.st_size, previous.st_mtime_ns, 1):
                raise ValueError('Candidate changed after preflight; preserve remaining files')
            record({'intent': 'remove_intermediate_only', 'path': str(path), 'bytes': current.st_size})
            path.unlink()
            report['deleted_count'] += 1;report['deleted_apparent_bytes'] += current.st_size
            record({'deleted': str(path)})
        if any(file_sha(path) != digest for path, digest in protected.items()):
            raise ValueError('Protected final changed; retain journal for investigation')
        report['free_bytes_after'] = shutil.disk_usage(base).free
        record({'complete': True, **report})
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-base', type=Path, default=Path('/root/autodl-tmp/outputs/CausalQ_DG'))
    p.add_argument('--apply', action='store_true', help='Delete the reviewed intermediate states; default previews only')
    args = p.parse_args(argv)
    if file_sha(FREEZE) != FREEZE_SHA:raise ValueError('Original protected-final freeze differs')
    report = prune(args.output_base, json.loads(PLAN.read_bytes()), json.loads(FREEZE.read_bytes()), apply=args.apply)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':raise SystemExit(main())
