"""Read-only AutoDL disk report; never delete or move files."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def disk_report(base, outputs, pretrained, datasets, *, home=None):
    freeze = json.loads((ROOT / 'experiments/STRICT_DG_PHASE26/final_evaluation_freeze.json').read_bytes())
    protected = [(outputs / r['checkpoint_relative_to_output_root']).resolve() for r in freeze['variants'].values()]
    candidates = []
    for final in protected:
        for path in sorted((final.parents[1]).glob('*/state.pth')):
            if path.resolve() in protected:continue
            candidates.append({'path': str(path), 'bytes': path.stat().st_size,
                'impact': 'candidate only; deleting breaks original segment recovery; preserve JSON/logs/final state'})
    groups = [datasets, pretrained, outputs]
    if home is not None:groups += [home / '.cache/pip', home / '.cache/huggingface', home / '.cache/torch']
    sizes = []
    for path in groups:
        if not path.exists():continue
        result = subprocess.run(['du', '-x', '-s', '-B1', '--', str(path)], capture_output=True, text=True)
        sizes.append({'path': str(path), 'bytes': int(result.stdout.split()[0]) if result.returncode == 0 else None,
                      'error': result.stderr.strip() or None})
    usage = shutil.disk_usage(base)
    return {'read_only': True, 'disk': {'total_bytes': usage.total, 'used_bytes': usage.used, 'free_bytes': usage.free},
        'top_level_sizes': sizes, 'protected_final_checkpoints': [str(p) for p in protected],
        'protected_file_status': [{'path': str(p), 'exists': p.is_file(), 'bytes': p.stat().st_size if p.is_file() else None}
                                  for p in protected],
        'protected_other': [str(datasets), str(pretrained), str(outputs / 'diagnostic_20261020')],
        'closed_study_intermediate_candidates': candidates,
        'candidate_bytes': sum(r['bytes'] for r in candidates), 'deletion_performed': False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=Path('/root/autodl-tmp'))
    p.add_argument('--outputs', type=Path, default=Path('/root/autodl-tmp/outputs/CausalQ_DG'))
    args = p.parse_args(argv)
    print(json.dumps(disk_report(args.base, args.outputs, args.base / 'pretrained', args.base / 'datasets', home=Path.home()), indent=2))
    return 0


if __name__ == '__main__':raise SystemExit(main())
