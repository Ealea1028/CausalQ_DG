"""Export Phase29 source-only reports; checkpoints and full traces stay remote."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import yaml

from causalq.protocols.diagnostic_formal import NEW_UNCENTERED, matched_baseline, release_contract, validate_training_config, verify_release
from causalq.protocols.diagnostic_20261020 import STUDY
from tools.inspect_gta5_source_structure import _git
from tools.train_formal_source import validate_final_report, write_json
from tools.train_strict_source import file_sha


def initialize(root):
    identity = {'phase': 29, 'study': STUDY, 'git_sha': _git('rev-parse', 'HEAD'),
                'phase29_contract_sha256': verify_release(), 'contract': release_contract()}
    path = root / 'run_manifest.json'
    if path.exists():
        previous = json.loads(path.read_bytes())
        if any(previous[k] != value for k, value in identity.items()):
            raise ValueError('Phase29 run manifest differs; preserve old outputs')
    else:
        write_json(path, {**identity, 'started_utc': datetime.now(timezone.utc).isoformat()})


def package(root):
    sha = _git('rev-parse', 'HEAD')
    contract_sha = verify_release()
    reports = {}
    initialize(root)
    files = [root / 'run_manifest.json']
    for variant in NEW_UNCENTERED:
        directory = root / variant
        metadata = json.loads((directory / 'metadata.json').read_bytes())
        validate_training_config(yaml.safe_load((directory / 'config.yaml').read_bytes()), variant)
        if (metadata['git_sha'] != sha or metadata['variant'] != variant
                or metadata['phase29_contract_sha256'] != contract_sha
                or metadata['centered_residuals'] is not False
                or metadata['config_sha256'] != file_sha(directory / 'config.yaml')):
            raise ValueError('Phase29 source run identity differs')
        report = validate_final_report(directory, metadata)
        if (report['phase'] != 29 or report['initial_head_sha256'] != matched_baseline()['initial_head_sha256']
                or report['source_evaluation']['training_state_unchanged'] is not True):
            raise ValueError('Phase29 initialization/final source policy differs')
        reports[variant] = report
        files += [directory / name for name in ('formal_report.json', 'metadata.json', 'config.yaml', 'initialization.json')]
    report = {'schema_version': 1, 'phase': 29, 'study': STUDY, 'ok': True, 'git_sha': sha,
              'finished_utc': datetime.now(timezone.utc).isoformat(),
              'phase29_contract_sha256': contract_sha, 'optimizer_updates_per_variant': 40000,
              'total_optimizer_updates': 160000, 'target_data_accessed': False, 'target_miou': None,
              'source_miou': {v: r['source_miou'] for v, r in reports.items()},
              'reports': reports, 'next_phase': 'Phase30 CA/CR formal40000; no target access'}
    write_json(root / 'phase29_report.json', report)
    files.append(root / 'phase29_report.json')
    sums = root / 'SHA256SUMS'
    sums.write_bytes(''.join(f'{file_sha(p)}  {p.relative_to(root).as_posix()}\n' for p in files).encode())
    archive_path = root / 'phase29_export.zip'
    temporary = root / 'phase29_export.zip.partial'
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*files, sums]:
            archive.write(path, path.relative_to(root).as_posix())
    temporary.replace(archive_path)
    print(json.dumps({'ok': True, 'source_miou': report['source_miou'],
                      'export': str(archive_path), 'sha256': file_sha(archive_path)}, indent=2), flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--initialize', action='store_true', help='Record immutable stage identity before formal training')
    args = p.parse_args(argv)
    if args.initialize:initialize(args.output_root)
    else:package(args.output_root)
    return 0


if __name__ == '__main__':raise SystemExit(main())
