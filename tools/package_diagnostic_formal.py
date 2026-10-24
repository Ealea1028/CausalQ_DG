"""Export formal source-only reports; checkpoints and full traces stay remote."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import yaml

from causalq.protocols.diagnostic_formal import NEW_UNCENTERED, matched_baseline, release_contract, validate_training_config, verify_release
from causalq.protocols.diagnostic_20261020 import STUDY
from causalq.protocols.diagnostic_centered_formal import (
    NEW_CENTERED, centered_baseline, centered_release_contract,
    validate_centered_training_config, verify_centered_release,
)
from tools.inspect_gta5_source_structure import _git
from tools.train_formal_source import validate_final_report, write_json
from tools.train_strict_source import file_sha


def stage(phase):
    if phase == 29:
        return NEW_UNCENTERED, matched_baseline, release_contract, validate_training_config, verify_release
    if phase == 30:
        return NEW_CENTERED, centered_baseline, centered_release_contract, validate_centered_training_config, verify_centered_release
    raise ValueError('Only formal phases29/30 are released')


def initialize(root, *, phase=29):
    _, _, contract, _, verify = stage(phase)
    identity = {'phase': phase, 'study': STUDY, 'git_sha': _git('rev-parse', 'HEAD'),
                f'phase{phase}_contract_sha256': verify(), 'contract': contract()}
    path = root / 'run_manifest.json'
    if path.exists():
        previous = json.loads(path.read_bytes())
        if any(previous[k] != value for k, value in identity.items()):
            raise ValueError('Formal run manifest differs; preserve old outputs')
    else:
        write_json(path, {**identity, 'started_utc': datetime.now(timezone.utc).isoformat()})


def package(root, *, phase=29):
    variants, baseline, _, validate_config, verify = stage(phase)
    sha = _git('rev-parse', 'HEAD')
    contract_sha = verify()
    reports = {}
    initialize(root, phase=phase)
    files = [root / 'run_manifest.json']
    for variant in variants:
        directory = root / variant
        metadata = json.loads((directory / 'metadata.json').read_bytes())
        validate_config(yaml.safe_load((directory / 'config.yaml').read_bytes()), variant)
        if (metadata['git_sha'] != sha or metadata['variant'] != variant
                or metadata[f'phase{phase}_contract_sha256'] != contract_sha
                or metadata['centered_residuals'] is not (phase == 30)
                or metadata['config_sha256'] != file_sha(directory / 'config.yaml')):
            raise ValueError('Formal source run identity differs')
        report = validate_final_report(directory, metadata)
        if (report['phase'] != phase or report['initial_head_sha256'] != baseline()['initial_head_sha256']
                or report['source_evaluation']['training_state_unchanged'] is not True):
            raise ValueError('Formal initialization/final source policy differs')
        reports[variant] = report
        files += [directory / name for name in ('formal_report.json', 'metadata.json', 'config.yaml', 'initialization.json')]
    report = {'schema_version': 1, 'phase': phase, 'study': STUDY, 'ok': True, 'git_sha': sha,
              'finished_utc': datetime.now(timezone.utc).isoformat(),
              f'phase{phase}_contract_sha256': contract_sha, 'optimizer_updates_per_variant': 40000,
              'total_optimizer_updates': 40000*len(variants), 'target_data_accessed': False, 'target_miou': None,
              'source_miou': {v: r['source_miou'] for v, r in reports.items()},
              'reports': reports, 'next_phase': ('Phase30 CA/CR formal40000; no target access' if phase == 29
                                               else 'Phase31 new-model source diagnostics; no target access')}
    report_path = root / f'phase{phase}_report.json'
    write_json(report_path, report)
    files.append(report_path)
    sums = root / 'SHA256SUMS'
    sums.write_bytes(''.join(f'{file_sha(p)}  {p.relative_to(root).as_posix()}\n' for p in files).encode())
    archive_path = root / f'phase{phase}_export.zip'
    temporary = root / f'phase{phase}_export.zip.partial'
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
    p.add_argument('--phase', type=int, choices=(29, 30), default=29)
    args = p.parse_args(argv)
    if args.initialize:initialize(args.output_root, phase=args.phase)
    else:package(args.output_root, phase=args.phase)
    return 0


if __name__ == '__main__':raise SystemExit(main())
