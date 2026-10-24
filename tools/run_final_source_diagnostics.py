"""Phase31 source-only final diagnostics; shared Phase28 measurements, no training."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

from causalq.protocols.diagnostic_20261020 import ROOT, STUDY
from causalq.protocols.diagnostic_final_source import (
    NEW_FINALS, final_records, final_source_report, final_config, final_contract, verify_final_contract,
)
from tools.inspect_gta5_source_structure import _git
from tools.run_diagnostic_20261020 import run_diagnostic, write_json
from tools.train_strict_source import file_sha


def initialize(root):
    identity = {'phase': 31, 'study': STUDY, 'git_sha': _git('rev-parse', 'HEAD'),
                'contract_sha256': verify_final_contract(), 'contract': final_contract()}
    path = root / 'run_manifest.json'
    if path.exists():
        previous = json.loads(path.read_bytes())
        if any(previous[k] != value for k, value in identity.items()):
            raise ValueError('Phase31 manifest differs; preserve original outputs')
    else:
        write_json(path, {**identity, 'started_utc': datetime.now(timezone.utc).isoformat()})


def package(root):
    initialize(root)
    sha = _git('rev-parse', 'HEAD');digest = verify_final_contract()
    files = [root / 'run_manifest.json'];reports = {};records = final_records()
    for v in NEW_FINALS:
        path = root / v / 'summary.json'
        summary = json.loads(path.read_bytes())
        identity = summary['identity']
        config = ROOT / f'configs/diagnostic_20261020/formal/{v.lower()}_seed0.yaml'
        if (summary['ok'] is not True or summary['sample_count'] != 2497 or summary['views'] != 5
                or identity != {'study': STUDY, 'phase': 31, 'variant': v, 'git_sha': sha,
                                'contract_sha256': digest, 'checkpoint_sha256': records[v]['checkpoint_sha256'],
                                'original_config_sha256': records[v]['config_sha256'],
                                'diagnostic_config_sha256': file_sha(config)}
                or summary['optimizer_updates_this_run'] != 0 or summary['target_data_accessed'] is not False
                or summary['model_rng_unchanged'] is not True or summary['checkpoint_training_updates'] != 40000
                or summary['training_producer_git_sha'] != records[v]['producer_git_sha']
                or summary['effect_geometry'] != 'longedge1024_aspect_pad16'
                or summary['gradient_geometry'] != 'deterministic_512_crop'):
            raise ValueError('Need six complete matching final source summaries: ' + v)
        reports[v] = {k: val for k, val in summary.items() if k not in
                      ('image_macro_statistics', 'pixel_weighted_responsibility')}
        files += [path, root / v / 'class_summary.csv']
    result = {'schema_version': 1, 'phase': 31, 'study': STUDY, 'ok': True, 'git_sha': sha,
              'contract_sha256': digest, 'reports': reports, 'optimizer_updates_this_run': 0,
              'target_data_accessed': False, 'baseline_diagnostics_reused': final_contract()['baseline_diagnostics_reused'],
              'intervention_examples_reused': final_contract()['intervention_examples'],
              'next_phase': final_contract()['next_phase']}
    report_path = root / 'job_report.json';write_json(report_path, result);files.append(report_path)
    sums = root / 'SHA256SUMS'
    sums.write_bytes(''.join(f'{file_sha(p)}  {p.relative_to(root).as_posix()}\n' for p in files).encode())
    archive_path = root / 'phase31_export.zip';temporary = root / 'phase31_export.zip.partial'
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*files, sums]:archive.write(path, path.relative_to(root).as_posix())
    temporary.replace(archive_path)
    print(json.dumps({'ok': True, 'export': str(archive_path), 'sha256': file_sha(archive_path)}, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=(*NEW_FINALS, 'initialize', 'package'), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--training-output-root', type=Path, default=Path('/root/autodl-tmp/outputs/CausalQ_DG'))
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets'))
    parser.add_argument('--pretrained-root', type=Path, default=Path('/root/autodl-tmp/pretrained'))
    args = parser.parse_args(argv)
    protected = [ROOT, args.data_root, args.pretrained_root]
    protected += [(args.training_output_root / r['checkpoint_relative_to_output_root']).parents[2]
                  for r in final_records().values()]
    if any(args.output_root.resolve().is_relative_to(p.resolve()) for p in protected):
        parser.error('Output must be outside code/data/weights/formal training runs')
    if _git('status', '--porcelain'):
        raise ValueError('Use clean exact checkout')
    args.output_root.mkdir(parents=True, exist_ok=True)
    initialize(args.output_root)
    if args.variant == 'package':
        package(args.output_root)
    elif args.variant != 'initialize':
        v = args.variant
        run_diagnostic(args, final_records()[v], final_source_report(v), final_config(v), phase=31,
                       contract_digest=verify_final_contract(),
                       config_path=ROOT / f'configs/diagnostic_20261020/formal/{v.lower()}_seed0.yaml')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
