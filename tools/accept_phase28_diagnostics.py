"""Accept returned source summaries without replaying data or whole-run transfer."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from causalq.protocols.diagnostic_20261020 import BASELINES, STUDY, baseline_records, contract_sha, preview_ids

PRODUCER = '1086bc36c2967bc9ff0f7480faad929b210d393f'


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accept(root):
    root = root.resolve()
    checked = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        rel = PurePosixPath(name)
        path = root.joinpath(*rel.parts).resolve()
        if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root) or name in checked:
            raise ValueError('Unsafe/duplicate returned checksum entry')
        if file_sha(path) != digest:
            raise ValueError('Returned file SHA256 differs: ' + name)
        checked[name] = digest
    if 'job_report.json' not in checked:
        raise ValueError('Missing checked job report')
    job = json.loads((root / 'job_report.json').read_bytes())
    if (job['ok'] is not True or job['phase'] != 28 or job['study'] != STUDY
            or job['git_sha'] != PRODUCER or job['target_data_accessed'] is not False
            or job['optimizer_updates_this_run'] != 0 or set(job['reports']) != set(BASELINES)
            or job['preview_ids'] != preview_ids()):
        raise ValueError('Phase28 source-only producer contract differs')
    compact = {}
    frozen = baseline_records()
    for variant in BASELINES:
        name = variant + '/summary.json'
        if name not in checked or variant + '/class_summary.csv' not in checked:
            raise ValueError('Missing checked variant results')
        summary = json.loads((root / name).read_bytes())
        expected_identity = job['reports'][variant]['identity']
        if (summary['ok'] is not True or summary['sample_count'] != 2497
                or summary['identity'] != expected_identity
                or summary['identity']['git_sha'] != PRODUCER
                or summary['identity']['variant'] != variant
                or summary['identity']['contract_sha256'] != contract_sha()
                or summary['identity']['checkpoint_sha256'] != frozen[variant]['checkpoint_sha256']
                or summary['checkpoint_training_updates'] != 40000
                or summary['optimizer_updates_this_run'] != 0 or summary['views'] != 5
                or summary['effect_geometry'] != 'longedge1024_aspect_pad16'
                or summary['gradient_geometry'] != 'deterministic_512_crop'
                or summary['target_data_accessed'] is not False or summary['model_rng_unchanged'] is not True):
            raise ValueError('Phase28 summary coverage/state differs')
        compact[variant] = {**job['reports'][variant], 'summary_sha256': checked[name],
                            'image_macro_statistics': [r for r in summary['image_macro_statistics']
                                if r['class_id'] == -1 and r['population'] == 'all_classes'],
                            'pixel_weighted_responsibility': summary['pixel_weighted_responsibility']}
    return {'schema_version': 1, 'phase': 28, 'accepted_date': '2026-10-23', 'accepted': True,
            'producer_git_sha': PRODUCER, 'files_verified': checked,
            'provided_archive_sha256': '4b01034eef7f4ddea00d368d23c8e1cb2d8bcd4c611ff9e08c7fa5a1ebb9d01b',
            'archive_bytes_available_locally': False, 'raw_per_image_records_available_locally': False,
            'local_scope': 'checksum/summary provenance and state; no independent raw-data recomputation',
            'new_training_gate': False, 'target_data_accessed': False, 'reports': compact}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    result = accept(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(f'Phase28 accepted: {len(result["files_verified"])} files, four models x2497 source images')
    return 0


if __name__ == '__main__':raise SystemExit(main())
