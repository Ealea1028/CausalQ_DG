"""Accept Phase31 summaries without replaying source images or remote checkpoints."""

import argparse
from datetime import date
import json
from pathlib import Path, PurePosixPath

from causalq.protocols.diagnostic_20261020 import STUDY
from causalq.protocols.diagnostic_final_source import (
    NEW_FINALS, final_records, final_source_report, final_contract, verify_final_contract,
)
from tools.train_strict_source import file_sha

PRODUCER = '69432981ca2cc5fa6885f4b5fbdc20ef9f3370df'


def accept(root):
    root = Path(root).resolve();checked = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1);rel = PurePosixPath(name)
        path = root.joinpath(*rel.parts).resolve()
        if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root) or name in checked:
            raise ValueError('Unsafe/duplicate returned checksum entry')
        if file_sha(path) != digest:raise ValueError('Returned SHA256 differs: ' + name)
        checked[name] = digest
    expected = {'job_report.json', 'run_manifest.json'}
    expected.update(f'{v}/{n}' for v in NEW_FINALS for n in ('summary.json', 'class_summary.csv'))
    if set(checked) != expected:raise ValueError('Missing/unexpected source export files')
    job = json.loads((root / 'job_report.json').read_bytes())
    manifest = json.loads((root / 'run_manifest.json').read_bytes())
    digest = verify_final_contract()
    if (job['ok'] is not True or job['phase'] != 31 or job['study'] != STUDY or job['git_sha'] != PRODUCER
            or job['contract_sha256'] != digest or job['target_data_accessed'] is not False
            or job['optimizer_updates_this_run'] != 0 or set(job['reports']) != set(NEW_FINALS)
            or manifest['git_sha'] != PRODUCER or manifest['phase'] != 31
            or manifest['contract_sha256'] != digest or manifest['contract'] != final_contract()):
        raise ValueError('Phase31 producer/source-only contract differs')
    reports = {};frozen = final_records()
    for v in NEW_FINALS:
        s = json.loads((root / v / 'summary.json').read_bytes())
        identity = {'study': STUDY, 'phase': 31, 'variant': v, 'git_sha': PRODUCER,
                    'contract_sha256': digest, 'checkpoint_sha256': frozen[v]['checkpoint_sha256'],
                    'original_config_sha256': frozen[v]['config_sha256'],
                    'diagnostic_config_sha256': frozen[v]['config_sha256']}
        compact = {k: val for k, val in s.items() if k not in ('image_macro_statistics', 'pixel_weighted_responsibility')}
        if (s['ok'] is not True or s['identity'] != identity or compact != job['reports'][v]
                or s['sample_count'] != 2497 or s['views'] != 5 or s['checkpoint_training_updates'] != 40000
                or s['training_producer_git_sha'] != frozen[v]['producer_git_sha']
                or s['environment'] != final_source_report(v)['metadata']['environment']
                or s['optimizer_updates_this_run'] != 0 or s['target_data_accessed'] is not False
                or s['model_rng_unchanged'] is not True or s['new_training_gate'] is not False
                or s['effect_geometry'] != 'longedge1024_aspect_pad16'
                or s['gradient_geometry'] != 'deterministic_512_crop'):
            raise ValueError('Phase31 identity/coverage/state differs: ' + v)
        reports[v] = {**compact, 'summary_sha256': checked[v+'/summary.json'],
                      'image_macro_statistics': [r for r in s['image_macro_statistics'] if r['class_id'] == -1],
                      'pixel_weighted_responsibility': s['pixel_weighted_responsibility']}
    return {'schema_version': 1, 'phase': 31, 'accepted': True, 'accepted_date': date.today().isoformat(),
            'producer_git_sha': PRODUCER, 'files_verified': checked,
            'provided_archive_sha256': '92d75e96dcf3e67214b69f81bb780960e487bba8da6538d28fd106d3521fb16f',
            'archive_bytes_available_locally': False, 'raw_per_image_records_available_locally': False,
            'checkpoint_bytes_available_locally': False,
            'local_scope': 'hashes, source summary provenance/coverage/state; no independent per-image recomputation',
            'target_data_accessed': False, 'optimizer_updates_this_run': 0, 'reports': reports}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv);result = accept(args.input)
    args.output.write_bytes((json.dumps(result, indent=2, allow_nan=False)+'\n').encode())
    print(f'Phase31 accepted: {len(result["files_verified"])} files; six models x2497 images; zero updates')
    return 0


if __name__ == '__main__':raise SystemExit(main())
