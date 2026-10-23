"""Small returned-metadata fixtures; no real source images or checkpoints."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tools.accept_phase28_diagnostics import accept

ROOT = Path(__file__).resolve().parents[1]


def make_export(root, mutate=None):
    accepted = json.loads((ROOT / 'experiments/DIAGNOSTIC_20261020/phase28_acceptance.json').read_bytes())
    reports = {}
    files = []
    for variant, report in accepted['reports'].items():
        summary = deepcopy(report)
        summary.pop('summary_sha256')
        if mutate:mutate(variant, summary)
        directory = root / variant
        directory.mkdir()
        for name, value in [('summary.json', json.dumps(summary)), ('class_summary.csv', 'class_id\n')]:
            path = directory / name
            path.write_text(value)
            files.append(path)
        reports[variant] = {k: v for k, v in summary.items() if k not in ('image_macro_statistics', 'pixel_weighted_responsibility')}
    job = {'ok': True, 'phase': 28, 'study': 'diagnostic_20261020', 'git_sha': accepted['producer_git_sha'],
           'target_data_accessed': False, 'optimizer_updates_this_run': 0,
           'reports': reports, 'preview_ids': ['18734', '12488', '08744']}
    path = root / 'job_report.json'
    path.write_text(json.dumps(job))
    files.append(path)
    (root / 'SHA256SUMS').write_text(''.join(
        f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n' for p in files))


def test_returned_summary_acceptance_never_requires_remote_raw_records(tmp_path):
    make_export(tmp_path)
    result = accept(tmp_path)
    assert result['accepted'] and not result['raw_per_image_records_available_locally']
    assert not result['new_training_gate']
    (tmp_path / 'U0/summary.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256'):accept(tmp_path)


@pytest.mark.parametrize('key,value', [('sample_count', 150), ('optimizer_updates_this_run', 1),
    ('model_rng_unchanged', False), ('target_data_accessed', True), ('views', 2)])
def test_returned_summary_rejects_wrong_coverage_or_state(tmp_path, key, value):
    make_export(tmp_path, lambda variant, s: s.update({key: value}) if variant == 'C0' else None)
    with pytest.raises(ValueError, match='coverage/state'):accept(tmp_path)
