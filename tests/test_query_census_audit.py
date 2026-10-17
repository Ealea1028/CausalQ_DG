import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from tools.audit_query_component_census import audit_census, validate_census_report, verify_archive_bytes

DIRECTORY = Path(__file__).resolve().parents[1] / 'experiments/STRICT_DG_PHASE20'


def test_complete_census_report_verifies_without_claiming_raw_review():
    result = audit_census(DIRECTORY)
    assert result['census_report_verified']
    assert not result['raw_csv_independently_reviewed'] and not result['actual_figures_visually_reviewed']


@pytest.mark.parametrize('change', ['updates', 'target', 'count', 'selection', 'figure', 'replay', 'aggregate', 'entropy'])
def test_census_rejects_changed_metadata(change):
    r = json.loads((DIRECTORY / 'census_report.json').read_bytes())
    if change == 'updates': r['optimizer_updates_this_run'] = 1
    elif change == 'target': r['target_data_accessed'] = 0
    elif change == 'count': r['backbone_forward_calls'] = 4993
    elif change == 'selection': r['selection']['ids'].pop()
    elif change == 'figure': r['figures'].pop()
    elif change == 'replay': r['samples'][0]['original_input_sha256'] = '0' * 64
    elif change == 'aggregate': r['effect_summary']['anchor']['original_mean_abs']['mean'] *= 2
    else: r['responsibility_summary']['original']['usage_entropy_nats']['mean'] = 1.
    with pytest.raises(ValueError): validate_census_report(r)


def test_archive_stream_checks_content_and_coverage_without_extracting(tmp_path):
    path = tmp_path / 'export.zip'
    payloads = {'result.csv': b'class_id,value\n0,1\n', 'sample_index.jsonl': b'{}\n'}
    with zipfile.ZipFile(path, 'w') as archive:
        for name, value in payloads.items(): archive.writestr(name, value)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
    assert verify_archive_bytes(path, digest, expected)['file_count'] == 2
    assert not (tmp_path / 'result.csv').exists()
    with pytest.raises(ValueError, match='coverage'): verify_archive_bytes(path, digest, {})
    expected['result.csv'] = '0' * 64
    with pytest.raises(ValueError, match='member'): verify_archive_bytes(path, digest, expected)
    with pytest.raises(ValueError, match='archive SHA256'): verify_archive_bytes(path, '0' * 64, expected)
