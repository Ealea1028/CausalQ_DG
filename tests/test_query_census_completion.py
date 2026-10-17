from pathlib import Path

import pytest

from tools.audit_query_census_completion import audit_completed_diagnostics
from tools.review_query_census import compare_summary, optional_number, summary

ROOT = Path(__file__).resolve().parents[1]


def test_independent_csv_and_visual_review_acceptance():
    result = audit_completed_diagnostics()
    assert result['priority3_engineering_complete']
    assert not result['raw_anchor_hypothesis_supported'] and not result['new_anchor_cqe_loss_authorized']


def test_review_pins_reject_modified_certificate(tmp_path):
    original = ROOT / 'experiments/STRICT_DG_PHASE20'
    for name in ('census_report.json', 'census_SHA256SUMS', 'raw_review.json', 'completion_review.json'):
        (tmp_path / name).write_bytes((original / name).read_bytes())
    path = tmp_path / 'completion_review.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='SHA256'): audit_completed_diagnostics(tmp_path)


def test_independent_reduction_and_missing_denominators():
    assert summary([1e16, 1., -1e16])['mean'] == 1 / 3
    assert optional_number('') is None
    with pytest.raises(ValueError): optional_number('nan')
    with pytest.raises(ValueError): compare_summary([1., 2.], {'mean': 1.5, 'defined_count': 3})
