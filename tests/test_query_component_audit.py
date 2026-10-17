import json
from pathlib import Path
import pytest
from tools.audit_query_component_diagnostics import audit_components, validate_component_report

DIRECTORY = Path(__file__).resolve().parents[1] / 'experiments/STRICT_DG_PHASE20'


def test_complete_component_evidence_acceptance():
    assert audit_components(DIRECTORY)['component_diagnostics_accepted']


@pytest.mark.parametrize('change', ['updates', 'target', 'sample', 'seed', 'replay', 'nan', 'aggregate', 'entropy', 'csv'])
def test_component_evidence_rejects_changed_claims(change):
    r = json.loads((DIRECTORY / 'component_effect_report.json').read_bytes())
    if change == 'updates': r['optimizer_updates_this_run'] = 1
    elif change == 'target': r['target_data_accessed'] = 0
    elif change == 'sample': r['samples'].pop()
    elif change == 'seed': r['samples'][0]['intervention_seed'] += 1
    elif change == 'replay': r['samples'][0]['appearance_replay_exact'] = False
    elif change == 'nan': r['samples'][0]['decomposition_max_error'] = float('nan')
    elif change == 'aggregate': r['effect_summary']['anchor']['original_mean_abs']['mean'] += .001
    elif change == 'entropy': r['responsibility_summary']['original']['usage_entropy_nats']['mean'] = 1.
    else: r['parameter_rows'][0]['anchor_norm'] += 1
    with pytest.raises(ValueError): validate_component_report(r)


def test_component_report_bytes_are_pinned(tmp_path):
    for name in ('component_effect_report.json', 'component_SHA256SUMS'):
        (tmp_path / name).write_bytes((DIRECTORY / name).read_bytes())
    path = tmp_path / 'component_SHA256SUMS'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='SHA256'): audit_components(tmp_path)
