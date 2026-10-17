"""Pin the independent local CSV recomputation and actual-source visual review."""

import hashlib
import json

from tools.audit_query_component_census import ROOT, audit_census, census_artifact_hashes

RAW_SHA = 'a94501521e071d4dffdfde685cc940add3a56fdd6e12f5ac434d45e66af8da79'
COMPLETION_SHA = 'b132365ad250eb8e5cf76b2d184601b9de57d6e9736d8310e3d04faa1a54168a'


def audit_completed_diagnostics(directory=ROOT / 'experiments/STRICT_DG_PHASE20'):
    audit_census(directory)
    raw = (directory / 'raw_review.json').read_bytes()
    certificate = (directory / 'completion_review.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != RAW_SHA or hashlib.sha256(certificate).hexdigest() != COMPLETION_SHA:
        raise ValueError('Local census review SHA256 differs')
    result, final = json.loads(raw), json.loads(certificate)
    report = json.loads((directory / 'census_report.json').read_bytes())
    expected = census_artifact_hashes(report)
    del expected['analysis_export.zip']
    if result['files_sha256'] != expected or result['effect_row_count'] != 142329 or result['responsibility_row_count'] != 189772:
        raise ValueError('Independent raw CSV coverage differs')
    if result['raw_csv_independently_reviewed'] is not True or final['raw_review_sha256'] != RAW_SHA:
        raise ValueError('Independent raw review missing')
    if any(final[k] is not True for k in ('actual_figures_visually_reviewed', 'priority3_engineering_complete',
            'source_geometry_alignment_accepted', 'shared_effect_scales_accepted', 'ignore255_regions_consistent')):
        raise ValueError('Source visual review missing')
    if any(final[k] is not False for k in ('formal_training_authorized', 'target_data_accessed',
            'raw_anchor_stability_hypothesis_supported_at_this_checkpoint', 'new_anchor_cqe_loss_authorized')):
        raise ValueError('Diagnostic review overclaims scientific/training authorization')
    return {'priority3_engineering_complete': True, 'raw_review_sha256': RAW_SHA,
            'completion_review_sha256': COMPLETION_SHA, 'raw_anchor_hypothesis_supported': False,
            'new_anchor_cqe_loss_authorized': False, 'target_data_accessed': False}
