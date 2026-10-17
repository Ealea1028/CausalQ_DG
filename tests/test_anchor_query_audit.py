from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.audit_anchor_query_effect import audit_anchor, validate_anchor_report

DIRECTORY = Path(__file__).resolve().parents[1] / "experiments/STRICT_DG_PHASE19"


def test_complete_anchor_evidence_acceptance():
    assert audit_anchor(DIRECTORY)["anchor_components_accepted"] is True


@pytest.mark.parametrize("key,value", [("ok", 1), ("git_sha", "0" * 40), ("optimizer_updates_this_run", 1),
    ("target_data_accessed", True), ("source_miou", .5), ("backbone_forward_calls", 32)])
def test_anchor_contract_rejects_changed_claims(key, value):
    report = json.loads((DIRECTORY / "anchor_report.json").read_bytes())
    report[key] = value
    with pytest.raises(ValueError):
        validate_anchor_report(report)


@pytest.mark.parametrize("change", ["coverage", "routing", "nan", "trace", "reference"])
def test_anchor_rows_reject_incomplete_or_changed_evidence(change):
    report = json.loads((DIRECTORY / "anchor_report.json").read_bytes())
    if change == "coverage": report["rows"].pop()
    elif change == "routing": report["rows"][0]["counterfactual_routing_exact"] = False
    elif change == "nan": report["rows"][0]["ce"] = float("nan")
    elif change == "trace": report["rows"][1]["ce"] += .001
    else: report["rows"][0]["ce"] += .001
    with pytest.raises(ValueError):
        validate_anchor_report(report)


def test_anchor_bytes_reject_mutation(tmp_path):
    for name in ("anchor_report.json", "anchor_SHA256SUMS"):
        (tmp_path / name).write_bytes((DIRECTORY / name).read_bytes())
    path = tmp_path / "anchor_report.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        audit_anchor(tmp_path)
