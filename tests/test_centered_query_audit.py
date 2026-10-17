from pathlib import Path
import json

import pytest

from tools.audit_centered_query import audit_centered, validate_centered_report

EVIDENCE = Path(__file__).resolve().parents[1] / "experiments/STRICT_DG_PHASE18"


def test_centered_return_passes_hash_and_semantic_audit(tmp_path):
    assert audit_centered(EVIDENCE)["centered_query_accepted"]
    for name in ("centered_report.json", "centered_SHA256SUMS"):
        (tmp_path / name).write_bytes((EVIDENCE / name).read_bytes())
    (tmp_path / "centered_report.json").write_bytes((EVIDENCE / "centered_report.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        audit_centered(tmp_path)


@pytest.mark.parametrize("key,value", [("target_data_accessed", True), ("optimizer_updates", 21),
    ("prediction_roundtrip_max_error", float("nan")), ("residual_gradient_sum_max_error", 0.01)])
def test_centered_audit_rejects_invalid_report(key, value):
    report = json.loads((EVIDENCE / "centered_report.json").read_bytes())
    report[key] = value
    with pytest.raises(ValueError):
        validate_centered_report(report)
