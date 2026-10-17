import json
from pathlib import Path

import pytest

from tools.audit_strict_source_pilot import audit_pilot, validate_pilot

EVIDENCE = Path(__file__).resolve().parents[1] / "experiments/STRICT_DG_PHASE17"


def test_complete_pilot_report_and_manifest_pass_byte_and_semantic_audit(tmp_path):
    result = audit_pilot(EVIDENCE)
    assert result["source_pilot_accepted"]
    assert result["source_miou"] == pytest.approx(0.35724184600067665)
    for name in ("pilot_report.json", "pilot_SHA256SUMS"):
        (tmp_path / name).write_bytes((EVIDENCE / name).read_bytes())
    (tmp_path / "pilot_report.json").write_bytes((EVIDENCE / "pilot_report.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        audit_pilot(tmp_path)


@pytest.mark.parametrize("key,value", [("optimizer_updates", 501), ("target_data_accessed", True),
    ("checkpoint_selection", "best_target"), ("source_miou", float("nan")), ("selected_checkpoint", "prefix/state.pth")])
def test_pilot_rejects_changed_contract(key, value):
    report = json.loads((EVIDENCE / "pilot_report.json").read_bytes())
    sums = {line.split("  ")[1].removeprefix("./"): line.split("  ")[0]
            for line in (EVIDENCE / "pilot_SHA256SUMS").read_text().splitlines()}
    report[key] = value
    with pytest.raises(ValueError):
        validate_pilot(report, sums)
