"""Pinned acceptance of the complete Phase19 read-only component evidence."""

import hashlib
import json
import math
from pathlib import Path

from causalq.datasets.strict_source import MANIFEST_HASHES, load_membership
from tools.audit_centered_query import audit_centered, REPORT_SHA as CENTERED_SHA, SUMS_SHA as CENTERED_SUMS_SHA, TRACE_SHA, CHECKPOINT_SHA

ROOT = Path(__file__).resolve().parents[1]
REPORT_SHA = "e45bd7bf4cb1b334ad02938c7052668b977c0ada0d3dea6a3c566ba0c3fc010c"
SUMS_SHA = "902ac0da623b8ee7d569d0fd18bae222a909ee768c883c241b04b48521f6b6d4"
COMPONENTS_SHA = "fa447c563066344b86d658ccf4c9a35339595cc311625b0a55e5c84b73518e97"
PRODUCER_SHA = "f5ca5b5ac604bdae0d233d631001e95ee4b6ac36"
HEAD_SHA = "dd2b5d4fe9631ca4f893f2377c3ae661bd901d2fce9e538e0a70955b2bab3c45"


def finite_range(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def validate_anchor_report(report):
    expected = {"schema_version": 1, "phase": 19, "scope": "anchor_component_readonly8x2",
        "git_sha": PRODUCER_SHA, "git_status": "", "ok": True,
        "optimizer_updates_this_run": 0, "checkpoint_training_updates": 20,
        "formal_training_authorized": False, "target_data_accessed": False,
        "new_loss_enabled": False, "source_miou": None, "target_miou": None,
        "sample_count": 8, "view_count": 16, "backbone_forward_calls": 16,
        "model_rng_unchanged": True, "original_files_unchanged": True,
        "manifest_sha256": MANIFEST_HASHES, "train_count": 22469, "val_count": 2497,
        "source_validation": "gta5_holdout", "checkpoint_sha256": CHECKPOINT_SHA, "head_sha256": HEAD_SHA}
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError(f"Anchor report mismatch: {key}")
    if report["centered_audit"] != audit_centered(ROOT / "experiments/STRICT_DG_PHASE18"):
        raise ValueError("Anchor prior acceptance mismatch")
    if report["original_artifacts"] != {"centered_report.json": CENTERED_SHA, "SHA256SUMS": CENTERED_SUMS_SHA,
                                        "train.jsonl": TRACE_SHA, "state.pth": CHECKPOINT_SHA}:
        raise ValueError("Anchor original artifact identity mismatch")
    for key, value in {"torch": "2.7.0+cu126", "cuda": "12.6", "transformers": "4.56.1",
                       "gpu": "NVIDIA GeForce RTX 4090 D"}.items():
        if report["environment"][key] != value:
            raise ValueError(f"Anchor runtime mismatch: {key}")
    if not finite_range(report["original_ce_max_error"], 0, 1e-5):
        raise ValueError("Anchor original CE replay mismatch")
    if any(not finite_range(value, 0, 1e-6) for value in report["centering"].values()):
        raise ValueError("Anchor centering mismatch")
    _, ids = load_membership()
    rows = report["rows"]
    if [(row["source_id"], row["view"]) for row in rows] != [(i, v) for i in ids[:8] for v in ("original", "photometric")]:
        raise ValueError("Anchor row coverage mismatch")
    original = json.loads((ROOT / "experiments/STRICT_DG_PHASE18/centered_report.json").read_bytes())
    for index, row in enumerate(rows):
        if row["counterfactual_routing_exact"] is not True or not finite_range(row["decomposition_max_error"], 0, 1e-6):
            raise ValueError("Anchor algebra/routing mismatch")
        for key in ("ce", "full_abs_max", "anchor_abs_max", "residual_abs_max", "base_addition_roundoff_max"):
            if not finite_range(row[key], 0, float("inf")):
                raise ValueError(f"Anchor row nonfinite: {key}")
        digest = row["input_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Anchor input digest invalid")
        if index % 2 == 0 and abs(row["ce"] - original["holdout_compatibility"][index // 2]["ce"]) > 1e-5:
            raise ValueError("Anchor reference CE mismatch")
    reconstructed = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    if hashlib.sha256(reconstructed).hexdigest() != COMPONENTS_SHA:
        raise ValueError("Anchor embedded rows differ from original trace hash")
    for key in ("peak_allocated_gib", "peak_reserved_gib"):
        if not finite_range(report[key], 0, float("inf")):
            raise ValueError("Anchor memory invalid")
    return {"anchor_components_accepted": True, "report_sha256": REPORT_SHA,
            "checkpoint_sha256": CHECKPOINT_SHA, "components_sha256": COMPONENTS_SHA,
            "target_data_accessed": False, "formal_training_authorized": False}


def audit_anchor(directory):
    raw = (directory / "anchor_report.json").read_bytes()
    sums = (directory / "anchor_SHA256SUMS").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA or hashlib.sha256(sums).hexdigest() != SUMS_SHA:
        raise ValueError("Anchor report/checksum SHA256 mismatch")
    entries = dict((line.split("  ", 1)[1], line.split("  ", 1)[0]) for line in sums.decode("ascii").splitlines())
    if entries != {"anchor_report.json": REPORT_SHA, "components.jsonl": COMPONENTS_SHA}:
        raise ValueError("Anchor original checksum manifest mismatch")
    return validate_anchor_report(json.loads(raw))
