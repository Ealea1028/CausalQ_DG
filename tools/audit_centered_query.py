"""Pinned acceptance of the Phase18 centered-query engineering return."""

import hashlib
import json
import math
from pathlib import Path

from causalq.datasets.strict_source import MANIFEST_HASHES, load_membership
from causalq.protocols.centered_query import validate_centered_config

REPORT_SHA = "3a09ece34eeb17718a90b2d2f53141aacecdce710b6f4aa4be53c3799979926e"
SUMS_SHA = "19b8e0f134c910739885e0409a0e359e56671daca427d93e48a513064a359a26"
PRODUCER_SHA = "958392541519cdad5f80d628c18959176966c6c8"
CHECKPOINT_SHA = "9b31426b13aaecb458866c010fdcf385774c1d085f3de48334485fa0d9996bf9"
TRACE_SHA = "50b77fd5be30417e38eea3355b2978186e1b2d1b54fa3fca31501f4b33cfdf03"


def validate_centered_report(report):
    expected = {"schema_version": 1, "phase": 18, "scope": "m1_centered_query_engineering20",
        "git_sha": PRODUCER_SHA, "git_status": "", "seed": 0, "ok": True,
        "authorized_optimizer_updates": 20, "authorized_holdout_samples": 8,
        "optimizer_updates": 20, "initial_iteration": 0, "final_iteration": 20,
        "formal_training_authorized": False, "target_data_accessed": False,
        "source_miou": None, "target_miou": None, "exact_resume_verified": False,
        "roundtrip_scope": "new_head_same_frozen_backbone", "alpha_zero_error": 0.0,
        "model_roundtrip_exact": True, "optimizer_roundtrip_exact": True,
        "backbone_unchanged": True, "manifest_sha256": MANIFEST_HASHES}
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError(f"Centered report mismatch: {key}")
    validate_centered_config(report["config"])
    if report["metadata"]["centered_residuals"] is not True:
        raise ValueError("Centered checkpoint semantic flag missing")
    for section in ("initial_centering", "final_centering"):
        for key in ("residual_mean_max_error", "query_mean_anchor_max_error"):
            if not math.isfinite(report[section][key]) or not 0 <= report[section][key] <= 1e-6:
                raise ValueError("Centered algebra acceptance failed")
    for key, limit in (("prediction_roundtrip_max_error", 1e-5), ("residual_gradient_sum_max_error", 1e-6)):
        if not math.isfinite(report[key]) or not 0 <= report[key] <= limit:
            raise ValueError("Centered prediction/gradient acceptance failed")
    for prefix in ("decoder.", "query_bank.class_anchors", "query_bank.query_residuals",
                   "query_head.feature_projection.", "query_head.query_projection.", "query_head.alpha"):
        if not any(math.isfinite(value) and value > 0 for name, value in report["gradient_max"].items() if name.startswith(prefix)):
            raise ValueError("Centered trainable gradient group missing")
    _, ids = load_membership()
    refs = report["holdout_compatibility"]
    if len(refs) != 8 or [row["source_id"] for row in refs] != ids[:8]:
        raise ValueError("Centered holdout coverage mismatch")
    if any(not math.isfinite(row["ce"]) or row["ce"] < 0 for row in refs):
        raise ValueError("Centered compatibility CE invalid")
    return {"centered_query_accepted": True, "report_sha256": REPORT_SHA,
            "checkpoint_sha256": CHECKPOINT_SHA, "trace_sha256": TRACE_SHA,
            "target_data_accessed": False, "formal_training_authorized": False}


def audit_centered(directory: Path):
    raw = (directory / "centered_report.json").read_bytes()
    sums = (directory / "centered_SHA256SUMS").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA or hashlib.sha256(sums).hexdigest() != SUMS_SHA:
        raise ValueError("Centered report/checksum SHA256 mismatch")
    entries = dict((line.split("  ", 1)[1], line.split("  ", 1)[0]) for line in sums.decode("ascii").splitlines())
    if entries != {"centered_report.json": REPORT_SHA, "train.jsonl": TRACE_SHA, "state.pth": CHECKPOINT_SHA}:
        raise ValueError("Centered original checksum manifest mismatch")
    return validate_centered_report(json.loads(raw))
