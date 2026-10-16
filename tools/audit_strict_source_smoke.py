"""Read-only verification of immutable Phase-17 GPU smoke evidence."""

import hashlib
import json
import math
from pathlib import Path
import re

from causalq.datasets.strict_source import load_membership, MANIFEST_HASHES
from causalq.protocols.strict_dg import validate_strict_config
from tools.audit_strict_source_data import DATA_REPORT_SHA256

SMOKE_PRODUCER_SHA = "b16236c9989cc85b802b85a8abacde56deca9db7"
SMOKE_REPORT_SHA = "82dbbccd90e13720103a47a265f8b60afff4b27e72f72a9b78934bb510e7c54b"
SMOKE_TRACE_SHA = "42a4319b2ea02d47984faff4ad890722fc75cdb0952f62fdae0230612e118c34"
CHECKPOINT_SHA = "701e0ce24d6c517baacd39d42d762b5cb1f9d2d1df5dd3465cf746642e62b383"
BACKBONE_STATE_SHA = "24e1950a1ad87645cf6f46532a80594b5102af3644ccb55062e689a5b16f71ad"


def validate_smoke(report: dict, rows: list[dict]) -> dict:
    expected = {"schema_version": 1, "phase": 17, "scope": "m0_strict_source_smoke_20_updates",
                "seed": 0, "git_sha": SMOKE_PRODUCER_SHA, "git_status": "", "ok": True,
                "formal_training_authorized": False, "target_data_accessed": False,
                "authorized_optimizer_updates": 20, "authorized_holdout_samples": 8,
                "optimizer_updates": 20, "alpha_zero_error": 0.0,
                "model_roundtrip_exact": True, "optimizer_roundtrip_exact": True,
                "prediction_roundtrip_max_error": 0.0, "backbone_unchanged": True,
                "backbone_state_sha256": BACKBONE_STATE_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
                "checkpoint_bytes": 45612179, "trace_sha256": SMOKE_TRACE_SHA,
                "schedule_horizon": 40000, "sample_workers": 0, "amp": True,
                "checkpoint_policy": "fixed_final_only_no_selection", "exact_resume_verified": False,
                "manifest_sha256": MANIFEST_HASHES, "train_count": 22469, "val_count": 2497,
                "source_miou": None, "target_miou": None}
    for key, value in expected.items():
        if key not in report or type(report[key]) is not type(value) or report[key] != value:
            raise ValueError(f"Smoke report mismatch: {key}")
    validate_strict_config(report["config"])
    if report["source_data_audit"]["report_sha256"] != DATA_REPORT_SHA256:
        raise ValueError("Smoke source-data provenance mismatch")
    if (report["pretrained_checkpoint_sha256"] != report["config"]["model"]["weights_sha256"]
            or report["environment"]["torch"] != "2.7.0+cu126"
            or report["environment"]["transformers"] != "4.56.1"):
        raise ValueError("Smoke runtime/weight identity mismatch")
    train, val = load_membership()
    if len(rows) != 20 or [row["update"] for row in rows] != list(range(1, 21)):
        raise ValueError("Smoke trace must contain exactly 20 contiguous updates")
    ids = [row["source_id"] for row in rows]
    if len(set(ids)) != 20 or not set(ids) <= set(train) or report["source_ids"] != ids:
        raise ValueError("Smoke train membership/coverage mismatch")
    maxima = {}
    for row in rows:
        step = row["update"]
        for key in ("loss_original", "loss_photometric", "objective", "gradient_norm_pre_clip", "alpha"):
            if not math.isfinite(row[key]):
                raise ValueError(f"Non-finite smoke trace field: {key}")
        if row["objective"] != row["loss_original"] + row["loss_photometric"]:
            raise ValueError("Smoke objective does not reproduce the two-view sum")
        for key, offset in (("lr_used", step - 1), ("next_lr", step)):
            if not math.isclose(row[key], 1e-4 * (1 - offset / 40000) ** 0.9,
                                rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("Smoke schedule horizon mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", row["input_sha256"]):
            raise ValueError("Malformed source input fingerprint")
        for name, value in row["gradients"].items():
            if not math.isfinite(value) or value < 0 or name.startswith("backbone."):
                raise ValueError("Invalid/frozen-backbone gradient in trace")
            maxima[name] = max(maxima.get(name, 0), value)
    if maxima != report["gradient_max"]:
        raise ValueError("Gradient summary mismatch")
    for prefix in ("decoder.", "query_bank.class_anchors", "query_bank.query_residuals",
                   "query_head.feature_projection.", "query_head.query_projection.", "query_head.alpha"):
        if not any(value > 0 for name, value in maxima.items() if name.startswith(prefix)):
            raise ValueError(f"Missing training gradient group: {prefix}")
    for name, value in rows[0]["gradients"].items():
        if (name.startswith("query_bank.") or name.startswith("query_head.")) and name != "query_head.alpha":
            if value != 0:
                raise ValueError("First-step query gradients disagree with zero alpha")
    compat = report["holdout_compatibility"]
    if len(compat) != 8 or [row["source_id"] for row in compat] != val[:8]:
        raise ValueError("Smoke holdout membership mismatch")
    if any(not math.isfinite(row["ce"]) or row["ce"] < 0
           or row["shape"] not in ([1, 560, 1024], [1, 576, 1024]) for row in compat):
        raise ValueError("Invalid source holdout compatibility record")
    if report["initial_trainable_state_sha256"] == report["final_trainable_state_sha256"]:
        raise ValueError("No trainable-state change")
    if not 0 < report["peak_allocated_gib"] <= report["peak_reserved_gib"] < 24:
        raise ValueError("Invalid/out-of-budget smoke memory")
    return {"smoke_accepted": True, "optimizer_updates": 20,
            "checkpoint_sha256": CHECKPOINT_SHA, "backbone_unchanged": True,
            "first5_objective_mean": sum(row["objective"] for row in rows[:5]) / 5,
            "last5_objective_mean": sum(row["objective"] for row in rows[-5:]) / 5,
            "peak_reserved_gib": report["peak_reserved_gib"],
            "formal_training_authorized": False, "target_data_accessed": False,
            "checkpoint_bytes_independently_audited": False}


def audit_smoke(directory: Path) -> dict:
    report_bytes = (directory / "smoke_report.json").read_bytes()
    trace_bytes = (directory / "train.jsonl").read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != SMOKE_REPORT_SHA:
        raise ValueError("Saved smoke report SHA256 mismatch")
    if hashlib.sha256(trace_bytes).hexdigest() != SMOKE_TRACE_SHA:
        raise ValueError("Saved smoke trace SHA256 mismatch")
    return validate_smoke(json.loads(report_bytes),
                          [json.loads(line) for line in trace_bytes.decode().splitlines()])
