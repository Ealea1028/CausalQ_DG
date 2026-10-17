"""Pinned read-only audit of the accepted full-source diagnostic report."""

import hashlib
import json
import math
from pathlib import Path

from causalq.datasets.strict_source import MANIFEST_HASHES
from tools.audit_strict_source_smoke import CHECKPOINT_SHA, BACKBONE_STATE_SHA

REPORT_SHA = "a70ddf23e11075b35c94e839c1cee69e2d2311d582a42992652008353ea0f29c"
TRACE_SHA = "96ee3ee2fce106b360833b68ce0c552b28cd4d41abb05be46edaf842364c63fc"
PRODUCER_SHA = "c7822fd78a43c4d3ca7796d068fed5b392fb8705"


def validate_evaluation(report):
    expected = {"schema_version": 1, "phase": 17, "ok": True,
                "scope": "full_source_holdout_20update_diagnostic",
                "git_sha": PRODUCER_SHA, "git_status": "", "optimizer_updates_this_run": 0,
                "checkpoint_training_updates": 20, "restored_iteration": 20,
                "formal_training_authorized": False, "diagnostic_only": True,
                "exact_resume_verified": False, "target_data_accessed": False, "target_miou": None,
                "checkpoint_selection": "immutable_preselected_smoke_final",
                "manifest_sha256": MANIFEST_HASHES, "train_count": 22469, "val_count": 2497,
                "sample_count": 2497, "seen_ids_sha256": MANIFEST_HASHES["gta5_source_val.txt"],
                "checkpoint_sha256": CHECKPOINT_SHA, "backbone_state_sha256": BACKBONE_STATE_SHA,
                "trace_sha256": TRACE_SHA, "confusion_counts_exact": True,
                "model_unchanged": True, "original_smoke_files_unchanged": True,
                "checkpoint_bytes_independently_audited": True, "evaluation_rng_unchanged": True,
                "compatibility_reference_count": 8, "compatibility_ce_max_error": 0.0}
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError(f"Source evaluation contract mismatch: {key}")
    audit = report["smoke_audit"]
    if not (audit["smoke_accepted"] is True and audit["checkpoint_bytes_independently_audited"] is True
            and audit["checkpoint_sha256"] == CHECKPOINT_SHA):
        raise ValueError("Original smoke audit is incomplete")
    matrix = report["confusion_matrix"]
    if len(matrix) != 19 or any(len(row) != 19 for row in matrix):
        raise ValueError("Expected 19 by 19 confusion matrix")
    if any(type(x) is not int or x < 0 for row in matrix for x in row):
        raise ValueError("Confusion counts must be nonnegative integers")
    if sum(map(sum, matrix)) != report["valid_pixel_count"]:
        raise ValueError("Confusion valid-pixel sum mismatch")
    ious = []
    for i in range(19):
        union = sum(matrix[i]) + sum(row[i] for row in matrix) - matrix[i][i]
        ious.append(matrix[i][i] / union if union else None)
    for actual, reference in zip(report["class_iou"], ious, strict=True):
        if ((actual is None) != (reference is None)
                or (reference is not None and (not math.isfinite(actual) or abs(actual - reference) > 1e-12))):
            raise ValueError("Per-class IoU mismatch")
    present = [x for x in ious if x is not None]
    miou = sum(present) / len(present)
    if report["valid_classes"] != len(present) or any(
            not math.isfinite(report[key]) or abs(report[key] - miou) > 1e-12
            for key in ("source_miou", "independent_miou")):
        raise ValueError("Source mIoU mismatch")
    for key in ("image_mean_ce", "pixel_weighted_ce", "peak_allocated_gib", "peak_reserved_gib"):
        if not math.isfinite(report[key]) or report[key] <= 0:
            raise ValueError(f"Invalid source summary: {key}")
    return {"source_evaluation_accepted": True, "report_sha256": REPORT_SHA,
            "sample_count": 2497, "source_miou": miou, "target_data_accessed": False,
            "formal_training_authorized": False}


def audit_evaluation(path: Path):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA:
        raise ValueError("Source evaluation report SHA256 mismatch")
    return validate_evaluation(json.loads(payload))
