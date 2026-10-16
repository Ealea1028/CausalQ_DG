"""Read-only acceptance of the exact returned Phase-17 source-data evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from causalq.protocols.strict_dg import validate_strict_config
from tools.build_gta5_source_split import PAIR_SHA256
from tools.import_gta5_source_split import MANIFEST_SHA256


DATA_REPORT_SHA256 = "cf01d813f6125c42a8e6712e449bfdba72ce5f287222357c975780d7cad8fab3"
DATA_PRODUCER_SHA = "cb5207101915c97b23da82792608b2e2c24c123e"


def validate_report(report: dict) -> None:
    expected = {
        "schema_version": 1, "phase": 17, "scope": "gta5_strict_source_data_only",
        "git_sha": DATA_PRODUCER_SHA, "git_status": "", "ok": True,
        "training_authorized": False, "optimizer_updates": 0, "target_data_accessed": False,
        "manifest_sha256": MANIFEST_SHA256, "train_count": 22469, "val_count": 2497,
        "source_validation": "gta5_holdout", "pair_manifest_sha256": PAIR_SHA256,
    }
    for key, value in expected.items():
        if key not in report or type(report[key]) is not type(value) or report[key] != value:
            raise ValueError(f"Returned data report mismatch: {key}")
    validate_strict_config(report["config"])
    if report["environment"]["torch"] != "2.7.0+cu126":
        raise ValueError("Unexpected source-data runtime")
    for split, count in (("train", 22469), ("val", 2497)):
        scan = report[f"{split}_scan"]
        for key, value in {"ok": True, "scanned_label_count": count, "scan_scope": "all",
                           "invalid_train_ids": [], "unreadable_count": 0, "unreadable": [],
                           "geometry_mismatch_count": 0, "all_ignore_label_count": 0,
                           "unique_label_ids": list(range(19)) + [255]}.items():
            if type(scan.get(key)) is not type(value) or scan[key] != value:
                raise ValueError(f"Source {split} scan mismatch: {key}")
        if scan["shape_mismatch_count"] != scan["scale_equivalent_mismatch_count"]:
            raise ValueError("Unaccounted source geometry differences")
        if not 0 < scan["valid_fraction_min"] <= scan["valid_fraction_mean"] <= 1:
            raise ValueError("Invalid source valid-pixel statistics")
    prep = report["preprocessing"]
    for key, value in {"ok": True, "holdout_checked": 2497, "train_crop_photo_checked": 32,
                       "holdout_torch_rng_unchanged": True,
                       "holdout_shapes": [[560, 1024], [576, 1024]]}.items():
        if type(prep.get(key)) is not type(value) or prep[key] != value:
            raise ValueError(f"Source preprocessing mismatch: {key}")
    for key in ("holdout_valid_fraction_min", "photo_mean_absolute_change"):
        if not math.isfinite(prep[key]) or prep[key] <= 0:
            raise ValueError(f"Invalid preprocessing statistic: {key}")


def audit_report(path: Path) -> dict:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != DATA_REPORT_SHA256:
        raise ValueError("Saved source-data report SHA256 mismatch")
    report = json.loads(payload)
    validate_report(report)
    return {"source_data_accepted": True, "report_sha256": DATA_REPORT_SHA256,
            "producer_git_sha": DATA_PRODUCER_SHA, "train_count": 22469, "val_count": 2497,
            "scale_equivalent_pairs": report["train_scan"]["shape_mismatch_count"]
                                      + report["val_scan"]["shape_mismatch_count"],
            "formal_training_authorized": False, "target_data_accessed": False}
