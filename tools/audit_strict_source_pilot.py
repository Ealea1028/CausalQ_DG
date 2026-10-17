"""Pin and audit the accepted 500-update source runner integration evidence."""

import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from causalq.datasets.strict_source import MANIFEST_HASHES

REPORT_SHA = "d5b4747b7200d40722728cd797da9f736dedb15d9310c3a293aa8a51b98ffd1b"
SUMS_SHA = "8ea73a530d6e43e78b1dea19795b898070fd0259f69f3dc86956730eeb3eb615"
PRODUCER_SHA = "482ef2663f3ecf0f2f5d5545710aefd0fd2a9520"


def validate_pilot(report, sums):
    expected = {"schema_version": 1, "phase": 17, "scope": "source_runner_pilot500",
                "stage": "summary", "git_sha": PRODUCER_SHA, "git_status": "", "ok": True,
                "formal_training_authorized": False, "target_data_accessed": False,
                "target_miou": None, "optimizer_updates": 500, "resumed_at": 250,
                "checkpoint_selection": "fixed_final", "selected_checkpoint": "finish/state.pth",
                "pilot_is_formal_m0": False, "exact_resume_previously_accepted": True,
                "manifest_sha256": MANIFEST_HASHES}
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError(f"Pilot contract mismatch: {key}")
    workers = report["workers"]
    if set(workers) != {"prefix", "finish"} or workers["prefix"]["process_id"] == workers["finish"]["process_id"]:
        raise ValueError("Pilot independent worker coverage mismatch")
    for stage, worker in workers.items():
        bounds = (0, 250) if stage == "prefix" else (250, 500)
        if (worker["ok"] is not True or worker["git_sha"] != PRODUCER_SHA or worker["git_status"]
                or worker["target_data_accessed"] is not False or worker["formal_training_authorized"] is not False
                or (worker["initial_iteration"], worker["final_iteration"]) != bounds
                or worker["optimizer_updates"] != 250 or worker["backbone_unchanged"] is not True):
            raise ValueError("Pilot worker contract mismatch")
        if worker["metadata"] != workers["prefix"]["metadata"]:
            raise ValueError("Pilot worker metadata differs")
        encoded = (json.dumps(worker, indent=2) + "\n").encode()
        if hashlib.sha256(encoded).hexdigest() != sums[f"{stage}/worker_report.json"]:
            raise ValueError("Nested worker bytes do not reproduce remote checksum")
        for name, digest in worker["files_sha256"].items():
            if sums[f"{stage}/{name}"] != digest:
                raise ValueError("Pilot artifact digest provenance mismatch")
    evaluation = report["evaluation"]
    if evaluation != workers["finish"]["evaluation"] or evaluation["sample_count"] != 2497:
        raise ValueError("Pilot full source evaluation mismatch")
    if evaluation["seen_ids_sha256"] != MANIFEST_HASHES["gta5_source_val.txt"] or any(
            evaluation[key] is not True for key in ("training_state_unchanged", "model_mode_restored",
                                                   "evaluation_rng_unchanged", "confusion_counts_exact")):
        raise ValueError("Pilot validation coverage/state isolation mismatch")
    matrix = evaluation["confusion_matrix"]
    if len(matrix) != 19 or any(len(row) != 19 for row in matrix) or any(type(x) is not int or x < 0 for row in matrix for x in row):
        raise ValueError("Invalid pilot integer confusion matrix")
    if sum(map(sum, matrix)) != evaluation["valid_pixel_count"]:
        raise ValueError("Pilot valid-pixel sum mismatch")
    ious = []
    for i in range(19):
        union = sum(matrix[i]) + sum(row[i] for row in matrix) - matrix[i][i]
        ious.append(matrix[i][i] / union if union else None)
    for actual, expected_iou in zip(evaluation["class_iou"], ious, strict=True):
        if ((actual is None) != (expected_iou is None)
                or (actual is not None and (not math.isfinite(actual) or abs(actual - expected_iou) > 1e-12))):
            raise ValueError("Pilot class IoU mismatch")
    mean = sum(x for x in ious if x is not None) / sum(x is not None for x in ious)
    if any(not math.isfinite(value) or abs(value - mean) > 1e-12 for value in
           (report["source_miou"], evaluation["source_miou"], evaluation["independent_miou"])):
        raise ValueError("Pilot mIoU mismatch")
    if sums[report["selected_checkpoint"]] != workers["finish"]["files_sha256"]["state.pth"]:
        raise ValueError("Pilot fixed-final checkpoint mismatch")
    return {"source_pilot_accepted": True, "report_sha256": REPORT_SHA, "checksums_sha256": SUMS_SHA,
            "source_miou": mean, "target_data_accessed": False, "formal_training_authorized": False}


def audit_pilot(directory: Path):
    raw = (directory / "pilot_report.json").read_bytes()
    checksums = (directory / "pilot_SHA256SUMS").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA or hashlib.sha256(checksums).hexdigest() != SUMS_SHA:
        raise ValueError("Pilot report/checksum SHA256 mismatch")
    sums = {}
    for line in checksums.decode("ascii").splitlines():
        digest, name = line.split("  ", 1)
        name = name.removeprefix("./")
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or name in sums:
            raise ValueError("Invalid or duplicate evidence checksum path")
        sums[name] = digest
    if sums["pilot_report.json"] != REPORT_SHA:
        raise ValueError("Report digest missing from original checksum manifest")
    return validate_pilot(json.loads(raw), sums)
