"""Read-only audit of the completed Phase 16 source-only REIN baseline."""

import argparse
import json
import math
from pathlib import Path

from tools.audit_rein_pilot import digest, finite, require
from tools.check_rein_backbone import WEIGHT_SHA
from tools.check_rein_runtime import EXPECTED_VERSIONS
from tools.rein_schedule_smoke import DATA_SHA, scheduled_lr

PRODUCER = "d6fc52c5af9dcf0d6f57218b8b448c7df7a74ea4"
SOURCE_SIZE = 24966
UPDATES = 40000
ACCUMULATION = 4
MICROBATCHES = UPDATES * ACCUMULATION


def audit_records(records, *, updates=UPDATES, accumulation=ACCUMULATION,
                  source_size=SOURCE_SIZE):
    expected_count = updates * accumulation
    require(len(records) == expected_count, "Training record count mismatch")
    for position, record in enumerate(records, 1):
        require(record.get("iteration") == position, "Noncontiguous trace")
        index = record.get("dataset_index")
        require(type(index) is int and 0 <= index < source_size, "Invalid source index")
        require(record.get("optimizer_update") is (position % accumulation == 0),
                "Accumulation boundary mismatch")
        require(all(finite(record.get(key)) for key in ("loss", "gradient_norm", "lr")),
                "Nonfinite training value")
        require(record["loss"] > 0 and record["gradient_norm"] > 0,
                "Nonpositive loss or gradient norm")
        require(abs(record["lr"] - scheduled_lr(position // accumulation)) <= 1e-10,
                "Optimizer-update LR mismatch")
    indices = [record["dataset_index"] for record in records]
    complete_epochs, remainder = divmod(expected_count, source_size)
    expected = set(range(source_size))
    for epoch in range(complete_epochs):
        start = epoch * source_size
        require(set(indices[start:start + source_size]) == expected,
                "Source epoch is not a complete permutation")
    require(len(set(indices[complete_epochs * source_size:])) == remainder,
            "Partial source epoch repeats indices")
    return dict(complete_source_epochs=complete_epochs, partial_epoch_samples=remainder)


def audit_validation(validation):
    require(validation.get("sample_count") == 500, "Incomplete target coverage")
    samples = validation.get("samples")
    require(isinstance(samples, list) and [sample.get("index") for sample in samples] == list(range(500)),
            "Target sample order mismatch")
    require(all(sample.get("prediction_shape") == [1024, 2048]
                and sample.get("gt_shape") == [1024, 2048]
                and type(sample.get("valid_pixel_count")) is int
                and sample["valid_pixel_count"] > 0 for sample in samples),
            "Target geometry or pixel coverage mismatch")
    matrix = validation.get("confusion_matrix")
    require(isinstance(matrix, list) and len(matrix) == 19
            and all(isinstance(row, list) and len(row) == 19 for row in matrix),
            "Confusion matrix shape mismatch")
    require(all(type(value) is int and value >= 0 for row in matrix for value in row),
            "Invalid confusion matrix count")
    total = sum(map(sum, matrix))
    require(total == validation.get("total_valid_pixels")
            == sum(sample["valid_pixel_count"] for sample in samples),
            "Target valid-pixel accounting mismatch")
    calculated = []
    for class_id in range(19):
        union = sum(matrix[class_id]) + sum(row[class_id] for row in matrix) - matrix[class_id][class_id]
        calculated.append(matrix[class_id][class_id] / union if union else None)
    saved = validation.get("class_iou")
    require(isinstance(saved, list) and len(saved) == 19, "Class metric count mismatch")
    for expected, actual in zip(calculated, saved):
        require(actual is None if expected is None
                else finite(actual) and abs(actual - expected) < 1e-10,
                "Class IoU mismatch")
    present = [value for value in calculated if value is not None]
    require(bool(present) and len(present) == validation.get("valid_classes"),
            "Valid class count mismatch")
    miou = sum(present) / len(present)
    require(finite(validation.get("diagnostic_miou"))
            and abs(validation["diagnostic_miou"] - miou) < 1e-10,
            "Independent mIoU mismatch")
    require(finite(validation.get("miou")) and abs(validation["miou"] - miou) < 1e-10,
            "Final mIoU alias mismatch")
    official = validation.get("official_diagnostic_metrics")
    require(isinstance(official, dict) and all(finite(value) for value in official.values()),
            "Nonfinite official metric")
    require(abs(official["mIoU"] / 100 - miou) <= 5.1e-5,
            "Official and independent mIoU mismatch")
    require(validation.get("iteration") == UPDATES, "Final evaluation iteration mismatch")
    return miou


def audit(report, records):
    expected = dict(ok=True, stage="complete", phase=16, git_sha=PRODUCER, seed=0,
                    optimizer_updates=UPDATES, microbatches=MICROBATCHES,
                    accumulation=ACCUMULATION, training_record_count=MICROBATCHES,
                    schedule_horizon_optimizer_updates=UPDATES,
                    target_labels_optimized=False, formal_training_authorized=True,
                    initialization="fresh_seed0_not_pilot_resume", dtype="float32",
                    pretrained_sha256=WEIGHT_SHA, data_report_sha256=DATA_SHA,
                    final_evaluation_samples=500, checkpoint_interval_updates=1000,
                    checkpoint_selection="fixed_final_update_no_target_selection",
                    exact_resume_verified=False, source="gta5",
                    validation_dataset="cityscapes_val", backbone="dinov2_vitl14_rein_patch16",
                    decoder="Mask2Former", physical_batch_size=1, num_workers=0,
                    train_crop=[512, 512], target_input=[1024, 512],
                    target_gt=[2048, 1024], slide_crop=[512, 512],
                    slide_stride=[341, 341], cqe_enabled=False,
                    trainable_parameters=23569877, retained_checkpoint_count=2,
                    sampler_cursor=10204, sampler_epoch=6)
    for key, value in expected.items():
        require(type(report.get(key)) is type(value) and report[key] == value,
                "Unexpected " + key)
    require(report.get("purpose") == "formal_source_only_rein_baseline_not_cqe_or_exact_paper_reproduction",
            "Unexpected experiment purpose")
    require(report.get("versions") == EXPECTED_VERSIONS, "Runtime pins changed")
    record_summary = audit_records(records)
    for key, subset in (("first_20_loss_mean", records[:20]),
                        ("last_20_loss_mean", records[-20:])):
        require(finite(report.get(key))
                and abs(report[key] - sum(record["loss"] for record in subset) / 20) < 1e-6,
                "Loss summary mismatch")
    require(finite(report.get("elapsed_seconds")) and report["elapsed_seconds"] > 0,
            "Invalid elapsed time")
    require(finite(report.get("peak_reserved_gib")) and report["peak_reserved_gib"] > 0,
            "Invalid memory measurement")
    checkpoints = report.get("checkpoints")
    require(isinstance(checkpoints, list) and len(checkpoints) == 40,
            "Checkpoint history length mismatch")
    require([entry.get("optimizer_updates") for entry in checkpoints]
            == list(range(1000, UPDATES + 1, 1000)), "Checkpoint update history mismatch")
    require(all(type(entry.get("bytes")) is int and 0 < entry["bytes"] <= 2**30
                and isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64
                for entry in checkpoints), "Checkpoint history integrity fields missing")
    miou = audit_validation(report.get("validation"))
    return dict(source40k_saved_audit_ok=True, training_git_sha=PRODUCER,
                optimizer_updates=UPDATES, microbatches=MICROBATCHES,
                final_miou=miou, first_20_loss_mean=report["first_20_loss_mean"],
                last_20_loss_mean=report["last_20_loss_mean"],
                peak_reserved_gib=report["peak_reserved_gib"],
                checkpoint_history_count=len(checkpoints), **record_summary,
                cqe_enabled=False, target_labels_optimized=False)


def audit_files(report_path, log_path, exit_path, run_dir):
    require(exit_path.read_text(encoding="utf-8").strip() == "0", "Training exit code is not zero")
    stderr = log_path.read_text(encoding="utf-8", errors="replace")
    require(not any(marker in stderr for marker in
                    ("Traceback", "FloatingPointError", "CUDA out of memory", "LR mismatch")),
            "Training stderr contains a fatal marker")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run = run_dir.resolve(strict=True)
    require(json.loads((run / "summary.json").read_text(encoding="utf-8")) == report,
            "Summary and report differ")
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    for key in ("git_sha", "seed", "microbatches", "accumulation", "initialization",
                "versions", "pretrained_sha256", "data_report_sha256",
                "formal_training_authorized", "checkpoint_selection"):
        require(metadata.get(key) == report.get(key), "Metadata mismatch: " + key)
    require({path.name for path in run.glob("*.pth")} == {"last.pth", "previous.pth"},
            "Retained checkpoint files mismatch")
    for name, entry in (("last.pth", report["checkpoints"][-1]),
                        ("previous.pth", report["checkpoints"][-2])):
        path = (run / name).resolve(strict=True)
        require(path.parent == run and path.stat().st_size == entry["bytes"]
                and digest(path) == entry["sha256"], "Retained checkpoint integrity mismatch")
    records = [json.loads(line) for line in
               (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    result = audit(report, records)
    result.update(report_sha256=digest(report_path), stderr_sha256=digest(log_path),
                  final_checkpoint_sha256=digest(run / "last.pth"),
                  previous_checkpoint_sha256=digest(run / "previous.pth"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "stderr-log", "exit-file", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit_files(args.report, args.stderr_log, args.exit_file, args.run_dir)
    except (ValueError, OSError, KeyError, TypeError, IndexError, json.JSONDecodeError) as error:
        result = dict(source40k_saved_audit_ok=False,
                      error=f"{type(error).__name__}: {error}")
    print(json.dumps(result, indent=2))
    return 0 if result["source40k_saved_audit_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
