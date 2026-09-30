"""Read-only saved-evidence audit of the Phase 16 REIN query-residual smoke."""

import argparse
import json
import math
from pathlib import Path

from tools.audit_rein_pilot import digest, require
from tools.check_rein_query_residual import (
    BASELINE_AUDIT_SHA,
    BASELINE_CHECKPOINT_SHA,
    BASELINE_MIOU,
    BASELINE_PRODUCER,
    SMOKE_SEED,
    smoke_budget,
)
from tools.check_rein_runtime import EXPECTED_VERSIONS
from tools.rein_schedule_smoke import DATA_SHA
from tools.check_rein_backbone import WEIGHT_SHA


PRODUCER = "14d3e5aa90571362eabb368c6216bb230a4421fa"
REPORT_SHA = "5b566bd4d764109e71a26b4da7abd10f267ad627377de0340ce3c616e1266510"
STDERR_SHA = "ebec21c1e0c8bd257fa9353337304a3c96e6d7361e7520ba6d43b3711bf040b2"
CHECKPOINT_SHA = "fade8f3893ca1b8e2559ec154450ad3cb6ff9efee84b17354e6f70149c68d852"


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def audit_records(records):
    budget = smoke_budget()
    require(len(records) == budget["microbatches"], "Training record count mismatch")
    indices = []
    for iteration, record in enumerate(records, 1):
        require(record.get("iteration") == iteration, "Noncontiguous training trace")
        index = record.get("dataset_index")
        require(type(index) is int and 0 <= index < 24966, "Invalid GTA5 index")
        indices.append(index)
        require(record.get("optimizer_update") is
                (iteration % budget["accumulation"] == 0),
                "Accumulation boundary mismatch")
        for key in ("loss", "gradient_norm", "alpha"):
            require(_finite(record.get(key)), "Nonfinite " + key)
        require(record["loss"] > 0 and record["gradient_norm"] > 0,
                "Nonpositive loss or gradient norm")
    require(len(set(indices)) == len(indices), "Source prefix repeats an index")
    return dict(training_record_count=len(records), unique_source_indices=len(set(indices)),
                first_20_loss_mean=sum(record["loss"] for record in records[:20]) / 20,
                last_20_loss_mean=sum(record["loss"] for record in records[-20:]) / 20,
                final_record_alpha=records[-1]["alpha"])


def audit(report, records):
    budget = smoke_budget()
    expected = dict(
        ok=True, phase=16, stage="complete",
        purpose="frozen_rein_class_query_residual_smoke_not_accuracy_or_cqe",
        git_sha=PRODUCER, seed=SMOKE_SEED,
        base_segmentor="fixed_accepted_rein_source40k_seed0",
        base_final_miou_reference=BASELINE_MIOU,
        native_mask2former_queries_relabelled=False,
        cqe_enabled=False, target_labels_optimized=False,
        formal_training_authorized=False,
        requested_optimizer_updates=budget["optimizer_updates"],
        optimizer_updates=budget["optimizer_updates"],
        accumulation=budget["accumulation"], microbatches=budget["microbatches"],
        versions=EXPECTED_VERSIONS,
        pretrained_sha256=WEIGHT_SHA, data_report_sha256=DATA_SHA,
        baseline_audit_sha256=BASELINE_AUDIT_SHA,
        baseline_checkpoint_sha256=BASELINE_CHECKPOINT_SHA,
        baseline_training_git_sha=BASELINE_PRODUCER,
        dtype="float32", branch_trainable_parameters=3713,
        training_record_count=budget["microbatches"],
        all_losses_finite=True, base_parameters_unchanged=True,
        branch_checkpoint_sha256=CHECKPOINT_SHA,
        branch_checkpoint_bytes=17887,
    )
    for key, value in expected.items():
        require(type(report.get(key)) is type(value) and report[key] == value,
                "Unexpected " + key)
    require(report.get("query") == dict(classes=19, queries_per_class=2,
                hidden_channels=64, temperature=0.07, alpha_init=0.0,
                interaction="logit_residual"), "Query protocol mismatch")
    require(report.get("gpu") == "NVIDIA GeForce RTX 4090 D"
            and report.get("cuda") == "11.8"
            and report.get("python") == "3.10.21", "Runtime identity mismatch")
    trace = audit_records(records)
    for key in ("first_20_loss_mean", "last_20_loss_mean"):
        require(_finite(report.get(key)) and abs(report[key] - trace[key]) < 1e-6,
                "Loss summary mismatch")
    require(_finite(report.get("final_alpha")) and report["final_alpha"] != 0
            and abs(report["final_alpha"] - trace["final_record_alpha"]) < 1e-10,
            "Residual scale mismatch")
    required = {"alpha", "query_bank", "pixel_projection.weight"}
    actual = report.get("nonzero_gradient_parameters")
    require(isinstance(actual, list) and required.issubset(actual)
            and all(isinstance(name, str) for name in actual),
            "Branch gradient coverage mismatch")
    class_id = report.get("intervention_class_id")
    require(type(class_id) is int and 0 <= class_id < 19,
            "Invalid intervention class")
    require(report.get("outside_selected_class_max_error") == 0.0,
            "Intervention changed another class")
    require(_finite(report.get("selected_class_effect_max_error"))
            and report["selected_class_effect_max_error"] <= 1e-5,
            "Selected-class effect mismatch")
    for key in ("elapsed_seconds", "peak_allocated_gib", "peak_reserved_gib"):
        require(_finite(report.get(key)) and report[key] > 0,
                "Invalid " + key)
    require(report["peak_reserved_gib"] >= report["peak_allocated_gib"],
            "GPU memory accounting mismatch")
    return dict(query_residual_saved_audit_ok=True, training_git_sha=PRODUCER,
                optimizer_updates=budget["optimizer_updates"], **trace,
                final_alpha=report["final_alpha"],
                nonzero_gradient_parameters=actual,
                outside_selected_class_max_error=report["outside_selected_class_max_error"],
                selected_class_effect_max_error=report["selected_class_effect_max_error"],
                base_parameters_unchanged=True, cqe_enabled=False,
                target_labels_optimized=False,
                peak_reserved_gib=report["peak_reserved_gib"])


def audit_files(report_path: Path, stderr_path: Path, run_dir: Path):
    require(digest(report_path) == REPORT_SHA, "Smoke report SHA mismatch")
    require(digest(stderr_path) == STDERR_SHA, "Smoke stderr SHA mismatch")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    require(not any(marker in stderr for marker in
                    ("Traceback", "FloatingPointError", "CUDA out of memory",
                     "AssertionError")), "Smoke stderr contains fatal marker")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run = run_dir.resolve(strict=True)
    require(json.loads((run / "summary.json").read_text(encoding="utf-8")) == report,
            "Summary and saved report differ")
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    for key in ("git_sha", "seed", "phase", "base_segmentor", "query",
                "cqe_enabled", "target_labels_optimized", "baseline_checkpoint_sha256",
                "baseline_audit_sha256", "branch_trainable_parameters", "versions"):
        require(metadata.get(key) == report.get(key), "Metadata mismatch: " + key)
    require({path.name for path in run.glob("*.pth")}
            == {"query_residual_20updates.pth"}, "Checkpoint file set mismatch")
    checkpoint = (run / "query_residual_20updates.pth").resolve(strict=True)
    require(checkpoint.parent == run and checkpoint.stat().st_size == 17887
            and digest(checkpoint) == CHECKPOINT_SHA
            and report.get("branch_checkpoint") == str(checkpoint),
            "Branch checkpoint integrity mismatch")
    records = [json.loads(line) for line in
               (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    result = audit(report, records)
    result.update(report_sha256=REPORT_SHA, stderr_sha256=STDERR_SHA,
                  branch_checkpoint_sha256=CHECKPOINT_SHA)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stderr-log", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit_files(args.report, args.stderr_log, args.run_dir)
    except (ValueError, OSError, KeyError, TypeError, IndexError,
            json.JSONDecodeError) as error:
        result = dict(query_residual_saved_audit_ok=False,
                      error=f"{type(error).__name__}: {error}")
    print(json.dumps(result, indent=2))
    return 0 if result["query_residual_saved_audit_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
