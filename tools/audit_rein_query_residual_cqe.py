"""Read-only integrity audit for the paired Phase 16 CQE smoke."""

import argparse
import json
import math
import struct
from pathlib import Path

from tools.audit_rein_pilot import digest, require
from tools.check_rein_backbone import WEIGHT_SHA
from tools.check_rein_query_residual import (
    BASELINE_AUDIT_SHA,
    BASELINE_CHECKPOINT_SHA,
    BASELINE_MIOU,
    BASELINE_PRODUCER,
)
from tools.check_rein_runtime import EXPECTED_VERSIONS
from tools.rein_schedule_smoke import DATA_SHA

PRODUCER = "3644785a22a4c9391bb9076b4cdad606c63be615"
REPORT_SHA = "c6a7ee96f1e2c4e2defd29e2dc07e8d439c48fde74a4f1e3f7776a4b428f5ddd"
STDERR_SHA = "15f7e575bf971c025c9cbe64b18e1051f9346458015edb8583f9965940ea591e"
NO_CQE_CHECKPOINT_SHA = "632929ccce21ee68e6a1b804de900ab30b4019f8391fda7710fd1b81153f78f1"
CQE_CHECKPOINT_SHA = "10410867c71843057798a45fb6a2ab86fe57c4e1b0e83491dea7836b79ff69a0"
NO_CQE_TRACE_SHA = "1eb60f2459ea9a5a077303ea6cd8c0f7a66ad1caf3d013c3daecf74ed624e620"
CQE_TRACE_SHA = "f5544d5fa82e81959b9806dbc315921ecfbef4296e48f8c62aa0c8800d488bef"
UPDATES = 20
ACCUMULATION = 4
MICROBATCHES = UPDATES * ACCUMULATION


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _close(left, right, tolerance=1e-7):
    return _finite(left) and _finite(right) and abs(left - right) <= tolerance


def _float32(value):
    """Round a Python scalar to the float32 value used by the training tensors."""
    return struct.unpack("!f", struct.pack("!f", value))[0]


def audit_arm(records, summary, *, enabled):
    require(len(records) == MICROBATCHES, "Microbatch record count mismatch")
    indices = []
    for iteration, record in enumerate(records, 1):
        require(record.get("iteration") == iteration, "Noncontiguous trace")
        index = record.get("dataset_index")
        require(type(index) is int and 0 <= index < 24966, "Invalid GTA5 index")
        indices.append(index)
        require(record.get("optimizer_update") is (iteration % ACCUMULATION == 0),
                "Accumulation boundary mismatch")
        for key in ("loss", "loss_seg", "loss_original", "loss_photometric",
                    "loss_cqe", "gradient_norm", "alpha"):
            require(_finite(record.get(key)), "Nonfinite " + key)
        require(record["loss"] > 0 and record["gradient_norm"] > 0
                and record["loss_cqe"] >= 0, "Invalid training scalar")
        # The producer adds these scalar tensors in float32 before serializing
        # ``loss``. Reproduce that rounding rather than summing JSON floats in
        # Python's float64 and spuriously rejecting a valid trace.
        objective = _float32(
            record["loss_seg"] + (record["loss_cqe"] if enabled else 0.0)
        )
        require(_close(record["loss"], objective), "Objective reconstruction mismatch")
    require(len(set(indices)) == MICROBATCHES, "Source prefix repeats an index")
    require(summary.get("optimizer_updates") == UPDATES
            and summary.get("record_count") == MICROBATCHES
            and summary.get("unique_source_indices") == MICROBATCHES,
            "Arm update/coverage summary mismatch")
    require(summary.get("cqe_enabled") is enabled, "Arm CQE isolation mismatch")
    for field, trace_key, subset in (
        ("first_20_loss_mean", "loss", records[:20]),
        ("last_20_loss_mean", "loss", records[-20:]),
        ("first_20_cqe_mean", "loss_cqe", records[:20]),
        ("last_20_cqe_mean", "loss_cqe", records[-20:]),
    ):
        expected = sum(record[trace_key] for record in subset) / len(subset)
        require(_close(summary.get(field), expected, 1e-6),
                "Trace summary mismatch: " + field)
    require(summary.get("objective_reconstruction_error") == 0.0,
            "Reported objective reconstruction error is nonzero")
    require(summary.get("branch_changed_from_init") is True,
            "Branch did not change from initialization")
    gradients = summary.get("nonzero_gradient_parameters")
    require(isinstance(gradients, list)
            and {"alpha", "query_bank", "pixel_projection.weight"}.issubset(gradients),
            "Required branch gradient missing")
    require(_close(summary.get("final_alpha"), records[-1]["alpha"], 1e-10)
            and summary["final_alpha"] != 0, "Final residual scale mismatch")
    return indices


def audit(report, traces):
    require(report.get("ok") is True and report.get("stage") == "complete",
            "Smoke did not complete")
    expected = dict(
        phase=16, purpose="matched_frozen_rein_query_residual_cqe_smoke_not_accuracy",
        git_sha=PRODUCER, seed=20260931,
        base_segmentor="fixed_accepted_rein_source40k_seed0",
        base_final_miou_reference=BASELINE_MIOU,
        native_mask2former_queries_relabelled=False,
        target_labels_optimized=False, formal_training_authorized=False,
        requested_optimizer_updates=UPDATES, accumulation=ACCUMULATION,
        microbatches=MICROBATCHES, pretrained_sha256=WEIGHT_SHA,
        data_report_sha256=DATA_SHA, baseline_audit_sha256=BASELINE_AUDIT_SHA,
        baseline_checkpoint_sha256=BASELINE_CHECKPOINT_SHA,
        baseline_training_git_sha=BASELINE_PRODUCER, dtype="float32",
        target_images_used=0, matched_input_count=MICROBATCHES,
        matched_inputs_identical=True, base_parameters_unchanged=True,
        all_losses_finite=True, versions=EXPECTED_VERSIONS,
        query=dict(classes=19, queries_per_class=2, hidden_channels=64,
                   temperature=0.07, alpha_init=0.0, interaction="logit_residual"),
        cqe=dict(control_enabled=False, candidate_enabled=True, lambda_cqe=1.0,
                 objective="segmentation_mean_plus_existing_cqe"),
    )
    for key, value in expected.items():
        require(type(report.get(key)) is type(value) and report[key] == value,
                "Unexpected report field: " + key)
    require(report.get("gpu") == "NVIDIA GeForce RTX 4090 D"
            and report.get("cuda") == "11.8"
            and report.get("python") == "3.10.21", "Runtime identity mismatch")
    require(report.get("initial_branch_fingerprint") ==
            "d8558b81dffc38ca6b1b71a349ba1e53eeee63e8356bd837ef9b0d0a9f7a0b24",
            "Paired branch initialization mismatch")
    require(report.get("views") == ["original", "photometric"]
            and report.get("geometry_preserved") is True,
            "Style protocol mismatch")
    arms = report.get("arms")
    require(isinstance(arms, dict) and set(arms) == {"no_cqe", "cqe"},
            "Paired arms missing")
    no_cqe_indices = audit_arm(traces["no_cqe"], arms["no_cqe"], enabled=False)
    cqe_indices = audit_arm(traces["cqe"], arms["cqe"], enabled=True)
    require(no_cqe_indices == cqe_indices, "Paired source indices differ")
    for key in ("elapsed_seconds", "peak_allocated_gib", "peak_reserved_gib"):
        require(_finite(report.get(key)) and report[key] > 0, "Invalid " + key)
    require(report["peak_reserved_gib"] >= report["peak_allocated_gib"],
            "GPU memory accounting mismatch")
    return dict(
        query_residual_cqe_saved_audit_ok=True,
        training_git_sha=PRODUCER,
        optimizer_updates_per_arm=UPDATES,
        matched_microbatches=MICROBATCHES,
        unique_source_indices_per_arm=MICROBATCHES,
        matched_inputs_identical=True,
        no_cqe_final_alpha=arms["no_cqe"]["final_alpha"],
        cqe_final_alpha=arms["cqe"]["final_alpha"],
        no_cqe_first_last_loss_mean=[arms["no_cqe"]["first_20_loss_mean"],
                                     arms["no_cqe"]["last_20_loss_mean"]],
        cqe_first_last_loss_mean=[arms["cqe"]["first_20_loss_mean"],
                                  arms["cqe"]["last_20_loss_mean"]],
        no_cqe_first_last_cqe_mean=[arms["no_cqe"]["first_20_cqe_mean"],
                                    arms["no_cqe"]["last_20_cqe_mean"]],
        cqe_first_last_cqe_mean=[arms["cqe"]["first_20_cqe_mean"],
                                 arms["cqe"]["last_20_cqe_mean"]],
        base_parameters_unchanged=True,
        target_labels_optimized=False,
        formal_training_authorized=False,
        peak_reserved_gib=report["peak_reserved_gib"],
    )


def _read_trace(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def audit_files(report_path, stderr_path, run_dir):
    require(digest(report_path) == REPORT_SHA, "Report SHA256 mismatch")
    require(digest(stderr_path) == STDERR_SHA, "stderr SHA256 mismatch")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    require(not any(marker in stderr for marker in
                    ("Traceback", "FloatingPointError", "CUDA out of memory",
                     "AssertionError")), "stderr contains fatal marker")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run = run_dir.resolve(strict=True)
    require(json.loads((run / "summary.json").read_text(encoding="utf-8")) == report,
            "Saved summary differs from report")
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    for key in ("git_sha", "seed", "phase", "base_segmentor", "query", "views",
                "cqe", "versions", "pretrained_sha256", "data_report_sha256",
                "baseline_audit_sha256", "baseline_checkpoint_sha256"):
        require(metadata.get(key) == report.get(key), "Metadata mismatch: " + key)
    expected_files = {"metadata.json", "summary.json", "no_cqe.jsonl", "cqe.jsonl",
                      "no_cqe_branch.pth", "cqe_branch.pth"}
    require({path.name for path in run.iterdir() if path.is_file()} == expected_files,
            "Run artifact set mismatch")
    hashes = {"no_cqe.jsonl": NO_CQE_TRACE_SHA, "cqe.jsonl": CQE_TRACE_SHA,
              "no_cqe_branch.pth": NO_CQE_CHECKPOINT_SHA,
              "cqe_branch.pth": CQE_CHECKPOINT_SHA}
    for filename, expected_hash in hashes.items():
        require(digest(run / filename) == expected_hash,
                "Artifact SHA256 mismatch: " + filename)
    result = audit(report, {"no_cqe": _read_trace(run / "no_cqe.jsonl"),
                            "cqe": _read_trace(run / "cqe.jsonl")})
    result.update(report_sha256=REPORT_SHA, stderr_sha256=STDERR_SHA,
                  no_cqe_trace_sha256=NO_CQE_TRACE_SHA,
                  cqe_trace_sha256=CQE_TRACE_SHA,
                  no_cqe_checkpoint_sha256=NO_CQE_CHECKPOINT_SHA,
                  cqe_checkpoint_sha256=CQE_CHECKPOINT_SHA)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stderr-log", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit_files(args.report, args.stderr_log, args.run_dir)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        result = dict(query_residual_cqe_saved_audit_ok=False,
                      error=f"{type(error).__name__}: {error}")
    print(json.dumps(result, indent=2))
    return 0 if result["query_residual_cqe_saved_audit_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
