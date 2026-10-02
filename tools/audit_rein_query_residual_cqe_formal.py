"""Read-only audit and paired image-bootstrap analysis for Phase 16 formal CQE."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.audit_rein_pilot import digest, require
from tools.check_rein_backbone import WEIGHT_SHA
from tools.check_rein_query_residual import (
    BASELINE_AUDIT_SHA,
    BASELINE_CHECKPOINT_SHA,
    BASELINE_MIOU,
    BASELINE_PRODUCER,
)
from tools.check_rein_runtime import EXPECTED_VERSIONS
from tools.rein_schedule_smoke import DATA_SHA, scheduled_lr


UPDATES = 40000
ACCUMULATION = 4
MICROBATCHES = UPDATES * ACCUMULATION
SOURCE_SIZE = 24966
CLASSES = 19
BASE_SEED = 20260931


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _miou(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    union = matrix.sum(0) + matrix.sum(1) - matrix.diagonal()
    valid = union > 0
    if not valid.any():
        raise ValueError("No valid classes in evaluation matrix")
    return float(np.mean(matrix.diagonal()[valid] / union[valid]))


def paired_image_bootstrap(no_cqe_samples, cqe_samples, *, seed=20260931,
                           replicates=2000):
    """Paired image bootstrap; uncertainty is conditional on these two models."""
    if len(no_cqe_samples) != 500 or len(cqe_samples) != 500:
        raise ValueError("Expected 500 paired Cityscapes images")
    no_matrices = np.asarray([sample["confusion_matrix"] for sample in no_cqe_samples],
                             dtype=np.int64)
    cqe_matrices = np.asarray([sample["confusion_matrix"] for sample in cqe_samples],
                              dtype=np.int64)
    if no_matrices.shape != (500, CLASSES, CLASSES) or cqe_matrices.shape != no_matrices.shape:
        raise ValueError("Per-image confusion matrix shape mismatch")
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, 500, size=500)
        deltas[index] = _miou(cqe_matrices[selected].sum(0)) - _miou(
            no_matrices[selected].sum(0))
    return dict(method="paired_image_cluster_bootstrap_percentile",
                seed=seed, replicates=replicates,
                confidence_level=0.95,
                delta_miou_cqe_minus_control_mean=float(deltas.mean()),
                delta_miou_cqe_minus_control_ci95=[
                    float(np.quantile(deltas, 0.025)),
                    float(np.quantile(deltas, 0.975))],
                interpretation="image-sampling uncertainty conditional on one training seed; not training-seed uncertainty")


def _read_trace(path):
    raw = Path(path).read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    return records, hashlib.sha256(raw).hexdigest()


def _optimizer_update_records(records):
    """Return only post-accumulation records used by producer summaries."""
    return [record for record in records if record.get("optimizer_update") is True]


def expected_augmentation_seed(training_seed, microbatch):
    return (training_seed * 1000003 + microbatch) % (2 ** 32)


def _audit_arm(name, arm, records, *, enabled, training_seed):
    require(len(records) == MICROBATCHES, f"{name}: microbatch count mismatch")
    indices = []
    seg_losses, cqe_losses = [], []
    gradients = set()
    for microbatch, record in enumerate(records, 1):
        require(record.get("microbatch") == microbatch, f"{name}: trace not contiguous")
        source_index = record.get("dataset_index")
        require(type(source_index) is int and 0 <= source_index < SOURCE_SIZE,
                f"{name}: invalid source index")
        indices.append(source_index)
        require(record.get("augmentation_seed") == expected_augmentation_seed(
                    training_seed, microbatch),
                f"{name}: augmentation seed mismatch")
        fingerprint = record.get("input_sha256")
        require(isinstance(fingerprint, str) and len(fingerprint) == 64,
                f"{name}: missing input fingerprint")
        update = microbatch % ACCUMULATION == 0
        require(record.get("optimizer_update") is update, f"{name}: update boundary mismatch")
        expected_lr = scheduled_lr((microbatch - 1) // ACCUMULATION)
        require(_finite(record.get("lr")) and abs(record["lr"] - expected_lr) <= 1e-12,
                f"{name}: PolyLR mismatch at microbatch {microbatch}")
        for key in ("loss", "loss_seg", "loss_original", "loss_photometric", "loss_cqe", "alpha"):
            require(_finite(record.get(key)), f"{name}: nonfinite {key}")
        require(record["loss"] > 0 and record["loss_cqe"] >= 0,
                f"{name}: invalid loss")
        expected_objective = record["loss_seg"] + (record["loss_cqe"] if enabled else 0.0)
        require(abs(record["loss"] - expected_objective) <= 1e-6,
                f"{name}: objective reconstruction mismatch")
        if update:
            require(_finite(record.get("gradient_norm")) and record["gradient_norm"] > 0,
                    f"{name}: invalid update gradient")
            seg_losses.append(record["loss_seg"])
            cqe_losses.append(record["loss_cqe"])
            if (record.get("gradient_parameters") is not None):
                gradients.update(record["gradient_parameters"])
        else:
            require(record.get("gradient_norm") is None, f"{name}: gradient outside update")
    # The SourceSampler emits a permutation for every complete epoch.
    for start in range(0, MICROBATCHES, SOURCE_SIZE):
        window = indices[start:min(start + SOURCE_SIZE, MICROBATCHES)]
        require(len(set(window)) == len(window), f"{name}: repeated index within source epoch")
    require(arm.get("optimizer_updates") == UPDATES
            and arm.get("microbatch_count") == MICROBATCHES
            and arm.get("accumulation") == ACCUMULATION,
            f"{name}: schedule metadata mismatch")
    require(arm.get("cqe_enabled") is enabled, f"{name}: CQE isolation mismatch")
    require(arm.get("checkpoint_roundtrip_ok") is True, f"{name}: checkpoint roundtrip missing")
    require(arm.get("final_alpha") != 0 and _finite(arm.get("final_alpha")),
            f"{name}: residual scale did not open")
    require({"alpha", "query_bank", "pixel_projection.weight"}.issubset(
        arm.get("nonzero_gradient_parameters", [])), f"{name}: required branch gradient missing")
    require(arm.get("trace_record_count") == MICROBATCHES,
            f"{name}: trace summary mismatch")
    update_records = _optimizer_update_records(records)
    require(len(update_records) == UPDATES, f"{name}: optimizer update count mismatch")
    for field, expected in (
            ("first_100_update_loss_mean", np.mean([r["loss"] for r in update_records[:100]])),
            ("last_100_update_loss_mean", np.mean([r["loss"] for r in update_records[-100:]])),
            ("first_100_update_seg_loss_mean", np.mean(seg_losses[:100])),
            ("last_100_update_seg_loss_mean", np.mean(seg_losses[-100:])),
            ("first_100_update_cqe_loss_mean", np.mean(cqe_losses[:100])),
            ("last_100_update_cqe_loss_mean", np.mean(cqe_losses[-100:]))):
        require(_finite(arm.get(field)) and abs(arm[field] - float(expected)) <= 1e-6,
                f"{name}: {field} mismatch")
    return indices


def _audit_validation(arm, name):
    validation = arm.get("validation", {})
    samples = validation.get("samples", [])
    require(validation.get("sample_count") == 500 and len(samples) == 500,
            f"{name}: full Cityscapes validation missing")
    matrices = []
    paths = []
    for index, sample in enumerate(samples):
        require(sample.get("index") == index, f"{name}: target order mismatch")
        matrix = np.asarray(sample.get("confusion_matrix"), dtype=np.int64)
        require(matrix.shape == (CLASSES, CLASSES) and np.all(matrix >= 0),
                f"{name}: invalid per-image confusion matrix")
        require(int(matrix.sum()) == sample.get("valid_pixels"),
                f"{name}: per-image valid-pixel count mismatch")
        matrices.append(matrix)
        paths.append(sample.get("img_path"))
    require(all(isinstance(path, str) and path for path in paths)
            and len(set(paths)) == 500, f"{name}: duplicate/missing target paths")
    combined = np.asarray(matrices, dtype=np.int64).sum(0)
    require(np.array_equal(combined, np.asarray(validation.get("confusion_matrix"), dtype=np.int64)),
            f"{name}: aggregate confusion matrix mismatch")
    require(abs(_miou(combined) - validation.get("miou", float("nan"))) <= 1e-12,
            f"{name}: mIoU reconstruction mismatch")
    class_iou = validation.get("class_iou")
    union = combined.sum(0) + combined.sum(1) - combined.diagonal()
    require(isinstance(class_iou, list) and len(class_iou) == CLASSES,
            f"{name}: class IoU coverage mismatch")
    for class_id in range(CLASSES):
        expected_iou = (float(combined[class_id, class_id] / union[class_id])
                        if union[class_id] > 0 else None)
        reported_iou = class_iou[class_id]
        require((expected_iou is None and reported_iou is None)
                or (expected_iou is not None and _finite(reported_iou)
                    and abs(expected_iou - reported_iou) <= 1e-12),
                f"{name}: class IoU mismatch at class {class_id}")
    require(validation.get("total_valid_pixels") == int(combined.sum()),
            f"{name}: total valid-pixel mismatch")
    return samples


def audit(report_path, run_dir, *, replicates=2000, expected_seed=BASE_SEED):
    report_path, run_dir = Path(report_path), Path(run_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report.get("ok") is True and report.get("stage") == "complete",
            "Formal run did not complete")
    require(report.get("purpose") == "matched_formal_frozen_rein_query_residual_cqe_comparison",
            "Unexpected experiment purpose")
    require(report.get("phase") == 16 and report.get("seed") == expected_seed,
            "Phase/seed mismatch")
    require(report.get("formal_training_authorized") is True
            and report.get("target_labels_optimized") is False,
            "Scope/target optimization mismatch")
    require(report.get("requested_optimizer_updates_per_arm") == UPDATES
            and report.get("microbatches_per_arm") == MICROBATCHES
            and report.get("accumulation") == ACCUMULATION,
            "Formal schedule mismatch")
    require(report.get("base_segmentor") == "fixed_accepted_rein_source40k_seed0"
            and report.get("base_final_miou_reference") == BASELINE_MIOU,
            "Fixed reference mismatch")
    require(report.get("pretrained_sha256") == WEIGHT_SHA
            and report.get("data_report_sha256") == DATA_SHA
            and report.get("baseline_audit_sha256") == BASELINE_AUDIT_SHA
            and report.get("baseline_checkpoint_sha256") == BASELINE_CHECKPOINT_SHA
            and report.get("baseline_training_git_sha") == BASELINE_PRODUCER,
            "Pinned input provenance mismatch")
    require(report.get("versions") == EXPECTED_VERSIONS
            and report.get("python", "").startswith("3.10.")
            and report.get("cuda") == "11.8"
            and report.get("gpu") == "NVIDIA GeForce RTX 4090 D",
            "Runtime identity mismatch")
    require(report.get("query") == dict(classes=19, queries_per_class=2,
                hidden_channels=64, temperature=0.07, alpha_init=0.0,
                interaction="logit_residual"), "Query protocol mismatch")
    require(report.get("views") == ["original", "photometric"]
            and report.get("geometry_preserved") is True
            and report.get("native_mask2former_queries_relabelled") is False,
            "Intervention protocol mismatch")
    arms = report.get("arms")
    require(isinstance(arms, dict) and set(arms) == {"no_cqe", "cqe"},
            "Matched arms are missing")
    require(report.get("optimizer") == dict(name="AdamW", lr=1e-4,
                weight_decay=0.01, scheduler="polynomial", poly_power=0.9,
                schedule_horizon=UPDATES), "Optimizer protocol mismatch")
    require(report.get("cqe") == dict(control_enabled=False, candidate_enabled=True,
                lambda_cqe=1.0,
                objective="two_view_segmentation_mean_plus_normalized_cqe"),
            "CQE isolation/objective mismatch")
    traces = {}
    trace_hashes = {}
    for name in ("no_cqe", "cqe"):
        trace_path = Path(arms[name]["trace"])
        checkpoint_path = Path(arms[name]["checkpoint"])
        require(trace_path.parent == run_dir / name and checkpoint_path.parent == run_dir / name,
                f"{name}: artifact path escapes run directory")
        traces[name], trace_hashes[name] = _read_trace(trace_path)
        require(trace_hashes[name] == arms[name].get("trace_sha256"),
                f"{name}: trace SHA256 mismatch")
        require(digest(checkpoint_path) == arms[name].get("checkpoint_sha256"),
                f"{name}: checkpoint SHA256 mismatch")
        import torch
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(payload.get("metadata", {}).get("optimizer_updates") == UPDATES
                and payload["metadata"].get("cqe_enabled") is (name == "cqe")
                and payload["metadata"].get("git_sha") == report.get("git_sha")
                and payload["metadata"].get("seed") == expected_seed,
                f"{name}: final checkpoint metadata mismatch")
        require(set(payload.get("model", {})) ==
                {"alpha", "query_bank", "pixel_projection.weight", "pixel_projection.bias"},
                f"{name}: branch-only checkpoint coverage mismatch")
        require(all(torch.isfinite(tensor).all().item() for tensor in payload["model"].values()),
                f"{name}: nonfinite branch checkpoint")
        require(abs(float(payload["model"]["alpha"].item())
                    - arms[name].get("final_alpha", float("nan"))) <= 1e-8
                and checkpoint_path.stat().st_size == arms[name].get("checkpoint_bytes"),
                f"{name}: checkpoint/report state mismatch")
    no_indices = _audit_arm("no_cqe", arms["no_cqe"], traces["no_cqe"],
                            enabled=False, training_seed=expected_seed)
    cqe_indices = _audit_arm("cqe", arms["cqe"], traces["cqe"],
                             enabled=True, training_seed=expected_seed)
    require(no_indices == cqe_indices, "Source indices differ across arms")
    require([row["augmentation_seed"] for row in traces["no_cqe"]]
            == [row["augmentation_seed"] for row in traces["cqe"]],
            "Augmentation seeds differ across arms")
    require([row["input_sha256"] for row in traces["no_cqe"]]
            == [row["input_sha256"] for row in traces["cqe"]],
            "Paired source tensors/style views differ")
    pair_hash = hashlib.sha256()
    for record in traces["no_cqe"]:
        pair_hash.update(record["input_sha256"].encode("ascii"))
    require(report.get("paired_input_hash_sha256") == pair_hash.hexdigest(),
            "Paired input digest mismatch")
    require(report.get("paired_inputs_identical") is True
            and report.get("base_parameters_unchanged") is True,
            "Matched/frozen base evidence missing")
    require(isinstance(report.get("fixed_base_state_sha256"), str)
            and len(report["fixed_base_state_sha256"]) == 64,
            "Complete frozen-base state fingerprint missing")
    require(arms["no_cqe"].get("initial_branch_sha256")
            == arms["cqe"].get("initial_branch_sha256")
            == report.get("initial_branch_sha256"), "Initial branches differ")
    samples_no = _audit_validation(arms["no_cqe"], "no_cqe")
    samples_cqe = _audit_validation(arms["cqe"], "cqe")
    require([row["img_path"] for row in samples_no]
            == [row["img_path"] for row in samples_cqe],
            "Target evaluation samples differ")
    observed_delta = arms["cqe"]["validation"]["miou"] - arms["no_cqe"]["validation"]["miou"]
    require(abs(report.get("final_miou_difference_cqe_minus_control", float("nan"))
                - observed_delta) <= 1e-12, "Paired mIoU delta mismatch")
    bootstrap = paired_image_bootstrap(samples_no, samples_cqe,
                                       seed=expected_seed, replicates=replicates)
    return dict(formal_paired_audit_ok=True, training_git_sha=report["git_sha"],
                seed=expected_seed, updates_per_arm=UPDATES,
                microbatches_per_arm=MICROBATCHES, source_indices_matched=True,
                per_microbatch_inputs_matched=True,
                no_cqe_miou=arms["no_cqe"]["validation"]["miou"],
                cqe_miou=arms["cqe"]["validation"]["miou"],
                cqe_minus_control_miou=observed_delta,
                baseline_miou_reference=BASELINE_MIOU,
                no_cqe_trace_sha256=trace_hashes["no_cqe"],
                cqe_trace_sha256=trace_hashes["cqe"],
                no_cqe_checkpoint_sha256=arms["no_cqe"]["checkpoint_sha256"],
                cqe_checkpoint_sha256=arms["cqe"]["checkpoint_sha256"],
                report_sha256=digest(report_path),
                paired_image_bootstrap=bootstrap,
                training_seed_uncertainty_reported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--expected-seed", type=int, default=BASE_SEED)
    args = parser.parse_args()
    try:
        result = audit(args.report, args.run_dir,
                       replicates=args.bootstrap_replicates,
                       expected_seed=args.expected_seed)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        result = dict(formal_paired_audit_ok=False,
                      error=f"{type(error).__name__}: {error}")
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["formal_paired_audit_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
