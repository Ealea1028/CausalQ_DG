import copy

import pytest

from tools.audit_rein_query_residual_cqe import audit, audit_arm
from tools.check_rein_backbone import WEIGHT_SHA
from tools.check_rein_query_residual import (
    BASELINE_AUDIT_SHA,
    BASELINE_CHECKPOINT_SHA,
    BASELINE_MIOU,
    BASELINE_PRODUCER,
)
from tools.check_rein_runtime import EXPECTED_VERSIONS
from tools.rein_schedule_smoke import DATA_SHA


def _records(enabled):
    records = []
    for i in range(1, 81):
        seg = 2.0 - i / 1000
        cqe = i / 100000 if i % 2 else 0.0
        records.append(dict(iteration=i, dataset_index=i - 1, loss=seg + (cqe if enabled else 0),
                            loss_seg=seg, loss_original=seg - 0.01,
                            loss_photometric=seg + 0.01, loss_cqe=cqe,
                            gradient_norm=0.1 + i / 1000,
                            optimizer_update=i % 4 == 0, alpha=i / 10000))
    return records


def _arm(records, enabled):
    return dict(cqe_enabled=enabled, optimizer_updates=20, record_count=80,
                unique_source_indices=80, objective_reconstruction_error=0.0,
                first_20_loss_mean=sum(r["loss"] for r in records[:20]) / 20,
                last_20_loss_mean=sum(r["loss"] for r in records[-20:]) / 20,
                first_20_cqe_mean=sum(r["loss_cqe"] for r in records[:20]) / 20,
                last_20_cqe_mean=sum(r["loss_cqe"] for r in records[-20:]) / 20,
                final_alpha=records[-1]["alpha"],
                branch_changed_from_init=True,
                nonzero_gradient_parameters=["alpha", "query_bank",
                                             "pixel_projection.weight"])


def _report(no_cqe, cqe):
    return dict(ok=True, stage="complete", phase=16,
                purpose="matched_frozen_rein_query_residual_cqe_smoke_not_accuracy",
                git_sha="3644785a22a4c9391bb9076b4cdad606c63be615",
                seed=20260931, base_segmentor="fixed_accepted_rein_source40k_seed0",
                base_final_miou_reference=BASELINE_MIOU,
                native_mask2former_queries_relabelled=False,
                target_labels_optimized=False, formal_training_authorized=False,
                requested_optimizer_updates=20, accumulation=4, microbatches=80,
                pretrained_sha256=WEIGHT_SHA, data_report_sha256=DATA_SHA,
                baseline_audit_sha256=BASELINE_AUDIT_SHA,
                baseline_checkpoint_sha256=BASELINE_CHECKPOINT_SHA,
                baseline_training_git_sha=BASELINE_PRODUCER, dtype="float32",
                target_images_used=0, matched_input_count=80,
                matched_inputs_identical=True, base_parameters_unchanged=True,
                all_losses_finite=True, versions=EXPECTED_VERSIONS,
                query=dict(classes=19, queries_per_class=2, hidden_channels=64,
                           temperature=0.07, alpha_init=0.0,
                           interaction="logit_residual"),
                cqe=dict(control_enabled=False, candidate_enabled=True,
                         lambda_cqe=1.0,
                         objective="segmentation_mean_plus_existing_cqe"),
                gpu="NVIDIA GeForce RTX 4090 D", cuda="11.8", python="3.10.21",
                initial_branch_fingerprint=(
                    "d8558b81dffc38ca6b1b71a349ba1e53eeee63e8356bd837ef9b0d0a9f7a0b24"),
                views=["original", "photometric"], geometry_preserved=True,
                elapsed_seconds=48.0, peak_allocated_gib=1.9, peak_reserved_gib=2.4,
                arms={"no_cqe": _arm(no_cqe, False), "cqe": _arm(cqe, True)})


def test_accepts_two_matched_finite_arms():
    no_cqe, cqe = _records(False), _records(True)
    result = audit(_report(no_cqe, cqe), {"no_cqe": no_cqe, "cqe": cqe})
    assert result["query_residual_cqe_saved_audit_ok"] is True
    assert result["matched_microbatches"] == 80


@pytest.mark.parametrize("edit", [
    lambda report, traces: report.update(matched_inputs_identical=False),
    lambda report, traces: report["arms"]["cqe"].update(cqe_enabled=False),
    lambda report, traces: traces["cqe"][4].update(dataset_index=900),
    lambda report, traces: traces["no_cqe"][3].update(optimizer_update=False),
    lambda report, traces: traces["cqe"][7].update(loss=float("nan")),
    lambda report, traces: report.update(formal_training_authorized=True),
])
def test_rejects_inconsistent_or_out_of_scope_pair(edit):
    no_cqe, cqe = _records(False), _records(True)
    report = copy.deepcopy(_report(no_cqe, cqe))
    traces = {"no_cqe": copy.deepcopy(no_cqe), "cqe": copy.deepcopy(cqe)}
    edit(report, traces)
    with pytest.raises(ValueError):
        audit(report, traces)


def test_rejects_objective_mismatch():
    records = _records(True)
    records[0]["loss"] += 0.1
    with pytest.raises(ValueError, match="Objective reconstruction"):
        audit_arm(records, _arm(records, True), enabled=True)
