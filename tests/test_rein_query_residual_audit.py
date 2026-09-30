import copy

import pytest

from tools.audit_rein_query_residual import (
    CHECKPOINT_SHA,
    PRODUCER,
    audit,
    audit_records,
)
from tools.check_rein_backbone import WEIGHT_SHA
from tools.check_rein_query_residual import (
    BASELINE_AUDIT_SHA,
    BASELINE_CHECKPOINT_SHA,
    BASELINE_MIOU,
    BASELINE_PRODUCER,
    SMOKE_SEED,
)
from tools.check_rein_runtime import EXPECTED_VERSIONS
from tools.rein_schedule_smoke import DATA_SHA


def records():
    return [dict(iteration=i, dataset_index=i - 1, loss=2.1 - i / 1000,
                 gradient_norm=0.1 + i / 1000, optimizer_update=i % 4 == 0,
                 alpha=i // 4 / 10000) for i in range(1, 81)]


def report(trace):
    return dict(
        ok=True, phase=16, stage="complete",
        purpose="frozen_rein_class_query_residual_smoke_not_accuracy_or_cqe",
        git_sha=PRODUCER, seed=SMOKE_SEED,
        base_segmentor="fixed_accepted_rein_source40k_seed0",
        base_final_miou_reference=BASELINE_MIOU,
        native_mask2former_queries_relabelled=False,
        query=dict(classes=19, queries_per_class=2, hidden_channels=64,
                   temperature=0.07, alpha_init=0.0,
                   interaction="logit_residual"),
        cqe_enabled=False, target_labels_optimized=False,
        formal_training_authorized=False,
        requested_optimizer_updates=20, optimizer_updates=20,
        accumulation=4, microbatches=80,
        versions=EXPECTED_VERSIONS, pretrained_sha256=WEIGHT_SHA,
        data_report_sha256=DATA_SHA, baseline_audit_sha256=BASELINE_AUDIT_SHA,
        baseline_checkpoint_sha256=BASELINE_CHECKPOINT_SHA,
        baseline_training_git_sha=BASELINE_PRODUCER,
        dtype="float32", branch_trainable_parameters=3713,
        training_record_count=80, all_losses_finite=True,
        base_parameters_unchanged=True,
        branch_checkpoint_sha256=CHECKPOINT_SHA,
        branch_checkpoint_bytes=17887,
        gpu="NVIDIA GeForce RTX 4090 D", cuda="11.8", python="3.10.21",
        first_20_loss_mean=sum(item["loss"] for item in trace[:20]) / 20,
        last_20_loss_mean=sum(item["loss"] for item in trace[-20:]) / 20,
        final_alpha=trace[-1]["alpha"],
        nonzero_gradient_parameters=["alpha", "query_bank",
                                     "pixel_projection.weight"],
        intervention_class_id=0,
        outside_selected_class_max_error=0.0,
        selected_class_effect_max_error=6e-8,
        elapsed_seconds=28.5, peak_allocated_gib=1.705,
        peak_reserved_gib=2.055,
    )


def test_accepts_complete_smoke_evidence():
    trace = records()
    result = audit(report(trace), trace)
    assert result["query_residual_saved_audit_ok"] is True
    assert result["training_record_count"] == 80


@pytest.mark.parametrize("change", [
    lambda data: data[17].update(iteration=99),
    lambda data: data[3].update(optimizer_update=False),
    lambda data: data[10].update(dataset_index=0),
    lambda data: data[20].update(loss=float("nan")),
])
def test_rejects_broken_trace(change):
    trace = records()
    change(trace)
    with pytest.raises(ValueError):
        audit_records(trace)


@pytest.mark.parametrize("key,value", [
    ("cqe_enabled", True),
    ("base_parameters_unchanged", False),
    ("branch_checkpoint_sha256", "wrong"),
    ("outside_selected_class_max_error", 0.01),
    ("selected_class_effect_max_error", 0.01),
])
def test_rejects_changed_mechanism_evidence(key, value):
    trace = records()
    evidence = copy.deepcopy(report(trace))
    evidence[key] = value
    with pytest.raises(ValueError):
        audit(evidence, trace)
