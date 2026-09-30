import pytest

from tools.check_rein_query_residual import (
    BASELINE_CHECKPOINT_SHA,
    BASELINE_MIOU,
    BASELINE_PRODUCER,
    smoke_budget,
    validate_baseline_audit,
)


def accepted_audit():
    return {
        "source40k_saved_audit_ok": True,
        "training_git_sha": BASELINE_PRODUCER,
        "optimizer_updates": 40000,
        "microbatches": 160000,
        "final_miou": BASELINE_MIOU,
        "final_checkpoint_sha256": BASELINE_CHECKPOINT_SHA,
        "cqe_enabled": False,
        "target_labels_optimized": False,
    }


def test_smoke_budget_is_bounded_and_accumulated():
    budget = smoke_budget()
    assert budget == {"optimizer_updates": 20, "accumulation": 4,
                      "microbatches": 80}
    assert budget["microbatches"] == (budget["optimizer_updates"]
                                      * budget["accumulation"])


def test_baseline_audit_gate_accepts_exact_evidence():
    validate_baseline_audit(accepted_audit())


@pytest.mark.parametrize("key", [
    "source40k_saved_audit_ok", "training_git_sha", "optimizer_updates",
    "microbatches", "final_miou", "final_checkpoint_sha256",
    "cqe_enabled", "target_labels_optimized",
])
def test_baseline_audit_gate_rejects_changed_evidence(key):
    report = accepted_audit()
    report[key] = None
    with pytest.raises(ValueError):
        validate_baseline_audit(report)
