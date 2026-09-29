import copy

import pytest

from tools.audit_rein_real_data import SOURCE_SHA, audit_report
from tools.check_rein_backbone import REIN_SHA, WEIGHT_SHA
from tools.check_rein_runtime import EXPECTED_VERSIONS


def fixture_report():
    """Fabricated schema fixture, not an AutoDL experiment result."""
    losses = {f"{stage}.loss_{kind}": 1. for stage in
              ["decode"] + [f"decode.d{i}" for i in range(9)]
              for kind in ("cls", "mask", "dice")}
    return dict(ok=True, real_data_smoke_ok=True, stage="complete", phase=16,
                git_sha=SOURCE_SHA, rein_sha=REIN_SHA, checkpoint_sha256=WEIGHT_SHA,
                optimizer_steps=20, adapter_parameter_changed=True,
                frozen_patch_weight_unchanged=True, frozen_backbone_gradients_absent=True,
                target_inference_count=5, loaded_tensor_count=343, dtype="float32",
                seed=20260929, versions=EXPECTED_VERSIONS.copy(),
                purpose="real_data_20step_optimization_smoke_not_accuracy_evaluation",
                data_inventory=dict(source_pairs=24966, target_pairs=500),
                smoke_protocol=dict(steps=20, target_inference_samples=5, data_seed=0,
                    optimizer="AdamW", lr=1e-4, weight_decay=0.05, betas=[0.9, 0.999],
                    clip_norm=1.0, batch_size=1, workers=0, dtype="float32",
                    crop_size=[512, 512], accuracy_evaluation=False,
                    checkpoint_saved=False, formal_baseline=False),
                real_training_records=[dict(iteration=i, loss=30., gradient_norm=1000.,
                    valid_pixel_count=10, losses=losses.copy()) for i in range(1, 21)],
                target_inference_records=[dict(semantic_score_shape=[1, 19, 512, 512])
                                          for _ in range(5)],
                max_normalization_roundtrip_error=1e-6,
                peak_allocated_gib=3., peak_reserved_gib=4.)


def test_saved_audit_accepts_large_finite_preclip_norm():
    result = audit_report(fixture_report())
    assert result["real_data_saved_report_audit_ok"]
    assert not result["formal_training_authorized"]


@pytest.mark.parametrize("key,value", [("target_inference_count", 0),
                                      ("frozen_patch_weight_unchanged", False),
                                      ("git_sha", "wrong")])
def test_saved_audit_rejects_missing_gate_or_wrong_provenance(key, value):
    report = fixture_report()
    report[key] = value
    with pytest.raises(ValueError):
        audit_report(report)


def test_saved_audit_rejects_nonfinite_and_incomplete_losses():
    report = fixture_report()
    report["real_training_records"][0]["losses"]["decode.loss_cls"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        audit_report(report)
    report = copy.deepcopy(fixture_report())
    report["real_training_records"][0]["losses"].pop("decode.loss_cls")
    with pytest.raises(ValueError, match="Incomplete"):
        audit_report(report)
