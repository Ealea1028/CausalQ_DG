"""Synthetic evidence and image tensors suffice for source evaluation tests."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from causalq.datasets.strict_source import load_membership, MANIFEST_HASHES
from causalq.protocols.source_split import manifest_bytes
from causalq.utils.checkpoint import save_training_checkpoint, trainable_state_dict
from tools import audit_strict_source_smoke as audit
from tools import evaluate_strict_source as launcher
from tools.strict_source_evaluation import evaluate_holdout
from tools.train import load_config
from tools.strict_source_smoke import tree_digest


ROOT = Path(__file__).resolve().parents[1]


def evidence_fixture():
    train, val = load_membership()
    config = load_config(ROOT / "configs/strict_dg/gta_dinov3l_m0.yaml")
    report = {
        "schema_version": 1, "phase": 17, "scope": "m0_strict_source_smoke_20_updates",
        "seed": 0, "git_sha": audit.SMOKE_PRODUCER_SHA, "git_status": "", "ok": True,
        "formal_training_authorized": False, "target_data_accessed": False,
        "authorized_optimizer_updates": 20, "authorized_holdout_samples": 8,
        "optimizer_updates": 20, "alpha_zero_error": 0.0,
        "model_roundtrip_exact": True, "optimizer_roundtrip_exact": True,
        "prediction_roundtrip_max_error": 0.0, "backbone_unchanged": True,
        "backbone_state_sha256": audit.BACKBONE_STATE_SHA,
        "checkpoint_sha256": audit.CHECKPOINT_SHA, "checkpoint_bytes": 45612179,
        "trace_sha256": audit.SMOKE_TRACE_SHA, "schedule_horizon": 40000, "sample_workers": 0,
        "amp": True, "checkpoint_policy": "fixed_final_only_no_selection", "exact_resume_verified": False,
        "manifest_sha256": MANIFEST_HASHES, "train_count": 22469, "val_count": 2497,
        "source_miou": None, "target_miou": None, "config": config,
        "source_data_audit": {"report_sha256": audit.DATA_REPORT_SHA256},
        "pretrained_checkpoint_sha256": config["model"]["weights_sha256"],
        "environment": {"torch": "2.7.0+cu126", "transformers": "4.56.1"},
        "source_ids": train[:20], "holdout_compatibility": [
            {"source_id": key, "ce": 1.0, "shape": [1, 576, 1024]} for key in val[:8]],
        "initial_trainable_state_sha256": "a" * 64, "final_trainable_state_sha256": "b" * 64,
        "peak_allocated_gib": 1.0, "peak_reserved_gib": 1.2,
    }
    gradients = {"decoder.weight": 1.0, "query_bank.class_anchors": 1.0,
                 "query_bank.query_residuals": 1.0, "query_head.feature_projection.weight": 1.0,
                 "query_head.query_projection.weight": 1.0, "query_head.alpha": 1.0}
    rows = []
    for step, key in enumerate(train[:20], 1):
        grad = dict(gradients)
        if step == 1:
            for name in grad:
                if name not in ("decoder.weight", "query_head.alpha"):
                    grad[name] = 0.0
        rows.append({"update": step, "loss_original": 1.0, "loss_photometric": 2.0,
                     "objective": 3.0, "gradient_norm_pre_clip": sum(x*x for x in grad.values()) ** 0.5,
                     "alpha": -step * 1e-4, "source_id": key, "input_sha256": "c" * 64,
                     "lr_used": 1e-4 * (1 - (step-1) / 40000) ** 0.9,
                     "next_lr": 1e-4 * (1 - step / 40000) ** 0.9, "gradients": grad})
    report["gradient_max"] = gradients
    return report, rows


def test_complete_smoke_evidence_contract():
    report, rows = evidence_fixture()
    assert audit.validate_smoke(report, rows)["smoke_accepted"]


@pytest.mark.parametrize("fault", ["schedule", "objective", "coverage", "gradient", "target", "reference"])
def test_smoke_audit_rejects_inconsistent_evidence(fault):
    report, rows = evidence_fixture()
    if fault == "schedule": rows[3]["lr_used"] *= 0.9
    if fault == "objective": rows[1]["objective"] += 1
    if fault == "coverage": rows.pop()
    if fault == "gradient": rows[0]["gradients"]["query_bank.class_anchors"] = 1
    if fault == "target": report["target_data_accessed"] = True
    if fault == "reference": report["holdout_compatibility"][0]["source_id"] = report["source_ids"][0]
    with pytest.raises(ValueError):
        audit.validate_smoke(report, rows)


def test_smoke_audit_checks_raw_bytes_before_metadata(tmp_path, monkeypatch):
    report, rows = evidence_fixture()
    rb = json.dumps(report).encode()
    tb = "\n".join(json.dumps(row) for row in rows).encode()
    (tmp_path / "smoke_report.json").write_bytes(rb)
    (tmp_path / "train.jsonl").write_bytes(tb)
    monkeypatch.setattr(audit, "SMOKE_REPORT_SHA", hashlib.sha256(rb).hexdigest())
    monkeypatch.setattr(audit, "SMOKE_TRACE_SHA", hashlib.sha256(tb).hexdigest())
    # Report's trace digest must agree with the pinned bytes as in production.
    report["trace_sha256"] = audit.SMOKE_TRACE_SHA
    rb = json.dumps(report).encode()
    (tmp_path / "smoke_report.json").write_bytes(rb)
    monkeypatch.setattr(audit, "SMOKE_REPORT_SHA", hashlib.sha256(rb).hexdigest())
    assert audit.audit_smoke(tmp_path)["smoke_accepted"]
    (tmp_path / "train.jsonl").write_bytes(tb + b" ")
    with pytest.raises(ValueError, match="trace SHA256"):
        audit.audit_smoke(tmp_path)


class ConstantModel(nn.Module):
    def forward(self, images):
        logits = images.new_zeros(images.shape[0], 3, *images.shape[-2:])
        logits[:, 0] = 2
        return logits


def holdout_fixture():
    labels = torch.zeros(16, 16, dtype=torch.long)
    labels[:4] = 1
    labels[-1] = 255
    return [{"id": "00001", "image": torch.zeros(3, 16, 16), "label": labels},
            {"id": "00002", "image": torch.zeros(3, 16, 16), "label": labels.clone()}]


def test_streaming_metrics_match_known_confusion_and_absent_classes(tmp_path):
    result = evaluate_holdout(ConstantModel(), holdout_fixture(), ["00001", "00002"], tmp_path,
                              num_classes=3, device="cpu", amp=False)
    assert result["confusion_matrix"] == [[352, 0, 0], [128, 0, 0], [0, 0, 0]]
    assert result["valid_pixel_count"] == 480
    assert result["class_iou"][2] is None
    assert result["source_miou"] == pytest.approx((352/480) / 2)
    assert result["confusion_counts_exact"] and result["evaluation_rng_unchanged"]
    assert result["seen_ids_sha256"] == hashlib.sha256(manifest_bytes(["00001", "00002"])).hexdigest()
    assert len((tmp_path / "source_eval.jsonl").read_text().splitlines()) == 2


def test_reference_replay_and_failures(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    dataset = holdout_fixture()
    evaluate_holdout(ConstantModel(), dataset, ["00001", "00002"], first,
                      num_classes=3, device="cpu", amp=False)
    row = json.loads((first / "source_eval.jsonl").read_text().splitlines()[0])
    reference = [{"source_id": row["source_id"], "shape": row["shape"], "ce": row["ce"]}]
    second = tmp_path / "second"
    second.mkdir()
    result = evaluate_holdout(ConstantModel(), dataset, ["00001", "00002"], second,
                              compatibility=reference, num_classes=3, device="cpu", amp=False)
    assert result["compatibility_ce_max_error"] == 0
    reference[0]["ce"] += 1
    third = tmp_path / "third"
    third.mkdir()
    with pytest.raises(ValueError, match="compatibility CE"):
        evaluate_holdout(ConstantModel(), dataset, ["00001", "00002"], third,
                          compatibility=reference, num_classes=3, device="cpu", amp=False)


@pytest.mark.parametrize("fault", ["coverage", "order", "labels", "nonfinite"])
def test_full_evaluation_fails_closed(tmp_path, fault):
    dataset = holdout_fixture()
    ids = ["00001", "00002"]
    model = ConstantModel()
    if fault == "coverage": ids.pop()
    if fault == "order": dataset[0]["id"] = "00003"
    if fault == "labels": dataset[0]["label"][:] = 255
    if fault == "nonfinite":
        class BadModel(nn.Module):
            def forward(self, x): return x * float("nan")
        model = BadModel()
    with pytest.raises((ValueError, RuntimeError)):
        evaluate_holdout(model, dataset, ids, tmp_path, num_classes=3, device="cpu", amp=False)


def test_launcher_requires_audit_before_cuda_or_model(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "_git", lambda *args: "" if args[0] == "status" else "d"*40)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA before smoke audit"))
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path / "checkout")
    output = tmp_path / "evidence"
    assert launcher.main(["--data-root", str(tmp_path/"data"), "--pretrained-root", str(tmp_path/"weights"),
                          "--smoke-dir", str(tmp_path/"absent"), "--output-dir", str(output)]) == 1
    report = json.loads((output / "source_evaluation.json").read_bytes())
    assert not report["ok"] and report["optimizer_updates_this_run"] == 0
    assert not report["target_data_accessed"]


def test_read_only_checkpoint_restore_checks_hash_metadata_and_state(tmp_path, monkeypatch):
    model = nn.Linear(3, 3)
    optimizer = torch.optim.AdamW(model.parameters())
    report, _ = evidence_fixture()
    report["final_trainable_state_sha256"] = tree_digest(trainable_state_dict(model))
    metadata = {"git_sha": audit.SMOKE_PRODUCER_SHA, "seed": 0, "target_data_accessed": False,
                "manifest_sha256": report["manifest_sha256"],
                "pretrained_checkpoint_sha256": report["pretrained_checkpoint_sha256"],
                "protocol": report["config"]["protocol"],
                "checkpoint_selection": "fixed_final_after_20_updates"}
    path = tmp_path / "final.pth"
    save_training_checkpoint(path, model=model, optimizer=optimizer, iteration=20, metadata=metadata)
    monkeypatch.setattr(launcher, "CHECKPOINT_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
    restored = nn.Linear(3, 3)
    assert launcher.restore_for_evaluation(restored, path, report) == 20
    assert tree_digest(trainable_state_dict(restored)) == report["final_trainable_state_sha256"]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        launcher.restore_for_evaluation(restored, path, report)
