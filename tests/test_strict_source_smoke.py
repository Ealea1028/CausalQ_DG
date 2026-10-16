"""CPU synthetic features verify the real smoke engine without DINO weights."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from causalq.interventions import StyleInterventionBank
from causalq.models import QuerySegmentor
from causalq.utils.checkpoint import save_training_checkpoint
from tools.audit_strict_source_data import audit_report, validate_report
from tools import check_strict_source_smoke as launcher
from tools.strict_source_smoke import (HORIZON, optimizer_update, restore_compact,
                                       run_bounded, tree_digest)


ROOT = Path(__file__).resolve().parents[1]


class TinyFrozenBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.freeze, self.hidden_size, self.intermediate_indices = True, 8, ()
        self.projection = nn.Conv2d(3, 8, 2, 2)
        self.requires_grad_(False)

    def forward(self, images):
        with torch.no_grad():
            feature = self.projection(images)
        return SimpleNamespace(patch_map=feature, patch_tokens=feature.flatten(2).transpose(1, 2),
                               intermediate_maps=())


def fixture():
    torch.manual_seed(0)
    backbone = TinyFrozenBackbone()

    def make_model():
        return QuerySegmentor(backbone, decoder_channels=32, num_classes=4, dropout=0.1,
                              queries_per_class=2, interaction="static", cross_attention_layers=0,
                              num_heads=2)

    config = {"style": {"lambda_cf": 1.0}, "optimizer": {"lr": 1e-4, "weight_decay": 0.01,
                                                           "poly_power": 0.9}}
    train = [{"id": f"train{i}", "image": torch.randn(3, 8, 8),
              "label": torch.randint(0, 4, (8, 8))} for i in range(24)]
    holdout = [{"id": f"holdout{i}", "image": torch.randn(3, 8, 8),
                "label": torch.randint(0, 4, (8, 8))} for i in range(8)]
    return make_model, config, train, holdout


def test_returned_data_evidence_passes_hash_and_protocol_audit(tmp_path):
    report_path = ROOT / "experiments/STRICT_DG_PHASE17/source_data_report.json"
    assert audit_report(report_path)["source_data_accepted"]
    modified = tmp_path / "report.json"
    modified.write_bytes(report_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        audit_report(modified)


@pytest.mark.parametrize("section,key,value", [
    (None, "target_data_accessed", True), (None, "optimizer_updates", 20),
    ("train_scan", "scanned_label_count", 100), ("val_scan", "invalid_train_ids", [33]),
    ("preprocessing", "holdout_torch_rng_unchanged", False),
])
def test_report_invariants_reject_false_acceptance(section, key, value):
    report = json.loads((ROOT / "experiments/STRICT_DG_PHASE17/source_data_report.json").read_bytes())
    (report if section is None else report[section])[key] = value
    with pytest.raises(ValueError):
        validate_report(report)


def test_digest_supports_bfloat16_scalar_and_optimizer_trees():
    value = {"a": torch.tensor(0.25, dtype=torch.bfloat16), "b": [0, 1.0, "x"]}
    assert tree_digest(value) == tree_digest(deepcopy(value))
    changed = deepcopy(value)
    changed["a"] += 1
    assert tree_digest(value) != tree_digest(changed)


def test_update_uses_two_view_sum_and_formal_schedule_horizon():
    make, config, train, _ = fixture()
    model = make().train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    image = train[0]["image"].unsqueeze(0)
    label = train[0]["label"].unsqueeze(0)
    record = optimizer_update(model, optimizer, image, image * 0.9, label, 1, config,
                              device="cpu", amp=False)
    assert record["objective"] == record["loss_original"] + record["loss_photometric"]
    assert record["lr_used"] == 1e-4
    assert record["next_lr"] == pytest.approx(1e-4 * (1 - 1 / HORIZON) ** 0.9)
    assert all(p.grad is None for p in model.backbone.parameters())
    with pytest.raises(ValueError, match="20 bounded"):
        optimizer_update(model, optimizer, image, image, label, 21, config, device="cpu", amp=False)


def test_full_20_update_engine_checkpoint_and_source_holdout(tmp_path):
    make, config, train, holdout = fixture()
    metadata = {"test": "synthetic_only", "target_data_accessed": False}
    result = run_bounded(make(), make, train, holdout, StyleInterventionBank(), config, tmp_path,
                         device="cpu", amp=False, metadata=metadata)
    assert result["ok"] and result["optimizer_updates"] == 20
    assert len(set(result["source_ids"])) == 20 and len(result["holdout_compatibility"]) == 8
    assert result["backbone_unchanged"]
    assert result["model_roundtrip_exact"] and result["optimizer_roundtrip_exact"]
    assert result["prediction_roundtrip_max_error"] == 0
    assert result["schedule_horizon"] == 40000
    assert result["source_miou"] is None and result["target_miou"] is None
    assert not result["exact_resume_verified"]
    assert len((tmp_path / "train.jsonl").read_text().splitlines()) == 20
    assert len(list(tmp_path.glob("*.pth"))) == 1


def test_compact_restore_rejects_incomplete_model_coverage(tmp_path):
    make, _, _, _ = fixture()
    model = make()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    path = tmp_path / "bad.pth"
    save_training_checkpoint(path, model=model, optimizer=optimizer, iteration=20, metadata={})
    payload = torch.load(path, weights_only=True)
    del payload["trainable_model"]["query_head.alpha"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="coverage mismatch"):
        restore_compact(make(), optimizer, path)


def test_real_launcher_refuses_cpu_before_loading_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "_git", lambda *args: "" if args[0] == "status" else "a" * 40)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(launcher.DINOv3Backbone, "from_pretrained",
                        lambda *a, **k: pytest.fail("local real weights attempted"))
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path / "checkout")
    # Keep real data-report/config audits, but make the synthetic workspace own
    # their small metadata files so no external outputs are required.
    repo = launcher.PROJECT_ROOT
    for relative in ("experiments/STRICT_DG_PHASE17/source_data_report.json",
                     "configs/strict_dg/gta_dinov3l_m0.yaml"):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    output = tmp_path / "cpu_failure"
    assert launcher.main(["--data-root", str(tmp_path / "data"),
                          "--pretrained-root", str(tmp_path / "weights"),
                          "--output-dir", str(output)]) == 1
    report = json.loads((output / "smoke_report.json").read_bytes())
    assert not report["ok"] and "AutoDL CUDA" in report["error"]
    assert not report["target_data_accessed"] and not report["formal_training_authorized"]
