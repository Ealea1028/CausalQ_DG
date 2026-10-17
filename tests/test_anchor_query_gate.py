from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from causalq.models import QuerySegmentorOutput
from causalq.utils.checkpoint import trainable_state_dict
from tools import check_anchor_query_effect as gate
from tools.strict_source_smoke import tree_digest
from test_anchor_query_effect import make_model

ROOT = Path(__file__).resolve().parents[1]


def test_component_checker_accepts_shared_feature_effects():
    model = make_model()
    with torch.no_grad():
        model.query_head.alpha.fill_(.3)
        output = model.forward_components(torch.randn(1, 3, 8, 8), include_anchor=True)
        result = gate.component_checks(model, output)
    assert result["counterfactual_routing_exact"] and result["decomposition_max_error"] <= 1e-6
    assert model.backbone.calls == 1


def test_quantized_base_addition_roundoff_is_not_effect_definition():
    model = make_model()
    base = torch.full((1, 4, 1, 1), 4., dtype=torch.bfloat16)
    full = torch.full_like(base, .015625)
    anchor = torch.full_like(base, .00390625)
    output = QuerySegmentorOutput(logits=base + full, base_logits=base, delta_logits=full,
        scaled_delta_logits=full, query_states=torch.zeros(1, 4, 2, 8),
        query_score_maps=torch.zeros(1, 4, 2, 1, 1), anchor_logits=base + anchor,
        anchor_scaled_delta_logits=anchor)
    result = gate.component_checks(model, output)
    assert result["decomposition_max_error"] == 0
    assert result["base_addition_roundoff_max"] == .00390625


def test_model_only_restore_checks_hash_mode_metadata_and_coverage(tmp_path, monkeypatch):
    model = make_model()
    payload = {"iteration": 20, "metadata": {"centered_residuals": True},
               "trainable_model": trainable_state_dict(model), "optimizer": {}}
    path = tmp_path / "state.pth"
    torch.save(payload, path)
    monkeypatch.setattr(gate, "CHECKPOINT_SHA", gate.file_sha(path))
    report = {"metadata": payload["metadata"], "head_sha256": tree_digest(payload["trainable_model"])}
    clone = make_model()
    gate.restore_anchor_source(clone, path, report)
    assert tree_digest(trainable_state_dict(clone)) == report["head_sha256"]
    with pytest.raises(ValueError, match="semantic/provenance"):
        gate.restore_anchor_source(make_model(centered=False), path, report)
    broken = deepcopy(payload)
    del broken["trainable_model"]["query_head.alpha"]
    torch.save(broken, path)
    monkeypatch.setattr(gate, "CHECKPOINT_SHA", gate.file_sha(path))
    with pytest.raises(ValueError, match="coverage"):
        gate.restore_anchor_source(clone, path, report)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        gate.restore_anchor_source(clone, path, report)


def test_gate_audits_original_before_cuda_or_real_weights(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(gate, "ROOT", tmp_path / "checkout")
    destination = gate.ROOT / "experiments/STRICT_DG_PHASE18"
    destination.mkdir(parents=True)
    for name in ("centered_report.json", "centered_SHA256SUMS"):
        (destination / name).write_bytes((ROOT / "experiments/STRICT_DG_PHASE18" / name).read_bytes())
    monkeypatch.setattr(gate, "_git", lambda *a: "" if a[0] == "status" else "a" * 40)
    monkeypatch.delitem(sys.modules, "tools.train", raising=False)
    monkeypatch.delitem(sys.modules, "causalq.datasets.cityscapes", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried before saved-artifact audit"))
    monkeypatch.setattr(gate.DINOv3Backbone, "from_pretrained", lambda *a, **k: pytest.fail("Local real weights attempted"))
    output = tmp_path / "out"
    assert gate.main(["--data-root", str(tmp_path / "data"), "--pretrained-root", str(tmp_path / "weights"),
                      "--centered-dir", str(tmp_path / "missing"), "--output-dir", str(output)]) == 1
    report = json.loads((output / "anchor_report.json").read_bytes())
    assert report["centered_audit"]["centered_query_accepted"]
    assert "FileNotFoundError" in report["error"] and report["optimizer_updates_this_run"] == 0
