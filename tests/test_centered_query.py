from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from causalq.models import GroupedCausalQueryBank, QuerySegmentor
from causalq.protocols.centered_query import validate_centered_config
from causalq.utils.checkpoint import save_training_checkpoint
from tools import check_centered_query as launcher
from tools.strict_source_smoke import tree_digest
from test_strict_source_smoke import TinyFrozenBackbone

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("batch", [1, 4])
def test_centered_mean_and_query_mean_anchor(count, batch):
    bank = GroupedCausalQueryBank(16, num_classes=4, queries_per_class=count, centered_residuals=True)
    queries = bank(batch)
    assert queries.shape == (batch, 4, count, 16)
    torch.testing.assert_close(bank.get_centered_residuals().mean(1), torch.zeros(4, 16), atol=1e-8, rtol=0)
    torch.testing.assert_close(queries.mean(2), bank.get_anchors().expand(batch, -1, -1), atol=1e-8, rtol=1e-6)
    assert bank.get_query_residuals() is bank.query_residuals
    assert bank.get_anchors() is bank.class_anchors


@pytest.mark.parametrize("count", [1, 2, 3])
def test_centering_projects_gradient_and_removes_shared_residual_shift(count):
    bank = GroupedCausalQueryBank(16, num_classes=4, queries_per_class=count, centered_residuals=True)
    before = bank(2).detach().clone()
    with torch.no_grad():
        bank.query_residuals += torch.randn(4, 1, 16) * .1
    torch.testing.assert_close(bank(2), before, atol=5e-8, rtol=1e-5)
    (bank(2) * torch.randn_like(before)).sum().backward()
    torch.testing.assert_close(bank.query_residuals.grad.sum(1), torch.zeros(4, 16), atol=1e-6, rtol=0)
    if count == 1:
        assert torch.count_nonzero(bank.query_residuals.grad) == 0
        assert torch.count_nonzero(bank.get_centered_residuals()) == 0


def test_legacy_default_keeps_same_initialization_parameters_and_forward():
    torch.manual_seed(41)
    default = GroupedCausalQueryBank(16, num_classes=4, queries_per_class=2)
    torch.manual_seed(41)
    centered = GroupedCausalQueryBank(16, num_classes=4, queries_per_class=2, centered_residuals=True)
    assert tree_digest(default.state_dict()) == tree_digest(centered.state_dict())
    assert set(default.state_dict()) == {"class_anchors", "query_residuals"}
    assert torch.equal(default(1)[0], default.class_anchors[:, None] + default.query_residuals)
    assert not torch.equal(default(1), centered(1))


def make_model(centered=True, count=2):
    return QuerySegmentor(TinyFrozenBackbone(), decoder_channels=32, num_classes=4, dropout=0.1,
                          queries_per_class=count, interaction="static", num_heads=2,
                          cross_attention_layers=0, centered_residuals=centered)


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("batch", [1, 2])
def test_centered_alpha_zero_is_base_path(count, batch):
    model = make_model(count=count).eval()
    with torch.no_grad():
        output = model.forward_components(torch.randn(batch, 3, 8, 8))
    assert torch.equal(output.logits, output.base_logits)
    assert model.get_query_residuals() is model.query_bank.query_residuals
    torch.testing.assert_close(model.get_centered_residuals().mean(1), torch.zeros(4, 8), atol=1e-8, rtol=0)


def test_checkpoint_roundtrip_requires_explicit_centered_mode(tmp_path):
    model = make_model().eval()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    with torch.no_grad():
        model.query_head.alpha.fill_(.25)
    metadata = {"centered_residuals": True, "seed": 0}
    path = tmp_path / "state.pth"
    save_training_checkpoint(path, model=model, optimizer=optimizer, iteration=20, metadata=metadata)
    clone = make_model().eval()
    # Compact checkpoints omit the foundation state, which must be independently fixed.
    clone.backbone.load_state_dict(model.backbone.state_dict())
    restored_optimizer = torch.optim.AdamW([p for p in clone.parameters() if p.requires_grad])
    launcher.restore_centered(clone, restored_optimizer, path, metadata)
    image = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        assert torch.equal(model(image), clone(image))
    with pytest.raises(ValueError, match="semantic mode"):
        launcher.restore_centered(make_model(False), restored_optimizer, path, metadata)
    with pytest.raises(ValueError, match="semantic mode"):
        launcher.restore_centered(clone, restored_optimizer, path, {"centered_residuals": False})


def test_m1_config_changes_only_parameterization():
    config = yaml.safe_load((ROOT / "configs/strict_dg/gta_dinov3l_m1_centered.yaml").read_text())
    validate_centered_config(config)
    for section, key, value in (("query", "queries_per_class", 3), ("query", "centered_residuals", False),
                                ("style", "lambda_cf", 2), ("protocol", "training_authorized", True)):
        modified = deepcopy(config)
        modified[section][key] = value
        with pytest.raises(ValueError):
            validate_centered_config(modified)


def test_centered_launcher_stops_before_weights_on_cpu(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(launcher, "ROOT", tmp_path / "checkout")
    for relative in ("configs/strict_dg/gta_dinov3l_m1_centered.yaml",
                     "experiments/STRICT_DG_PHASE17/pilot_report.json",
                     "experiments/STRICT_DG_PHASE17/pilot_SHA256SUMS"):
        path = launcher.ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    monkeypatch.setattr(launcher, "_git", lambda *a: "" if a[0] == "status" else "a" * 40)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(launcher.DINOv3Backbone, "from_pretrained", lambda *a, **k: pytest.fail("Local real weights attempted"))
    monkeypatch.delitem(sys.modules, "tools.train", raising=False)
    monkeypatch.delitem(sys.modules, "causalq.datasets.cityscapes", raising=False)
    output = tmp_path / "gate"
    assert launcher.main(["--data-root", str(tmp_path / "data"), "--pretrained-root", str(tmp_path / "weights"),
                          "--output-dir", str(output)]) == 1
    import json
    report = json.loads((output / "centered_report.json").read_bytes())
    assert report["source_pilot_audit"]["source_pilot_accepted"]
    assert not report["ok"] and "AutoDL CUDA" in report["error"]
