from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from causalq.losses import null_query_centroid_loss
from causalq.models import DINOv3Backbone, NullQueryBank, QuerySegmentor
from tools.train import load_config


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            patch_size=2,
            hidden_size=16,
            num_register_tokens=2,
        )
        self.projection = nn.Conv2d(3, 16, kernel_size=2, stride=2)

    def forward(self, *, pixel_values, output_hidden_states, return_dict):
        del return_dict
        patch_map = self.projection(pixel_values)
        patch_tokens = patch_map.flatten(2).transpose(1, 2)
        prefix = patch_tokens.new_zeros(patch_tokens.shape[0], 3, 16)
        hidden = torch.cat((prefix, patch_tokens), dim=1)
        states = (hidden, hidden * 0.5) if output_hidden_states else None
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=states)


def test_null_query_bank_shares_slots_across_classes() -> None:
    bank = NullQueryBank(8, queries_per_class=2)

    states = bank(batch_size=3, num_classes=4)

    assert states.shape == (3, 4, 2, 8)
    assert torch.equal(states[:, 0], states[:, 3])
    assert bank.null_queries.shape == (2, 8)


def test_null_intervention_replaces_only_selected_class() -> None:
    backbone = DINOv3Backbone(FakeBackbone(), freeze=True)
    model = QuerySegmentor(
        backbone,
        decoder_channels=32,
        num_classes=4,
        dropout=0.0,
        queries_per_class=2,
        num_heads=4,
        cross_attention_layers=0,
        interaction="static",
        learned_null=True,
        alpha_init=0.5,
    ).eval()

    output = model.forward_components(torch.randn(1, 3, 8, 8))
    counterfactual = output.null_counterfactual_logits(2)
    effect = model.get_null_query_effect(output=output)

    assert output.null_query_states is not None
    assert output.null_query_states.shape == (1, 4, 2, 16)
    assert torch.equal(counterfactual[:, :2], output.logits[:, :2])
    assert torch.equal(counterfactual[:, 3:], output.logits[:, 3:])
    assert torch.allclose(
        output.logits[:, 2] - counterfactual[:, 2],
        effect[:, 2],
        atol=1e-6,
    )


def test_null_centroid_loss_detaches_factual_target() -> None:
    null = torch.randn(1, 3, 2, 4, requires_grad=True)
    factual = torch.randn(1, 3, 2, 4, requires_grad=True)

    loss = null_query_centroid_loss(null, factual)
    loss.backward()

    assert loss.item() > 0
    assert null.grad is not None
    assert factual.grad is None


def test_learned_null_requires_static_interaction() -> None:
    backbone = DINOv3Backbone(FakeBackbone(), freeze=True)
    with pytest.raises(ValueError, match="requires static"):
        QuerySegmentor(backbone, learned_null=True, interaction="one_way")


def test_learned_null_preserves_same_seed_factual_initialization() -> None:
    def build(enabled: bool) -> QuerySegmentor:
        torch.manual_seed(20260926)
        backbone = DINOv3Backbone(FakeBackbone(), freeze=True)
        return QuerySegmentor(
            backbone,
            decoder_channels=32,
            num_classes=4,
            dropout=0.0,
            queries_per_class=2,
            num_heads=4,
            cross_attention_layers=0,
            interaction="static",
            learned_null=enabled,
        )

    reference = dict(build(False).named_parameters())
    candidate = dict(build(True).named_parameters())

    assert set(candidate) - set(reference) == {"null_query_bank.null_queries"}
    assert all(
        torch.equal(parameter, candidate[name])
        for name, parameter in reference.items()
    )


def test_phase14_config_isolates_learned_null() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs/learned_null/gta_dinov3l_static_null.yaml"
    )

    assert config["experiment"]["phase"] == 14
    assert config["query"]["queries_per_class"] == 2
    assert config["query"]["interaction"] == "static"
    assert config["style"]["views"] == ["original", "photometric"]
    assert config["learned_null"]["enabled"] is True
    assert config["learned_null"]["shared_across_classes"] is True
    assert config["learned_null"]["stop_gradient_target"] is True
    assert "prediction_consistency" not in config
    assert "causal_query_effect" not in config
    assert "query_diversity" not in config
