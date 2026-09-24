from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from torch import nn

from causalq.models import (
    DINOv3Backbone,
    QueryResidualHead,
    QuerySegmentor,
)
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


def test_query_residual_head_shape_and_scale() -> None:
    head = QueryResidualHead(
        16,
        num_classes=4,
        queries_per_class=3,
        alpha_init=0.0,
    )
    image_tokens = torch.randn(2, 20, 16)
    queries = torch.randn(2, 4, 3, 16)

    scores = head.query_scores(image_tokens, queries, grid_size=(4, 5))
    delta = head.delta_logits(image_tokens, queries, grid_size=(4, 5))
    scaled = head(image_tokens, queries, grid_size=(4, 5))

    assert scores.shape == (2, 4, 3, 4, 5)
    assert delta.shape == (2, 4, 4, 5)
    assert torch.allclose(delta, torch.logsumexp(scores, dim=2))
    assert torch.count_nonzero(delta) > 0
    assert torch.count_nonzero(scaled) == 0


def test_alpha_zero_preserves_baseline_output() -> None:
    backbone = DINOv3Backbone(
        FakeBackbone(),
        freeze=True,
        intermediate_indices=(0, 1),
    )
    model = QuerySegmentor(
        backbone,
        decoder_channels=32,
        num_classes=4,
        dropout=0.0,
        queries_per_class=3,
        num_heads=4,
        alpha_init=0.0,
    ).eval()
    images = torch.randn(2, 3, 8, 8)

    output = model.forward_components(images)

    assert output.logits.shape == (2, 4, 8, 8)
    assert output.delta_logits.shape == (2, 4, 8, 8)
    assert output.query_states.shape == (2, 4, 3, 16)
    assert output.query_score_maps.shape == (2, 4, 3, 4, 4)
    assert torch.equal(
        model.get_query_score_maps(output=output),
        output.query_score_maps,
    )
    assert torch.max(torch.abs(output.logits - output.base_logits)).item() < 1e-6


def test_query_score_map_accessor_requires_one_source() -> None:
    backbone = DINOv3Backbone(FakeBackbone(), freeze=True)
    model = QuerySegmentor(
        backbone,
        decoder_channels=32,
        num_classes=4,
        dropout=0.0,
        queries_per_class=2,
        num_heads=4,
    ).eval()
    images = torch.randn(1, 3, 8, 8)

    from_images = model.get_query_score_maps(images)

    assert from_images.shape == (1, 4, 2, 4, 4)
    with pytest.raises(ValueError, match="exactly one"):
        model.get_query_score_maps()


def test_counterfactual_removes_only_selected_class_residual() -> None:
    backbone = DINOv3Backbone(FakeBackbone(), freeze=True)
    model = QuerySegmentor(
        backbone,
        decoder_channels=32,
        num_classes=4,
        dropout=0.0,
        queries_per_class=3,
        num_heads=4,
        alpha_init=0.5,
    ).eval()
    output = model.forward_components(torch.randn(1, 3, 8, 8))

    counterfactual = output.counterfactual_logits(2)

    assert torch.allclose(counterfactual[:, 2], output.base_logits[:, 2])
    assert torch.equal(counterfactual[:, :2], output.logits[:, :2])
    assert torch.equal(counterfactual[:, 3:], output.logits[:, 3:])


def test_static_queries_skip_image_cross_attention() -> None:
    backbone = DINOv3Backbone(FakeBackbone(), freeze=True)
    model = QuerySegmentor(
        backbone,
        decoder_channels=32,
        num_classes=4,
        dropout=0.0,
        queries_per_class=2,
        num_heads=4,
        interaction="static",
    ).eval()
    images = torch.randn(2, 3, 8, 8)

    output = model.forward_components(images)
    expected = model.query_bank(2)

    assert model.query_attention is None
    assert model.interaction == "static"
    assert torch.equal(output.query_states, expected)
    assert output.logits.shape == (2, 4, 8, 8)
    assert not any("query_attention" in name for name, _ in model.named_parameters())


def test_query_interaction_rejects_unknown_mode() -> None:
    backbone = DINOv3Backbone(FakeBackbone(), freeze=True)

    with pytest.raises(ValueError, match="one_way.*static"):
        QuerySegmentor(backbone, interaction="bidirectional")


def test_phase6_config_enables_only_query_mechanism() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/query/gta_dinov3l_query.yaml")

    assert config["experiment"]["phase"] == 6
    assert config["model"]["freeze_backbone"] is True
    assert config["model"]["query"] is True
    assert config["query"]["queries_per_class"] == 3
    assert config["query"]["cross_attention_layers"] == 1
    assert config["query"]["aggregation"] == "logsumexp"
    assert config["train"]["style"] is False


def test_phase13_static_query_config_is_isolated() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs/query_interaction/gta_dinov3l_static.yaml"
    )

    assert config["experiment"]["phase"] == 13
    assert config["query"]["queries_per_class"] == 2
    assert config["query"]["interaction"] == "static"
    assert config["query"]["cross_attention_layers"] == 0
    assert config["style"]["views"] == ["original", "photometric"]
    assert "prediction_consistency" not in config
    assert "causal_query_effect" not in config
    assert "query_diversity" not in config
