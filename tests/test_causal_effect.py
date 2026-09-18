from pathlib import Path

import torch
from torch import nn

from causalq.models import DINOv3Backbone, QuerySegmentor
from tools.train import load_config


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {
            "patch_size": 2,
            "hidden_size": 8,
            "num_register_tokens": 0,
        })()
        self.projection = nn.Conv2d(3, 8, kernel_size=2, stride=2)

    def forward(self, *, pixel_values, output_hidden_states, return_dict):
        del return_dict
        patch_map = self.projection(pixel_values)
        tokens = patch_map.flatten(2).transpose(1, 2)
        prefix = tokens.new_zeros(tokens.shape[0], 1, tokens.shape[2])
        hidden = torch.cat((prefix, tokens), dim=1)
        states = (hidden, hidden) if output_hidden_states else None
        return type("Output", (), {
            "last_hidden_state": hidden,
            "hidden_states": states,
        })()


ROOT = Path(__file__).resolve().parents[1]


def test_causal_effect_contract_is_recorded() -> None:
    method = (ROOT / "docs/METHOD.md").read_text(encoding="utf-8")
    config = (ROOT / "configs/causalq/gta_dinov3l_causalq.yaml").read_text(encoding="utf-8")
    assert "do(Q_c = 0)" in method
    assert "effect_space: logits" in config
    assert "present_classes_only: true" in config

    parsed = load_config(ROOT / "configs/causalq/gta_dinov3l_causalq.yaml")
    assert parsed["experiment"]["phase"] == 9
    assert parsed["causal_query_effect"]["enabled"] is True
    assert "prediction_consistency" not in parsed


def test_public_query_effect_matches_explicit_intervention() -> None:
    backbone = DINOv3Backbone(
        _FakeBackbone(), freeze=True, intermediate_indices=(0, 1)
    )
    model = QuerySegmentor(
        backbone,
        decoder_channels=32,
        num_classes=3,
        queries_per_class=2,
        num_heads=2,
        dropout=0.0,
    )
    model.query_head.alpha.data.fill_(0.5)
    images = torch.randn(1, 3, 8, 8)

    output = model.forward_components(images)
    effect = model.get_query_effect(output=output)

    assert torch.equal(effect, output.scaled_delta_logits)
    for class_index in range(3):
        counterfactual = output.counterfactual_logits(class_index)
        assert torch.allclose(
            output.logits[:, class_index] - counterfactual[:, class_index],
            effect[:, class_index],
        )
