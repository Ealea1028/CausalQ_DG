from types import SimpleNamespace

import pytest
import torch
from torch import nn

from causalq.models.dinov3_wrapper import DINOv3Backbone
from tools.check_backbone import checkpoint_hashes


class FakeDINOv3(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int = 16,
        hidden_size: int = 8,
        num_register_tokens: int = 2,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_register_tokens=num_register_tokens,
        )
        self.projection = nn.Linear(3, hidden_size)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        del return_dict
        batch, _, height, width = pixel_values.shape
        patches = (height // self.config.patch_size) * (
            width // self.config.patch_size
        )
        prefix = 1 + self.config.num_register_tokens
        base = self.projection(pixel_values.mean(dim=(-2, -1))).unsqueeze(1)
        hidden = base.expand(batch, prefix + patches, self.config.hidden_size)
        hidden_states = (hidden * 0.5, hidden) if output_hidden_states else None
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=hidden_states,
        )


def test_wrapper_removes_dynamic_prefix_and_returns_patch_map() -> None:
    wrapper = DINOv3Backbone(
        FakeDINOv3(num_register_tokens=2),
        freeze=True,
        intermediate_indices=(0,),
    )

    features = wrapper(torch.randn(2, 3, 32, 48))

    assert wrapper.num_prefix_tokens == 3
    assert features.patch_tokens.shape == (2, 6, 8)
    assert features.patch_map.shape == (2, 8, 2, 3)
    assert features.intermediate_maps[0].shape == (2, 8, 2, 3)
    assert not features.patch_map.requires_grad
    assert all(not parameter.requires_grad for parameter in wrapper.parameters())


def test_wrapper_keeps_frozen_model_in_eval_mode() -> None:
    wrapper = DINOv3Backbone(FakeDINOv3(), freeze=True)

    wrapper.train()

    assert wrapper.training is True
    assert wrapper.model.training is False


def test_wrapper_rejects_non_divisible_image_size() -> None:
    wrapper = DINOv3Backbone(FakeDINOv3(), freeze=True)

    with pytest.raises(ValueError, match="must be divisible"):
        wrapper(torch.randn(1, 3, 31, 32))


def test_wrapper_rejects_unexpected_token_count() -> None:
    model = FakeDINOv3()
    wrapper = DINOv3Backbone(model, freeze=True)
    model.config.num_register_tokens = 3

    with pytest.raises(RuntimeError, match="Unexpected DINOv3 token shape"):
        wrapper(torch.randn(1, 3, 32, 32))


def test_checkpoint_hashes_safetensors_only(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"abc")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    assert checkpoint_hashes(tmp_path) == {
        "model.safetensors": (
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad"
        )
    }
