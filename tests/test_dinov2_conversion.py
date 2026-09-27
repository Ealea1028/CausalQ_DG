"""CPU checks for the REIN-specific DINOv2 tensor conversion."""

import pytest
import torch
import torch.nn.functional as F

from tools.convert_dinov2_for_rein import convert_state_dict


def test_conversion_matches_rein_interpolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tools.convert_dinov2_for_rein.inspect_state_dict", lambda _: {}
    )
    patch = torch.arange(3 * 14 * 14, dtype=torch.float32).reshape(1, 3, 14, 14)
    position = torch.arange(1370, dtype=torch.float32).reshape(1, 1370, 1)
    position = position.expand(1, 1370, 1024).clone()
    state = {"patch_embed.proj.weight": patch, "pos_embed": position,
             "unchanged": torch.tensor([7.0])}

    converted = convert_state_dict(state)

    expected_patch = F.interpolate(
        patch, size=(16, 16), mode="bicubic", align_corners=False
    )
    expected_positions = F.interpolate(
        position[:, 1:, :].reshape(1, 37, 37, 1024).permute(0, 3, 1, 2),
        size=(32, 32), mode="bicubic", align_corners=False,
    ).permute(0, 2, 3, 1).reshape(1, 1024, 1024)
    assert torch.equal(converted["patch_embed.proj.weight"], expected_patch)
    assert torch.equal(converted["pos_embed"][:, :1, :], position[:, :1, :])
    assert torch.equal(converted["pos_embed"][:, 1:, :], expected_positions)
    assert converted["unchanged"] is state["unchanged"]
    assert tuple(state["patch_embed.proj.weight"].shape) == (1, 3, 14, 14)


def test_conversion_rejects_wrong_position_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tools.convert_dinov2_for_rein.inspect_state_dict", lambda _: {}
    )
    state = {
        "patch_embed.proj.weight": torch.empty(1, 3, 14, 14),
        "pos_embed": torch.empty(1, 257, 1024),
    }
    with pytest.raises(ValueError, match="37x37"):
        convert_state_dict(state)
