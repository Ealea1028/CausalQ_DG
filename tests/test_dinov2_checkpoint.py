"""CPU checks for the staged DINOv2-L checkpoint inspector."""

import pytest
import torch

from tools.check_dinov2_checkpoint import inspect_state_dict


def fake_state_dict() -> dict[str, torch.Tensor]:
    tensors = {
        "patch_embed.proj.weight": torch.empty(1024, 3, 14, 14, device="meta"),
        "cls_token": torch.empty(1, 1, 1024, device="meta"),
        "pos_embed": torch.empty(1, 257, 1024, device="meta"),
        "norm.weight": torch.empty(1024, device="meta"),
    }
    for index in range(24):
        tensors[f"blocks.{index}.attn.qkv.weight"] = torch.empty(
            3072, 1024, device="meta"
        )
    return tensors


def test_inspector_accepts_expected_nonregister_architecture() -> None:
    report = inspect_state_dict(fake_state_dict())

    assert report["architecture"] == "dinov2_vitl14_no_registers"
    assert report["block_count"] == 24
    assert report["representative_shapes"]["patch_embed.proj.weight"] == [1024, 3, 14, 14]


def test_inspector_accepts_state_dict_wrapper() -> None:
    assert inspect_state_dict({"state_dict": fake_state_dict()})["block_count"] == 24


@pytest.mark.parametrize("bad_key", ["blocks.23.attn.qkv.weight", "pos_embed"])
def test_inspector_rejects_missing_required_tensor(bad_key: str) -> None:
    tensors = fake_state_dict()
    del tensors[bad_key]

    with pytest.raises(ValueError):
        inspect_state_dict(tensors)


def test_inspector_rejects_register_variant() -> None:
    tensors = fake_state_dict()
    tensors["register_tokens"] = torch.empty(1, 4, 1024, device="meta")

    with pytest.raises(ValueError, match="non-register"):
        inspect_state_dict(tensors)


def test_inspector_rejects_non_tensor_payload() -> None:
    tensors = fake_state_dict()
    tensors["note"] = "not a tensor"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="string-to-tensor"):
        inspect_state_dict(tensors)
