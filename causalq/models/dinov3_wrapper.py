"""DINOv3 ViT feature wrapper for dense prediction."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn


DINOV3_MODEL_IDS = {
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

DINOV3_MODEL_SPECS = {
    "dinov3_vitb16": {"patch_size": (16, 16), "hidden_size": 768},
    "dinov3_vitl16": {"patch_size": (16, 16), "hidden_size": 1024},
}


@dataclass(frozen=True)
class DINOv3Features:
    """Dense patch features returned without CLS or register tokens."""

    patch_map: Tensor
    patch_tokens: Tensor
    intermediate_maps: tuple[Tensor, ...] = ()


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"Expected a scalar or pair, got {value!r}")
    return int(value[0]), int(value[1])


def _read_num_prefix_tokens(model: nn.Module) -> int:
    """Read the prefix-token count from model metadata, never a fixed slice."""
    for owner in (model, getattr(model, "embeddings", None)):
        value = getattr(owner, "num_prefix_tokens", None)
        if value is not None:
            return int(value)

    config = getattr(model, "config", None)
    value = getattr(config, "num_prefix_tokens", None)
    if value is not None:
        return int(value)

    register_count = getattr(config, "num_register_tokens", None)
    if register_count is None:
        raise ValueError(
            "DINOv3 model does not expose num_prefix_tokens or "
            "config.num_register_tokens"
        )
    # Hugging Face DINOv3 ViTs always prepend one CLS token and expose the
    # variable register-token count in their own configuration.
    return 1 + int(register_count)


class DINOv3Backbone(nn.Module):
    """Expose DINOv3 ViT patch tokens as a BCHW feature map."""

    def __init__(
        self,
        model: nn.Module,
        *,
        freeze: bool = True,
        intermediate_indices: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.model = model
        self.freeze = freeze
        self.intermediate_indices = tuple(int(index) for index in intermediate_indices)
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("DINOv3 model must expose a config object")
        self.patch_size = _pair(config.patch_size)
        self.hidden_size = int(config.hidden_size)
        self.num_prefix_tokens = _read_num_prefix_tokens(model)
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        weights: str | Path | None = None,
        freeze: bool = True,
        intermediate_indices: Sequence[int] = (),
        dtype: torch.dtype | None = None,
        local_files_only: bool = True,
    ) -> "DINOv3Backbone":
        """Load an official Hugging Face DINOv3 B/16 or L/16 checkpoint."""
        if model not in DINOV3_MODEL_IDS:
            choices = ", ".join(sorted(DINOV3_MODEL_IDS))
            raise ValueError(f"Unsupported DINOv3 model {model!r}; choose from {choices}")
        from transformers import AutoModel

        source = str(weights) if weights is not None else DINOV3_MODEL_IDS[model]
        backbone = AutoModel.from_pretrained(
            source,
            dtype=dtype,
            local_files_only=local_files_only,
        )
        wrapper = cls(
            backbone,
            freeze=freeze,
            intermediate_indices=intermediate_indices,
        )
        expected = DINOV3_MODEL_SPECS[model]
        if (
            wrapper.patch_size != expected["patch_size"]
            or wrapper.hidden_size != expected["hidden_size"]
        ):
            raise RuntimeError(
                f"Checkpoint architecture does not match {model}: got patch size "
                f"{wrapper.patch_size} and hidden size {wrapper.hidden_size}"
            )
        return wrapper

    def train(self, mode: bool = True) -> "DINOv3Backbone":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _to_patch_map(
        self,
        hidden_state: Tensor,
        grid_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        expected_patches = grid_size[0] * grid_size[1]
        expected_tokens = self.num_prefix_tokens + expected_patches
        if hidden_state.ndim != 3 or hidden_state.shape[1] != expected_tokens:
            raise RuntimeError(
                "Unexpected DINOv3 token shape: "
                f"got {tuple(hidden_state.shape)}, expected sequence length "
                f"{self.num_prefix_tokens} prefix + {expected_patches} patches"
            )
        patch_tokens = hidden_state[:, self.num_prefix_tokens :, :]
        patch_map = patch_tokens.transpose(1, 2).reshape(
            hidden_state.shape[0], self.hidden_size, *grid_size
        )
        return patch_map.contiguous(), patch_tokens

    def forward(self, images: Tensor) -> DINOv3Features:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected BCHW RGB images, got {tuple(images.shape)}")
        patch_height, patch_width = self.patch_size
        image_height, image_width = images.shape[-2:]
        if image_height % patch_height or image_width % patch_width:
            raise ValueError(
                f"Image size {(image_height, image_width)} must be divisible by "
                f"patch size {self.patch_size}"
            )
        grid_size = image_height // patch_height, image_width // patch_width
        context = torch.no_grad() if self.freeze else nullcontext()
        with context:
            outputs = self.model(
                pixel_values=images,
                output_hidden_states=bool(self.intermediate_indices),
                return_dict=True,
            )
        patch_map, patch_tokens = self._to_patch_map(
            outputs.last_hidden_state, grid_size
        )
        intermediate_maps: list[Tensor] = []
        if self.intermediate_indices:
            if outputs.hidden_states is None:
                raise RuntimeError("DINOv3 did not return requested hidden states")
            for index in self.intermediate_indices:
                feature_map, _ = self._to_patch_map(outputs.hidden_states[index], grid_size)
                intermediate_maps.append(feature_map)
        return DINOv3Features(
            patch_map=patch_map,
            patch_tokens=patch_tokens,
            intermediate_maps=tuple(intermediate_maps),
        )
