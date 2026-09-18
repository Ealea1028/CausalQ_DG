"""Compact checkpoints that exclude the frozen foundation-model weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.torch_version import TorchVersion


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable
    }


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "trainable_model": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_training_checkpoint(path: Path) -> dict[str, Any]:
    """Safely load a project checkpoint without enabling arbitrary pickle code."""
    # PyTorch exposes ``torch.__version__`` as a ``TorchVersion`` string
    # subclass. Training metadata stores that value, so PyTorch 2.6+ needs the
    # exact class in its weights-only allowlist.
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Training checkpoint payload must be a dictionary")
    required = {"iteration", "trainable_model", "optimizer", "metadata"}
    missing = required - payload.keys()
    if missing:
        raise KeyError(f"Training checkpoint is missing keys: {sorted(missing)}")
    return payload
