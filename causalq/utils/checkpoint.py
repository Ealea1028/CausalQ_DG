"""Compact checkpoints that exclude the frozen foundation-model weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


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
