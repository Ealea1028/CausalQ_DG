"""REIN optimizer, paired-view backward, and compact SCS training state."""

import torch
from torch import Tensor, nn

from causalq.models.scs_segmentor import SCSMask2Former


TOTAL_UPDATES = 40_000


def build_scs_optimizer(model: SCSMask2Former):
    norms = (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._NormBase)
    no_decay = {
        id(parameter)
        for name, module in model.named_modules()
        if isinstance(module, norms) or any(key in name for key in ("norm", "level_embed", "query_embed"))
        for parameter in module.parameters(recurse=False)
    }
    if model.backbone.injector is not None:
        no_decay.add(id(model.backbone.injector.query_bank))
    decay, exempt = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (exempt if id(parameter) in no_decay else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.05}, {"params": exempt, "weight_decay": 0.0}],
        lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda update: max(0.0, 1 - update / TOTAL_UPDATES)**0.9
    )
    return optimizer, scheduler


def paired_scs_backward(
    model: SCSMask2Former, original: Tensor, styled: Tensor, labels: Tensor,
    *, auxiliary_weight: float = 0.1, scale: float = 1.0, amp: bool = False,
    patch_valid: Tensor | None = None,
) -> Tensor:
    detached_losses = []
    for images in (original, styled):
        with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=amp):
            losses = model.loss(images, labels, auxiliary_weight=auxiliary_weight, patch_valid=patch_valid)
            objective = sum(losses.values()) * (0.5 * scale)
        objective.backward()
        detached_losses.append(objective.detach())
    return detached_losses[0] + detached_losses[1]


def scs_trainable_state(model: SCSMask2Former) -> dict:
    state = {"head": model.head.state_dict(), "query_embed": model.query_embed.state_dict()}
    if model.backbone.injector is not None:
        state["injector"] = model.backbone.injector.state_dict()
    return state


def load_scs_trainable_state(model: SCSMask2Former, state: dict) -> None:
    if state.keys() != scs_trainable_state(model).keys():
        raise ValueError("Checkpoint does not match the SCS experiment arm")
    if model.backbone.injector is not None:
        model.backbone.injector.load_state_dict(state["injector"], strict=True)
    model.head.load_state_dict(state["head"], strict=True)
    model.query_embed.load_state_dict(state["query_embed"], strict=True)
