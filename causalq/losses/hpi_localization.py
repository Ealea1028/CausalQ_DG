"""Weak B3 supervision on the router's class competition at each patch."""
import torch
from torch.nn import functional as F


def patch_localization_loss(match_logits, labels, grid, *, ignore_index=255):
    batch, classes, patches = match_logits.shape
    if labels.shape[0] != batch or patches != grid[0] * grid[1]:
        raise ValueError('Localization shapes do not match')
    category = torch.arange(classes, device=labels.device)[None, :, None, None]
    valid_pixels = labels[:, None] != ignore_index
    class_area = F.adaptive_avg_pool2d(((labels[:, None] == category) & valid_pixels).float(), grid)
    valid_area = class_area.sum(1, keepdim=True)
    targets = class_area / valid_area.clamp_min(1e-12)
    log_prob = match_logits.float().reshape(batch, classes, *grid).log_softmax(dim=1)
    per_patch = -(targets * log_prob).sum(1)
    counts = (valid_area[:, 0] > 0).sum((1, 2)).clamp_min(1)
    return (per_patch.sum((1, 2)) / counts).mean()
