"""The frozen valid-pixel FP32 CE contract, usable without legacy tools."""
import torch
from torch.nn import functional as F
BACKEND = 'fp32_log_softmax_gather_valid_mean_v1'


def deterministic_source_ce(logits, labels, *, ignore_index=255):
    if logits.ndim != 4 or labels.ndim != 3 or logits.shape[0] != labels.shape[0] or logits.shape[-2:] != labels.shape[-2:]:
        raise ValueError('Expected matching BCHW logits and BHW labels')
    if labels.dtype != torch.long:
        raise ValueError('Source labels must have torch.long dtype')
    valid = labels != ignore_index
    count = valid.sum()
    if not count:
        raise ValueError('Segmentation batch contains no valid pixels')
    if ((labels[valid] < 0) | (labels[valid] >= logits.shape[1])).any():
        raise ValueError('Source class ID outside logits class range')
    loss = -F.log_softmax(logits.float(), dim=1).gather(1, labels.masked_fill(~valid, 0)[:, None]).squeeze(1)
    return loss.masked_fill(~valid, 0).sum() / count
