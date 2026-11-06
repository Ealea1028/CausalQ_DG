"""Two-view prediction effect of deleting one GT-present class path."""
import torch


def gt_region_prediction_effect(on_logits, off_logits, labels, category):
    """FP32 probability difference with the off branch as a fixed reference."""
    with torch.autocast(on_logits.device.type, enabled=False):
        on = on_logits.float().softmax(dim=1).index_select(1, category)[:, 0]
        off = off_logits.detach().float().softmax(dim=1).index_select(1, category)[:, 0]
        mask = labels == category.reshape(1, 1, 1)
        return ((on - off) * mask).sum() / mask.sum()


def effect_invariance_loss(effects):
    """Population variance of the original and photometric effects."""
    original, photo = effects.float().unbind()
    return (original - photo).square() / 4
