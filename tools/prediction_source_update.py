"""P1 source-only optimizer update: original/photo CE plus detached-teacher KL."""

import torch

from causalq.losses.prediction_consistency import prediction_consistency_kl
from tools.deterministic_source_ce import deterministic_source_ce
from tools.strict_source_smoke import HORIZON, precision


def prediction_source_update(model, optimizer, image, photo, label, step, config, *, device, amp,
                             loss_fn=deterministic_source_ce):
    if not 1 <= step <= HORIZON:
        raise ValueError('P1 source optimizer step must be within the40000-step horizon')
    recipe = config['prediction_consistency']
    if (recipe['enabled'] is not True or recipe['reference_stop_gradient'] is not True
            or recipe['valid_pixels_only'] is not True or recipe['divergence'] != 'kl_reference_to_view'
            or recipe['lambda_pred'] < 0 or recipe['temperature'] <= 0):
        raise ValueError('Unsupported P1 consistency contract')
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    lr_used = optimizer.param_groups[0]['lr']
    with precision(device, amp):
        logits = model(image)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite P1 original logits')
        original = loss_fn(logits, label)
    if not torch.isfinite(original):
        raise FloatingPointError('Nonfinite P1 original CE')
    reference = logits.detach()
    original_value = float(original.detach())
    original.backward()
    del logits, original
    with precision(device, amp):
        logits = model(photo)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite P1 photometric logits')
        photometric = loss_fn(logits, label)
    # FP32 KL, valid labels only; original forward is reused without a third
    # backbone pass or an original-view consistency gradient.
    consistency = prediction_consistency_kl(logits, reference, labels=label, temperature=recipe['temperature'])
    if not torch.isfinite(photometric) or not torch.isfinite(consistency):
        raise FloatingPointError('Nonfinite P1 photometric CE/KL')
    weight = float(recipe['lambda_pred'])  # exactly one style view; no hidden /2
    objective = float(config['style']['lambda_cf']) * photometric + weight * consistency
    objective.backward()
    gradients = {name: float(p.grad.detach().float().norm()) for name, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None}
    norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True))
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in trainable):
        raise FloatingPointError('Nonfinite P1 parameter after update')
    next_lr = float(config['optimizer']['lr']) * (1 - step / HORIZON) ** float(config['optimizer']['poly_power'])
    for group in optimizer.param_groups:
        group['lr'] = next_lr
    photo_value, kl_value = float(photometric.detach()), float(consistency.detach())
    return {'update': step, 'loss_original': original_value, 'loss_photometric': photo_value,
            'loss_prediction_kl': kl_value, 'prediction_weight': weight,
            'objective': original_value + float(config['style']['lambda_cf']) * photo_value + weight * kl_value,
            'gradient_norm_pre_clip': norm, 'lr_used': lr_used, 'next_lr': next_lr,
            'alpha': float(model.query_head.alpha.detach()), 'gradients': gradients}
