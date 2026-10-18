"""M2 Full CQE control with the original public alpha*delta effect and loss."""

import torch

from causalq.losses.causal_query_effect import causal_query_effect_loss
from tools.deterministic_source_ce import deterministic_source_ce
from tools.strict_source_smoke import HORIZON, precision


def full_cqe_source_update(model, optimizer, image, photo, label, step, config, *, device, amp,
                           loss_fn=deterministic_source_ce):
    return _cqe_source_update(model, optimizer, image, photo, label, step, config,
                              device=device, amp=amp, loss_fn=loss_fn)


def anchor_cqe_source_update(model, optimizer, image, photo, label, step, config, *, device, amp,
                             loss_fn=deterministic_source_ce):
    if config['causal_query_effect'].get('component') != 'anchor_only':
        raise ValueError('M3 requires explicit anchor_only CQE')
    return _cqe_source_update(model, optimizer, image, photo, label, step, config,
                              device=device, amp=amp, loss_fn=loss_fn, anchor=True)


def _cqe_source_update(model, optimizer, image, photo, label, step, config, *, device, amp,
                       loss_fn, anchor=False):
    if not 1 <= step <= HORIZON:
        raise ValueError('M2 source optimizer step must be within the40000-step horizon')
    recipe = config['causal_query_effect']
    required = {'enabled': True, 'effect_space': 'logits', 'reference_stop_gradient': True,
                'normalization': 'l2_per_class_map', 'present_classes_only': True,
                'valid_pixels_only': True, 'loss': 'smooth_l1'}
    if (any(recipe.get(k) != v for k, v in required.items())
            or recipe['lambda_cqe'] < 0 or recipe['smooth_l1_beta'] <= 0):
        raise ValueError('Unsupported M2 Full CQE contract')
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    lr_used = optimizer.param_groups[0]['lr']
    with precision(device, amp):
        output = model.forward_components(image, include_anchor=True) if anchor else model.forward_components(image)
        effect = model.get_anchor_query_effect(output=output) if anchor else model.get_query_effect(output=output)
        if not torch.isfinite(output.logits).all() or not torch.isfinite(effect).all():
            raise FloatingPointError('Nonfinite M2 original logits/effect')
        original = loss_fn(output.logits, label)
    if not torch.isfinite(original):
        raise FloatingPointError('Nonfinite M2 original CE')
    reference = effect.detach()
    original_value = float(original.detach())
    original.backward()
    del output, effect, original
    with precision(device, amp):
        output = model.forward_components(photo, include_anchor=True) if anchor else model.forward_components(photo)
        effect = model.get_anchor_query_effect(output=output) if anchor else model.get_query_effect(output=output)
        if not torch.isfinite(output.logits).all() or not torch.isfinite(effect).all():
            raise FloatingPointError('Nonfinite M2 photometric logits/effect')
        photometric = loss_fn(output.logits, label)
    # Existing loss casts effect maps to FP32, masks ignore pixels, L2-normalizes
    # each class map, sums spatial SmoothL1 and averages GT-present classes.
    consistency = causal_query_effect_loss(effect, reference, label, beta=recipe['smooth_l1_beta'])
    if not torch.isfinite(photometric) or not torch.isfinite(consistency):
        raise FloatingPointError('Nonfinite M2 photometric CE/Full CQE')
    weight = float(recipe['lambda_cqe'])  # one photometric view; no hidden /2
    (float(config['style']['lambda_cf']) * photometric + weight * consistency).backward()
    gradients = {name: float(p.grad.detach().float().norm()) for name, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None}
    norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True))
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in trainable):
        raise FloatingPointError('Nonfinite M2 parameter after update')
    next_lr = float(config['optimizer']['lr']) * (1 - step / HORIZON) ** float(config['optimizer']['poly_power'])
    for group in optimizer.param_groups:
        group['lr'] = next_lr
    photo_value, cqe_value = float(photometric.detach()), float(consistency.detach())
    return {'update': step, 'loss_original': original_value, 'loss_photometric': photo_value,
            ('loss_anchor_cqe' if anchor else 'loss_full_cqe'): cqe_value, 'cqe_weight': weight,
            'objective': original_value + float(config['style']['lambda_cf']) * photo_value + weight * cqe_value,
            'gradient_norm_pre_clip': norm, 'lr_used': lr_used, 'next_lr': next_lr,
            'alpha': float(model.query_head.alpha.detach()), 'gradients': gradients}
