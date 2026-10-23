"""New-study Anchor/Residual CQE; reuse legacy two-view optimizer semantics."""

import torch

from causalq.losses.causal_query_effect import causal_query_effect_loss
from causalq.protocols.diagnostic_20261020 import STUDY
from tools.deterministic_source_ce import deterministic_source_ce
from tools.strict_source_smoke import HORIZON, precision


def selective_effect(model, output, component):
    fn = {'anchor': model.get_anchor_query_effect,
          'residual': model.get_residual_contribution}[component]
    return fn(output=output, component_protocol=STUDY)


def selective_cqe_source_update(model, optimizer, image, photo, label, step, config, *,
                                device, amp, loss_fn=deterministic_source_ce):
    recipe = config['causal_query_effect']
    component = recipe.get('component')
    required = {'enabled': True, 'effect_space': 'logits', 'reference_stop_gradient': True,
                'normalization': 'l2_per_class_map', 'present_classes_only': True,
                'valid_pixels_only': True, 'loss': 'smooth_l1',
                'lambda_cqe': 1., 'smooth_l1_beta': 1.}
    if (component not in ('anchor', 'residual') or config['protocol'].get('study') != STUDY
            or any(recipe.get(k) != value for k, value in required.items())
            or config.get('prediction_consistency', {}).get('enabled', False)):
        raise ValueError('Unsupported selective CQE contract')
    if not 1 <= step <= HORIZON:
        raise ValueError('Selective optimizer step outside40000-step horizon')
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    lr_used = optimizer.param_groups[0]['lr']
    with precision(device, amp):
        output = model.forward_components(image, include_anchor=True, component_protocol=STUDY)
        effect = selective_effect(model, output, component)
        if not torch.isfinite(output.logits).all() or not torch.isfinite(effect).all():
            raise FloatingPointError('Nonfinite selective original logits/effect')
        original = loss_fn(output.logits, label)
    if not torch.isfinite(original):
        raise FloatingPointError('Nonfinite original CE')
    reference = effect.detach()
    original_value = float(original.detach())
    original.backward()
    del output, effect, original
    with precision(device, amp):
        output = model.forward_components(photo, include_anchor=True, component_protocol=STUDY)
        effect = selective_effect(model, output, component)
        if not torch.isfinite(output.logits).all() or not torch.isfinite(effect).all():
            raise FloatingPointError('Nonfinite selective photometric logits/effect')
        photometric = loss_fn(output.logits, label)
    # Both style-side paths differentiate in Full-Anchor; only the reference
    # is detached. No residual recentering, projection, or extra backbone pass.
    consistency = causal_query_effect_loss(effect, reference, label, beta=recipe['smooth_l1_beta'])
    if not torch.isfinite(photometric) or not torch.isfinite(consistency):
        raise FloatingPointError('Nonfinite photometric CE/selective CQE')
    weight = float(recipe['lambda_cqe'])
    (float(config['style']['lambda_cf']) * photometric + weight * consistency).backward()
    gradients = {name: float(p.grad.detach().float().norm()) for name, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None}
    norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True))
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in trainable):
        raise FloatingPointError('Nonfinite selective parameter after update')
    next_lr = float(config['optimizer']['lr']) * (1-step/HORIZON) ** float(config['optimizer']['poly_power'])
    for group in optimizer.param_groups:
        group['lr'] = next_lr
    photo_value, cqe_value = float(photometric.detach()), float(consistency.detach())
    return {'update': step, 'loss_original': original_value, 'loss_photometric': photo_value,
            f'loss_{component}_cqe': cqe_value, 'cqe_weight': weight,
            'objective': original_value + float(config['style']['lambda_cf']) * photo_value + weight * cqe_value,
            'gradient_norm_pre_clip': norm, 'lr_used': lr_used, 'next_lr': next_lr,
            'alpha': float(model.query_head.alpha.detach()), 'gradients': gradients}
