"""Read-only Query/Anchor chain-rule attribution; no optimizer construction."""

import torch

from causalq.losses.causal_query_effect import causal_query_effect_loss
from causalq.losses.prediction_consistency import prediction_consistency_kl
from causalq.protocols.diagnostic_20261020 import STUDY, mechanism
from tools.deterministic_source_ce import deterministic_source_ce
from tools.strict_source_smoke import precision

EPS = 1e-12


def cosine(x, y):
    nx, ny = x.norm(), y.norm()
    return float((x @ y) / (nx * ny)) if min(nx, ny) > EPS else None


def projection(x, total):
    return float((x @ total) / total.square().sum()) if total.norm() > EPS else None


def loss_terms(model, outputs, label, variant):
    mode = mechanism(variant)
    terms = {'ce_original': deterministic_source_ce(outputs[0].logits, label),
             'ce_photometric': deterministic_source_ce(outputs[1].logits, label)}
    if mode == 'prediction':
        terms['consistency'] = prediction_consistency_kl(outputs[1].logits, outputs[0].logits, labels=label)
    elif mode != 'none':
        def effect(output):
            if mode == 'full':
                return model.get_query_effect(output=output)
            fn = model.get_anchor_query_effect if mode == 'anchor' else model.get_residual_contribution
            return fn(output=output, component_protocol=STUDY)
        terms['consistency'] = causal_query_effect_loss(effect(outputs[1]), effect(outputs[0]).detach(), label, beta=1.)
    terms['total'] = sum(terms.values())
    return terms


@torch.enable_grad()
def query_gradient_rows(model, image, photo, label, variant, *, amp=False):
    if model.training or image.shape[0] != 1:
        raise ValueError('Read-only gradients require eval mode and batch1')
    outputs = []
    with precision(str(image.device), amp):
        for view in (image, photo):
            outputs.append(model.forward_components(view, include_anchor=True, component_protocol=STUDY))
    with precision(str(image.device), amp):
        terms = loss_terms(model, outputs, label, variant)
    anchors = model.query_bank.class_anchors
    residuals = model.query_bank.query_residuals
    q_nodes = [o.query_states for o in outputs]
    a_nodes = [o.anchor_query_states for o in outputs]
    present_classes = set(label.detach().cpu().unique().tolist())
    alpha = float(model.query_head.alpha.detach())
    rows = []
    for term_index, (name, loss) in enumerate(terms.items()):
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite diagnostic objective')
        nodes = [*q_nodes, *a_nodes, anchors, residuals]
        grads = torch.autograd.grad(loss, nodes, allow_unused=True, retain_graph=term_index < len(terms) - 1)
        # Small gradient vectors leave CUDA once per term. Do not synchronize
        # thousands of tiny GPU norms/cosines for every source image.
        values = [torch.zeros(node.shape, dtype=torch.float64) if grad is None
                  else grad.detach().to(device='cpu', dtype=torch.float64) for node, grad in zip(nodes, grads)]
        h_views = torch.stack([v.sum(0) for v in values[:2]])
        h = h_views.sum(0)
        via = h.sum(1)
        direct = sum(v.sum((0, 2)) for v in values[2:4])
        total, raw = values[-2:]
        if not torch.allclose(via + direct, total, atol=1e-6, rtol=1e-3):
            raise ValueError('Query/direct-anchor gradient chain rule differs')
        expected_residual = h - h.mean(1, keepdim=True) if model.query_bank.centered_residuals else h
        if not torch.allclose(raw, expected_residual, atol=1e-6, rtol=1e-3):
            raise ValueError('Residual parameterization chain rule differs')
        objective = float(loss.detach())
        for c in range(h.shape[0]):
            denominator = h[c].norm(dim=-1).sum()
            for slot in range(h.shape[1]):
                rows.append({'class_id': c, 'query_r': slot, 'term': name,
                    'present': c in present_classes, 'objective': objective,
                    'query_gradient_norm': float(h[c, slot].norm()),
                    'raw_residual_gradient_norm': float(raw[c, slot].norm()),
                    'per_view_gradient_norm': h_views[:, c, slot].norm(dim=-1).tolist(),
                    'cross_view_gradient_cosine': cosine(h_views[0, c, slot], h_views[1, c, slot]),
                    'query_pair_cosines': [cosine(h[c, slot], v) for v in h[c]],
                    'cosine_with_via_anchor': cosine(h[c, slot], via[c]),
                    'norm_share': float(h[c, slot].norm() / denominator) if denominator > EPS else None,
                    'projection_via_anchor': projection(h[c, slot], via[c]),
                    'projection_total_anchor': projection(h[c, slot], total[c]),
                    'direct_projection_total': projection(direct[c], total[c]),
                    'via_anchor_norm': float(via[c].norm()), 'direct_anchor_norm': float(direct[c].norm()),
                    'total_anchor_norm': float(total[c].norm()),
                    'gradient_cancellation_ratio': float(via[c].norm() / denominator) if denominator > EPS else None,
                    'chain_rule_max_error': float((via[c] + direct[c] - total[c]).abs().max()),
                    'near_zero_query_gradient': bool(h[c, slot].norm() <= EPS),
                    'alpha': alpha})
    return rows


@torch.no_grad()
def parameter_rows(model):
    a = model.get_anchors().float().cpu()
    e = model.get_query_residuals().float().cpu()
    centered = e - e.mean(1, keepdim=True)
    q = a[:, None] + (centered if model.query_bank.centered_residuals else e)
    return [{'class_id': c, 'anchor_norm': float(a[c].norm()),
             'residual_mean_norm': float(e[c].mean(0).norm()), 'raw_residual_norms': e[c].norm(dim=-1).tolist(),
             'centered_residual_norms': centered[c].norm(dim=-1).tolist(), 'full_query_norms': q[c].norm(dim=-1).tolist()}
            for c in range(a.shape[0])]
