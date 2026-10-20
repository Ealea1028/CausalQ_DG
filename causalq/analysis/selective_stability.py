"""Multi-style output-space diagnostics, not semantic/causal identification."""

import math
import torch
from torch.nn import functional as F

EPS = 1e-6
COMPONENTS = ('full', 'anchor', 'residual')


def _ratio(numerator, denominator):
    return float(numerator / denominator) if float(denominator) > EPS else None


def _correlation(x, y):
    x, y = x.double(), y.double()
    xc, yc = x - x.mean(), y - y.mean()
    rx, ry = xc.square().mean().sqrt(), yc.square().mean().sqrt()
    covariance = (xc * yc).mean()
    ux, uy = x.square().mean().sqrt(), y.square().mean().sqrt()
    return {'pearson': float((covariance / (rx * ry)).clamp(-1, 1)) if min(rx, ry) > EPS else None,
            'cosine': float(((x * y).mean() / (ux * uy)).clamp(-1, 1)) if min(ux, uy) > EPS else None,
            'covariance': float(covariance)}


@torch.no_grad()
def effect_and_correlation_rows(views, labels):
    """One source image; masks never depend on model predictions."""
    if len(views) < 2 or labels.ndim != 3 or labels.shape[0] != 1:
        raise ValueError('Need paired/multiple single-image effects and BHW labels')
    shape = views[0]['full'].shape
    if shape[0] != 1 or shape[2:] != labels.shape[1:] or any(
            set(v) != set(COMPONENTS) or any(x.shape != shape for x in v.values()) for v in views):
        raise ValueError('Multi-style effect shapes differ')
    valid = labels[0] != 255
    if not valid.any():
        raise ValueError('No valid source pixels')
    effects, correlations = [], []
    for c in range(shape[1]):
        maps = {k: torch.stack([v[k][0, c][valid].float().double() for v in views]) for k in COMPONENTS}
        if any(not torch.isfinite(x).all() for x in maps.values()):
            raise ValueError('Nonfinite effect')
        common = {'class_id': c, 'present': bool((labels[0] == c).any()), 'valid_pixels': int(valid.sum())}
        for component, x in maps.items():
            rms = x.square().mean(1).sqrt()
            normalized = x / rms.clamp_min(EPS)[:, None]
            all_nonzero = bool((rms > EPS).all())
            pair_nonzero = bool((rms[:2] > EPS).all())
            effects.append({**common, 'component': component,
                'mean_abs': float(x.abs().mean()), 'joint_rms': float(x.square().mean().sqrt()),
                'view_rms': rms.tolist(), 'view_mean_abs': x.abs().mean(1).tolist(), 'view_signed_mean': x.mean(1).tolist(),
                'raw_variance': float(x.var(dim=0, unbiased=False).mean()),
                'rms_normalized_variance': float(normalized.var(dim=0, unbiased=False).mean()) if all_nonzero else None,
                'pair_raw_variance': float((x[0] - x[1]).square().mean() / 4),
                'pair_rms_normalized_variance': float((normalized[0] - normalized[1]).square().mean() / 4) if pair_nonzero else None,
                'near_zero_view_count': int((rms <= EPS).sum()),
                'rms_style_cv': _ratio(rms.std(unbiased=False), rms.mean())})
        a, r, f = maps['anchor'], maps['residual'], maps['full']
        va, vr, vf = [x.var(dim=0, unbiased=False).mean() for x in (a, r, f)]
        covariance = ((a - a.mean(0)) * (r - r.mean(0))).mean()
        correlations.append({**common, 'component': 'variance_decomposition', 'view_index': -1,
            'anchor_variance': float(va), 'residual_variance': float(vr), 'full_variance': float(vf),
            'style_covariance': float(covariance), 'variance_identity_abs_error': float((vf - va - vr - 2 * covariance).abs())})
        for s in range(len(views)):
            ar = _correlation(a[s], r[s]); fa = _correlation(f[s], a[s])
            af, rr, ff = [x[s].square().mean().sqrt() for x in (a, r, f)]
            row = {**common, 'component': 'spatial', 'view_index': s,
                **{f'anchor_residual_{k}': v for k, v in ar.items()},
                **{f'full_anchor_{k}': v for k, v in fa.items()},
                'full_mean_abs': float(f[s].abs().mean()), 'anchor_mean_abs': float(a[s].abs().mean()),
                'difference_mean_abs': float(r[s].abs().mean()),
                'full_rms': float(ff), 'anchor_rms': float(af), 'difference_rms': float(rr),
                'residual_full_ratio': _ratio(rr, ff), 'residual_anchor_ratio': _ratio(rr, af),
                'addition_cancellation_ratio': _ratio(ff, af + rr), 'difference_shrink_ratio': _ratio(rr, ff + af)}
            if s:
                for name, x, y in (('anchor_cross_style', a[0], a[s]), ('residual_cross_style', r[0], r[s]),
                                   ('delta_anchor_residual', a[s] - a[0], r[s] - r[0])):
                    row.update({name + '_' + k: v for k, v in _correlation(x, y).items()})
            correlations.append(row)
    return effects, correlations


@torch.no_grad()
def multiview_responsibility_rows(scores, labels):
    """Native score-grid GT pixels and all-valid supplemental populations."""
    if len(scores) < 2 or any(s.shape != scores[0].shape for s in scores) or scores[0].shape[0] != 1:
        raise ValueError('Responsibility needs aligned single-image BCRHW scores')
    _, classes, slots, h, w = scores[0].shape
    if slots < 2 or any(not torch.isfinite(s).all() for s in scores):
        raise ValueError('Need finite multiple query scores')
    gt = F.interpolate(labels[:, None].float(), (h, w), mode='nearest')[0, 0].long()
    probabilities = torch.stack([s[0].float().softmax(1).double() for s in scores])
    rows = []
    for c in range(classes):
        for scope, mask in (('gt_class', gt == c), ('all_valid', gt != 255)):
            count = int(mask.sum())
            p = probabilities[:, c, :, mask]
            if count:
                entropies = -(p * p.clamp_min(1e-30).log()).sum(1).mean(1)
                means = p.mean(-1)
                usage = -(means * means.clamp_min(1e-30).log()).sum(1)
                tops = p == p.max(1, keepdim=True).values
                credits = tops.double() / tops.sum(1, keepdim=True)
            for s in range(len(scores)):
                js = switch = None
                if count:
                    middle = (p[0] + p[s]) / 2
                    js = float((.5 * (p[0] * (p[0].clamp_min(1e-30) / middle.clamp_min(1e-30)).log()
                                          + p[s] * (p[s].clamp_min(1e-30) / middle.clamp_min(1e-30)).log()).sum(0)).mean())
                    switch = float((p[0].argmax(0) != p[s].argmax(0)).double().mean())
                for slot in range(slots):
                    rows.append({'class_id': c, 'query_r': slot, 'scope': scope, 'view_index': s,
                        'present': bool((gt == c).any()), 'pixels': count,
                        'mean_responsibility': float(means[s, slot]) if count else None,
                        'pixel_entropy': float(entropies[s]) if count else None,
                        'usage_entropy': float(usage[s]) if count else None,
                        'normalized_pixel_entropy': float(entropies[s] / math.log(slots)) if count else None,
                        'effective_query_count': float(usage[s].exp()) if count else None,
                        'top1_share': float(credits[s, slot].mean()) if count else None,
                        'top1_tie_rate': float((tops[s].sum(0) > 1).double().mean()) if count else None,
                        'mean_responsibility_delta': float(means[s, slot] - means[0, slot]) if count else None,
                        'mean_abs_delta': float((p[s, slot] - p[0, slot]).abs().mean()) if count else None,
                        'pair_population_variance': float((p[s, slot] - p[0, slot]).square().mean() / 4) if count else None,
                        'js_divergence': js, 'top1_switch_rate': switch})
    return rows
