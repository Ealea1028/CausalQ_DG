"""Read-only centered-query diagnostics; responsibility is not causal effect."""

from itertools import combinations
import math

import torch
from torch.nn import functional as F


def fixed_source_ids(ids, count=20):
    """Filename-hash sample, independent of labels, predictions and file order."""
    import hashlib
    if count < 1 or count > len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Invalid diagnostic source membership/count")
    return sorted(ids, key=lambda i: (hashlib.sha256(("causalq-components-v1:" + i).encode()).hexdigest(), i))[:count]


def intervention_seed(source_id):
    import hashlib
    return int(hashlib.sha256(("causalq-components-photo-v1:" + source_id).encode()).hexdigest()[:8], 16)


def _finite(*values):
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError("Nonfinite component diagnostic input")


def _labels(labels, classes):
    if labels.ndim != 3 or labels.dtype not in (torch.int64, torch.int32, torch.uint8):
        raise ValueError("Labels must be integer BxHxW")
    if ((labels != 255) & ((labels < 0) | (labels >= classes))).any():
        raise ValueError("Invalid source train ID")


@torch.no_grad()
def query_parameter_rows(anchors, centered_residuals):
    anchors, residuals = anchors.detach().float(), centered_residuals.detach().float()
    if anchors.ndim != 2 or residuals.ndim != 3 or (residuals.shape[0], residuals.shape[2]) != anchors.shape or residuals.shape[1] < 1:
        raise ValueError("Expected anchors CxD and centered residuals CxRxD")
    _finite(anchors, residuals)
    queries = anchors[:, None] + residuals
    rows = []
    for c in range(anchors.shape[0]):
        a = float(anchors[c].norm())
        norms = residuals[c].norm(dim=-1)
        mean_norm = float(norms.mean())
        pairs = list(combinations(range(residuals.shape[1]), 2)) or [(None, None)]
        for r, s in pairs:
            nonzero = r is not None and float(queries[c, r].norm()) > 0 and float(queries[c, s].norm()) > 0
            rows.append({"class_id": c, "query_r": r, "query_s": s,
                "query_pair_cosine": float(F.cosine_similarity(queries[c, r], queries[c, s], dim=0)) if nonzero else None,
                "anchor_norm": a, "mean_centered_residual_norm": mean_norm,
                "anchor_residual_norm_ratio": a / mean_norm if mean_norm > 0 else None,
                "residual_r_norm": float(norms[r]) if r is not None else float(norms[0]),
                "residual_s_norm": float(norms[s]) if s is not None else None})
    return rows


@torch.no_grad()
def responsibility_rows(original_scores, photo_scores, labels):
    """GT-class pixels on native score grid; no upsampled pseudo-precision.

    Activation is rho > 1/R (uniform baseline). Top1 ties split credit.
    Slot contribution is leave-one-slot logsumexp drop, before alpha/base;
    for R=1 deletion is undefined and reported as None.
    """
    if original_scores.ndim != 5 or original_scores.shape != photo_scores.shape or min(original_scores.shape) < 1:
        raise ValueError("Paired scores must share BxCxRxHxW shape")
    b, classes, slots, h, w = original_scores.shape
    _labels(labels, classes)
    if labels.shape[0] != b:
        raise ValueError("Score/label batch mismatch")
    _finite(original_scores, photo_scores)
    labels = F.interpolate(labels[:, None].float(), (h, w), mode="nearest")[:, 0].long()
    scores = [original_scores.detach().float(), photo_scores.detach().float()]
    probabilities = [score.softmax(dim=2) for score in scores]
    rows = []
    for batch in range(b):
        for c in range(classes):
            mask = labels[batch] == c
            count = int(mask.sum())
            values = [p[batch, c, :, mask] for p in probabilities]
            variance = ((values[0] - values[1]).square() / 4).mean(1) if count else None
            for view_index, view in enumerate(("original", "photometric")):
                p = values[view_index]
                entropy = float(-(p * p.clamp_min(1e-30).log()).sum(0).mean()) if count else None
                mean = p.mean(1) if count else None
                usage_entropy = float(-(mean * mean.clamp_min(1e-30).log()).sum()) if count else None
                top = p == p.max(0, keepdim=True).values if count else None
                for r in range(slots):
                    deletion = None
                    if count and slots > 1:
                        score = scores[view_index][batch, c, :, mask]
                        kept = torch.cat((score[:r], score[r + 1:]), dim=0)
                        deletion = float((score.logsumexp(0) - kept.logsumexp(0)).mean())
                    rows.append({"batch_index": batch, "class_id": c, "query_r": r, "view": view,
                        "gt_score_pixels": count, "mean_responsibility": float(mean[r]) if count else None,
                        "mean_pixel_entropy_nats": entropy, "usage_entropy_nats": usage_entropy,
                        "effective_query_count": math.exp(usage_entropy) if count else None,
                        "above_uniform_activation_rate": float((p[r] > 1 / slots).float().mean()) if count else None,
                        "top1_share_ties_split": float((top[r].float() / top.sum(0)).mean()) if count else None,
                        "cross_view_population_variance": float(variance[r]) if count else None,
                        "slot_logsumexp_deletion_drop": deletion})
    return rows


@torch.no_grad()
def effect_rows(original, photometric, labels):
    """Full-resolution valid source pixels; population variance of two views.

    Empty regions return None, not zero. Metrics are output-space diagnostics.
    RMS-normalized pair MSE is absent if either map has zero energy.
    """
    if set(original) != {"full", "anchor", "residual"} or set(original) != set(photometric):
        raise ValueError("Expected full/anchor/residual effect maps")
    first = original["full"]
    if first.ndim != 4 or min(first.shape) < 1:
        raise ValueError("Effects must be BxCxHxW")
    b, classes, h, w = first.shape
    _labels(labels, classes)
    if labels.shape != (b, h, w):
        raise ValueError("Effect/label shape mismatch")
    if any(value.shape != first.shape for value in (*original.values(), *photometric.values())):
        raise ValueError("Paired effect shapes differ")
    _finite(*original.values(), *photometric.values())
    rows = []
    for batch in range(b):
        valid = labels[batch] != 255
        for c in range(classes):
            inside = valid & (labels[batch] == c)
            outside = valid & ~inside
            for component in original:
                x, y = (maps[component][batch, c].detach().float() for maps in (original, photometric))
                count = int(valid.sum())
                ix, ox = (float(x[m].abs().mean()) if m.any() else None for m in (inside, outside))
                iy, oy = (float(y[m].abs().mean()) if m.any() else None for m in (inside, outside))
                rms_x = float(x[valid].square().mean().sqrt()) if count else 0.
                rms_y = float(y[valid].square().mean().sqrt()) if count else 0.
                rows.append({"batch_index": batch, "class_id": c, "component": component,
                    "valid_pixels": count, "inside_pixels": int(inside.sum()), "outside_pixels": int(outside.sum()),
                    "original_mean_abs": float(x[valid].abs().mean()) if count else None,
                    "photometric_mean_abs": float(y[valid].abs().mean()) if count else None,
                    "original_rms": rms_x if count else None, "photometric_rms": rms_y if count else None,
                    "cross_view_population_variance": float(((x[valid] - y[valid]).square() / 4).mean()) if count else None,
                    "rms_normalized_pair_mse": float(((x[valid] / rms_x - y[valid] / rms_y).square()).mean()) if rms_x > 0 and rms_y > 0 else None,
                    "original_inside_abs": ix, "original_outside_abs": ox,
                    "photometric_inside_abs": iy, "photometric_outside_abs": oy,
                    "original_inside_outside_ratio": ix / (ox + 1e-12) if ix is not None and ox is not None else None,
                    "photometric_inside_outside_ratio": iy / (oy + 1e-12) if iy is not None and oy is not None else None})
    return rows
