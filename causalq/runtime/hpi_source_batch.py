"""Budgeted exact-geometry source readout; keep reductions image-local."""
from collections import OrderedDict
from statistics import fmean
import torch
from causalq.protocols.hpi_dinov2_20261031 import STUDY
from causalq.analysis.selective_stability import (
    effect_and_correlation_rows, multiview_responsibility_rows, structural_effect_and_correlation_rows,
)
from causalq.models.dinov3_wrapper import DINOv3Features
from .feature_pipeline import feature_tensors


def source_batches(items,*,max_images=2,budget_bytes=8<<30,max_buckets=4):
    """Items are (ID, VCHW views, BHW label); no resizing or drop-last."""
    buckets=OrderedDict()
    for item in items:
        shape=tuple(item[1].shape)
        # Conservative peak for input, full/anchor/base/delta logits and readout temporaries.
        cost=item[1].numel()*item[1].element_size()+shape[0]*19*shape[-2]*shape[-1]*4*8
        if cost>budget_bytes:raise ValueError('One source image exceeds frozen readout budget')
        cap=min(max_images,budget_bytes//cost)
        if shape not in buckets and len(buckets)>=max_buckets:
            yield buckets.popitem(last=False)[1]
        batch=buckets.setdefault(shape,[]);batch.append(item)
        if len(batch)==cap:yield buckets.pop(shape)
    yield from buckets.values()


def main_path_rows(model, features, factual_logits, label, offset, chunk):
    """Class deletion changes the route and the selected additive readout."""
    label = label.to(factual_logits.device)[0]
    classes = [int(c) for c in torch.unique(label).cpu() if c != 255]
    views = len(factual_logits)
    pairs = [(category, view) for category in classes for view in range(views)]
    factual_prob = factual_logits.float().softmax(1)
    values = []
    for start in range(0, len(pairs), chunk):
        batch = pairs[start:start + chunk]
        category, view = torch.tensor(batch, device=label.device).unbind(1)
        picked = [tensor.index_select(0, view + offset) for tensor in feature_tensors(features)]
        local_features = DINOv3Features(picked[0], picked[1], tuple(picked[2:]))
        alternative = model.closed_class_logits(local_features, label.shape, category).float().softmax(1)
        index = torch.arange(len(batch), device=label.device)
        mask = label == category[:, None, None]
        count = mask.sum((1, 2))
        on = (factual_prob[view, category] * mask).sum((1, 2)) / count
        off = (alternative[index, category] * mask).sum((1, 2)) / count
        values.append(torch.stack((on, off, on - off), dim=1).double())
    if not values:
        return []
    values = torch.cat(values).reshape(len(classes), views, 3).cpu()
    rows = []
    for category, metrics in zip(classes, values):
        on, off, effect = metrics.unbind(1)
        rows.append({'class_id': category, 'on': on.tolist(), 'off': off.tolist(), 'tau': effect.tolist(),
                     'mean_tau': float(effect.mean()), 'variance': float(effect.var(unbiased=False)),
                     'positive_fraction': float((effect > 1e-6).double().mean()),
                     'negative_fraction': float((effect < -1e-6).double().mean()),
                     'near_zero_fraction': float((effect.abs() <= 1e-6).double().mean())})
    return rows


def summarize_main_path(by_image):
    """Weight each image equally after averaging its GT-present classes."""
    images = tuple(by_image.values())
    if not images:
        return None
    if any(not rows for rows in images):
        raise ValueError('Main-path summary requires GT-present classes for every image')
    scalar_keys = ('mean_tau', 'variance', 'positive_fraction', 'negative_fraction', 'near_zero_fraction')
    return {
        'population': 'equal_weight_images_over_gt_present_classes',
        **{key: fmean(fmean(row[key] for row in rows) for rows in images) for key in scalar_keys},
        **{key: [fmean(fmean(row[key][view] for row in rows) for rows in images) for view in range(5)]
           for key in ('on', 'off', 'tau')},
    }


@torch.inference_mode()
def batched_source_readout(heads,backbone,cases,*,device='cuda',amp=True,intervention_ids=(),counterfactual_chunk=4):
    """One batched backbone and head forward; transfer view maps in packs."""
    from .hpi_training import class_responsibility
    images=torch.cat([c[1] for c in cases]).to(device,non_blocking=True)
    v=cases[0][1].shape[0];h,w=images.shape[-2:]
    if v!=5 or any(c[1].shape!=cases[0][1].shape for c in cases):raise ValueError('Expected matching original+4 view geometry')
    result={c[0]:{} for c in cases}
    with torch.autocast(torch.device(device).type,dtype=torch.bfloat16,enabled=amp):
        features=backbone(images)
        for head in heads:
            structural = getattr(head.model, 'residual_status', None) == 'structurally_absent'
            route = None
            if hasattr(head.model, 'forward_routed'):
                prediction, route = head.model.forward_routed(features, (h,w), include_anchor=not structural, component_protocol=STUDY)
            else:
                prediction=head.model.forward_features(features,(h,w),include_anchor=not structural,component_protocol=STUDY)
            if not torch.isfinite(prediction.logits).all():raise ValueError('Nonfinite batched source prediction')
            classes=prediction.logits.shape[1]
            labels=torch.cat([c[2] for c in cases]).to(device)
            predicted=prediction.logits[::v].argmax(1)
            valid=labels!=255
            if ((labels[valid]<0)|(labels[valid]>=classes)).any():raise ValueError('Source labels outside train IDs')
            encoded=(labels*classes+predicted).masked_fill(~valid,classes**2).flatten(1)
            confusion=torch.zeros(len(cases),classes**2+1,dtype=torch.int64,device=device)
            confusion.scatter_add_(1,encoded,torch.ones_like(encoded))
            confusion=confusion[:,:-1].reshape(len(cases),classes,classes).cpu().tolist()
            full=prediction.scaled_delta_logits.float()
            if structural:
                maps = full.cpu()
            else:
                anchor = prediction.anchor_scaled_delta_logits.float()
                maps = torch.stack((full, anchor, full-anchor)).cpu()
            scores=prediction.query_score_maps.float().cpu()
            if route is not None:
                attention = route.patch_attention.float()
                route_values = torch.stack((route.injected.float().square().mean((1, 2)).sqrt(),
                    route.gate.float().mean((1, 2, 3)),
                    -(attention * attention.clamp_min(1e-12).log()).sum(-1).mean((1, 2))),dim=1).double().cpu()
                gamma = float(route.gamma)
            for j,(identity,_,label) in enumerate(cases):
                sl=slice(j*v,(j+1)*v)
                image_scores=[scores[i:i+1] for i in range(sl.start,sl.stop)]
                if structural:
                    effect_rows, correlation_rows = structural_effect_and_correlation_rows(maps[sl], label)
                else:
                    effects = [dict(zip(('full','anchor','residual'), maps[:, i:i+1]))
                               for i in range(sl.start, sl.stop)]
                    effect_rows, correlation_rows = effect_and_correlation_rows(effects, label)
                record={'effects':effect_rows,'correlations':correlation_rows,'gradients':[],
                    'responsibility':multiview_responsibility_rows(image_scores,label),
                    'class_responsibility':class_responsibility(image_scores,label),
                    'confusion':confusion[j]}
                if route is not None:
                    record['routing'] = {'gamma': gamma,
                        'injection_feature_update_rms': route_values[sl, 0].tolist(),
                        'gate_mean': route_values[sl, 1].tolist(),
                        'gate_mean_style_variance': float(route_values[sl, 1].var(unbiased=False)),
                        'route_patch_entropy': route_values[sl, 2].tolist()}
                    if identity in intervention_ids:
                        record['main_path'] = main_path_rows(head.model, features, prediction.logits[sl], label,
                                                            sl.start, counterfactual_chunk)
                result[identity][head.name] = record
    return result
