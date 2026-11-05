"""Budgeted exact-geometry source readout; keep reductions image-local."""
from collections import OrderedDict
import torch
from causalq.protocols.hpi_dinov2_20261031 import STUDY
from causalq.analysis.selective_stability import effect_and_correlation_rows, multiview_responsibility_rows


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


@torch.inference_mode()
def batched_source_readout(heads,backbone,cases,*,device='cuda',amp=True):
    """One batched backbone and head forward; transfer view maps in packs."""
    from .hpi_training import class_responsibility
    images=torch.cat([c[1] for c in cases]).to(device,non_blocking=True)
    v=cases[0][1].shape[0];h,w=images.shape[-2:]
    if v!=5 or any(c[1].shape!=cases[0][1].shape for c in cases):raise ValueError('Expected matching original+4 view geometry')
    result={c[0]:{} for c in cases}
    with torch.autocast(torch.device(device).type,dtype=torch.bfloat16,enabled=amp):
        features=backbone(images)
        for head in heads:
            prediction=head.model.forward_features(features,(h,w),include_anchor=True,component_protocol=STUDY)
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
            full=prediction.scaled_delta_logits.float();anchor=prediction.anchor_scaled_delta_logits.float()
            maps=torch.stack((full,anchor,full-anchor)).cpu()
            scores=prediction.query_score_maps.float().cpu()
            for j,(identity,_,label) in enumerate(cases):
                sl=slice(j*v,(j+1)*v)
                effects=[dict(zip(('full','anchor','residual'),maps[:,i:i+1])) for i in range(sl.start,sl.stop)]
                image_scores=[scores[i:i+1] for i in range(sl.start,sl.stop)]
                effect_rows,correlation_rows=effect_and_correlation_rows(effects,label)
                result[identity][head.name]={'effects':effect_rows,'correlations':correlation_rows,
                    'responsibility':multiview_responsibility_rows(image_scores,label),
                    'class_responsibility':class_responsibility(image_scores,label),
                    'confusion':confusion[j]}
    return result
