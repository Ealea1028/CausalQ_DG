"""Phase41 L0 execution candidates; no scientific objective changes."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from causalq.runtime.feature_pipeline import slice_features
from causalq.runtime.lora_training import sample_rng,learning_rate


@contextmanager
def paired_dropout(model,positions,view):
    drop=model.decoder.fuse[-1]
    if not isinstance(drop,torch.nn.Dropout2d): raise ValueError('Expected original decoder Dropout2d')
    original=drop.forward
    had_override='forward' in drop.__dict__
    def forward(x):
        if len(x)!=len(positions): raise ValueError('Dropout/source mapping mismatch')
        rows=[]
        for j,index in enumerate(positions):
            with sample_rng(index,view): rows.append(original(x[j:j+1]))
        return torch.cat(rows)
    drop.forward=forward
    try: yield
    finally:
        if had_override:drop.forward=original
        else:del drop.forward


def per_image_ce(logits,labels):
    valid=labels!=255
    ce=-F.log_softmax(logits.float(),1).gather(1,labels.masked_fill(~valid,0)[:,None]).squeeze(1)
    return ce.masked_fill(~valid,0).flatten(1).sum(1)/valid.flatten(1).sum(1)


def per_image_cqe(effect,reference,labels):
    b,c,_,_=effect.shape; valid=labels!=255
    present=(F.one_hot(labels.masked_fill(~valid,0),num_classes=c).bool()&valid.unsqueeze(-1)).flatten(1,2).any(1)
    mask=valid.reshape(b,1,-1)
    a=F.normalize(effect.float().reshape(b,c,-1).masked_fill(~mask,0),dim=-1,eps=1e-6)
    ref=F.normalize(reference.detach().float().reshape(b,c,-1).masked_fill(~mask,0),dim=-1,eps=1e-6)
    loss=F.smooth_l1_loss(a,ref,reduction='none',beta=1.).sum(-1)
    return loss.masked_fill(~present,0).sum(1)/present.sum(1)


def features(backbone,packets,forward_images,amp):
    if forward_images not in (8,16) or len(packets)!=8: raise ValueError('L0 batch8/16 only')
    if getattr(backbone,'lora_enabled',False) or any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('Frozen L0 only')
    device=next(backbone.parameters()).device;backbone.eval()
    with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
        if forward_images==16:
            # View-major order: original0..7, photo0..7, including real CLS/register.
            all_features=backbone(torch.stack([p['views'][v] for v in (0,1) for p in packets]).to(device))
            return tuple(slice_features(all_features,v*8,(v+1)*8) for v in (0,1))
        return tuple(backbone(torch.stack([p['views'][v] for p in packets]).to(device)) for v in (0,1))


def update_batched(model,optimizer,packets,step,variant,views,*,head_batch=8,amp=False):
    if variant not in ('U0','UA','C0','CA') or head_batch not in (4,8) or len(packets)!=8:
        raise ValueError('Invalid L0 execution')
    if any(p.requires_grad for p in model.backbone.parameters()) or getattr(model.backbone,'lora_enabled',False):
        raise ValueError('L0 only')
    model.train();optimizer.zero_grad(set_to_none=True)
    for g in optimizer.param_groups:g['lr']=learning_rate(step)
    device=next(model.parameters()).device;values=[]
    for start in range(0,8,head_batch):
        rows=packets[start:start+head_batch]
        labels=torch.stack([p['label'] for p in rows]).to(device)
        positions=[p['metadata']['step_zero_based'] for p in rows]
        reference=None
        for view in (0,1):
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp),paired_dropout(model,positions,view):
                out=model.forward_features(slice_features(views[view],start,start+head_batch),labels.shape[-2:],
                    include_anchor=variant in ('UA','CA'),component_protocol='lora_causalq_1114')
                loss=per_image_ce(out.logits,labels)
                if variant in ('UA','CA'):
                    if view==0:reference=out.anchor_scaled_delta_logits.detach().float()
                    else:loss=loss+per_image_cqe(out.anchor_scaled_delta_logits.float(),reference,labels)
                total=loss.sum()/8
            values.append(total.detach());total.backward()
    params=[p for p in model.parameters() if p.requires_grad]
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    if not torch.isfinite(torch.stack(values)).all(): raise FloatingPointError('Nonfinite source loss')
    optimizer.step()
    return torch.stack((torch.stack(values).sum(),norm.detach()))
