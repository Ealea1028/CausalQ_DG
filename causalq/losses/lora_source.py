"""Equivalent FP32 source losses after CanonicalInputStream CPU validation.

Avoid repeated GPU-to-host boolean guards and dynamic masked-selection syncs.
Only use for the controlled source stream: labels in 0..C-1/255 and nonempty GT.
The original public losses retain their strict validation behavior.
"""
import torch
from torch.nn import functional as F


def source_ce_prevalidated(logits,labels):
    valid=labels!=255
    safe=labels.masked_fill(~valid,0)
    loss=-F.log_softmax(logits.float(),dim=1).gather(1,safe[:,None]).squeeze(1)
    return loss.masked_fill(~valid,0).sum()/valid.sum()


def anchor_cqe_prevalidated(effect,reference,labels):
    batch,classes,_,_=effect.shape
    valid=labels!=255
    present=(F.one_hot(labels.masked_fill(~valid,0),num_classes=classes).bool() & valid.unsqueeze(-1)).any(2).any(1)
    mask=valid.reshape(batch,1,-1)
    a=F.normalize(effect.float().reshape(batch,classes,-1).masked_fill(~mask,0),p=2,dim=-1,eps=1e-6)
    b=F.normalize(reference.detach().float().reshape(batch,classes,-1).masked_fill(~mask,0),p=2,dim=-1,eps=1e-6)
    per_class=F.smooth_l1_loss(a,b,reduction='none',beta=1.).sum(-1)
    return per_class.masked_fill(~present,0).sum()/present.sum()
