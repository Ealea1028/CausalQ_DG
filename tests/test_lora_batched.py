import copy
import pytest
import torch
from test_lora_formal import cohort
from test_lora_phase40 import packet
from tools.check_lora_resume import compare
from causalq.runtime.lora_training import update,compact_state,make_optimizer
from causalq.runtime.lora_batched import features,update_batched,per_image_ce,per_image_cqe,paired_dropout
from causalq.losses.lora_source import source_ce_prevalidated,anchor_cqe_prevalidated


def test_per_image_losses_preserve_unequal_valid_pixels_classes_and_detach():
    labels=torch.randint(0,3,(3,4,5));labels[0,:3]=255;labels[1]=1
    x=torch.randn(3,3,4,5,requires_grad=True);y=x.detach().clone().requires_grad_();ref=torch.randn_like(x,requires_grad=True)
    for batch,single in [(lambda a:per_image_ce(a,labels),lambda a,i:source_ce_prevalidated(a,labels[i:i+1])),
        (lambda a:per_image_cqe(a,ref,labels),lambda a,i:anchor_cqe_prevalidated(a,ref[i:i+1],labels[i:i+1]))]:
        a=batch(x).mean();b=torch.stack([single(y[i:i+1],i) for i in range(3)]).mean()
        torch.testing.assert_close(a,b);a.backward();b.backward();torch.testing.assert_close(x.grad,y.grad)
        x.grad=None;y.grad=None
    assert ref.grad is None


@pytest.mark.parametrize('size',[4,8])
@pytest.mark.parametrize('backbone_batch',[8,16])
def test_batched_updates_preserve_dropout_gradients_optimizer_and_cls(size,backbone_batch):
    torch.manual_seed(0)
    c=cohort();samples=packet();views=features(c.backbone,samples,backbone_batch,False)
    reference=c.backbone(torch.stack([p['views'][0] for p in samples]))
    torch.testing.assert_close(views[0].cls_token,reference.cls_token)
    for name,model in c.heads.items():
        other=copy.deepcopy(model);opt=make_optimizer(other)
        update(model,c.optimizers[name],samples,1500,microbatch=4,variant=name,prevalidated_source=True)
        update_batched(other,opt,samples,1500,name,views,head_batch=size)
        assert compare(compact_state(model),compact_state(other))<1e-6
        assert compare(c.optimizers[name].state_dict(),opt.state_dict())<1e-6
        for (n,p),(nn,q) in zip(model.named_parameters(),other.named_parameters()):
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)


def test_dropout_hook_restored_after_exception():
    torch.manual_seed(0)
    c=cohort();m=c.heads['UA'];original=m.decoder.fuse[-1].forward
    with pytest.raises(RuntimeError):
        with paired_dropout(m,[0],0):raise RuntimeError('interrupted')
    assert m.decoder.fuse[-1].forward==original
    assert 'forward' not in m.decoder.fuse[-1].__dict__


def test_batched_checkpoint_resume_and_snapshot_are_independent(tmp_path):
    from causalq.runtime.lora_training import save_checkpoint,restore_checkpoint,rng_digest
    from causalq.runtime.cohort_runner import capture_rng
    from tools.check_lora_l0_batching import snapshot
    torch.manual_seed(0)
    c=cohort();initial=copy.deepcopy(c.heads['UA']);m=c.heads['UA'];o=c.optimizers['UA']
    for step in (0,1):
        samples=packet(step*8);v=features(c.backbone,samples,16,False)
        update_batched(m,o,samples,step,'UA',v)
        if step==0:save_checkpoint(tmp_path/'prefix',m,o,1,{'test':True})
    expected=rng_digest(capture_rng());opt=make_optimizer(initial)
    assert restore_checkpoint(tmp_path/'prefix',initial,opt,{'test':True})==1
    update_batched(initial,opt,samples,1,'UA',v)
    assert compare(compact_state(m),compact_state(initial))==0
    assert compare(o.state_dict(),opt.state_dict())==0
    assert rng_digest(capture_rng())==expected
    snap=snapshot(c);before=copy.deepcopy(snap)
    next(iter(o.state.values()))['exp_avg'].add_(1)
    assert compare(snap,before)==0
