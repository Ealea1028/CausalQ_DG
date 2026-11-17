import copy
import pytest
import torch
from torch.nn import functional as F
from causalq.runtime.lora_batched import SamplewiseConvGradient,native_samplewise_conv_gradients,features,update_batched
from causalq.runtime.lora_training import update,compact_state,make_optimizer
from tools.check_lora_resume import compare
from tools.check_lora_l0_batching import snapshot,errors,ComparisonFailure
from test_lora_formal import cohort
from test_lora_phase40 import packet


@pytest.mark.parametrize('amp',[False,True])
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('batch',[4,8])
def test_native_per_sample_conv_vjp_matches_legacy_cast_boundary(amp,bias,batch):
    torch.manual_seed(7)
    x=torch.randn(batch,3,7,7,requires_grad=True);w=torch.randn(5,3,3,3,requires_grad=True)
    b=torch.randn(5,requires_grad=True) if bias else None
    xx=x.detach().clone().requires_grad_();ww=w.detach().clone().requires_grad_();bb=b.detach().clone().requires_grad_() if bias else None
    refs=[]
    for start in range(0,batch,4):
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
            refs.extend(F.conv2d(x[j:j+1],w,b,padding=1) for j in range(start,start+4))
    ref=torch.cat(refs)
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        out=SamplewiseConvGradient.apply(xx,ww,bb,(1,1),(1,1),(1,1),1)
    torch.testing.assert_close(ref,out,atol=1e-5,rtol=1e-5)
    grad=torch.randn_like(ref)
    ref.backward(grad);out.backward(grad)
    for a,z in [(x,xx),(w,ww),*([(b,bb)] if bias else [])]:torch.testing.assert_close(a.grad,z.grad,atol=1e-5,rtol=1e-5)


def test_frozen_input_and_native_context_cleanup():
    torch.manual_seed(0);c=cohort();m=c.heads['UA'];x=torch.randn(3,2,6,6);w=torch.randn(4,2,1,1,requires_grad=True)
    out=SamplewiseConvGradient.apply(x,w,None,1,0,1,1);out.sum().backward();assert w.grad is not None
    with pytest.raises(RuntimeError):
        with native_samplewise_conv_gradients(m):raise RuntimeError('interruption')
    assert all('forward' not in mod.__dict__ for mod in m.decoder.modules() if isinstance(mod,torch.nn.Conv2d))


def test_repaired_head_preserves_full_fp32_update_and_optimizer():
    torch.manual_seed(0);c=cohort();p=packet();f=features(c.backbone,p,16,False)
    for name,m in c.heads.items():
        other=copy.deepcopy(m);opt=make_optimizer(other)
        update(m,c.optimizers[name],p,1500,microbatch=4,variant=name,prevalidated_source=True)
        update_batched(other,opt,p,1500,name,f,preserve_conv_reduction=True)
        assert compare(compact_state(m),compact_state(other))<1e-6
        assert compare(c.optimizers[name].state_dict(),opt.state_dict())<1e-6


def test_failed_comparison_retains_all_heads_with_unchanged_tolerances():
    torch.manual_seed(0);c=cohort();c.tick(packet());ref=snapshot(c);actual=copy.deepcopy(ref)
    for name,row in actual.items():next(iter(row['optimizer']['state'].values()))['exp_avg'].view(-1)[0]+=.0001
    with pytest.raises(ComparisonFailure) as exc:errors(ref,actual)
    assert set(exc.value.rows)==set(c.heads)
    assert all(not row['ok'] and row['optimizer_failed_tensors'] for row in exc.value.rows.values())
