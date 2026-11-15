import copy
import pytest
import torch

from test_hpi_backbone import tiny_backbone
from test_lora_phase40 import packet
from causalq.models.query_segmentor import QuerySegmentor
from causalq.models.hpi_anchor_only import freeze_query_residuals
from causalq.models.lora import inject_dino_lora
from causalq.models.lora_routing import LoRARoutedQuerySegmentor
from causalq.runtime.hpi_phase35 import build_ablation_head
from causalq.runtime.lora_training import update,make_optimizer,compact_state
from causalq.runtime.lora_performance import InputPrefetch,select,telemetry
from causalq.losses.lora_source import source_ce_prevalidated,anchor_cqe_prevalidated
from causalq.losses.source_ce import deterministic_source_ce
from causalq.losses.causal_query_effect import causal_query_effect_loss
from tools.check_lora_resume import compare


def test_prevalidated_losses_preserve_values_gradients_and_detach():
    g=torch.Generator().manual_seed(8)
    labels=torch.randint(0,3,(1,5,7),generator=g); labels[:,0]=255
    for base,fast in [(deterministic_source_ce,source_ce_prevalidated),(causal_query_effect_loss,anchor_cqe_prevalidated)]:
        x=torch.randn(1,4,5,7,generator=g,requires_grad=True)
        y=x.detach().clone().requires_grad_()
        ref=torch.randn_like(x,requires_grad=True)
        a,b=(base(x,labels),fast(y,labels)) if base==deterministic_source_ce else (base(x,ref,labels),fast(y,ref,labels))
        torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
        a.backward(); b.backward()
        torch.testing.assert_close(x.grad,y.grad,atol=1e-7,rtol=1e-6)
        assert ref.grad is None


def test_new_router_matches_frozen_reference_and_preserves_cls(monkeypatch):
    old=build_ablation_head(tiny_backbone(),'B3',None,decoder_channels=32,num_classes=3,dropout=0.)
    new=LoRARoutedQuerySegmentor(copy.deepcopy(old.backbone),decoder_channels=32,num_classes=3,
        queries_per_class=2,interaction='static',dropout=0.)
    new.query_router=copy.deepcopy(old.query_router); freeze_query_residuals(new)
    new.load_state_dict(old.state_dict(),strict=True)
    features=old.backbone(torch.randn(2,3,32,48))
    reference,route0=old.forward_routed(features,(32,48))
    seen=[]; real=QuerySegmentor.forward_features
    def capture(self,f,*args,**kwargs):
        seen.append(f.cls_token)
        return real(self,f,*args,**kwargs)
    monkeypatch.setattr(QuerySegmentor,'forward_features',capture)
    actual,route1=new.forward_routed(features,(32,48))
    assert seen[-1] is features.cls_token
    assert torch.equal(reference.logits,actual.logits)
    assert torch.equal(route0.injected,route1.injected)


def test_b3_microbatch_and_fast_losses_preserve_update():
    m=LoRARoutedQuerySegmentor(tiny_backbone(),decoder_channels=32,num_classes=3,
        queries_per_class=2,interaction='static',dropout=.1)
    from causalq.models.hpi_query_route import QueryFeatureRouter
    m.query_router=QueryFeatureRouter(32); freeze_query_residuals(m)
    inject_dino_lora(m.backbone,blocks=(2,3))
    n=copy.deepcopy(m); a,b=make_optimizer(m),make_optimizer(n)
    update(m,a,packet(),1500,microbatch=1,variant='B3')
    values=update(n,b,packet(),1500,microbatch=8,variant='B3',prevalidated_source=True,return_tensors=True)
    assert torch.isfinite(values).all()
    assert compare(compact_state(m),compact_state(n))<1e-6
    assert compare(a.state_dict(),b.state_dict())<1e-6
    assert not n.query_bank.query_residuals.requires_grad and torch.count_nonzero(n.query_bank.query_residuals)==0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for name,p in n.backbone.named_parameters() if name.endswith('.B'))


def test_input_prefetch_is_ordered_and_bounded_cpu():
    packets=[{'views':torch.full((8,2,3,2,2),float(i)),'labels':torch.zeros(8,2,2,dtype=torch.long),
              'metadata':[{'id':str(8*i+j)} for j in range(8)]} for i in range(5)]
    pipeline=InputPrefetch(packets,'cpu',2)
    seen=[]
    for p in pipeline: seen.extend(m['id'] for m in p['metadata'])
    assert seen==list(map(str,range(40))) and pipeline.peak_live_slots==2
    with pytest.raises(ValueError): InputPrefetch(packets,'cpu',3)


def test_performance_selection_rejects_invalid_and_applies_memory_tie():
    def row(name,speed,memory): return {'name':name,'ok':True,'numerical_ok':True,'timed_seconds':121,
        'peak_reserved_bytes':memory,'source_images_per_second':speed,'data_wait_fraction':.05,
        'telemetry':{'samples':110,'median':75,'p10':65}}
    a=row('micro1_serial',80,10); b=row('micro4_double',118,30); c=row('micro8_double',120,45)
    result=select([a,b,c],100)
    assert result['selected']=='micro4_double' and result['performance_target_met'] is True
    b['numerical_ok']=False; c['peak_reserved_bytes']=90
    assert select([a,b,c],100)['performance_target_met'] is False


def test_telemetry_missing_and_nonfinite_samples_are_not_success(tmp_path):
    p=tmp_path/'telemetry.csv'
    p.write_text('utilization_percent,memory_used_mib,power_w\n70,2,3\nN/A,N/A,N/A\nnan,nan,nan\n')
    result=telemetry(p)
    assert result['samples']==1 and result['median']==70 and result['power_w_mean']==3
