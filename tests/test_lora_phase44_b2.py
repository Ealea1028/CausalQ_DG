import json
from types import SimpleNamespace
import pytest
import torch
from test_hpi_backbone import tiny_backbone
from test_lora_backbones import FakeHF
from test_lora_phase40 import packet
from causalq.models.dinov3_wrapper import DINOv3Backbone
from causalq.runtime.lora_phase44_b2 import b2_head,B2Trainer,load_contract,SPECS
from causalq.runtime.lora_training import load_state_file
from tools.train_lora_core_d3_l1 import identity_for
from tools.train_lora_phase44_b2 import progress

@pytest.mark.parametrize('kind',['d2','d3'])
def test_b2_real_update_resume_keeps_zero_residual_and_frozen_dino(kind,tmp_path):
    def build():
        base=tiny_backbone() if kind=='d2' else DINOv3Backbone(FakeHF(),intermediate_indices=(1,2,3,4))
        return B2Trainer(b2_head(base,decoder_channels=32,num_classes=3,blocks=(2,3)),horizon=3)
    t=build();before=[p.clone() for p in t.frozen];t.advance(packet())
    assert not t.model.query_bank.query_residuals.requires_grad
    assert not torch.count_nonzero(t.model.query_bank.query_residuals)
    assert all(torch.equal(a,b) for a,b in zip(before,t.frozen))
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in t.model.query_router.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in t.model.query_router.parameters())
    job='D2_B2' if kind=='d2' else 'D3_B2';args=SimpleNamespace(expected_sha='a'*40,variant='B2',data_root=tmp_path,pretrained_root=tmp_path,output=tmp_path)
    i=identity_for(t,args,{'synthetic':True},phase=44,backbone_name=SPECS[job]['backbone'])
    t.commit(tmp_path/job/'checkpoints',i);r=build();assert r.restore(tmp_path/job/'checkpoints',i)==1
    r.advance(packet(8));r.commit(tmp_path/job/'checkpoints',i)
    assert progress(args)[job]==2
    s=load_state_file(tmp_path/job/'checkpoints/00000002/state.pt');assert s['source_cursor']==16
    with torch.no_grad():r.model.query_bank.query_residuals.add_(1)
    with pytest.raises(ValueError):r.commit(tmp_path/'bad',i)

def test_b2_contract_releases_only_two_residual_free_ce_models():
    c,f=load_contract();assert c['variants']==['B2'] and len(c['backbones'])==2
    assert c['query']['anchor_cqe_variants']==[] and c['frozen_zero_residual']
    assert c['router_seed']==20261031 and c['localization_weight']==0
    assert c['formal_steps']*len(SPECS)==80000 and not c['target_access_authorized']

def test_completed_b2_job_key_resolves_to_variant_and_backbone(tmp_path,monkeypatch):
    import tools.train_lora_phase42_parallel as cli
    from causalq.runtime.identity import file_sha
    root=tmp_path/'D2_B2';root.mkdir();payload=root/'state.pt';payload.write_bytes(b'synthetic')
    policy=tmp_path/'policy.json';policy.write_text('{}')
    identity={'training_producer_git_sha':'a'*40,'variant':'B2','phase':44,
              'backbone':SPECS['D2_B2']['backbone'],'parallel_policy_sha256':file_sha(policy),
              'data_root':str(tmp_path.resolve()),'pretrained_root':str(tmp_path.resolve())}
    marker={'sha256':file_sha(payload)}
    (root/'training_report.json').write_text(json.dumps({'ok':True,'identity':identity,
        'base_unchanged':True,'adapter_changed':True,'checkpoint':marker}))
    monkeypatch.setattr(cli,'checkpoint_rows',lambda *a:[(40000,root,marker)])
    args=SimpleNamespace(expected_sha='a'*40,phase=44,job_specs=SPECS,policy_path=policy,
                         data_root=tmp_path,pretrained_root=tmp_path)
    assert cli.verified_final(tmp_path,'D2_B2',args)['ok']
