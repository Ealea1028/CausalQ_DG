from copy import deepcopy
import json
import subprocess
import sys

import pytest
import torch
import yaml

from causalq.interventions import StyleInterventionBank
from causalq.losses.causal_query_effect import causal_query_effect_loss
from causalq.protocols.formal_source import ROOT, formal_config, validate_formal_config
from causalq.datasets.strict_source import load_membership
from causalq.utils.checkpoint import load_training_checkpoint
from tools.audit_full_cqe_source_result import audit_full_cqe, validate_full_cqe_result
from tools.full_cqe_source_update import anchor_cqe_source_update
from tools.strict_source_resume import capture_rng, restore_rng
from tools.strict_source_runner import train_segment
from tools.strict_source_smoke import source_optimizer_update, tree_digest
from tools.train_strict_source import file_sha
from tools import trained_anchor_stability as stability
from tools import run_anchor_formal_phase25 as job
from tools import train_formal_source as trainer
from test_full_cqe_source_formal import preparation
from test_strict_source_runner import make_state


def test_m2_returned_counts_and_hashes(tmp_path):
    result = audit_full_cqe()
    assert result['formal_m2_accepted']
    assert result['delta_vs_m1_percentage_points'] == pytest.approx(-.1359578169468)
    for name in ('formal_report.json','REPORT_SHA256SUMS'):
        (tmp_path/name).write_bytes((ROOT/'experiments/STRICT_DG_PHASE24'/name).read_bytes())
    path=tmp_path/'formal_report.json';path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='SHA256'): audit_full_cqe(tmp_path)


def test_m2_rejects_false_metrics():
    report=json.loads((ROOT/'experiments/STRICT_DG_PHASE24/formal_report.json').read_bytes())
    report['source_evaluation']['class_iou'][0]+=.01
    with pytest.raises(ValueError): validate_full_cqe_result(report)


def test_m3_recipe_only_changes_cqe_component_and_conditional_identity():
    config=yaml.safe_load((ROOT/'configs/strict_dg/formal/m3_seed0.yaml').read_bytes())
    validate_formal_config(config,'M3')
    assert config['causal_query_effect'].pop('component')=='anchor_only'
    config['experiment']=formal_config('M2')['experiment']
    assert config==formal_config('M2')


def test_anchor_step_matches_explicit_loss_with_two_backbone_forwards():
    model,optimizer,config,image,photo,label=preparation()
    config['causal_query_effect']=formal_config('M3')['causal_query_effect']
    reference=deepcopy(model)
    opt=torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    rng=capture_rng();calls=[]
    hook=model.backbone.register_forward_hook(lambda *a:calls.append(True))
    record=anchor_cqe_source_update(model,optimizer,image,photo,label,1,config,device='cpu',amp=False)
    hook.remove();restore_rng(rng)
    original=reference.forward_components(image,include_anchor=True)
    styled=reference.forward_components(photo,include_anchor=True)
    from tools.deterministic_source_ce import deterministic_source_ce as ce
    consistency=causal_query_effect_loss(styled.anchor_scaled_delta_logits.float(),original.anchor_scaled_delta_logits.detach().float(),label)
    objective=ce(original.logits,label)+ce(styled.logits,label)+consistency
    objective.backward();torch.nn.utils.clip_grad_norm_([p for p in reference.parameters() if p.requires_grad],1);opt.step()
    assert len(calls)==2 and record['loss_anchor_cqe']==pytest.approx(float(consistency.detach()))
    for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a,b,atol=2e-7,rtol=1e-5)


def test_anchor_only_consistency_has_no_direct_residual_parameter_gradient():
    model,optimizer,config,image,photo,label=preparation()
    original=model.forward_components(image,include_anchor=True)
    styled=model.forward_components(photo,include_anchor=True)
    causal_query_effect_loss(model.get_anchor_query_effect(output=styled),model.get_anchor_query_effect(output=original).detach(),label).backward()
    assert model.query_bank.query_residuals.grad is None
    assert model.query_bank.class_anchors.grad is not None
    assert model.query_bank.class_anchors.grad.norm()>0


def test_anchor_weight_zero_reproduces_m1_update():
    values=[]
    from tools.deterministic_source_ce import deterministic_source_ce
    for function in (source_optimizer_update,anchor_cqe_source_update):
        model,opt,config,image,photo,label=preparation()
        config['causal_query_effect']=formal_config('M3')['causal_query_effect'];config['causal_query_effect']['lambda_cqe']=0
        function(model,opt,image,photo,label,1,config,device='cpu',amp=False,loss_fn=deterministic_source_ce)
        values.append(tree_digest((model.state_dict(),opt.state_dict())))
    assert values[0]==values[1]


def test_m3_exact_resume_crosses_epoch(tmp_path):
    for mode,start,stop in [('full',0,8),('prefix',0,4),('resume',4,8)]:
        output=tmp_path/mode;output.mkdir()
        model,opt,dataset,config=make_state();model.query_bank.centered_residuals=True
        config['causal_query_effect']=formal_config('M3')['causal_query_effect']
        train_segment(model,opt,dataset,StyleInterventionBank(),config,output,start=start,stop=stop,
            metadata={'seed':0},resume=tmp_path/'prefix/state.pth' if start else None,
            device='cpu',amp=False,update_fn=anchor_cqe_source_update)
    a=load_training_checkpoint(tmp_path/'full/state.pth');b=load_training_checkpoint(tmp_path/'resume/state.pth')
    for key in ('trainable_model','optimizer','rng','sampler'):assert tree_digest(a[key])==tree_digest(b[key])


def test_raw_decision_rejects_ties_collapse_and_normalized_substitution():
    def row(a,r):return {'source_id':'x','anchor_variance_by_class':[a]*19,'residual_variance_by_class':[r]*19}
    assert stability.decision([row(1,2)])['m3_training_authorized']
    for a,r in [(2,1),(1,1),(0,0),(0,1)]:assert not stability.decision([row(a,r)])['m3_training_authorized']
    with pytest.raises(ValueError):stability.decision([row(float('nan'),1)])
    with pytest.raises(ValueError):stability.decision([row(1,2),row(1,2)])


def test_measure_pair_is_read_only_and_preserves_rng():
    model,_,_,image,_,label=preparation();model.eval();style=StyleInterventionBank()
    before=tree_digest((model.state_dict(),capture_rng()))
    result=stability.measure_pair(model,image,label,style,12)
    assert len(result['anchor_variance_by_class'])==4  # synthetic model has4 classes
    assert before==tree_digest((model.state_dict(),capture_rng()))


def evidence(tmp_path,allow):
    _,ids=load_membership()
    rows=[{'source_id':i,'anchor_variance_by_class':[1. if allow else 3.]*19,'residual_variance_by_class':[2.]*19} for i in ids]
    (tmp_path/'sample_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    original=stability.accepted_m1()
    report={'ok':True,'phase':25,'git_sha':'a'*40,'sample_count':2497,'backbone_forward_calls':4994,
        'checkpoint_sha256':original['checkpoint_sha256'],'checkpoint_training_updates':40000,
        'baseline_pair_report_sha256':stability.PAIR_SHA,'optimizer_updates_this_run':0,
        'read_only_state_unchanged':True,'target_data_accessed':False,
        'sample_index_sha256':file_sha(tmp_path/'sample_index.jsonl'),'decision':stability.decision(rows)}
    path=tmp_path/'stability_report.json';path.write_text(json.dumps(report));return path,report


def test_stability_validation_recomputes_raw_decision(tmp_path):
    path,report=evidence(tmp_path,False)
    assert not stability.validate_stability(path,'a'*40)['decision']['m3_training_authorized']
    report['decision']['m3_training_authorized']=True;path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='decision'):stability.validate_stability(path,'a'*40)


def test_direct_m3_cli_cannot_bypass_failed_hypothesis(tmp_path,monkeypatch):
    path,_=evidence(tmp_path,False)
    checkout=tmp_path/'checkout';target=checkout/'configs/strict_dg/formal/m3_seed0.yaml';target.parent.mkdir(parents=True)
    target.write_bytes((ROOT/'configs/strict_dg/formal/m3_seed0.yaml').read_bytes())
    monkeypatch.setattr(trainer,'ROOT',checkout);monkeypatch.setattr(trainer,'_git',lambda *a:'a'*40)
    with pytest.raises(ValueError,match='forbidden'):
        trainer.main(['--variant','M3','--output-root',str(tmp_path/'out'),'--stability-report',str(path)])


@pytest.mark.parametrize('allow',[False,True])
def test_conditional_job_starts_training_only_on_supported_hypothesis(tmp_path,monkeypatch,allow):
    calls=[];report={'git_sha':'a'*40,'decision':{'m3_training_authorized':allow}}
    monkeypatch.setattr(job,'validate_stability',lambda *a:report);monkeypatch.setattr(job,'_git',lambda *a:'a'*40)
    def run(command,check):
        calls.append(command)
        if 'tools.train_formal_source' in command:
            root=tmp_path/'M3';root.mkdir();(root/'formal_report.json').write_text('{"ok":true}')
    monkeypatch.setattr(job.subprocess,'run',run)
    job.main(['--output-root',str(tmp_path)])
    result=json.loads((tmp_path/'job_report.json').read_bytes())
    assert len(calls)==(2 if allow else 1) and result['m3_started']==allow


def test_fresh_imports_block_target_and_legacy():
    code="""
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*a):
  if fullname in ('tools.train','causalq.datasets.cityscapes'):raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import tools.trained_anchor_stability,tools.run_anchor_formal_phase25
"""
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_stability_cli_refuses_local_real_data_and_weights(tmp_path,monkeypatch):
    original=stability.accepted_m1()
    monkeypatch.setattr(stability,'ROOT',tmp_path/'checkout')
    monkeypatch.setattr(stability,'_git',lambda *a:'' if a[0]=='status' else 'a'*40)
    monkeypatch.setattr(stability,'accepted_m1',lambda:original)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.delitem(sys.modules,'tools.train',raising=False)
    monkeypatch.delitem(sys.modules,'causalq.datasets.cityscapes',raising=False)
    monkeypatch.setattr(stability,'build_source_datasets',lambda *a:pytest.fail('real source opened locally'))
    monkeypatch.setattr(stability.DINOv3Backbone,'from_pretrained',lambda *a,**k:pytest.fail('real weights opened locally'))
    with pytest.raises(RuntimeError,match='AutoDL CUDA'):
        stability.main(['--output-dir',str(tmp_path/'out')])
