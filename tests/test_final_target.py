from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from tools import evaluate_target as evaluator
from tools.audit_trained_anchor_stop import audit_stop,validate_stop
from tools.train_strict_source import file_sha
from tools.strict_source_resume import capture_rng
from tools.strict_source_smoke import tree_digest

ROOT=Path(__file__).resolve().parents[1]


def test_returned_stop_hashes_and_ratio(tmp_path):
    result=audit_stop()
    assert result['m3_started'] is False and result['raw_anchor_residual_ratio']==pytest.approx(27.601217631413384)
    for name in ('job_report.json','stability_report.json','REPORT_SHA256SUMS'):
        (tmp_path/name).write_bytes((ROOT/'experiments/STRICT_DG_PHASE25'/name).read_bytes())
    (tmp_path/'job_report.json').write_bytes((tmp_path/'job_report.json').read_bytes()+b' ')
    with pytest.raises(ValueError,match='SHA256'):audit_stop(tmp_path)


def test_stop_rejects_false_m3_start():
    report=json.loads((ROOT/'experiments/STRICT_DG_PHASE25/job_report.json').read_bytes());report['m3_started']=True
    with pytest.raises(ValueError):validate_stop(report)


def test_freeze_contains_all_completed_models_and_fixed_membership(tmp_path):
    freeze=evaluator.frozen_contract()
    assert set(freeze['variants'])=={'M0','M1','P1','M2'} and freeze['m3_status']=='not_run_hypothesis_unsupported'
    ids=evaluator.target_ids();assert len(ids)==len(set(ids))==500
    assert {city:sum(i.startswith(city+'/') for i in ids) for city in ('frankfurt','lindau','munster')}=={'frankfurt':267,'lindau':59,'munster':174}
    path=tmp_path/'bad.txt';path.write_text('\n'.join(ids[:-1])+'\n')
    with pytest.raises(ValueError,match='SHA256'):evaluator.target_ids(path)


def make_native_pair(root,*,label_shape=(1024,2048)):
    identity='frankfurt/frankfurt_000000_000294';city,stem=identity.split('/')
    a=root/f'leftImg8bit/val/{city}/{stem}_leftImg8bit.png';b=root/f'gtFine/val/{city}/{stem}_gtFine_labelTrainIds.png'
    a.parent.mkdir(parents=True);b.parent.mkdir(parents=True)
    Image.new('RGB',(2048,1024),(100,120,130)).save(a)
    label=np.zeros(label_shape,dtype=np.uint8);label[0,0]=255;Image.fromarray(label).save(b)
    return identity


def test_native_dataset_keeps_ignore_pixels_and_never_resizes_gt(tmp_path):
    identity=make_native_pair(tmp_path)
    data=evaluator.FinalCityscapes(tmp_path,[identity]);sample=data[0]
    assert sample['image'].shape==(3,1024,2048) and sample['label'].shape==(1024,2048) and sample['label'][0,0]==255
    with pytest.raises(ValueError,match='frozen500'):evaluator.FinalCityscapes(tmp_path,[])


def test_target_dataset_rejects_scale_mismatch_instead_of_resizing(tmp_path):
    identity=make_native_pair(tmp_path,label_shape=(512,1024))
    with pytest.raises(ValueError,match='geometry'):evaluator.FinalCityscapes(tmp_path,[identity])[0]


class NativeModel(nn.Module):
    def __init__(self):super().__init__();self.bias=nn.Parameter(torch.tensor(1.))
    def forward(self,x):
        # Class0 predicted everywhere; native output without materialized logits.
        return self.bias.expand(1,19,1024,2048)


def test_final_metric_matches_independent_counts_without_optimizer_or_rng_changes(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('optimizer constructed by evaluator'))
    model=NativeModel().eval();label=torch.zeros(1024,2048,dtype=torch.long);label[0,0]=255;label[0,1]=1
    data=[{'id':'synthetic','image':torch.zeros(3,1024,2048),'label':label}]
    before=tree_digest((model.state_dict(),capture_rng()))
    result=evaluator.evaluate(model,data,['synthetic'],tmp_path,device='cpu',amp=False)
    assert result['valid_pixel_count']==1024*2048-1 and result['confusion_matrix'][1][0]==1
    assert result['target_miou']==pytest.approx((1024*2048-2)/(1024*2048-1)/2)
    assert result['target_labels_optimized'] is False and before==tree_digest((model.state_dict(),capture_rng()))


def test_cached_target_result_rejects_changed_metrics(tmp_path):
    trace=tmp_path/'target_eval.jsonl';trace.write_text('fixed trace\n')
    matrix=np.eye(19,dtype=np.int64)
    result={'ok':True,'identity':{'x':1},'sample_count':500,'target_labels_optimized':False,'model_rng_unchanged':True,
            'trace_sha256':file_sha(trace),'confusion_matrix':matrix.tolist(),'valid_pixel_count':19,'target_miou':1.}
    path=tmp_path/'metrics.json';path.write_text(json.dumps(result))
    evaluator.validate_result(tmp_path,{'x':1})
    result['target_miou']=.9;path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='mIoU'):evaluator.validate_result(tmp_path,{'x':1})


def test_cli_refuses_real_target_and_weights_locally(tmp_path,monkeypatch):
    frozen=evaluator.frozen_contract()
    monkeypatch.setattr(evaluator,'ROOT',tmp_path/'checkout')
    monkeypatch.setattr(evaluator,'_git',lambda *a:'' if a[0]=='status' else 'a'*40)
    monkeypatch.delitem(sys.modules,'tools.train',raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(evaluator,'FinalCityscapes',lambda *a:pytest.fail('real target data opened locally'))
    monkeypatch.setattr(evaluator.DINOv3Backbone,'from_pretrained',lambda *a,**k:pytest.fail('real weights loaded locally'))
    # Frozen metadata uses its original absolute path; no real data is needed.
    monkeypatch.setattr(evaluator,'source_report',lambda *a:{})
    monkeypatch.setattr(evaluator,'frozen_contract',lambda:frozen)
    with pytest.raises(RuntimeError,match='AutoDL CUDA'):
        evaluator.main(['--variant','M0','--output-root',str(tmp_path/'out')])


def test_fresh_evaluator_import_has_no_legacy_trainer():
    code="""
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*a):
  if fullname=='tools.train':raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import tools.evaluate_target
"""
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
