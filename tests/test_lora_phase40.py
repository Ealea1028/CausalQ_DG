import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_hpi_backbone import tiny_backbone
from causalq.models.query_segmentor import QuerySegmentor
from causalq.models.lora import inject_dino_lora
from causalq.runtime.lora_training import (learning_rate,make_optimizer,update,compact_state,
    save_checkpoint,restore_checkpoint,load_state_file,rng_digest)
from causalq.runtime.cohort_runner import capture_rng
from tools.accept_lora_phase39 import validate
from tools.check_lora_resume import compare


def model():
    torch.manual_seed(11)
    m=QuerySegmentor(tiny_backbone(),decoder_channels=32,num_classes=3,queries_per_class=2,
                      interaction='static',dropout=.1)
    inject_dino_lora(m.backbone,blocks=(2,3))
    return m


def packet(start=0):
    g=torch.Generator().manual_seed(9)
    return [{'views':torch.randn(2,3,32,32,generator=g),
             'label':torch.randint(0,3,(32,32),generator=g),
             'metadata':{'step_zero_based':start+i}} for i in range(8)]


def test_hpi_schedule_and_norm_parameter_coverage():
    assert learning_rate(0)==pytest.approx(1e-10)
    assert learning_rate(1500)==pytest.approx(1e-6+(1e-4-1e-6)*(1-1500/40000)**.9)
    assert learning_rate(40000)==1e-6
    with pytest.raises(ValueError): learning_rate(40001)
    m=model(); opt=make_optimizer(m)
    params=[p for group in opt.param_groups for p in group['params']]
    assert len(params)==len({id(p) for p in params})
    assert {id(p) for p in params}=={id(p) for p in m.parameters() if p.requires_grad}
    assert {g['weight_decay'] for g in opt.param_groups}=={0.,1e-5}


def test_batch_accumulation_preserves_dropout_stream_and_moments():
    a=model(); b=copy.deepcopy(a)
    oa,ob=make_optimizer(a),make_optimizer(b)
    update(a,oa,packet(),1500,microbatch=1)
    update(b,ob,packet(),1500,microbatch=8)
    assert compare(compact_state(a),compact_state(b))<1e-6
    assert compare(oa.state_dict(),ob.state_dict())<1e-6


def test_compact_restore_includes_lora_and_replays_optimizer(tmp_path):
    m=model(); opt=make_optimizer(m)
    identity={'backbone':'synthetic','lora':m.backbone.lora_manifest,'batch':8}
    update(m,opt,packet(),0,microbatch=2)
    save_checkpoint(tmp_path/'one',m,opt,1,identity)
    saved=load_state_file(tmp_path/'one/state.pt')
    assert any(k.startswith('backbone.') and k.endswith('.B') for k in saved['model'])
    assert not any('.base.weight' in k for k in saved['model'])
    update(m,opt,packet(8),1,microbatch=2)
    clone=model(); oc=make_optimizer(clone)
    assert restore_checkpoint(tmp_path/'one',clone,oc,identity)==1
    update(clone,oc,packet(8),1,microbatch=2)
    assert compare(compact_state(m),compact_state(clone))==0
    assert compare(opt.state_dict(),oc.state_dict())==0
    with pytest.raises(ValueError,match='identity'):
        restore_checkpoint(tmp_path/'one',clone,oc,{'batch':1})
    with pytest.raises(ValueError,match='already'):
        save_checkpoint(tmp_path/'one',m,opt,2,identity)
    with (tmp_path/'one/state.pt').open('ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError,match='hash'):
        restore_checkpoint(tmp_path/'one',clone,oc,identity)


def test_rng_digest_checks_numpy_middle_bytes():
    a=capture_rng(); b=copy.deepcopy(a)
    b['numpy'][1][100]^=np.uint32(1)
    assert rng_digest(a)!=rng_digest(b)


def test_phase39_acceptance_rejects_duplicate_models_and_nonfinite_gradients():
    report=json.loads(Path('experiments/LORA_CAUSALQ_1114/phase39_report.json').read_text())
    validate(report)
    bad=copy.deepcopy(report); bad['models'][1]=bad['models'][0]
    with pytest.raises(ValueError): validate(bad)
    bad=copy.deepcopy(report)
    key=next(iter(bad['models'][0]['final_adapter_gradient_norms']))
    bad['models'][0]['final_adapter_gradient_norms'][key]=float('nan')
    with pytest.raises(ValueError): validate(bad)
