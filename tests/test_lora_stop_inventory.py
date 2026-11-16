import copy
import random
import pytest
import torch

from test_lora_formal import cohort
from test_lora_phase40 import packet
from causalq.runtime.cohort_runner import capture_rng
from causalq.runtime.lora_training import compact_state,learning_rate,rng_digest
from tools.audit_lora_stopped import safe,protected,validate_state,inventory


def example():
    c=cohort();c.tick(packet());model=c.heads['UA']
    identity={'test':'synthetic'}
    state={'schema':1,'identity':identity,'step':1,'source_cursor':8,'next_lr':learning_rate(1),
        'model':compact_state(model),'optimizer':copy.deepcopy(c.optimizers['UA'].state_dict()),'rng':capture_rng()}
    return state,{'step':1,'identity':identity},model


def test_readonly_schema_optimizer_and_rng_inventory():
    s,m,model=example();before=rng_digest(capture_rng())
    r=validate_state(s,m,model)
    assert r['source_cursor']==8 and len(r['optimizer_sha256'])==64
    assert before==rng_digest(capture_rng())


@pytest.mark.parametrize('fault',['cursor','moment','model','rng','lr'])
def test_rejects_invalid_recovery_state(fault):
    s,m,model=example()
    if fault=='cursor':s['source_cursor']=16
    if fault=='model':s['model'].pop(next(iter(s['model'])))
    if fault=='rng':s['rng'].pop('numpy')
    if fault=='lr':s['optimizer']['param_groups'][0]['lr']=.1
    if fault=='moment':next(iter(s['optimizer']['state'].values()))['exp_avg'].fill_(float('nan'))
    with pytest.raises((ValueError,KeyError)):validate_state(s,m,model)


def test_provisional_latest_common_final_protections():
    keep,common=protected(dict(U0=[1000,2000,3000,40000],UA=[1000,2000],C0=[1000,2000,3000],CA=[1000,2000,4000]))
    assert common==[1000,2000]
    assert keep['U0']==[1000,2000,3000,40000] and keep['CA']==[1000,2000,4000]
    keep,common=protected(dict(U0=[1000],UA=[],C0=[],CA=[]))
    assert not common and keep['U0']==[1000]


def test_path_escape_and_missing_run_fail_closed(tmp_path):
    with pytest.raises(ValueError):safe(tmp_path,tmp_path.parent/'outside')
    with pytest.raises(ValueError):safe(tmp_path,tmp_path/'..'/'outside')
    with pytest.raises(ValueError,match='missing'):inventory(tmp_path/'missing')


def test_symlink_refused(tmp_path):
    link=tmp_path/'link'
    try:link.symlink_to(tmp_path/'target')
    except OSError:pytest.skip('Host does not permit unprivileged symlinks')
    with pytest.raises(ValueError):safe(tmp_path,link)
