import copy
import pytest
import torch
from test_lora_formal import cohort
from test_lora_phase40 import packet
from causalq.runtime.lora_training import compact_state,rng_digest
from causalq.runtime.cohort_runner import capture_rng
from causalq.runtime.identity import tree_digest
from tools.diagnose_lora_batch_states import tensor_difference,compare_prefixes


def states():
    torch.manual_seed(0);c=cohort();m=c.heads['UA'];o=c.optimizers['UA']
    def snapshot(step):
        return {'step':step,'source_cursor':step*8,'model':compact_state(m),
            'optimizer':copy.deepcopy(o.state_dict()),'rng':capture_rng()}
    c.tick(packet());original=snapshot(1);c.tick(packet(8));old=snapshot(2)
    return old,copy.deepcopy(old),original,m


def test_records_all_optimizer_differences_without_accepting_migration():
    old,new,original,m=states();before=tree_digest(old)
    slot=next(iter(new['optimizer']['state'].values()))
    slot['exp_avg'].view(-1)[0]+=2e-5
    rng=rng_digest(capture_rng());r=compare_prefixes(old,new,original,m)
    assert r['model_allclose'] and not r['optimizer_allclose'] and r['rng_equal']
    assert len(r['adamw'])==sum(len(g['params']) for g in old['optimizer']['param_groups'])
    assert any(v['exp_avg']['failed_elements']>0 for v in r['adamw'].values())
    inferred=max(v['first_moment_inferred_clipped_gradient']['max_abs'] for v in r['adamw'].values())
    assert inferred==pytest.approx(2e-4,abs=1e-9)
    assert tree_digest(old)==before and rng_digest(capture_rng())==rng


def test_reports_parameter_names_and_zero_size_scalar_statistics():
    old,new,original,m=states();key=next(iter(new['model']));new['model'][key].view(-1)[0]+=.1
    r=compare_prefixes(old,new,original,m)
    assert not r['model_allclose'] and r['parameters'][key]['failed_elements']>0
    assert tensor_difference(torch.empty(0),torch.empty(0))['max_abs']==0
    assert tensor_difference(torch.tensor(1.),torch.tensor(2.))['max_abs_flat_index']==0


@pytest.mark.parametrize('kind',['shape','nonfinite','dtype'])
def test_rejects_invalid_tensor_comparisons(kind):
    a=torch.ones(2);b=a.clone()
    if kind=='shape':b=torch.ones(3)
    if kind=='nonfinite':b[0]=float('nan')
    if kind=='dtype':b=b.double()
    with pytest.raises(ValueError):tensor_difference(a,b)


def test_refuses_different_optimizer_groups_or_steps():
    old,new,original,m=states();new['optimizer']['param_groups'][0]['lr']=.1
    with pytest.raises(ValueError,match='groups'):compare_prefixes(old,new,original,m)
    new=copy.deepcopy(old);new['step']=3
    with pytest.raises(ValueError,match='prefix'):compare_prefixes(old,new,original,m)
