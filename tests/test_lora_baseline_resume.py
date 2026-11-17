import json
import pytest
import torch
from test_lora_formal import cohort
from test_lora_phase40 import packet
from causalq.runtime.lora_training import save_checkpoint
from tools.resume_lora_baseline import selected_states,active_trainers
from tools.check_lora_l0_batching import RUN


def checkpoints(tmp_path):
    torch.manual_seed(0);c=cohort();identity={'synthetic_original':True}
    for step in (1,2):
        c.tick(packet((step-1)*8));save_checkpoint(tmp_path/f'{step:08d}',c.heads['UA'],c.optimizers['UA'],step,identity)
    return c.heads['UA'],identity


def test_validates_latest_two_complete_boundaries_without_mutation(tmp_path):
    m,identity=checkpoints(tmp_path)
    before={p.name+p.parent.name:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    rows=selected_states(tmp_path,identity,m,2)
    assert [r['step'] for r in rows]==[2,1]
    assert before=={p.name+p.parent.name:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault',['corrupt','identity','missing','orphan','regressed'])
def test_refuses_unsafe_or_unexpected_resume(tmp_path,fault):
    m,identity=checkpoints(tmp_path);folder=tmp_path/'00000002'
    if fault=='corrupt':
        with (folder/'state.pt').open('ab') as f:f.write(b'corrupt')
    if fault=='identity':
        p=folder/'complete.json';j=json.loads(p.read_text());j['identity']={'wrong':True};p.write_text(json.dumps(j))
    if fault=='missing':(folder/'state.pt').unlink()
    if fault=='orphan':(folder/'state.pt.partial').write_bytes(b'interrupted')
    with pytest.raises((ValueError,OSError)):
        selected_states(tmp_path,identity,m,3 if fault=='regressed' else 2)


def test_detects_existing_original_training_worker_only(tmp_path):
    for pid,cmd in [(101,f'python -m tools.train_lora_core_l0 --output {RUN}'),(102,'python unrelated_job.py')]:
        p=tmp_path/str(pid);p.mkdir();(p/'cmdline').write_bytes(cmd.replace(' ','\0').encode())
    rows=active_trainers(tmp_path)
    assert len(rows)==1 and rows[0]['pid']=='101'
