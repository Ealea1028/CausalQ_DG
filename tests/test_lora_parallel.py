from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from causalq.runtime.lora_parallel import GIB, resource_plan, MPS
from tools.train_lora_phase42_parallel import measured_window, pool


def base():
    return json.load(open('experiments/LORA_CAUSALQ_1114/execution_profile.json'))


@pytest.mark.parametrize('cpu,ram,jobs', [(64,128,4), (16,32,4), (8,8,1)])
def test_realistic_cpu_ram_budgets_without_changing_effective_batch(cpu, ram, jobs):
    plan = resource_plan(4, {'cpu_budget':cpu,'available_host_bytes':ram*GIB,'dev_shm_free_bytes':64*GIB}, base())
    assert plan['jobs'] == jobs
    assert plan['total_workers'] <= max(0,cpu-1)
    assert plan['profile']['microbatch_source'] == 4 and plan['profile']['effective_source_batch'] == 8
    assert plan['jobs']*plan['loader_policy_estimate']['host_bound_bytes'] <= plan['host_pipeline_budget_bytes']


def test_host_memory_guard_rejects_unsafe_even_single_model():
    with pytest.raises(RuntimeError):
        resource_plan(4, {'cpu_budget':64,'available_host_bytes':2*GIB,'dev_shm_free_bytes':GIB}, base())


@pytest.mark.parametrize('updates,memory,passed', [(90,.2,True), (50,.2,False), (90,.9,False)])
def test_selection_uses_aggregate_update_speed_and_actual_gpu_memory(updates,memory,passed):
    start={n:{'step':0,'data_wait_seconds':0} for n in ('U0','UA','C0','CA')}
    end={n:{'step':updates,'data_wait_seconds':1} for n in start}
    util={'samples':120,'median':94,'p10':90}
    result=measured_window(end,start,120,util,memory,23.606389638098957/8,{'minimum_baseline_fraction':.95})
    assert result['target_met'] is passed
    assert result['aggregate_updates_per_second'] == pytest.approx(updates*4/120)


def test_mps_only_starts_and_stops_private_service(tmp_path,monkeypatch):
    import causalq.runtime.lora_parallel as resource
    calls=[]
    monkeypatch.setattr(resource.shutil,'which',lambda _: '/usr/bin/nvidia-cuda-mps-control')
    pipe=tmp_path/'private'; pipe.mkdir()
    monkeypatch.setattr(resource.tempfile,'mkdtemp',lambda **kwargs:str(pipe))
    def command(args,**kwargs):
        calls.append((args,kwargs))
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(resource.subprocess,'run',command)
    with MPS(tmp_path) as service:
        assert service.record['enabled']
        assert service.env['CUDA_MPS_PIPE_DIRECTORY'].startswith(str(tmp_path))
    assert [kw.get('input') for _,kw in calls] == [None,'get_server_list\n','quit\n']
    assert all(kw['env']['CUDA_MPS_PIPE_DIRECTORY'].startswith(str(tmp_path)) for _,kw in calls)


def test_missing_mps_falls_back_without_changing_parent_environment(tmp_path,monkeypatch):
    import causalq.runtime.lora_parallel as resource
    monkeypatch.setattr(resource.shutil,'which',lambda _: None)
    with MPS(tmp_path) as service:
        assert not service.record['enabled'] and not service.started


@pytest.mark.parametrize('fail',[False,True])
def test_four_workers_are_launched_before_wait_and_failure_stops_owned_peers(tmp_path,monkeypatch,fail):
    import tools.train_lora_phase42_parallel as cli
    from causalq.runtime.lora_l1_formal import VARIANTS
    args=SimpleNamespace(output=tmp_path,expected_sha='a'*40,data_root=tmp_path,pretrained_root=tmp_path,lock_fds=(8,9))
    monkeypatch.setattr(cli,'verified_final',lambda *a: None)
    monkeypatch.setattr(cli,'hints',lambda *a:{v:0 for v in VARIANTS})
    monkeypatch.setattr(cli.signal,'signal',lambda *a:None)
    monkeypatch.setattr(cli,'Telemetry',lambda *a:nullcontext(SimpleNamespace(started=0)))
    monkeypatch.setattr(cli.time,'sleep',lambda *a:None)
    children=[]; signaled=[]
    class Child:
        def __init__(self,variant):
            self.variant=variant; self.pid=100+len(children); self.returncode=None
        def poll(self):
            assert len(children)==4, 'All four must be submitted before waiting'
            if self.returncode is None and (not fail or self.variant=='U0'):
                self.returncode=2 if fail else 0; self.report()
            return self.returncode
        def report(self):
            path=tmp_path/self.variant/'training_report.json'
            path.write_text(json.dumps({'ok':self.returncode==0,'status':'complete' if self.returncode==0 else 'failed'}))
        def send_signal(self,sig):
            signaled.append(self.variant); self.returncode=130; self.report()
    def launch(command,**kwargs):
        assert kwargs['pass_fds']==(8,9) and kwargs['start_new_session'] is True
        variant=command[command.index('--variant')+1]
        child=Child(variant); children.append(child); return child
    monkeypatch.setattr(cli.subprocess,'Popen',launch)
    leg=tmp_path/'leg';leg.mkdir()
    result=pool(args,{'jobs':4},{},leg,{'single_ua_source_images_per_second':23.606389638098957})
    assert len(children)==4 and result['status']==('failed' if fail else 'complete')
    if fail: assert set(signaled)=={'UA','C0','CA'}
