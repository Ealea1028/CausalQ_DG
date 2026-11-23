"""Phase44 B2 only: two independent LoRA backbones, formal updates throughout."""
import argparse,json,os,sys,subprocess,time,traceback,uuid
from pathlib import Path
from contextlib import ExitStack
from causalq.runtime.lora_phase44_b2 import *
from causalq.runtime.lora_parallel import MPS,hardware,resource_plan
from causalq.runtime.lora_phase43 import check_runtime
from causalq.runtime.identity import write_json,file_sha
from tools.train_lora_core_d3_l1 import BASE,worker,git_sha,package,require_retirement
from tools.train_lora_phase42_parallel import pool
from tools.retire_lora_l0 import safe

def progress(args):
    result={}
    for job,spec in SPECS.items():
        steps=[]
        for p in (args.output/job/'checkpoints').glob('*/complete.json'):
            m=json.loads(p.read_text());i=m['identity']
            if (i['phase']!=44 or i['variant']!='B2' or i['backbone']!=spec['backbone']
                or i['training_producer_git_sha']!=args.expected_sha or not 0<m['step']<=40000):
                raise ValueError('Foreign B2 checkpoint')
            steps.append(m['step'])
        result[job]=max(steps,default=0)
    return result

def run(args):
    from tools.train_lora_core_d3_l1 import runtime
    load_contract();require_retirement();check_runtime(runtime());verify_references()
    initial=progress(args);out=args.output;start=time.monotonic()
    import shutil
    need=sum((40000-s+999)//1000 for s in initial.values())*(80<<20)+(3<<30)
    if shutil.disk_usage(out).free<need:raise RuntimeError('Insufficient full-history checkpoint budget')
    old=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else {}
    report={'phase':44,'module':'B2','git_sha':args.expected_sha,'ok':False,'status':'running',
            'initial_committed_steps':initial,'target_data_accessed':False,'models':{},'legs':[],
            'expected_formal_optimizer_updates':80000,'previous_good_windows':old.get('previous_good_windows',0)+sum(
                w['target_met'] for l in old.get('legs',[]) for w in l['windows'])}
    selected=old.get('selected_jobs',2);policy=json.loads(POLICY.read_text())
    base=json.loads((DIRECTORY/'execution_profile.json').read_text());machine=hardware()
    attempt=out/'cohort_attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
    try:
        with MPS(out) as mps:
            report['mps']=mps.record
            while True:
                e=resource_plan(selected,machine,base)
                e.update(phase=44,config_path=CONFIG,freeze_path=FREEZE.relative_to(ROOT).as_posix(),
                         policy_sha256=file_sha(POLICY),variants=list(SPECS),original_donor_migration=False)
                report['selected_jobs']=e['jobs'];write_json(out/'resolved_execution.json',e)
                write_json(out/'report.json',report);leg=attempt/f"jobs{e['jobs']}";leg.mkdir()
                result=pool(args,e,mps.env,leg,policy);report['legs'].append(result)
                report['models'].update(result['models']);report['status']=result['status']
                if result['status'] in ('oom','needs_performance_review') and e['jobs']>1:
                    selected=1;continue
                break
        good=report['previous_good_windows']+sum(w['target_met'] for l in report['legs'] for w in l['windows'])
        report['ok']=(report['status']=='complete' and progress(args)==dict.fromkeys(SPECS,40000)
                      and len(report['models'])==2 and all(m['ok'] for m in report['models'].values()) and good>0)
    except BaseException:report['status']='failed';report['error']=traceback.format_exc()
    try:verify_references();report['core_finals_unchanged']=True
    except Exception:report['ok']=False;report['status']='failed';report['reference_error']=traceback.format_exc()
    report['committed_steps']=progress(args);report['formal_optimizer_updates_total']=sum(report['committed_steps'].values())
    report['formal_optimizer_updates_this_invocation']=report['formal_optimizer_updates_total']-sum(initial.values())
    report['seconds']=time.monotonic()-start
    if report['status']=='complete' and not report['ok']:report['status']='needs_performance_review'
    write_json(out/'report.json',report);print(json.dumps({k:report[k] for k in ('ok','status','committed_steps')}),flush=True)
    return 0 if report['ok'] else 130 if report['status']=='interrupted' else 2

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-sha',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--variant',choices=SPECS);p.add_argument('--parent-pid',type=int)
    p.add_argument('--status',action='store_true');p.add_argument('--package',action='store_true');p.add_argument('--archive',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/datasets'))
    p.add_argument('--pretrained-root',type=Path,default=Path('/root/autodl-tmp/pretrained'));a=p.parse_args()
    if sys.platform!='linux' or git_sha()!=a.expected_sha: p.error('AutoDL exact SHA required')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():p.error('Tracked source modified')
    allowed=BASE/f'phase44_b2_{a.expected_sha[:7]}'
    if a.output.absolute()!=allowed:p.error('Use exact-SHA B2 output')
    if Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip()).resolve()!=ROOT.resolve():p.error('Selected checkout required')
    safe(BASE,allowed);a.phase=44;a.job_specs=SPECS;a.worker_module='tools.train_lora_phase44_b2';a.policy_path=POLICY;a.progress_reader=progress
    if a.status:
        print((allowed/'report.json').read_text() if (allowed/'report.json').exists() else 'not started');print(progress(a));return 0
    import fcntl
    if a.package:
        if a.archive is None:p.error('--archive required')
        with (allowed/'run.lock').open('r+') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);package(allowed,a.archive)
        return 0
    load_contract()
    if a.worker:
        if a.variant is None or a.parent_pid!=os.getppid():p.error('Coordinator-owned worker only')
        job=a.variant;e=json.loads((allowed/'resolved_execution.json').read_text())
        if (e['phase']!=44 or e['config_path']!=CONFIG or e['policy_sha256']!=file_sha(POLICY)
            or e['freeze_path']!=FREEZE.relative_to(ROOT).as_posix() or e['original_donor_migration']
            or e['profile']['microbatch_source']!=4 or e['profile']['effective_source_batch']!=8):
            p.error('Changed B2 execution')
        e.update(job_key=job,backbone=SPECS[job]['backbone']);a.variant='B2';return worker(a,execution=e)
    allowed.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        handles=[];locks=[BASE/'gpu0.lock',allowed/'run.lock']+sorted({Path(pin['path']).parents[3]/'run.lock' for _,_,_,pin in references()})
        for path in locks:
            f=stack.enter_context(safe(BASE,path).open('a+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
        a.lock_fds=tuple(f.fileno() for f in handles);return run(a)
if __name__=='__main__':raise SystemExit(main())
