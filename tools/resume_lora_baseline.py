"""Verified orchestration of the original Phase41 producer, not state migration."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback
from causalq.runtime.identity import file_sha,write_json
from causalq.runtime.lora_formal import DIRECTORY,VARIANTS,core_head
from tools.audit_lora_stopped import safe,ShapeOnlyBackbone,validate_state
from tools.check_lora_l0_batching import RUN,NAME

TRAIN_SHA='1af6b2e91e54083be433341f9fc8958954536272'


def selected_states(folder,identity,model,floor):
    folder=safe(folder,folder)
    if not folder.is_dir():raise ValueError('Missing original checkpoint directory')
    records=[]
    for p in folder.glob('*/complete.json'):
        safe(folder,p);m=json.loads(p.read_text());step=m['step']
        if m['identity']!=identity or type(step) is not int or not 0<step<=40000 or p.parent.name!=f'{step:08d}':
            raise ValueError('Original marker identity/step changed')
        if not safe(folder,p.parent/'state.pt').is_file():raise ValueError('Committed payload missing')
        records.append((step,p))
    for p in folder.rglob('*'):
        safe(folder,p)
        if p.is_file() and (p.name.endswith('.partial') or p.name=='state.pt' and not (p.parent/'complete.json').exists()):
            raise ValueError('Uncommitted payload needs review: '+str(p))
    records.sort(key=lambda x:x[0],reverse=True)
    if not records or records[0][0]<floor:raise ValueError('Verified progress regressed; refuse fresh initialization')
    out=[]
    for step,p in records[:2]:
        m=json.loads(p.read_text());payload=p.parent/'state.pt'
        if file_sha(payload)!=m['sha256']:raise ValueError('Original payload SHA mismatch')
        state=__import__('torch').load(payload,map_location='cpu',weights_only=False)
        validate_state(state,m,model);del state
        out.append({'step':step,'sha256':m['sha256'],'bytes':payload.stat().st_size,'path':str(payload)})
    return out


def preflight(code):
    if (RUN/'L0_RETIRED.json').exists():
        raise ValueError('L0 study retired by user; baseline continuation is disabled')
    code=safe(code,code)
    sha=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
    if sha!=TRAIN_SHA or subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Training must use clean original exact commit')
    freeze=json.loads((code/'experiments/LORA_CAUSALQ_1114/phase41_freeze.json').read_text())
    for name,digest in freeze['files'].items():
        if hashlib.sha256((code/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise ValueError('Original frozen source changed: '+name)
    acceptance=json.loads((DIRECTORY/'phase41_stop_acceptance.json').read_text())
    result={'training_git_sha':sha,'training_code_root':str(code),'run_root':str(RUN),'states':{},'cursors':{},'all_payloads_retained':True}
    for backbone,heads in acceptance['latest'].items():
        result['states'][backbone]={};result['cursors'][backbone]={}
        for name,row in heads.items():
            model=core_head(ShapeOnlyBackbone(),name)
            states=selected_states(RUN/backbone/'checkpoints'/name,row['identity'],model,row['step'])
            result['states'][backbone][name]=states;result['cursors'][backbone][name]=states[0]['step']
        report=RUN/backbone/'training_report.json'
        if backbone!=NAME:
            r=json.loads(report.read_text())
            if not r['ok'] or r['git_sha']!=TRAIN_SHA or any(v!=40000 for v in result['cursors'][backbone].values()):
                raise ValueError('D3 final not verified; never retrain it')
        elif report.exists() and json.loads(report.read_text()).get('status')=='needs_performance_review':
            raise ValueError('Original worker stopped for performance; return report for explicit review')
    result['committed_updates']=sum(sum(c.values()) for c in result['cursors'].values())
    result['remaining_updates']=320000-result['committed_updates']
    segments=sum((40000-step+999)//1000 for c in result['cursors'].values() for step in c.values())
    largest=max(s['bytes'] for heads in result['states'].values() for rows in heads.values() for s in rows)
    result['required_free_bytes']=int(segments*max(50<<20,largest*1.2))+(2<<30)
    result['free_bytes']=shutil.disk_usage(RUN).free
    if result['free_bytes']<result['required_free_bytes']:raise ValueError('Insufficient space; no checkpoint cleanup authorized')
    return result


def active_trainers(proc_root=Path('/proc')):
    active=[]
    for p in proc_root.glob('[0-9]*/cmdline'):
        try:
            cmd=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if 'tools.train_lora_core_l0' in cmd and str(RUN) in cmd:active.append({'pid':p.parent.name,'command':cmd})
        except OSError:pass
    return active


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--code-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if sys.platform!='linux':p.error('Execute this controller only on AutoDL')
    if RUN in a.output.absolute().parents or a.output.absolute()==RUN:p.error('Fresh controller output must be outside formal run')
    a.output.mkdir(parents=True,exist_ok=True)
    if any((a.output/n).exists() for n in ('report.json','preflight.json','training.log')):p.error('Fresh controller directory required')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():p.error('Controller code modified')
    report={'phase':41,'ok':False,'training_git_sha':TRAIN_SHA,'controller_git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'selected_mode':'baseline','profile_changed':False,'cross_sha_state_migration':False,'cleanup_applied':False}
    def stop(*_):raise KeyboardInterrupt('Controller interrupted')
    signal.signal(signal.SIGTERM,stop)
    try:
        import fcntl
        with ExitStack() as stack:
            for lock in (RUN.parent/'gpu0.lock',RUN/'run.lock'):
                safe(RUN.parent,lock);f=stack.enter_context(lock.open('r+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if active_trainers():raise ValueError('Original worker still active; do not start duplicate training')
            prepared=preflight(a.code_root);write_json(a.output/'preflight.json',prepared)
            report['initial_committed_updates']=prepared['committed_updates']
            print(json.dumps({'cursors':prepared['cursors'],'remaining_formal_updates':prepared['remaining_updates']}),flush=True)
            # Inspect the old environment in a disposable subprocess: no idle parent CUDA context.
            live=json.loads(subprocess.check_output([sys.executable,'-c','import json;from tools.train_lora_core_l0 import runtime;print(json.dumps(runtime()))'],cwd=a.code_root,text=True,timeout=60))
            expected=json.loads((DIRECTORY/'phase41_stop_acceptance.json').read_text())['latest'][NAME]['U0']['identity']['runtime']
            report['runtime']=live
            if live!=expected:raise ValueError('Exact recovery environment differs; preserve old identity and return differences')
            old_report=RUN/'report.json';previous_mtime=old_report.stat().st_mtime_ns if old_report.exists() else -1
            with (a.output/'training.log').open('w') as log:
                child=subprocess.Popen([sys.executable,'-u','-m','tools.train_lora_core_l0','--output',str(RUN)],cwd=a.code_root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                report['training_pid']=child.pid
                try:report['training_exit_code']=child.wait()
                except BaseException:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
            if not old_report.exists() or old_report.stat().st_mtime_ns==previous_mtime:
                raise ValueError('Original coordinator produced no fresh final report; committed progress remains unknown until re-audit')
            final=json.loads(old_report.read_text())
            report['formal_report']=final
            report['final_committed_updates']=final.get('formal_optimizer_updates_total')
            if report['final_committed_updates'] is not None:report['new_committed_updates']=report['final_committed_updates']-prepared['committed_updates']
            report['ok']=report['training_exit_code']==0 and final['ok'] and final['git_sha']==TRAIN_SHA and report['final_committed_updates']==320000
    except (Exception,KeyboardInterrupt):report['error']=traceback.format_exc()
    write_json(a.output/'report.json',report)
    print(json.dumps({'ok':report['ok'],'report':str(a.output/'report.json'),'error':report.get('error')}),flush=True)
    return 0 if report['ok'] else 2


if __name__=='__main__':raise SystemExit(main())
