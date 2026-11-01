"""Exact-commit Phase33 launcher; future training remains explicitly unreleased."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

from causalq.protocols.hpi_dinov2_20261031 import ROOT, STUDY, RELEASED_ACTIONS, load_profile
from causalq.runtime.identity import file_sha, write_json


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def verify_checkout(expected):
    import causalq
    if len(expected) != 40 or any(c not in '0123456789abcdef' for c in expected) or git('rev-parse','HEAD') != expected:
        raise ValueError('Expected exact full release SHA')
    if git('status','--porcelain'):
        raise ValueError('Code checkout is dirty; do not edit AutoDL sources/configuration')
    if Path(causalq.__file__).resolve().parent != ROOT/'causalq':
        raise ValueError('Imported causalq does not belong to this exact checkout')


def package(out):
    status=out/'status.json'
    if status.exists() and json.loads(status.read_text()).get('status') in ('running','submitted'):
        raise RuntimeError('Package is read-only and requires a finished or failed preparation')
    files = sorted(p for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.log'))
    checksums = ''.join(f'{file_sha(p)}  {p.name}\n' for p in files)
    (out/'SHA256SUMS').write_text(checksums, encoding='ascii', newline='\n')
    stage = out/'export.zip.partial'
    with zipfile.ZipFile(stage,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (*files,out/'SHA256SUMS'):
            archive.write(path,path.name)
    os.replace(stage,out/'export.zip')


def run(out, expected):
    from causalq.runtime.hpi_prepare import Telemetry, prepare, telemetry_summary
    verify_checkout(expected)
    profile=load_profile(os.environ.get('CAUSALQ_RUNTIME_PROFILE'))
    write_json(out/'status.json',{'status':'running','git_sha':expected,'formal_optimizer_updates':0})
    write_json(out/'run_manifest.json',{'git_sha':expected,'study':STUDY,'action':'prepare','phase':33,
        'target_data_accessed':False,'formal_optimizer_updates':0,'runtime_profile':profile,
        'tracked_file_sha256':{name:file_sha(ROOT/name) for name in git('ls-files').splitlines() if (ROOT/name).is_file()},
        'python':sys.executable,'code_root':str(ROOT),'data_root':os.environ['CAUSALQ_DATA_ROOT'],
        'pretrained_root':os.environ['CAUSALQ_PRETRAINED_ROOT']})
    code=0
    try:
        with Telemetry(out/'gpu_telemetry.csv',profile['gpu_telemetry_interval_seconds']):
            prepare(out,profile,data_root=os.environ['CAUSALQ_DATA_ROOT'],pretrained_root=os.environ['CAUSALQ_PRETRAINED_ROOT'])
    except Exception as exc:
        code=1
        traceback.print_exc()
        write_json(out/'failure.json',{'error':str(exc),'type':type(exc).__name__,'git_sha':expected})
    if (out/'gpu_telemetry.csv').exists():
        write_json(out/'gpu_telemetry_summary.json',telemetry_summary(out/'gpu_telemetry.csv'))
    write_json(out/'status.json',{'status':'complete' if code==0 else 'failed','exit_code':code,
        'git_sha':expected,'formal_optimizer_updates':0,'target_data_accessed':False})
    print(f'phase33_exit_code={code}\noutput_directory={out}\nexport={out / "export.zip"}',flush=True)
    package(out)
    return code


def launch(out, expected):
    import fcntl
    verify_checkout(expected)
    out.mkdir(parents=True,exist_ok=True)
    # Shared across release SHAs/actions, retained by the detached coordinator.
    lock_path=Path(os.environ['CAUSALQ_OUTPUT_ROOT'])/'phase33.lock'
    lock=lock_path.open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Phase33 coordinator already running; use status')
    status=out/'status.json'
    if status.exists() and json.loads(status.read_text()).get('status')=='complete':
        print(f'already_complete\noutput_directory={out}\nexport={out / "export.zip"}')
        return
    # Keep failed evidence instead of overwriting it on a same-SHA retry.
    if (out/'run_manifest.json').exists():
        from datetime import datetime,timezone
        attempt=out/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        attempt.mkdir(parents=True)
        for path in out.iterdir():
            if path.is_file(): path.rename(attempt/path.name)
    env={**os.environ,'PYTHONPATH':str(ROOT),'PYTHONUNBUFFERED':'1'}
    write_json(status,{'status':'submitted','git_sha':expected,'formal_optimizer_updates':0})
    with (out/'launcher.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-m','tools.run_hpi_dinov2_20261031',expected,'prepare','--run',
            '--output',str(out)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=(lock.fileno(),))
    print(f'PID={child.pid}\noutput_directory={out}\nlauncher_log={out / "launcher.log"}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('expected_sha',nargs='?')
    parser.add_argument('action',nargs='?',choices=(*RELEASED_ACTIONS,'train-baselines','train-b1-b3','train-b4-b5','target-evaluation'))
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--compile-probe',type=Path)
    parser.add_argument('--probe-profile',type=Path)
    args=parser.parse_args()
    if args.compile_probe:
        from causalq.runtime.hpi_prepare import compile_probe
        compile_probe(args.compile_probe,args.output,json.loads(args.probe_profile.read_text()))
        return 0
    if args.action not in RELEASED_ACTIONS:
        parser.error('Phase33 releases prepare/status/package only; training and target access await later phases')
    verify_checkout(args.expected_sha)
    output=Path(os.environ['CAUSALQ_OUTPUT_ROOT'])/'phase33'/f'PREPARE_{args.expected_sha[:7]}'
    if args.output is not None and args.output.resolve()!=output.resolve():
        parser.error('Output must be the exact phase/SHA directory')
    if args.action=='status':
        print(f'output_directory={output}')
        print((output/'status.json').read_text() if (output/'status.json').exists() else 'not_started')
    elif args.action=='package':
        package(output)
        print(output/'export.zip')
    elif args.run:
        return run(output,args.expected_sha)
    else:
        launch(output,args.expected_sha)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
