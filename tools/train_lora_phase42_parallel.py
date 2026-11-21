"""Fresh Phase42 parallel execution; all measured updates are formal training."""
import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

from causalq.runtime.hpi_prepare import Telemetry
from causalq.runtime.identity import file_sha, write_json
from causalq.runtime.lora_l1_formal import ROOT, DIRECTORY, NAME, VARIANTS, checkpoint_rows
from causalq.runtime.lora_parallel import hardware, resource_plan, MPS
from causalq.runtime.lora_performance import telemetry
from tools.retire_lora_l0 import safe
from tools.train_lora_core_d3_l1 import BASE, git_sha, hints, require_retirement, worker, status, package

POLICY = DIRECTORY/'phase42_parallel_policy.json'


def contract():
    freeze = json.loads((DIRECTORY/'phase42_parallel_freeze.json').read_text())
    for path, digest in freeze['files'].items():
        if hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError('Parallel release source changed: '+path)
    return json.loads(POLICY.read_text())


def verified_final(output, variant, args):
    path = output/variant/'training_report.json'
    if not path.exists(): return None
    report = json.loads(path.read_text())
    if report.get('recovery_blocked'): raise ValueError('Failed base invariant; local review required')
    if not report.get('ok'): return None
    identity = report['identity']
    spec = getattr(args, 'job_specs', {}).get(variant, {})
    if (identity.get('training_producer_git_sha') != args.expected_sha or identity.get('variant') != spec.get('variant', variant)
            or identity.get('phase') != getattr(args, 'phase', 42)
            or identity.get('backbone') != spec.get('backbone', getattr(args, 'backbone_name', NAME))
            or identity.get('parallel_policy_sha256') != file_sha(getattr(args, 'policy_path', POLICY))
            or identity.get('data_root') != str(args.data_root.resolve())
            or identity.get('pretrained_root') != str(args.pretrained_root.resolve())
            or not report.get('base_unchanged') or not report.get('adapter_changed')):
        raise ValueError('Completed parallel model identity changed')
    rows = checkpoint_rows(output/variant/'checkpoints', identity)
    if not rows or rows[0][0] != 40000: raise ValueError('Final checkpoint missing')
    for _, folder, marker in rows[:2]:
        if file_sha(folder/'state.pt') != marker['sha256']: raise ValueError('Final hash mismatch')
    if rows[0][2]['sha256'] != report['checkpoint']['sha256']: raise ValueError('Final report/marker mismatch')
    return report


def read_progress(folder, pid):
    path = folder/'execution_progress.json'
    if not path.exists(): return None
    row = json.loads(path.read_text())
    return row if row.get('pid') == pid else None


def committed_progress(args):
    if hasattr(args, 'progress_reader'): return args.progress_reader(args)
    if getattr(args, 'phase', 42) == 42 and getattr(args, 'backbone_name', NAME) == NAME:
        return hints(args.output, args.expected_sha)
    return hints(args.output, args.expected_sha, phase=args.phase, backbone=args.backbone_name)


def measured_window(rows, start_rows, seconds, util, memory_fraction, baseline, policy):
    updates = sum(rows[n]['step']-start_rows[n]['step'] for n in rows)
    wait = sum(max(0., rows[n]['data_wait_seconds']-start_rows[n]['data_wait_seconds']) for n in rows)
    rate = updates/max(seconds, 1e-9)
    met = (util['samples'] >= 100 and util['median'] >= 70 and util['p10'] >= 60
           and wait/max(seconds*len(rows), 1e-9) < .1 and memory_fraction <= .85
           and rate >= baseline*policy['minimum_baseline_fraction'])
    return {'updates': updates, 'seconds': seconds, 'aggregate_updates_per_second': rate,
            'aggregate_source_images_per_second': rate*8, 'baseline_updates_per_second': baseline,
            'data_wait_fraction': wait/max(seconds*len(rows), 1e-9),
            'memory_fraction': memory_fraction, 'telemetry': util, 'target_met': met}


def terminate_owned(active, stop_time):
    for item in active.values():
        child = item['child']
        if child.poll() is not None: continue
        try:
            if time.monotonic()-stop_time > 60: os.killpg(child.pid, signal.SIGKILL)
            else: child.send_signal(signal.SIGTERM)
        except ProcessLookupError: pass


def pool(args, execution, environment, leg, policy):
    active = {}; pending = []; models = {}; stop = {'at': None, 'reason': None}
    result = {'status': 'running', 'windows': [], 'models': models, 'execution': execution}
    for variant in execution.get('variants', VARIANTS):
        existing = verified_final(args.output, variant, args)
        if existing is not None: models[variant] = existing
        else: pending.append(variant)
    def request_stop(signum, frame):
        if stop['at'] is None: stop.update(at=time.monotonic(), reason='interrupted')
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, 'SIGHUP'): signal.signal(signal.SIGHUP, request_stop)
    baseline = policy['single_ua_source_images_per_second']/8
    start_rows = None; window_start = None
    previous_progress = 0.
    try:
        with Telemetry(leg/'telemetry.csv', 1) as monitor:
            while pending or active:
                if stop['at'] is not None: terminate_owned(active, stop['at'])
                while pending and len(active) < execution['jobs'] and stop['at'] is None:
                    variant = pending.pop(0); folder = args.output/variant; folder.mkdir(exist_ok=True)
                    progress = folder/'training_report.json'
                    stamp = progress.stat().st_mtime_ns if progress.exists() else -1
                    log = (folder/'worker.log').open('a')
                    command = [sys.executable, '-u', '-m', getattr(args, 'worker_module', 'tools.train_lora_phase42_parallel'), '--worker',
                               '--variant', variant, '--parent-pid', str(os.getpid()), '--output', str(args.output),
                               '--expected-sha', args.expected_sha, '--data-root', str(args.data_root),
                               '--pretrained-root', str(args.pretrained_root)]
                    try:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                                 env=environment, start_new_session=True, pass_fds=getattr(args, 'lock_fds', ()))
                    except BaseException:
                        log.close(); raise
                    active[variant] = {'child': child, 'log': log, 'stamp': stamp}
                    print('Parallel start:', variant, 'PID', child.pid, 'active', len(active), flush=True)
                    start_rows = None
                for variant, item in list(active.items()):
                    child = item['child']
                    if child.poll() is None: continue
                    item['log'].close(); del active[variant]; start_rows = None
                    path = args.output/variant/'training_report.json'
                    if not path.exists() or path.stat().st_mtime_ns == item['stamp']:
                        raise RuntimeError(f'{variant} exited {child.returncode} without a fresh report')
                    report = json.loads(path.read_text()); models[variant] = report
                    if child.returncode != 0 or not report['ok']:
                        if stop['at'] is None:
                            oom = 'out of memory' in report.get('error', '').lower() or 'OutOfMemoryError' in report.get('error', '')
                            stop.update(at=time.monotonic(), reason='oom' if oom else 'failed')
                rows = {n: read_progress(args.output/n, item['child'].pid) for n, item in active.items()}
                if active and stop['at'] is None and all(r and r['ready'] for r in rows.values()):
                    now = time.monotonic()-monitor.started
                    if start_rows is None:
                        start_rows = rows; window_start = now
                    elif set(start_rows) == set(rows) and now-window_start >= 120:
                        util = telemetry(leg/'telemetry.csv', start=window_start, end=now)
                        memory = []
                        with (leg/'telemetry.csv').open() as stream:
                            for row in csv.DictReader(stream):
                                if window_start <= float(row['elapsed_seconds']) <= now:
                                    try: memory.append(float(row['memory_used_mib'])*(1 << 20))
                                    except ValueError: pass
                        total_memory = policy['visible_memory_bytes']
                        window = measured_window(rows, start_rows, now-window_start, util,
                                                 max(memory, default=total_memory)/total_memory, baseline, policy)
                        if execution.get('baseline_calibration'):
                            # A single job may leave spare GPU capacity: its speed is the comparison,
                            # while activity/wait targets apply to the subsequent concurrent cohort.
                            import math
                            window['calibration_valid'] = (util['samples'] >= 100
                                and math.isfinite(window['aggregate_source_images_per_second'])
                                and window['aggregate_source_images_per_second'] > 0
                                and window['memory_fraction'] <= .85)
                            window['target_met'] = False
                        result['windows'].append(window); write_json(leg/'cohort.json', result)
                        if execution.get('baseline_calibration'):
                            print('Single-UA baseline:', window['aggregate_source_images_per_second'],
                                  'source images/s; valid', window['calibration_valid'], flush=True)
                        else:
                            print('Cohort:', execution['jobs'], 'jobs', window['aggregate_source_images_per_second'],
                                  'source images/s; GPU', util['median'], 'target', window['target_met'], flush=True)
                        if execution.get('baseline_calibration'):
                            stop.update(at=time.monotonic(), reason='baseline_complete' if window['calibration_valid']
                                        else 'needs_performance_review')
                        elif not window['target_met']:
                            stop.update(at=time.monotonic(), reason='needs_performance_review')
                        start_rows = rows; window_start = now
                if time.monotonic()-previous_progress > 10:
                    write_json(args.output/'parallel_progress.json', {'active_variants': list(active),
                        'jobs': execution['jobs'], 'steps': {n: r['step'] for n, r in rows.items() if r},
                        'committed_steps': committed_progress(args)})
                    previous_progress = time.monotonic()
                if not active and stop['at'] is not None: break
                time.sleep(.25)
    except BaseException:
        result['error'] = traceback.format_exc(); stop.update(at=time.monotonic(), reason='failed')
    finally:
        if active:
            when = stop['at'] or time.monotonic()
            while any(i['child'].poll() is None for i in active.values()):
                terminate_owned(active, when); time.sleep(.25)
            for item in active.values(): item['log'].close()
        result['status'] = stop['reason'] or 'complete'
        result['committed_steps'] = committed_progress(args)
        write_json(leg/'cohort.json', result)
    return result


def run(args):
    started = time.monotonic()
    policy = contract(); require_retirement()
    initial = hints(args.output, args.expected_sha)
    required = sum((40000-step+999)//1000 for step in initial.values())*(50 << 20)+(2 << 30)
    if __import__('shutil').disk_usage(args.output).free < required: raise RuntimeError('Insufficient full-history checkpoint budget')
    old_file = args.output/'report.json'
    old = json.loads(old_file.read_text()) if old_file.exists() else {}
    selected = old.get('selected_jobs', 4) if args.jobs == 'auto' else int(args.jobs)
    report = {'phase': 42, 'git_sha': args.expected_sha, 'mode': 'parallel', 'ok': False, 'status': 'starting',
              'initial_committed_steps': initial, 'target_data_accessed': False, 'models': {}, 'legs': [],
              'checkpoint_budget_bytes': required, 'expected_formal_optimizer_updates': 160000,
              'previous_good_windows': old.get('previous_good_windows', 0)+sum(
                  w['target_met'] for leg in old.get('legs', []) for w in leg.get('windows', []))}
    launch = args.output/'cohort_attempts'/uuid.uuid4().hex; launch.mkdir(parents=True)
    machine = hardware(); base = json.loads((DIRECTORY/'execution_profile.json').read_text())
    try:
        with MPS(args.output, args.mps) as mps:
            report['mps'] = mps.record
            while True:
                execution = resource_plan(selected, machine, base)
                execution.update(config_path='configs/lora_causalq_1114/phase42_parallel.json',
                                 policy_sha256=file_sha(POLICY), mps_enabled=mps.record['enabled'])
                report['selected_jobs'] = execution['jobs']; report['status'] = 'running'
                write_json(args.output/'resolved_execution.json', execution)
                write_json(args.output/'report.json', report)
                leg = launch/f"jobs{execution['jobs']}"; leg.mkdir()
                result = pool(args, execution, mps.env, leg, policy)
                report['legs'].append(result); report['models'].update(result['models'])
                report['status'] = result['status']
                if result['status'] in ('oom', 'needs_performance_review') and execution['jobs'] > 1:
                    selected = execution['jobs']//2
                    print('Saved full updates; reducing concurrency to', selected, flush=True)
                    continue
                break
        good = report['previous_good_windows']+sum(w['target_met'] for leg in report['legs'] for w in leg['windows'])
        final = hints(args.output, args.expected_sha)
        report['ok'] = (report['status'] == 'complete' and all(s == 40000 for s in final.values())
                        and len(report['models']) == 4 and all(r['ok'] for r in report['models'].values()) and good > 0)
        if report['status'] == 'complete' and not report['ok']: report['status'] = 'needs_performance_review'
    except BaseException:
        report['status'] = 'failed'; report['error'] = traceback.format_exc()
    report['committed_steps'] = hints(args.output, args.expected_sha)
    report['formal_optimizer_updates_total'] = sum(report['committed_steps'].values())
    report['formal_optimizer_updates_this_invocation'] = sum(report['committed_steps'].values())-sum(initial.values())
    report['seconds'] = time.monotonic()-started
    report['committed_updates_per_second_including_startup_and_io'] = report['formal_optimizer_updates_this_invocation']/max(report['seconds'],1e-9)
    report['exit_code'] = 0 if report['ok'] else 130 if report['status'] == 'interrupted' else 2
    write_json(args.output/'report.json', report)
    print(json.dumps({k: report[k] for k in ('ok', 'status', 'selected_jobs', 'committed_steps') if k in report}), flush=True)
    return report['exit_code']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--jobs', choices=('auto', '1', '2', '4'), default='auto')
    parser.add_argument('--mps', choices=('auto', 'off'), default='auto')
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--parent-pid', type=int)
    parser.add_argument('--status', action='store_true'); parser.add_argument('--package', action='store_true')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets'))
    parser.add_argument('--pretrained-root', type=Path, default=Path('/root/autodl-tmp/pretrained'))
    args = parser.parse_args()
    if sys.platform != 'linux' or git_sha() != args.expected_sha: parser.error('AutoDL exact SHA required')
    if Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip()).resolve() != ROOT.resolve():
        parser.error('Run from the selected code checkout')
    if __import__('subprocess').check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        parser.error('Tracked code modified')
    allowed = BASE/f'phase42_parallel_d3_l1_{args.expected_sha[:7]}'
    if args.output.absolute() != allowed: parser.error('Use this SHA-specific parallel output')
    safe(BASE, allowed)
    if args.status:
        status(allowed, args.expected_sha)
        p = allowed/'parallel_progress.json'
        if p.exists(): print(p.read_text())
        return 0
    import fcntl
    if args.package:
        if args.archive is None: parser.error('--archive required')
        with (allowed/'run.lock').open('r+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); package(allowed, args.archive)
        return 0
    contract()
    if args.worker:
        if args.variant is None or args.parent_pid != os.getppid(): parser.error('Coordinator-owned worker only')
        return worker(args, execution=json.loads((allowed/'resolved_execution.json').read_text()))
    allowed.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handles = []
        for path in (BASE/'gpu0.lock', allowed/'run.lock'):
            handle = stack.enter_context(safe(BASE, path).open('a+')); handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.lock_fds = tuple(h.fileno() for h in handles)
        return run(args)


if __name__ == '__main__': raise SystemExit(main())
