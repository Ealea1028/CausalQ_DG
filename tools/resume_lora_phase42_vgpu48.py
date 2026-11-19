"""Phase42 formal continuation on the GPU accepted by the recovery gate."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from causalq.runtime.identity import file_sha, write_json
from causalq.runtime.lora_l1_formal import ROOT, DIRECTORY, VARIANTS, load_contract
from causalq.runtime.lora_migration import accepted, check_runtime, effective_progress
from causalq.runtime.lora_parallel import MPS, hardware, resource_plan
from tools.check_lora_vgpu48_resume import verify_donor
from tools.retire_lora_l0 import safe
from tools.train_lora_core_d3_l1 import (BASE, git_sha, hints, package, require_retirement,
                                         runtime, status, worker)
from tools.train_lora_phase42_parallel import pool

POLICY = DIRECTORY/'phase42_vgpu48_policy.json'
FREEZE = DIRECTORY/'phase42_vgpu48_freeze.json'
CONFIG = 'configs/lora_causalq_1114/phase42_vgpu48.json'
MODULE = 'tools.resume_lora_phase42_vgpu48'


def contract():
    load_contract()
    freeze = json.loads(FREEZE.read_text())
    for name, digest in freeze['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError('48GB continuation source changed: '+name)
    audit, recovery = accepted()
    policy = json.loads(POLICY.read_text())
    if (policy['visible_memory_bytes'] != recovery['runtime']['visible_memory_bytes']
            or policy['source_producer'] != audit['source_producer']
            or policy['source_root'] != audit['source_root']):
        raise ValueError('48GB continuation policy/evidence mismatch')
    return policy


def execution_plan(jobs, machine, base, mps, *, calibration=False):
    execution = resource_plan(jobs, machine, base)
    execution.update(config_path=CONFIG, freeze_path=FREEZE.relative_to(ROOT).as_posix(),
                     policy_sha256=file_sha(POLICY), original_donor_migration=True,
                     mps_enabled=mps, baseline_calibration=calibration)
    if calibration: execution['variants'] = ['UA']
    return execution


def validate_execution(execution):
    if (execution['config_path'] != CONFIG or execution['freeze_path'] != FREEZE.relative_to(ROOT).as_posix()
            or execution['policy_sha256'] != file_sha(POLICY) or execution['original_donor_migration'] is not True
            or execution['jobs'] not in (1, 2, 4)
            or execution['profile']['microbatch_source'] != 4
            or execution['profile']['effective_source_batch'] != 8
            or execution['profile']['buffer_slots'] != 2
            or execution.get('variants', list(VARIANTS)) not in (list(VARIANTS), ['UA'])):
        raise ValueError('Resolved continuation execution changed')


def baseline_record(args, result, mps_enabled):
    if result['status'] != 'baseline_complete' or len(result['windows']) != 1:
        raise ValueError('No complete same-GPU UA calibration window')
    window = result['windows'][0]
    model = result['models']['UA']
    if (not window.get('calibration_valid') or model.get('status') != 'interrupted'
            or not model.get('base_unchanged') or model.get('tainted_update')
            or model['committed_step'] <= model['restored_step']):
        raise ValueError('UA calibration did not save a valid full-update boundary')
    return {'git_sha': args.expected_sha, 'policy_sha256': file_sha(getattr(args, 'policy_path', POLICY)),
            'phase': getattr(args, 'phase', 42), 'backbone': getattr(args, 'backbone_name', 'dinov3_vitl16'),
            'runtime': model['runtime'], 'mps_enabled': mps_enabled, 'variant': 'UA',
            'window': window, 'source_images_per_second': window['aggregate_source_images_per_second'],
            'initial_step': model['restored_step'], 'committed_step': model['committed_step'],
            'checkpoint': model['checkpoint'], 'all_updates_formal': True, 'discarded_updates': 0}


def load_baseline(path, args, mps_enabled):
    if not path.exists(): return None
    record = json.loads(path.read_text())
    rate = record['source_images_per_second']
    if (record['git_sha'] != args.expected_sha or record['policy_sha256'] != file_sha(getattr(args, 'policy_path', POLICY))
            or record.get('phase', 42) != getattr(args, 'phase', 42)
            or record.get('backbone', 'dinov3_vitl16') != getattr(args, 'backbone_name', 'dinov3_vitl16')
            or record['mps_enabled'] != mps_enabled or not record['all_updates_formal']
            or not math.isfinite(rate) or rate <= 0):
        raise ValueError('Stored single-UA calibration identity changed')
    check_runtime(record['runtime'])
    return record


def run(args):
    started = time.monotonic()
    policy = contract(); audit, _ = accepted(); require_retirement()
    check_runtime(runtime())
    import torch
    props = torch.cuda.get_device_properties(0)
    if ([props.major, props.minor] != audit['target_gpu']['capability']
            or props.multi_processor_count != audit['target_gpu']['sm_count']
            or not torch.cuda.is_bf16_supported()):
        raise ValueError('Audited CUDA capability/SM/BF16 changed')
    for variant, row in audit['donors'].items(): verify_donor(Path(audit['source_root']), variant, row)
    initial = effective_progress(hints(args.output, args.expected_sha))
    # Native migrated segments, all future 1000-step states, emergency boundaries,
    # and staging space. Original donor files remain untouched on disk.
    required = sum((40000-s+999)//1000 for s in initial.values())*(50 << 20)+(3 << 30)
    if shutil.disk_usage(args.output).free < required: raise RuntimeError('Insufficient full-history checkpoint budget')
    previous = args.output/'report.json'
    old = json.loads(previous.read_text()) if previous.exists() else {}
    if old and (old['git_sha'] != args.expected_sha or old['policy_sha256'] != file_sha(POLICY)):
        raise ValueError('Continuation report identity changed')
    selected = old.get('selected_jobs', 4) if args.jobs == 'auto' else int(args.jobs)
    report = {'phase': 42, 'git_sha': args.expected_sha, 'policy_sha256': file_sha(POLICY),
              'mode': 'vgpu48_parallel_continuation', 'ok': False, 'status': 'starting',
              'initial_committed_steps': initial, 'source_producer': audit['source_producer'],
              'original_donor_steps': {v: r['step'] for v, r in audit['donors'].items()},
              'preserved_original_updates': 3238, 'migration_optimizer_updates': 0,
              'engineering_states_used': False, 'target_data_accessed': False,
              'checkpoint_budget_bytes': required, 'expected_formal_optimizer_updates': 160000,
              'models': {}, 'legs': [], 'previous_good_windows': old.get('previous_good_windows', 0)+sum(
                  w['target_met'] for leg in old.get('legs', []) for w in leg['windows'])}
    launch = args.output/'cohort_attempts'/uuid.uuid4().hex; launch.mkdir(parents=True)
    machine = hardware(); base = json.loads((DIRECTORY/'execution_profile.json').read_text())
    report['hardware'] = machine
    write_json(args.output/'report.json', report)
    try:
        with MPS(args.output, args.mps) as service:
            report['mps'] = service.record
            baseline_path = args.output/'single_ua_baseline.json'
            baseline = load_baseline(baseline_path, args, service.record['enabled'])
            if baseline is None:
                execution = execution_plan(1, machine, base, service.record['enabled'], calibration=True)
                write_json(args.output/'resolved_execution.json', execution)
                leg = launch/'baseline_ua'; leg.mkdir()
                print('Calibrating this48GB GPU using formal UA updates; no throwaway training', flush=True)
                result = pool(args, execution, service.env, leg, dict(policy, single_ua_source_images_per_second=0.))
                report['legs'].append(result); report['models'].update(result['models'])
                report['status'] = result['status']
                if result['status'] == 'baseline_complete':
                    baseline = baseline_record(args, result, service.record['enabled'])
                    write_json(baseline_path, baseline)
            if baseline is not None:
                report['single_ua_baseline'] = baseline
                policy = dict(policy, single_ua_source_images_per_second=baseline['source_images_per_second'])
                while True:
                    execution = execution_plan(selected, machine, base, service.record['enabled'])
                    report['selected_jobs'] = execution['jobs']; report['status'] = 'running'
                    write_json(args.output/'resolved_execution.json', execution)
                    write_json(args.output/'report.json', report)
                    leg = launch/f"jobs{execution['jobs']}"; leg.mkdir()
                    result = pool(args, execution, service.env, leg, policy)
                    report['legs'].append(result); report['models'].update(result['models'])
                    report['status'] = result['status']
                    if result['status'] in ('oom', 'needs_performance_review') and execution['jobs'] > 1:
                        selected = execution['jobs']//2
                        print('Saved complete states; reducing concurrency to', selected, flush=True)
                        continue
                    break
        good = report['previous_good_windows']+sum(w['target_met'] for leg in report['legs'] for w in leg['windows'])
        final = effective_progress(hints(args.output, args.expected_sha))
        report['ok'] = (report['status'] == 'complete' and all(s == 40000 for s in final.values())
                        and len(report['models']) == 4 and all(r['ok'] for r in report['models'].values()) and good > 0)
        if report['status'] == 'complete' and not report['ok']: report['status'] = 'needs_performance_review'
    except BaseException:
        report['status'] = 'failed'; report['error'] = traceback.format_exc()
    for variant, row in audit['donors'].items(): verify_donor(Path(audit['source_root']), variant, row)
    report['original_donors_unchanged'] = True
    report['committed_steps'] = effective_progress(hints(args.output, args.expected_sha))
    report['formal_optimizer_updates_total'] = sum(report['committed_steps'].values())
    report['formal_optimizer_updates_this_invocation'] = sum(report['committed_steps'].values())-sum(initial.values())
    report['seconds'] = time.monotonic()-started
    report['committed_updates_per_second_including_startup_and_io'] = report['formal_optimizer_updates_this_invocation']/max(report['seconds'], 1e-9)
    report['exit_code'] = 0 if report['ok'] else 130 if report['status'] == 'interrupted' else 2
    write_json(args.output/'report.json', report)
    print(json.dumps({k: report[k] for k in ('ok', 'status', 'selected_jobs', 'committed_steps') if k in report}), flush=True)
    return report['exit_code']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--expected-sha', required=True)
    p.add_argument('--jobs', choices=('auto', '1', '2', '4'), default='auto')
    p.add_argument('--mps', choices=('auto', 'off'), default='auto')
    p.add_argument('--worker', action='store_true'); p.add_argument('--variant', choices=VARIANTS)
    p.add_argument('--parent-pid', type=int)
    p.add_argument('--status', action='store_true'); p.add_argument('--package', action='store_true')
    p.add_argument('--archive', type=Path)
    p.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets'))
    p.add_argument('--pretrained-root', type=Path, default=Path('/root/autodl-tmp/pretrained'))
    args = p.parse_args()
    if sys.platform != 'linux' or git_sha() != args.expected_sha: p.error('AutoDL exact SHA required')
    if Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()).resolve() != ROOT.resolve():
        p.error('Run from this exact checkout')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        p.error('Tracked code changed')
    allowed = BASE/f'phase42_vgpu48_d3_l1_{args.expected_sha[:7]}'
    if args.output.absolute() != allowed: p.error('Use the SHA-specific48GB continuation output')
    safe(BASE, allowed)
    if args.status:
        status(allowed, args.expected_sha)
        path = allowed/'parallel_progress.json'
        if path.exists(): print(path.read_text())
        return 0
    import fcntl
    if args.package:
        if args.archive is None: p.error('--archive required')
        with (allowed/'run.lock').open('r+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); package(allowed, args.archive)
        return 0
    contract()
    args.policy_path = POLICY; args.worker_module = MODULE
    if args.worker:
        if args.variant is None or args.parent_pid != os.getppid(): p.error('Coordinator-owned worker only')
        execution = json.loads((allowed/'resolved_execution.json').read_text()); validate_execution(execution)
        return worker(args, execution=execution)
    allowed.mkdir(parents=True, exist_ok=True)
    audit, _ = accepted()
    with ExitStack() as stack:
        handles = []
        for path in (BASE/'gpu0.lock', allowed/'run.lock', Path(audit['source_root'])/'run.lock'):
            handle = stack.enter_context(safe(BASE, path).open('a+')); handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.lock_fds = tuple(h.fileno() for h in handles)
        return run(args)


if __name__ == '__main__': raise SystemExit(main())
