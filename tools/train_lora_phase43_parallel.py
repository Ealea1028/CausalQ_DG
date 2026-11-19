"""Phase43: four fresh independent DINOv2 LoRA models, source training only."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from causalq.runtime.identity import file_sha, write_json
from causalq.runtime.lora_l1_formal import ROOT, DIRECTORY, VARIANTS
from causalq.runtime.lora_parallel import MPS, hardware, resource_plan
from causalq.runtime.lora_phase43 import (PHASE, NAME, CONFIG, FREEZE, POLICY, completion,
                                         check_runtime, load_contract, reference_root, verify_references)
from tools.retire_lora_l0 import safe
from tools.train_lora_core_d3_l1 import (BASE, git_sha, hints, package, require_retirement,
                                         runtime, status, worker)
from tools.train_lora_phase42_parallel import pool
from tools.resume_lora_phase42_vgpu48 import baseline_record, load_baseline

MODULE = 'tools.train_lora_phase43_parallel'


def execution_plan(jobs, machine, base, mps, *, calibration=False):
    execution = resource_plan(jobs, machine, base)
    execution.update(phase=PHASE, backbone=NAME, config_path=CONFIG,
                     freeze_path=FREEZE.relative_to(ROOT).as_posix(), policy_sha256=file_sha(POLICY),
                     original_donor_migration=False, mps_enabled=mps, baseline_calibration=calibration)
    if calibration: execution['variants'] = ['UA']
    return execution


def validate_execution(execution):
    if (execution['phase'] != PHASE or execution['backbone'] != NAME
            or execution['config_path'] != CONFIG or execution['freeze_path'] != FREEZE.relative_to(ROOT).as_posix()
            or execution['policy_sha256'] != file_sha(POLICY) or execution['original_donor_migration'] is not False
            or execution['jobs'] not in (1, 2, 4) or execution['profile']['microbatch_source'] != 4
            or execution['profile']['effective_source_batch'] != 8 or execution['profile']['buffer_slots'] != 2
            or execution.get('variants', list(VARIANTS)) not in (list(VARIANTS), ['UA'])):
        raise ValueError('Resolved Phase43 execution changed')


def progress(args):
    return hints(args.output, args.expected_sha, phase=PHASE, backbone=NAME)


def run(args):
    started = time.monotonic()
    load_contract(); accepted = completion(); require_retirement(); check_runtime(runtime())
    audit = json.loads((DIRECTORY/'phase42_vgpu48_audit_acceptance.json').read_text())
    import torch
    props = torch.cuda.get_device_properties(0)
    if ([props.major, props.minor] != audit['target_gpu']['capability']
            or props.multi_processor_count != audit['target_gpu']['sm_count']
            or not torch.cuda.is_bf16_supported()):
        raise ValueError('Accepted CUDA capability/SM/BF16 changed')
    verify_references()
    initial = progress(args)
    required = sum((40000-step+999)//1000 for step in initial.values())*(50 << 20)+(3 << 30)
    if shutil.disk_usage(args.output).free < required: raise RuntimeError('Insufficient full-history checkpoint budget')
    old_path = args.output/'report.json'
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    if old and (old['git_sha'] != args.expected_sha or old['policy_sha256'] != file_sha(POLICY)):
        raise ValueError('Existing Phase43 report identity changed')
    selected = old.get('selected_jobs', 4) if args.jobs == 'auto' else int(args.jobs)
    report = {'phase': PHASE, 'backbone': NAME, 'git_sha': args.expected_sha,
              'policy_sha256': file_sha(POLICY), 'mode': 'fresh_d2_parallel_lora', 'ok': False, 'status': 'starting',
              'initial_committed_steps': initial, 'expected_formal_optimizer_updates': 160000,
              'initialization': 'fresh_common_seed0_head_and_zero_delta_lora',
              'trained_l0_or_d3_initialization': False, 'phase42_producer': accepted['producer'],
              'phase42_acceptance_sha256': file_sha(DIRECTORY/'phase42_completion_acceptance.json'),
              'target_data_accessed': False, 'source_miou': None, 'target_miou': None,
              'source_diagnostics_deferred_to_phase45': True, 'checkpoint_budget_bytes': required,
              'models': {}, 'legs': [], 'previous_good_windows': old.get('previous_good_windows', 0)+sum(
                  w['target_met'] for leg in old.get('legs', []) for w in leg['windows'])}
    launch = args.output/'cohort_attempts'/uuid.uuid4().hex; launch.mkdir(parents=True)
    machine = hardware(); base = json.loads((DIRECTORY/'execution_profile.json').read_text())
    policy = json.loads(POLICY.read_text()); report['hardware'] = machine
    write_json(args.output/'report.json', report)
    try:
        with MPS(args.output, args.mps) as service:
            report['mps'] = service.record
            baseline_path = args.output/'single_ua_baseline.json'
            baseline = load_baseline(baseline_path, args, service.record['enabled'])
            if baseline is None:
                execution = execution_plan(1, machine, base, service.record['enabled'], calibration=True)
                write_json(args.output/'resolved_execution.json', execution)
                leg = launch/'baseline_d2_ua'; leg.mkdir()
                print('Measuring DINOv2 UA on this GPU using retained formal updates', flush=True)
                result = pool(args, execution, service.env, leg, dict(policy, single_ua_source_images_per_second=0.))
                report['legs'].append(result); report['models'].update(result['models']); report['status'] = result['status']
                if result['status'] == 'baseline_complete':
                    baseline = baseline_record(args, result, service.record['enabled'])
                    write_json(baseline_path, baseline)
            if baseline is not None:
                report['single_ua_baseline'] = baseline
                policy = dict(policy, single_ua_source_images_per_second=baseline['source_images_per_second'])
                while True:
                    execution = execution_plan(selected, machine, base, service.record['enabled'])
                    report['selected_jobs'] = execution['jobs']; report['status'] = 'running'
                    write_json(args.output/'resolved_execution.json', execution); write_json(args.output/'report.json', report)
                    leg = launch/f"jobs{execution['jobs']}"; leg.mkdir()
                    result = pool(args, execution, service.env, leg, policy)
                    report['legs'].append(result); report['models'].update(result['models']); report['status'] = result['status']
                    if result['status'] in ('oom', 'needs_performance_review') and execution['jobs'] > 1:
                        selected = execution['jobs']//2
                        print('Saved complete D2 states; reducing concurrency to', selected, flush=True)
                        continue
                    break
        good = report['previous_good_windows']+sum(w['target_met'] for leg in report['legs'] for w in leg['windows'])
        final = progress(args)
        models = list(report['models'].values())
        common_heads = {m.get('identity', {}).get('initial_head_sha256') for m in models}
        common_adapters = {m.get('identity', {}).get('initial_adapter_sha256') for m in models}
        report['common_initialization'] = len(common_heads) == len(common_adapters) == 1 and None not in common_heads | common_adapters
        report['ok'] = (report['status'] == 'complete' and final == dict.fromkeys(VARIANTS, 40000)
                        and len(models) == 4 and all(m['ok'] for m in models) and good > 0 and report['common_initialization'])
        if report['status'] == 'complete' and not report['ok']: report['status'] = 'needs_performance_review'
    except BaseException:
        report['status'] = 'failed'; report['error'] = traceback.format_exc()
    try:
        verify_references(); report['phase42_finals_unchanged'] = True
        final = progress(args)
    except Exception:
        report['ok'] = False; report['status'] = 'failed'; report['reference_or_inventory_error'] = traceback.format_exc()
        final = initial
    report['committed_steps'] = final
    report['formal_optimizer_updates_total'] = sum(final.values())
    report['formal_optimizer_updates_this_invocation'] = sum(final.values())-sum(initial.values())
    report['seconds'] = time.monotonic()-started
    report['committed_updates_per_second_including_startup_and_io'] = report['formal_optimizer_updates_this_invocation']/max(report['seconds'], 1e-9)
    report['exit_code'] = 0 if report['ok'] else 130 if report['status'] == 'interrupted' else 2
    write_json(args.output/'report.json', report)
    print(json.dumps({k: report[k] for k in ('ok', 'status', 'selected_jobs', 'committed_steps') if k in report}), flush=True)
    return report['exit_code']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--expected-sha', required=True)
    p.add_argument('--jobs', choices=('auto', '1', '2', '4'), default='auto'); p.add_argument('--mps', choices=('auto', 'off'), default='auto')
    p.add_argument('--worker', action='store_true'); p.add_argument('--variant', choices=VARIANTS); p.add_argument('--parent-pid', type=int)
    p.add_argument('--status', action='store_true'); p.add_argument('--package', action='store_true'); p.add_argument('--archive', type=Path)
    p.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets'))
    p.add_argument('--pretrained-root', type=Path, default=Path('/root/autodl-tmp/pretrained'))
    args = p.parse_args()
    if sys.platform != 'linux' or git_sha() != args.expected_sha: p.error('AutoDL exact code SHA required')
    if Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()).resolve() != ROOT.resolve():
        p.error('Run from the selected checkout')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip(): p.error('Tracked code changed')
    allowed = BASE/f'phase43_parallel_d2_l1_{args.expected_sha[:7]}'
    if args.output.absolute() != allowed: p.error('Use the SHA-specific Phase43 output')
    safe(BASE, allowed)
    if args.status:
        status(allowed, args.expected_sha, phase=PHASE, backbone=NAME)
        path = allowed/'parallel_progress.json'
        if path.exists(): print(path.read_text())
        return 0
    import fcntl
    if args.package:
        if args.archive is None: p.error('--archive required')
        with (allowed/'run.lock').open('r+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); package(allowed, args.archive)
        return 0
    load_contract()
    args.phase = PHASE; args.backbone_name = NAME; args.policy_path = POLICY; args.worker_module = MODULE
    if args.worker:
        if args.variant is None or args.parent_pid != os.getppid(): p.error('Coordinator-owned worker only')
        execution = json.loads((allowed/'resolved_execution.json').read_text()); validate_execution(execution)
        return worker(args, execution=execution)
    allowed.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handles = []
        for path in (BASE/'gpu0.lock', allowed/'run.lock', reference_root()/'run.lock'):
            handle = stack.enter_context(safe(BASE, path).open('a+')); handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.lock_fds = tuple(h.fileno() for h in handles)
        return run(args)


if __name__ == '__main__': raise SystemExit(main())
