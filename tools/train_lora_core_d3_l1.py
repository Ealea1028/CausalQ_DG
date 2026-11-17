"""Phase42: four sequential independent DINOv3 LoRA models, source training only."""
import argparse
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import uuid

import torch

from causalq.datasets.segmentation import TrainTransform
from causalq.datasets.strict_source import build_source_datasets, SourceEvalTransform, MANIFEST_HASHES
from causalq.runtime.hpi_packets import packet_loader
from causalq.runtime.hpi_prepare import configure, Telemetry
from causalq.runtime.identity import file_sha, tree_digest, write_json
from causalq.runtime.input_stream import CanonicalInputStream
from causalq.runtime.lora_formal import build_backbone
from causalq.runtime.lora_l1_formal import (ROOT, DIRECTORY, NAME, VARIANTS, L1Trainer,
                                          checkpoint_rows, core_l1_head, load_contract)
from causalq.runtime.lora_performance import InputPrefetch, telemetry
from causalq.runtime.lora_training import compact_state, seed_training_rng
from tools.check_lora_backbones import frozen_digest
from tools.retire_lora_l0 import RUN as RETIRED_RUN, POLICY, safe
from tools.train_lora_core_l0 import runtime

BASE = RETIRED_RUN.parent


def git_sha():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def identity_for(trainer, args, environment):
    return {'study': 'lora_causalq_1114', 'phase': 42, 'backbone': NAME, 'variant': args.variant,
            'lora_enabled': True, 'lora': trainer.model.backbone.lora_manifest,
            'centered': args.variant in ('C0', 'CA'), 'anchor_cqe': args.variant in ('UA', 'CA'),
            'training_producer_git_sha': args.expected_sha,
            'freeze_sha256': file_sha(DIRECTORY/'phase42_freeze.json'),
            'config_sha256': file_sha(ROOT/'configs/lora_causalq_1114/phase42_core_d3_l1.json'),
            'execution_profile_sha256': file_sha(DIRECTORY/'execution_profile.json'),
            'phase39_report_sha256': file_sha(DIRECTORY/'phase39_report.json'),
            'source_manifests': MANIFEST_HASHES, 'source_stream': 'hpi_step_sha256_v1',
            'data_root': str(args.data_root.resolve()), 'pretrained_root': str(args.pretrained_root.resolve()),
            'runtime': environment, 'seed': 0, 'effective_source_batch': 8, 'microbatch': 4, 'horizon': 40000,
            'initial_head_sha256': trainer.initial_head_sha256,
            'initial_adapter_sha256': trainer.initial_adapter_sha256,
            'rng_initialization': 'python_numpy_torch_seed0_v1',
            'optimizer_recipe': 'hpi_40k_poly09_warm1500_ratio1e-6_wd1e-5_norm0_v1'}


def feature_record(features):
    """Keep shapes only, never retain an adapted tensor/graph in a log or cache."""
    patch, cls = features.patch_tokens, features.cls_token
    if (patch.ndim != 3 or patch.shape[1:] != (1024, 1024) or cls is None
            or tuple(cls.shape) != (patch.shape[0], 1024) or len(features.intermediate_maps) != 4
            or not patch.requires_grad):
        raise ValueError('Unexpected D3 training Patch/CLS/intermediate features')
    return {'patch_tokens': list(patch.shape), 'cls_token': list(cls.shape),
            'register_tokens': None if features.register_tokens is None else list(features.register_tokens.shape),
            'intermediate_maps': [list(t.shape) for t in features.intermediate_maps],
            'patch_requires_grad': patch.requires_grad, 'dtype': str(patch.dtype)}


def adapter_gradients(model):
    pairs = [(name, p) for name, p in model.backbone.named_parameters() if p.requires_grad]
    if any(p.grad is None for _, p in pairs):
        raise ValueError('A LoRA parameter is disconnected')
    values = torch.stack([p.grad.detach().float().norm() for _, p in pairs]).cpu().tolist()
    if not all(__import__('math').isfinite(v) for v in values):
        raise ValueError('Nonfinite LoRA gradient')
    return dict(zip((name for name, _ in pairs), values))


def worker(args, *, execution=None):
    config, _ = load_contract()
    parallel = execution is not None
    if parallel: config = json.loads((ROOT/execution['config_path']).read_text())
    profile = execution['profile'] if parallel else json.loads((DIRECTORY/'execution_profile.json').read_text())
    folder = args.output/args.variant
    folder.mkdir(parents=True, exist_ok=True)
    attempt = folder/'attempts'/uuid.uuid4().hex
    attempt.mkdir(parents=True)
    report = {'study': 'lora_causalq_1114', 'phase': 42, 'git_sha': args.expected_sha,
              'variant': args.variant, 'backbone': NAME, 'ok': False, 'status': 'starting',
              'lora_enabled': True, 'target_data_accessed': False, 'source_miou': None, 'target_miou': None,
              'source_diagnostics_deferred_to_phase45': True, 'attempt_directory': str(attempt),
              'formal_optimizer_updates_this_invocation': 0, 'checkpoint_retention': 'all segments; conservative full-history budget'}
    trainer = None; trace = None; iterator = None; hook = None
    initial = 0; record = None; windows = []; history = []; checkpoint_seconds = 0.
    started = time.monotonic(); stop = {'requested': False}
    def request_stop(signum, frame): stop['requested'] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, 'SIGHUP'): signal.signal(signal.SIGHUP, request_stop)
    try:
        configure(); seed_training_rng(0)
        if parallel: torch.set_num_threads(1)
        environment = runtime()
        accepted = json.loads((DIRECTORY/'phase39_report.json').read_text())['environment']
        for key in ('python', 'torch', 'cuda', 'transformers', 'gpu', 'visible_memory_bytes'):
            if environment[key] != accepted[key]: raise ValueError('Accepted runtime changed: '+key)
        original_runtime = json.loads((DIRECTORY/'phase41_stop_acceptance.json').read_text())['latest'][NAME]['U0']['identity']['runtime']
        if environment != original_runtime:
            raise ValueError('Full recorded runtime changed: '+repr([k for k in environment if environment[k] != original_runtime.get(k)]))
        report['runtime'] = environment
        train, _ = build_source_datasets(args.data_root/'gta5', TrainTransform((512, 512)), SourceEvalTransform())
        backbone = build_backbone(NAME, args.pretrained_root)
        model = core_l1_head(backbone, args.variant)
        if backbone.lora_manifest['trainable_parameters'] != 98304:
            raise ValueError('Expected exactly98304 LoRA parameters')
        trainer = L1Trainer(model, args.variant)
        identity = identity_for(trainer, args, environment); report['identity'] = identity
        if parallel:
            identity['config_sha256'] = file_sha(ROOT/execution['config_path'])
            identity['freeze_sha256'] = file_sha(DIRECTORY/'phase42_parallel_freeze.json')
            identity['parallel_policy_sha256'] = execution['policy_sha256']
            identity['performance_scope'] = 'whole_gpu_cohort'
            report['resolved_execution'] = execution
        previous = folder/'training_report.json'
        if previous.exists():
            old = json.loads(previous.read_text())
            if old.get('identity') != identity: raise ValueError('Existing model/run identity changed')
            if old.get('recovery_blocked'): raise ValueError('Return the failed invariant check for local review')
            if old.get('status') == 'needs_performance_review': raise ValueError('Return existing performance stop before retry')
            history = old.get('steady_window_history', []) + old.get('steady_windows', [])
        initial = trainer.restore(folder/'checkpoints', identity)
        report['restored_step'] = initial
        if initial: record = trainer.commit(folder/'checkpoints', identity)
        write_json(folder/'resolved_config.json', {'config': config, 'identity': identity, 'execution_profile': profile})
        before = frozen_digest(backbone); report['base_digest_before'] = before
        def observe(module, inputs, output):
            if 'first_features' not in report: report['first_features'] = feature_record(output)
        hook = backbone.register_forward_hook(observe)
        recent = []; ticks = 0; window_start = None; wait_start = 0.; pauses = []
        if initial < 40000 and not stop['requested']:
            stream = CanonicalInputStream(train.pairs, start=initial*8, stop=40000*8)
            loader, policy = packet_loader(stream, profile); report['loader_policy'] = policy
            pipeline = InputPrefetch(loader, 'cuda', profile['buffer_slots']); iterator = iter(pipeline)
            trace = (attempt/'input_trace.jsonl').open('w')
            from contextlib import nullcontext
            from types import SimpleNamespace
            monitoring = nullcontext(SimpleNamespace(started=time.monotonic())) if parallel else Telemetry(attempt/'telemetry.csv', 1)
            with monitoring as monitor:
                while trainer.step < 40000 and not stop['requested']:
                    if os.getppid() != args.parent_pid:
                        stop['requested'] = True; break
                    data = next(iterator)
                    samples = [{'views': data['views'][j], 'label': data['labels'][j], 'metadata': meta}
                               for j, meta in enumerate(data['metadata'])]
                    values = trainer.advance(samples, amp=True); ticks += 1
                    trace.write(json.dumps({'step_zero_based': trainer.step-1, 'first_source_position': (trainer.step-1)*8,
                        'ids': [m['id'] for m in data['metadata']], 'canonical_metadata_sha256':
                        hashlib.sha256(json.dumps(data['metadata'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()},
                        separators=(',', ':'))+'\n')
                    recent.append(values.detach())
                    samples.clear(); del samples, data, values
                    if parallel:
                        write_json(folder/'execution_progress.json', {'step': trainer.step,
                            'pid': os.getpid(),
                            'initial_step': initial, 'ready': ticks >= 20, 'monotonic': time.monotonic(),
                            'data_wait_seconds': pipeline.total_data_wait_seconds,
                            'checkpoint_seconds': checkpoint_seconds, 'attempt_directory': str(attempt)})
                    if ticks in (1, 20):
                        grads = adapter_gradients(model)
                        report[f'gradient_probe_tick_{ticks}'] = grads
                        if not any(v > 0 for n, v in grads.items() if n.endswith('.B')):
                            raise ValueError('LoRA B has no learning signal')
                        if ticks == 20 and not any(v > 0 for n, v in grads.items() if n.endswith('.A')):
                            raise ValueError('LoRA A has no learning signal after20 formal updates')
                    if ticks == 20:
                        torch.cuda.synchronize(); window_start = time.monotonic()-monitor.started
                        wait_start = pipeline.total_data_wait_seconds
                    if ticks % 50 == 0:
                        summary = torch.stack(recent).float().mean(0).cpu().tolist(); recent.clear()
                        write_json(folder/'progress.json', {'phase': 42, 'variant': args.variant, 'status': 'running',
                            'git_sha': args.expected_sha, 'step_in_memory': trainer.step,
                            'committed_step': record['step'] if record else 0,
                            'loss_and_gradient_norm_mean': summary, 'attempt_directory': str(attempt),
                            'seconds': time.monotonic()-started})
                        print(NAME, args.variant, 'step', trainer.step, 'committed', record['step'] if record else 0, flush=True)
                    end = time.monotonic()-monitor.started
                    active = end-window_start-sum(b-a for a, b in pauses) if window_start is not None else 0.
                    if not parallel and window_start is not None and active >= profile['steady_seconds']:
                        util = telemetry(attempt/'telemetry.csv', start=window_start, end=end, exclude_intervals=pauses)
                        wait = (pipeline.total_data_wait_seconds-wait_start)/active
                        reserved = torch.cuda.max_memory_reserved()/environment['visible_memory_bytes']
                        met = (util['samples'] >= profile['telemetry_samples_min'] and util['median'] >= 70
                               and util['p10'] >= 60 and wait < .1 and reserved <= .85)
                        windows.append({'attempt_directory': str(attempt), 'start': window_start, 'end': end,
                                        'active_seconds': active, 'summary': util, 'data_wait_fraction': wait,
                                        'reserved_fraction': reserved, 'checkpoint_exclusions': pauses.copy(), 'target_met': met})
                        write_json(attempt/'steady_windows.json', windows)
                        if not met:
                            report['status'] = 'needs_performance_review'; break
                        window_start = end; wait_start = pipeline.total_data_wait_seconds; pauses = []
                    if trainer.step % 1000 == 0:
                        trace.flush()
                        pause_start = time.monotonic()-monitor.started
                        if shutil.disk_usage(folder).free < (512 << 20): raise RuntimeError('Insufficient checkpoint staging space')
                        record = trainer.commit(folder/'checkpoints', identity)
                        pause_end = time.monotonic()-monitor.started; checkpoint_seconds += pause_end-pause_start
                        if window_start is not None: pauses.append((pause_start, pause_end))
                report['peak_live_input_slots'] = pipeline.peak_live_slots
                report['data_wait_seconds'] = pipeline.total_data_wait_seconds
            iterator.close(); iterator = None
        record = trainer.commit(folder/'checkpoints', identity) or record
        after = frozen_digest(backbone); report['base_digest_after'] = after
        if after != before:
            report['recovery_blocked'] = True
            raise ValueError('Original DINO base hash changed; do not reuse this trajectory')
        adapters = {k: v for k, v in compact_state(model).items() if k.startswith('backbone.')}
        report['adapter_sha256'] = tree_digest(adapters)
        report['adapter_changed'] = report['adapter_sha256'] != trainer.initial_adapter_sha256
        report['base_unchanged'] = True
        report['performance_target_met'] = None if parallel else bool(history+windows) and all(w['target_met'] for w in history+windows)
        report['performance_scope'] = 'whole_gpu_cohort' if parallel else 'single_model'
        report['ok'] = (trainer.step == 40000 and record is not None and record['step'] == 40000
                        and report['adapter_changed'] and (parallel or report['performance_target_met']))
        report['status'] = 'complete' if report['ok'] else report.get('status', 'incomplete')
        if stop['requested'] and not report['ok']: report['status'] = 'interrupted'
        if report['status'] == 'starting': report['status'] = 'incomplete'
    except BaseException:
        report['error'] = traceback.format_exc(); report['status'] = 'failed'
        if trainer is not None and 'identity' in report:
            try: record = trainer.commit(folder/'checkpoints', report['identity']) or record
            except Exception: report['checkpoint_error'] = traceback.format_exc()
    finally:
        if iterator is not None:
            try: iterator.close()
            except Exception: report['input_close_error'] = traceback.format_exc()
        if hook is not None: hook.remove()
        if trace is not None:
            trace.close(); report['input_trace_sha256'] = file_sha(attempt/'input_trace.jsonl')
        report['checkpoint'] = record; report['steady_windows'] = windows; report['steady_window_history'] = history
        report['checkpoint_seconds'] = checkpoint_seconds; report['seconds'] = time.monotonic()-started
        report['step_in_memory'] = trainer.step if trainer is not None else 0
        report['committed_step'] = record['step'] if record else 0
        report['formal_optimizer_updates_this_invocation'] = max(0, report['committed_step']-initial)
        report['successful_update_image_views_lower_bound'] = max(0, report['step_in_memory']-initial)*16
        report['actual_backbone_image_views_this_invocation'] = (None if trainer is not None and trainer.tainted
                                                               else report['successful_update_image_views_lower_bound'])
        report['tainted_update'] = trainer.tainted if trainer is not None else False
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
        report['exit_code'] = 0 if report['ok'] else 130 if report['status'] == 'interrupted' else 2
        write_json(attempt/'training_report.json', report); write_json(folder/'training_report.json', report)
    return report['exit_code']


def hints(output, sha):
    cursors = {}
    for variant in VARIANTS:
        steps = []
        for path in (output/variant/'checkpoints').glob('*/complete.json'):
            marker = json.loads(path.read_text()); identity = marker['identity']; step = marker['step']
            if (identity.get('training_producer_git_sha') != sha or identity.get('phase') != 42
                    or identity.get('variant') != variant or identity.get('backbone') != NAME
                    or identity.get('lora_enabled') is not True or type(step) is not int or not 0 < step <= 40000):
                raise ValueError('Unexpected existing Phase42 checkpoint')
            steps.append(step)
        cursors[variant] = max(steps, default=0)
    return cursors


def completed(output, variant, args):
    path = output/variant/'training_report.json'
    if not path.exists(): return None
    report = json.loads(path.read_text())
    if report.get('recovery_blocked'): raise ValueError('Failed invariant requires local review')
    if report.get('status') == 'needs_performance_review': raise ValueError('Return performance stop before retry')
    if not report.get('ok'): return None
    identity = report['identity']
    if (report['git_sha'] != args.expected_sha or identity['training_producer_git_sha'] != args.expected_sha
            or identity.get('phase') != 42 or identity.get('variant') != variant
            or identity.get('backbone') != NAME or identity.get('lora_enabled') is not True
            or identity['data_root'] != str(args.data_root.resolve())
            or identity['pretrained_root'] != str(args.pretrained_root.resolve())
            or identity['freeze_sha256'] != file_sha(DIRECTORY/'phase42_freeze.json')
            or not report['base_unchanged'] or not report['adapter_changed'] or not report['performance_target_met']):
        raise ValueError('Completed model identity/acceptance mismatch')
    rows = checkpoint_rows(output/variant/'checkpoints', identity)
    if not rows or rows[0][0] != 40000: raise ValueError('Completed final missing')
    for _, folder, marker in rows[:2]:
        if file_sha(folder/'state.pt') != marker['sha256']: raise ValueError('Completed payload hash changed')
    if rows[0][2]['sha256'] != report['checkpoint']['sha256']: raise ValueError('Final report/marker mismatch')
    return report


def require_retirement():
    freeze = json.loads((DIRECTORY/'phase41_retirement_apply_freeze.json').read_text())
    cancellation = safe(BASE, RETIRED_RUN/'L0_RETIRED.json')
    if json.loads(cancellation.read_text()).get('policy') != POLICY: raise ValueError('L0 retirement not applied')
    for row in freeze['protected']:
        if file_sha(safe(BASE, RETIRED_RUN/row['path'])) != row['sha256']: raise ValueError('Retained L0 reference changed')
    if any(safe(BASE, RETIRED_RUN/row['path']).exists() for row in freeze['candidates']):
        raise ValueError('Approved L0 retirement incomplete')


def run(args):
    load_contract(); require_retirement()
    initial = hints(args.output, args.expected_sha)
    required = sum((40000-s+999)//1000 for s in initial.values())*(50 << 20)+(2 << 30)
    if shutil.disk_usage(args.output).free < required: raise RuntimeError(f'Need {required} bytes free; all L1 segments retained')
    report = {'phase': 42, 'git_sha': args.expected_sha, 'ok': False, 'status': 'running', 'models': {},
              'target_data_accessed': False, 'initial_committed_steps': initial,
              'checkpoint_budget_bytes': required, 'expected_formal_optimizer_updates': 160000}
    stop = {'at': None}; child = None
    def request_stop(signum, frame):
        if stop['at'] is None: stop['at'] = time.monotonic()
        if child is not None and child.poll() is None:
            try: child.send_signal(signal.SIGTERM)
            except ProcessLookupError: pass
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, 'SIGHUP'): signal.signal(signal.SIGHUP, request_stop)
    write_json(args.output/'report.json', report)
    try:
        for variant in VARIANTS:
            if stop['at'] is not None: break
            old = completed(args.output, variant, args)
            if old is not None:
                report['models'][variant] = old; continue
            report['active_variant'] = variant
            write_json(args.output/'report.json', report)
            print('Starting independent LoRA model:', variant, flush=True)
            folder = args.output/variant; folder.mkdir(exist_ok=True)
            previous = folder/'training_report.json'
            previous_stamp = previous.stat().st_mtime_ns if previous.exists() else -1
            command = [sys.executable, '-u', '-m', 'tools.train_lora_core_d3_l1', '--worker', '--variant', variant,
                       '--parent-pid', str(os.getpid()), '--output', str(args.output), '--expected-sha', args.expected_sha,
                       '--data-root', str(args.data_root), '--pretrained-root', str(args.pretrained_root)]
            with (folder/'worker.log').open('a') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True,
                                         pass_fds=getattr(args, 'lock_fds', ()))
                last_progress = None; last_poll = 0.
                while child.poll() is None:
                    if stop['at'] is not None and time.monotonic()-stop['at'] > 60:
                        try: os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                    if time.monotonic()-last_poll > 10:
                        progress = folder/'progress.json'
                        if progress.exists():
                            value = json.loads(progress.read_text())
                            if value.get('step_in_memory') != last_progress:
                                last_progress = value.get('step_in_memory')
                                print(variant, 'step', last_progress, 'committed', value.get('committed_step'), flush=True)
                        last_poll = time.monotonic()
                    time.sleep(.25)
                result = child.returncode; child = None
            if not previous.exists() or previous.stat().st_mtime_ns == previous_stamp:
                raise RuntimeError(f'Worker exited {result}; return {folder}/worker.log')
            current = json.loads(previous.read_text()); report['models'][variant] = current
            write_json(args.output/'report.json', report)
            if result != 0 or not current['ok']:
                report['status'] = current['status']; break
        report['ok'] = len(report['models']) == 4 and all(r['ok'] for r in report['models'].values())
        if report['ok']: report['status'] = 'complete'
        elif stop['at'] is not None: report['status'] = 'interrupted'
    except BaseException:
        report['error'] = traceback.format_exc(); report['status'] = 'failed'
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=60)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
    finally:
        try: final = hints(args.output, args.expected_sha)
        except Exception:
            final = initial.copy(); report['inventory_error'] = traceback.format_exc()
            report['ok'] = False; report['status'] = 'failed'
        report['committed_steps'] = final
        report['formal_optimizer_updates_total'] = sum(final.values())
        report['formal_optimizer_updates_this_invocation'] = sum(final.values())-sum(initial.values())
        report['exit_code'] = 0 if report['ok'] else 130 if report['status'] == 'interrupted' else 2
        write_json(args.output/'report.json', report)
        print(json.dumps({'ok': report['ok'], 'status': report['status'], 'committed_steps': final}), flush=True)
    return report['exit_code']


def package(output, archive):
    """Only small evidence; checkpoint tensors and complete raw traces stay on AutoDL."""
    archive = safe(output.parent, archive)
    if archive == output or output in archive.parents or archive.exists(): raise ValueError('Fresh sibling archive required')
    manifest = {}
    with archive.open('xb') as destination, tarfile.open(fileobj=destination, mode='w:gz') as bundle:
        for path in sorted(output.rglob('*')):
            safe(output, path)
            if not path.is_file(): continue
            rel = path.relative_to(output).as_posix()
            if path.suffix in ('.json', '.csv', '.txt'):
                manifest[rel] = {'sha256': file_sha(path), 'bytes': path.stat().st_size, 'included': True}
                bundle.add(path, arcname=rel, recursive=False)
            elif path.suffix in ('.jsonl', '.log'):
                manifest[rel] = {'sha256': file_sha(path), 'bytes': path.stat().st_size, 'included': False}
                if path.suffix == '.log':
                    with path.open('rb') as source:
                        source.seek(max(0, path.stat().st_size-65536)); tail = source.read()
                    info = tarfile.TarInfo(rel+'.tail'); info.size = len(tail); bundle.addfile(info, io.BytesIO(tail))
        data = (json.dumps(manifest, indent=2)+'\n').encode()
        info = tarfile.TarInfo('export_manifest.json'); info.size = len(data); bundle.addfile(info, io.BytesIO(data))
    print('Archive:', archive, 'SHA256:', file_sha(archive))


def status(output, sha):
    path = output/'report.json'
    report = json.loads(path.read_text()) if path.exists() else {'status': 'not_started'}
    result = {k: report[k] for k in ('status', 'ok', 'git_sha', 'active_variant',
                                    'formal_optimizer_updates_total', 'error') if k in report}
    result['committed_steps'] = hints(output, sha)
    result['models'] = {}
    for variant in VARIANTS:
        candidates = [p for p in (output/variant/'progress.json', output/variant/'training_report.json') if p.exists()]
        if not candidates: continue
        latest = json.loads(max(candidates, key=lambda p: p.stat().st_mtime_ns).read_text())
        result['models'][variant] = {k: latest[k] for k in ('status', 'step_in_memory', 'committed_step', 'error') if k in latest}
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--parent-pid', type=int)
    parser.add_argument('--status', action='store_true'); parser.add_argument('--package', action='store_true')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets'))
    parser.add_argument('--pretrained-root', type=Path, default=Path('/root/autodl-tmp/pretrained'))
    args = parser.parse_args()
    if sys.platform != 'linux': parser.error('Execute on AutoDL Linux')
    if git_sha() != args.expected_sha or Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()).resolve() != ROOT.resolve():
        parser.error('Exact code checkout required')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip(): parser.error('Tracked code modified')
    allowed = BASE/f'phase42_core_d3_l1_{args.expected_sha[:7]}'
    if args.output.absolute() != allowed: parser.error('Use this commit-specific Phase42 output')
    safe(BASE, allowed)
    if args.status:
        status(allowed, args.expected_sha); return 0
    import fcntl
    if args.package:
        if args.archive is None: parser.error('--archive required')
        with (allowed/'run.lock').open('r+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); package(allowed, args.archive)
        return 0
    if args.worker:
        if args.variant is None or args.parent_pid != os.getppid(): parser.error('Only the coordinator may launch a worker')
        return worker(args)
    allowed.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handles = []
        for path in (BASE/'gpu0.lock', allowed/'run.lock'):
            handle = stack.enter_context(safe(BASE, path).open('a+'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles.append(handle)
        args.lock_fds = tuple(handle.fileno() for handle in handles)
        return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
