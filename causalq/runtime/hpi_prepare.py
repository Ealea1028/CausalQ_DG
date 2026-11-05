"""AutoDL-only bounded Phase33 weight, CUDA, source and throughput preparation."""
import csv
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import torch
from causalq.datasets.segmentation import TrainTransform
from causalq.datasets.strict_source import SourceEvalTransform, build_source_datasets, source_manifest_metadata
from causalq.models.dinov2_wrapper import DINOv2Backbone, convert_hpi_state
from causalq.protocols.hpi_dinov2_20261031 import ROOT, STUDY, build_head, initialization_identity
from .cohort_runner import IndependentHead, baseline_backward, capture_rng, head_state
from .feature_pipeline import FeaturePipeline, slice_features, feature_tensors
from .identity import file_sha, tree_digest, write_json
from .input_stream import CanonicalInputStream, make_loader, loader_policy


def telemetry_summary(path):
    with Path(path).open() as stream:
        rows=list(csv.DictReader(stream))
    result={'samples':len(rows),'nvml_activity_is_not_tensor_core_utilization':True}
    for field in ('utilization_percent','memory_used_mib','power_w'):
        values=[]
        for row in rows:
            try: values.append(float(row[field]))
            except (ValueError,TypeError): pass
        values.sort()
        result[field]={'count':len(values),'mean':sum(values)/len(values) if values else None,
                       'min':min(values) if values else None,'max':max(values) if values else None,
                       'median':values[len(values)//2] if values else None}
    return result


def configure():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Phase33 prepare requires AutoDL CUDA; no CPU substitute')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported; report hardware instead of changing precision')
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)


class Telemetry:
    def __init__(self, path, interval=2):
        self.path, self.interval, self.stop = Path(path), interval, threading.Event()

    def __enter__(self):
        self.started = time.monotonic()
        def worker():
            with self.path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['elapsed_seconds', 'utilization_percent', 'memory_used_mib', 'power_w', 'status'])
                started = self.started
                while not self.stop.is_set():
                    try:
                        line = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
                            '--format=csv,noheader,nounits', '--id=0'], text=True, timeout=5).strip()
                        writer.writerow([time.monotonic()-started, *line.split(', '), 'ok'])
                    except Exception as exc:
                        writer.writerow([time.monotonic()-started, '', '', '', str(exc)])
                    stream.flush()
                    self.stop.wait(self.interval)
        self.thread = threading.Thread(target=worker, daemon=True); self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set(); self.thread.join(timeout=6)


def feature_error(reference, actual, profile):
    errors = []
    if len(feature_tensors(reference)) != len(feature_tensors(actual)):
        raise ValueError('Batched/compiled feature layer coverage differs')
    for a, b in zip(feature_tensors(reference), feature_tensors(actual)):
        a, b = a.float(), b.float()
        if a.shape != b.shape or not torch.isfinite(b).all():
            raise ValueError('Batched/compiled features have invalid shape or values')
        relative = float((a-b).norm() / a.norm().clamp_min(1e-6))
        if not torch.allclose(a, b, atol=profile['numerical_atol'], rtol=profile['numerical_rtol']) or relative > profile['numerical_relative_l2']:
            raise ValueError('Feature numerical equivalence exceeds frozen tolerance')
        errors.append({'max_abs': float((a-b).abs().max()), 'relative_l2': relative})
    return errors


def timed_forward(backbone, images, *, warmup=2, repeats=5):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(warmup):
            result = backbone(images)
            del result
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        first.record()
        for _ in range(repeats):
            result = backbone(images)
            del result
        last.record(); last.synchronize()
        elapsed = time.perf_counter() - start
    return {'gpu_ms_per_forward': first.elapsed_time(last)/repeats,
            'end_to_end_views_per_second': images.shape[0]*repeats/elapsed,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def compile_probe(weights, output, profile):
    configure()
    backbone = DINOv2Backbone.from_pretrained(weights, dtype=torch.bfloat16).cuda()
    x = torch.randn(profile['backbone_microbatch_views'], 3, 512, 512, device='cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        reference = slice_features(backbone(x[:1]), 0, 1)
    eager = timed_forward(backbone, x)
    started = time.monotonic()
    compiled = torch.compile(backbone, mode='default', dynamic=False, fullgraph=True)
    with torch.no_grad():
        actual = compiled(x)
    errors = feature_error(reference, slice_features(actual, 0, 1), profile)
    del actual
    compile_seconds = time.monotonic()-started
    measured = timed_forward(compiled, x)
    write_json(output, {'ok': True, 'compile_seconds': compile_seconds, 'eager': eager,
                        'compiled': measured, 'numerical_errors': errors,
                        'selected': measured['end_to_end_views_per_second'] > eager['end_to_end_views_per_second']})


def convert_weights(pretrained_root, out):
    source = Path(pretrained_root) / 'dinov2_vitl14/dinov2_vitl14_pretrain.pth'
    source_sha = file_sha(source)
    state = torch.load(source, map_location='cpu', weights_only=True)
    converted = convert_hpi_state(state)
    destination = source.with_name('dinov2_vitl14_hpi_p16.pth')
    identity = {'source_sha256': source_sha, 'converted_tensor_digest': tree_digest(converted),
                'recipe': 'hpi_clip_fp32_bicubic_align_corners_false_v1',
                'changed_keys': ['patch_embed.proj.weight', 'pos_embed'],
                'source_url': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'}
    if destination.exists():
        if tree_digest(torch.load(destination, map_location='cpu', weights_only=True)) != identity['converted_tensor_digest']:
            raise ValueError('Existing converted backbone differs; refusing overwrite')
    else:
        stage = destination.with_suffix('.pth.partial')
        torch.save(converted, stage)
        os.replace(stage, destination)
    identity['converted_file_sha256'] = file_sha(destination)
    write_json(out / 'weight_identity.json', identity)
    del state, converted
    gc.collect()
    return destination, identity


def prepare(out, profile, *, data_root, pretrained_root):
    configure()
    device = torch.cuda.get_device_properties(0)
    environment = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'gpu': device.name, 'compute_capability': list(torch.cuda.get_device_capability()),
        'visible_memory_bytes': device.total_memory, 'torch_arch_list': torch.cuda.get_arch_list(),
        'deterministic_algorithms': True, 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'pip_freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines(),
        'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True)}
    write_json(out / 'runtime.json', environment)
    train, val = build_source_datasets(Path(data_root) / 'gta5', TrainTransform((512, 512)), SourceEvalTransform())
    write_json(out / 'source_pairing.json', {**source_manifest_metadata(), 'paired_files': len(train)+len(val),
        'decoded_full_census': False, 'examples_in_holdout': all(s in {p.sample_id for p in val.pairs} for s in ('18734','12488','08744'))})
    weights, identity = convert_weights(pretrained_root, out)
    backbone = DINOv2Backbone.from_pretrained(weights, dtype=torch.bfloat16).cuda()
    frozen_versions = tuple(p._version for p in backbone.parameters())
    # Bound all real-source reads to 48 canonical crops for each worker candidate.
    worker_rows = []
    for workers in profile['worker_candidates']:
        candidate = {**profile, 'data_workers': workers}
        dataset = CanonicalInputStream(train.pairs, stop=48)
        loader = make_loader(dataset, candidate)
        started = time.perf_counter(); previous = started; waits = []; first = None
        for sample in loader:
            now = time.perf_counter(); waits.append(now-previous); previous = now
            if first is None:
                first = sample
        row = {'workers': workers, **loader_policy(candidate), 'samples': len(waits),
               'total_seconds': time.perf_counter()-started, 'cold_first_seconds': waits[0],
               'steady_seconds': sum(waits[1:]), 'steady_samples_per_second': (len(waits)-1)/max(sum(waits[1:]),1e-9),
               'first_tensor_sha256': tree_digest((first['views'],first['label'])), 'first_input': first['metadata']}
        worker_rows.append(row)
        del loader, dataset
    if len({r['first_tensor_sha256'] for r in worker_rows}) != 1:
        raise ValueError('Worker count changed canonical input')
    write_json(out / 'input_benchmark.json', worker_rows)
    best_worker = max(worker_rows, key=lambda r:r['steady_samples_per_second'])['workers']
    image = first['views'][:1].cuda().to(torch.bfloat16)
    with torch.no_grad():
        reference = backbone(image)
    single = timed_forward(backbone, image)
    candidates = []
    cap = int(device.total_memory * profile['prepare_backbone_memory_fraction'])
    with (out / 'throughput.jsonl').open('a') as log:
        for batch in profile['batch_candidates']:
            # Stop before likely pressure; no requirement to provoke OOM.
            if candidates and candidates[-1].get('ok') and candidates[-1]['peak_reserved_bytes'] * batch / candidates[-1]['batch'] > cap:
                break
            views = None
            try:
                views = image.expand(batch, -1, -1, -1).contiguous()
                with torch.no_grad():
                    actual = backbone(views)
                errors = feature_error(reference, slice_features(actual,0,1), profile)
                # Two windows, each includes four raw maps plus normalized tokens.
                feature_bytes = sum(x.numel()*x.element_size() for x in (*actual.intermediate_maps,actual.patch_tokens))
                del actual
                measured = timed_forward(backbone, views, warmup=profile['warmup_iterations'], repeats=profile['timed_iterations'])
                projected = measured['peak_reserved_bytes'] + 2*feature_bytes
                row = {'batch': batch, 'ok': projected <= cap, **measured, 'two_window_conservative_peak_bytes': projected,
                       'numerical_errors': errors}
                candidates.append(row)
                log.write(json.dumps(row)+'\n'); log.flush()
                print('backbone_benchmark', json.dumps(row), flush=True)
                if not row['ok']: break
            except torch.cuda.OutOfMemoryError as exc:
                row = {'batch':batch,'ok':False,'reason':str(exc)}
                candidates.append(row); log.write(json.dumps(row)+'\n'); log.flush()
                break
            finally:
                del views
                gc.collect(); torch.cuda.empty_cache()
    valid = [r for r in candidates if r['ok']]
    if not valid:
        raise RuntimeError('No permitted physical batch fits the frozen prepare memory budget')
    best = max(valid,key=lambda r:r['end_to_end_views_per_second'])
    actual_profile = {**profile, 'data_workers':best_worker,'backbone_microbatch_views':best['batch']}
    write_json(out / 'compile_profile.json', actual_profile)
    compile_result = {'selected':False, 'reason':'not attempted'}
    command = [sys.executable, '-m', 'tools.run_hpi_dinov2_20261031', '--compile-probe', str(weights),
               '--output', str(out / 'compile_result.json'), '--probe-profile', str(out / 'compile_profile.json')]
    # Child isolates unsupported compile/driver failures and enforces the wall limit.
    with (out / 'compile.log').open('w') as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=profile['compile_time_limit_seconds'], check=False)
            if result.returncode == 0:
                compile_result = json.loads((out / 'compile_result.json').read_text())
                if compile_result['compiled']['peak_reserved_bytes'] + 2*best['peak_allocated_bytes'] > cap:
                    compile_result.update(selected=False, reason='compiled path exceeds prepare memory headroom')
            else:
                compile_result = {'selected':False,'reason':f'compile child exit {result.returncode}; see compile.log'}
        except subprocess.TimeoutExpired:
            compile_result = {'selected':False,'reason':'compile wall limit180s exceeded; eager+SDPA retained'}
    actual_profile['compile_frozen_backbone'] = bool(compile_result['selected'])
    write_json(out / 'hardware_profile.json', {'measured':True,'execution':actual_profile,
        'single_image_eager':single, 'batch_candidates':candidates, 'worker_candidates':worker_rows,
        'compile':compile_result, 'visible_memory_bytes':device.total_memory, 'prepare_cap_bytes':cap,
        'hardware_selection_only':True,'target_data_accessed':False})
    # One source pair, independent identical heads; backward only, zero steps.
    u0, ua = build_head(backbone).cuda(), build_head(backbone).cuda()
    init = initialization_identity(u0)
    if tree_digest(head_state(u0)) != tree_digest(head_state(ua)):
        raise ValueError('U0/UA initialization is not paired')
    source_stream=CanonicalInputStream(train.pairs,stop=best['batch']//2)
    source_loader=make_loader(source_stream,actual_profile)
    pipeline=iter(FeaturePipeline(backbone,actual_profile)(source_loader))
    window=next(pipeline)
    first=window.samples[0]
    features=window.features
    original, photo = slice_features(features,0,1), slice_features(features,1,2)
    with torch.no_grad():
        single_original=backbone(first['views'][:1].cuda().to(torch.bfloat16))
        single_photo=backbone(first['views'][1:].cuda().to(torch.bfloat16))
    pair_errors={'original':feature_error(single_original,original,profile),
                 'photo':feature_error(single_photo,photo,profile)}
    del single_original,single_photo
    label = first['label'][None].cuda()
    head_rows = []
    head_peak_bytes = 0
    head_parameters = sum(v.numel() for v in head_state(u0).values())
    for name, model, anchor in [('U0',u0,False),('UA',ua,True)]:
        head = IndependentHead(name,model,None,capture_rng(),anchor_cqe=anchor)
        before = tree_digest(head_state(model))
        for alpha in (0., .1):
            model.zero_grad(set_to_none=True)
            with torch.no_grad(): model.query_head.alpha.fill_(alpha)
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
            loss = baseline_backward(head,original,photo,label,(512,512),amp=True)
            end_event.record();end_event.synchronize()
            head_peak_bytes = max(head_peak_bytes,torch.cuda.max_memory_allocated())
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not torch.isfinite(loss).all() or not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise ValueError('Nonfinite loss/gradient or missing trainable gradient')
            if alpha and not model.query_bank.class_anchors.grad.abs().sum():
                raise ValueError('Query path failed to receive gradient')
            head_rows.append({'variant':name,'diagnostic_alpha':alpha,'losses':loss.tolist(),
                              'forward_backward_ms':start_event.elapsed_time(end_event),
                              'gradient_l2':float(torch.stack([g.float().norm() for g in gradients]).norm())})
        with torch.no_grad(): model.query_head.alpha.zero_()
        model.zero_grad(set_to_none=True)
        if before != tree_digest(head_state(model)):
            raise ValueError('Backward-only prepare changed head parameters')
    window.timing_events[-1].synchronize()
    pipeline_timing={'data_wait_seconds':window.data_wait_seconds,
        'physical_views':features.patch_tokens.shape[0],
        'h2d_ms':window.timing_events[0].elapsed_time(window.timing_events[1]),
        'backbone_ms':window.timing_events[1].elapsed_time(window.timing_events[2]),
        'head_updates':0,'bounded_head_backward_only':True}
    pipeline.close()
    del features,original,photo,u0,ua,label,head,model,gradients,window,source_loader,pipeline
    gc.collect(); torch.cuda.empty_cache()
    # Synthetic native target geometry only: never open Cityscapes here.
    shape_rows=[]
    for batch in (1,2,4):
        synthetic = torch.zeros(batch,3,1024,2048,device='cuda',dtype=torch.bfloat16)
        try:
            measured=timed_forward(backbone,synthetic,warmup=0,repeats=1)
            shape_rows.append({'batch':batch,'synthetic_only':True,**measured})
            if measured['peak_reserved_bytes'] >= device.total_memory*.6: break
        except torch.cuda.OutOfMemoryError:
            shape_rows.append({'batch':batch,'synthetic_only':True,'fits':False})
            if batch <= 2:
                raise
            break
        finally:
            del synthetic
            gc.collect();torch.cuda.empty_cache()
    if tuple(p._version for p in backbone.parameters()) != frozen_versions or any(p.grad is not None for p in backbone.parameters()):
        raise ValueError('Frozen backbone changed or acquired gradients')
    # One close-time digest, never per parameter per step.
    write_json(out/'prepare_report.json', {'ok':True,'study':STUDY,'formal_optimizer_updates':0,
        'optimizer_steps':0,'target_data_accessed':False,'weight_identity':identity,'initialization':init,
        'real_source_single_vs_paired_feature_errors':pair_errors,
        'shared_feature_pipeline':pipeline_timing,
        'backward_checks':head_rows,'native_geometry_synthetic':shape_rows,'frozen_backbone_unchanged':True,
        'runtime_backbone_tensor_digest':tree_digest(backbone.state_dict()),
        'memory_accounting':'batch measurements include shared backbone/cache; head parameter bytes separate',
        'head_trainable_parameters':head_parameters,
        'per_head_parameter_bytes':4*head_parameters,'per_head_adamw_moment_bytes_when_initialized':8*head_parameters,
        'cohort_backward_peak_allocated_bytes':head_peak_bytes,
        'prepare_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'prepare_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
