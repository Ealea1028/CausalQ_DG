"""Measure batch-1 CUDA forwards using the historical Phase32 model implementation."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from causalq.protocols.diagnostic_20261020 import ROOT, VARIANTS, BASELINES, baseline_source_report
from causalq.protocols.diagnostic_final_source import final_source_report, final_config
from causalq.protocols.diagnostic_target import target_contract, verify_target_contract
from causalq.protocols.formal_source import formal_config
from tools.inspect_gta5_source_structure import _git
from tools.run_diagnostic_20261020 import load_model, write_json
from tools.strict_source_smoke import precision

INFERENCE_SHA = '40c73bcd6914604368c1006e6c3d7ff0e2ad2a0a'
SIZES = ((512, 1024), (1024, 2048))
WARMUP = 30
REPEATS = 3
FRAMES = 100


def latency_statistics(samples):
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError('Need positive finite latency samples')
    ordered = sorted(samples)
    mean = statistics.fmean(ordered)
    return {'frames': len(samples), 'mean_ms': mean, 'p50_ms': statistics.median(ordered),
            'p95_ms': ordered[math.ceil(.95 * len(ordered)) - 1], 'fps': 1000 / mean}


def measure(model, image):
    """One forward per sample, no pipeline or overlapping frames."""
    expected = (1, 19, *image.shape[-2:])
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(FRAMES)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(FRAMES)]
    device = str(image.device)
    with torch.inference_mode():
        with precision(device, True):
            output = model(image)
        if output.shape != expected or not torch.isfinite(output).all().item():
            raise ValueError('Benchmark forward returned invalid logits')
        del output
        # Materialize CUDA event handles before timing so their first-use cost
        # does not enter the wall-clock samples.
        for start, end in zip(starts, ends):
            start.record()
            end.record()
        for _ in range(WARMUP):
            with precision(device, True):
                model(image)
        torch.cuda.synchronize(image.device)

        repeats = []
        for _ in range(REPEATS):
            wall_ms = []
            for start, end in zip(starts, ends):
                before = time.perf_counter()
                start.record()
                with precision(device, True):
                    logits = model(image)
                end.record()
                end.synchronize()
                after = time.perf_counter()
                wall_ms.append((after-before)*1000)
                del logits
            gpu_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
            repeats.append({'gpu_ms': gpu_ms, 'wall_ms': wall_ms,
                            'gpu': latency_statistics(gpu_ms), 'wall': latency_statistics(wall_ms)})
    return {**{kind: latency_statistics([ms for repeat in repeats for ms in repeat[kind+'_ms']])
               for kind in ('gpu', 'wall')}, 'repeats': repeats}


def table(results):
    lines = ['| 模型 | 尺寸 H×W | GPU 均值 ms | GPU P50 ms | GPU P95 ms | GPU FPS | 同步墙钟均值 ms | 同步墙钟 FPS |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for size in SIZES:
        shape = '×'.join(map(str, size))
        for variant in VARIANTS:
            metrics = results[variant]['sizes'][shape];gpu = metrics['gpu'];wall = metrics['wall']
            lines.append(f'| {variant} | {shape} | {gpu["mean_ms"]:.3f} | {gpu["p50_ms"]:.3f} | '
                         f'{gpu["p95_ms"]:.3f} | {gpu["fps"]:.2f} | {wall["mean_ms"]:.3f} | {wall["fps"]:.2f} |')
    return '\n'.join(lines) + '\n'


def benchmark(args):
    if _git('rev-parse', 'HEAD') != INFERENCE_SHA or _git('status', '--porcelain'):
        raise ValueError('Run from the clean historical Phase32 inference checkout')
    if not torch.cuda.is_available():
        raise RuntimeError('The benchmark needs AutoDL CUDA')
    if torch.cuda.device_count() != 1 or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Expose exactly one GPU and use CUBLAS_WORKSPACE_CONFIG=:4096:8')
    digest = verify_target_contract()
    freeze = target_contract()
    benchmark_root = Path(__file__).resolve().parents[1]
    benchmark_sha = subprocess.check_output(['git', '-C', str(benchmark_root), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(benchmark_root), 'status', '--porcelain'], text=True).strip()
    if status or benchmark_sha != args.benchmark_sha:
        raise ValueError('Use the clean, pinned benchmark checkout')
    output = args.output_root.resolve()
    protected = (ROOT, benchmark_root, args.pretrained_root.resolve())
    if any(output.is_relative_to(path) for path in protected):
        raise ValueError('Write benchmark outputs outside code and weights')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'started.json').open('x', encoding='utf-8') as stream:
        json.dump({'created_utc': datetime.now(timezone.utc).isoformat(), 'benchmark_sha': benchmark_sha}, stream)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    images = {}
    generator = torch.Generator(device='cpu').manual_seed(20261120)
    for height, width in SIZES:
        images[height, width] = torch.randn((1, 3, height, width), generator=generator).to('cuda:0')
    results = {}
    for variant in VARIANTS:
        if variant in BASELINES:
            record = freeze['reused_models'][variant]
            source = baseline_source_report(variant)
            config = formal_config(BASELINES[variant])
        else:
            record = freeze['new_models'][variant]
            source = final_source_report(variant)
            config = final_config(variant)
        model, style, environment = load_model(args, variant, frozen=record, source=source, config=config)
        del style
        result = {'checkpoint_sha256': record['checkpoint_sha256'], 'training_sha': record['producer_git_sha'],
                  'source_miou': record['source_miou'], 'environment': environment, 'sizes': {}}
        for size, image in images.items():
            shape = '×'.join(map(str, size))
            measurement = measure(model, image)
            result['sizes'][shape] = measurement
            print(f'{variant} {shape} GPU {measurement["gpu"]["mean_ms"]:.3f} ms / '
                  f'{measurement["gpu"]["fps"]:.2f} FPS', flush=True)
        write_json(output/(variant+'.json'), result)
        results[variant] = result
        del model
        torch.cuda.empty_cache()
    report = {'schema_version': 1, 'ok': True, 'inference_sha': INFERENCE_SHA, 'benchmark_sha': benchmark_sha,
              'phase32_freeze_sha256': digest, 'batch_size': 1, 'warmup_frames_per_shape': WARMUP,
              'frames_per_repeat': FRAMES, 'repeat_count': REPEATS, 'input': 'fixed synthetic FP32 CUDA tensor per size',
              'autocast': 'BF16; original context per frame', 'timed_scope': 'model(image) to native-size logits',
              'fps_definition': '1000 / arithmetic mean of per-frame milliseconds',
              'excluded_from_timing': ['model loading', 'preprocessing', 'transfer', 'argmax', 'IO'],
              'optimizer_updates': 0, 'dataset_images_read': 0, 'results': results}
    (output/'forward_benchmark.md').write_text(table(results), encoding='utf-8')
    write_json(output/'forward_benchmark.json', report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-sha', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--training-output-root', type=Path, default=Path('/root/autodl-tmp/outputs/CausalQ_DG'))
    parser.add_argument('--pretrained-root', type=Path, default=Path('/root/autodl-tmp/pretrained'))
    args = parser.parse_args(argv)
    benchmark(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
