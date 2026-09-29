"""Two independent process restores of accepted update 20; no formal training."""

import argparse
import contextlib
import importlib
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import traceback

from tools.rein_schedule_smoke import digest, DATA_SHA, restore_protocol_config
from tools.check_rein_backbone import REIN_SHA, WEIGHT_SHA, validate_load_keys
from tools.check_rein_runtime import EXPECTED_VERSIONS, git_output, validate_versions

CHECKPOINT_SHA = 'e160cb17c3a708d2592016961bcde1eaa061abdddb5d9e0980dfe2d823e84678'
PRODUCER = '5fee730e98d0a62535d93dd310a7a6dd334a36a3'


def compare_snapshots(first, second):
    from tools.rein_training_state import compare_trees
    if first['indices'] != second['indices'] or first['indices'] != [22892, 5750, 13022, 5583]:
        raise ValueError('Fresh-process source indices mismatch')
    if len(first['losses']) != 4 or len(second['losses']) != 4:
        raise ValueError('Expected four continuation losses')
    import math
    if not all(math.isfinite(value) for value in first['losses'] + second['losses']):
        raise ValueError('Nonfinite fresh-process loss')
    loss_error = max(abs(a-b) for a, b in zip(first['losses'], second['losses']))
    if not math.isfinite(loss_error) or loss_error > 1e-4:
        raise ValueError('Fresh-process loss mismatch')
    parameter_error = compare_trees(first['model'], second['model'], 'model')
    if parameter_error > 1e-5:
        raise ValueError('Fresh-process parameter mismatch')
    optimizer_error = compare_trees(first['optimizer'], second['optimizer'], 'optimizer')
    compare_trees(first['scheduler'], second['scheduler'], 'scheduler')
    compare_trees(first['sampler'], second['sampler'], 'sampler')
    compare_trees(first['rng'], second['rng'], 'rng')
    return dict(maximum_loss_difference=loss_error, maximum_parameter_difference=parameter_error,
                maximum_optimizer_tensor_difference=optimizer_error, source_indices=first['indices'])


def worker(args):
    import torch
    import numpy as np
    from tools.rein_training_state import SourceSampler, verify_next_update
    if platform.python_version_tuple()[:2] != ('3', '10'):
        raise ValueError('Isolated Python 3.10 required')
    versions = {name: str(importlib.import_module(name).__version__) for name in EXPECTED_VERSIONS}
    validate_versions(versions)
    if torch.version.cuda != '11.8' or not torch.cuda.is_available():
        raise RuntimeError('CUDA 11.8 GPU required')
    torch.manual_seed(args.worker_seed)
    torch.cuda.manual_seed_all(args.worker_seed)
    np.random.seed(args.worker_seed)
    random.seed(args.worker_seed)
    sys.path.insert(0, str(args.rein_root.resolve()))
    with contextlib.redirect_stdout(sys.stderr):
        import rein  # noqa: F401
        from mmengine.registry import init_default_scope
        from mmengine.optim import build_optim_wrapper, PolyLR
        from mmseg.registry import DATASETS, MODELS
        from tools.rein_protocol_adapter import register_data_transforms
        init_default_scope('mmseg')
        register_data_transforms()
        config = restore_protocol_config(json.loads(args.data_report.read_text()), args.rein_root)
        model = MODELS.build(config['model'])
        model.decode_head.init_weights()
        state = torch.load(args.weights, map_location='cpu', weights_only=True)
        keys = set(dict(model.backbone.named_parameters())) | set(dict(model.backbone.named_buffers()))
        loaded = model.backbone.load_state_dict(state, strict=False)
        validate_load_keys(state, keys, loaded.missing_keys, loaded.unexpected_keys)
        del state
        model.train(True)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad != name.startswith(('backbone.reins.', 'decode_head.')):
                raise ValueError('Unexpected trainability: ' + name)
        model.to('cuda:0')
        frozen = model.backbone.patch_embed.proj.weight.detach().clone()
        wrapper = build_optim_wrapper(model, config['optim_wrapper'])
        if wrapper._accumulative_counts != 4:
            raise ValueError('Expected accumulation 4')
        wrapper.initialize_count_status(model, 0, 84)
        scheduler = PolyLR(wrapper, eta_min=0, power=.9, begin=0, end=40000, by_epoch=False)
        dataset = DATASETS.build(config['train_dataloader']['dataset'])
        if len(dataset) != 24966:
            raise ValueError('Source coverage changed')
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if payload['metadata']['git_sha'] != PRODUCER or payload['metadata']['optimizer_updates'] != 20:
            raise ValueError('Wrong checkpoint producer or update')
        torch.cuda.reset_peak_memory_stats(0)
        result = verify_next_update(model, dataset, wrapper, scheduler, SourceSampler(len(dataset)),
                                    payload, args.output_dir / f'worker_{args.worker_seed}.pth')
        if not torch.equal(frozen, model.backbone.patch_embed.proj.weight):
            raise ValueError('Frozen backbone changed')
    return dict(ok=True, initialization_seed=args.worker_seed, versions=versions,
                python=platform.python_version(), gpu=torch.cuda.get_device_name(0),
                peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30, 3), continuation=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rein-root', 'weights', 'data-report', 'checkpoint', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--worker-seed', type=int, choices=(17, 91))
    args = parser.parse_args()
    report = dict(ok=False, purpose='fresh_process_restore_replay_not_uninterrupted_equivalence_or_accuracy',
                  evaluation_git_sha=git_output(Path(__file__).resolve().parents[1], 'rev-parse', 'HEAD'),
                  formal_training_authorized=False, target_labels_optimized=False)
    try:
        for path, expected in ((args.checkpoint, CHECKPOINT_SHA), (args.weights, WEIGHT_SHA), (args.data_report, DATA_SHA)):
            if digest(path) != expected:
                raise ValueError('Input hash mismatch: ' + str(path))
        if git_output(args.rein_root, 'rev-parse', 'HEAD') != REIN_SHA or git_output(args.rein_root, 'status', '--porcelain'):
            raise ValueError('REIN source must be pinned and clean')
        if args.worker_seed is not None:
            report.update(worker(args))
        else:
            if shutil.disk_usage(args.output_dir.parent).free < 2*2**30:
                raise OSError('Need 2 GiB free')
            args.output_dir.mkdir()
            commands = [sys.executable, '-m', 'tools.rein_fresh_process']
            for name in ('rein_root', 'weights', 'data_report', 'checkpoint', 'output_dir'):
                commands += ['--' + name.replace('_', '-'), str(getattr(args, name))]
            workers = []
            for seed in (17, 91):
                with (args.output_dir / f'worker_{seed}.json').open('x') as stdout, (args.output_dir / f'worker_{seed}.stderr.log').open('x') as stderr:
                    completed = subprocess.run(commands + ['--worker-seed', str(seed)], stdout=stdout, stderr=stderr)
                if completed.returncode:
                    raise RuntimeError(f'Fresh worker {seed} failed; see retained logs')
                workers.append(json.loads((args.output_dir / f'worker_{seed}.json').read_text()))
            import torch
            paths = [args.output_dir / f'worker_{seed}.pth' for seed in (17, 91)]
            if any(path.stat().st_size > 2**30 for path in paths):
                raise ValueError('Snapshot exceeds 1 GiB budget')
            comparison = compare_snapshots(*(torch.load(path, map_location='cpu', weights_only=True) for path in paths))
            if digest(args.checkpoint) != CHECKPOINT_SHA:
                raise ValueError('Input checkpoint changed')
            report.update(ok=True, fresh_process_restore_replay_verified=True, workers=workers,
                          independent_process_count=2, extra_microbatches_executed=16,
                          checkpoint_sha256=CHECKPOINT_SHA, training_git_sha=PRODUCER,
                          snapshot_sha256=[digest(path) for path in paths], comparison=comparison)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc(file=sys.stderr)
    print(json.dumps(report, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
