"""Fresh source-only runner integration: cyclic sampler and bounded rolling saves."""

import argparse
import contextlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import traceback
import time

from tools.rein_schedule_smoke import DATA_SHA, digest, restore_protocol_config, compact_state, scheduled_lr
from tools.check_rein_backbone import REIN_SHA, WEIGHT_SHA, validate_load_keys
from tools.check_rein_runtime import EXPECTED_VERSIONS, git_output, validate_versions

FRESH_REPORT_SHA = '586e96ec63c481a9f9bf2b3c62754d96c98001123bc57fa34254f2361a9d01e7'


def runner_budget(formal):
    if type(formal) is not bool:
        raise ValueError('Explicit boolean mode required')
    return (40000, 1000, 500) if formal else (20, 5, 5)


def validate_fresh_report(report):
    if (report.get('ok') is not True or report.get('fresh_process_restore_replay_verified') is not True
            or report.get('evaluation_git_sha') != '1671f1a07441868e464776cfbf9305308187c289'
            or report.get('independent_process_count') != 2
            or report.get('target_labels_optimized') is not False):
        raise ValueError('Accepted fresh-process report required')


def rolling_save(directory, payload):
    """Only rotates files created inside this new run; keeps last and previous."""
    import torch
    from tools.rein_training_state import compare_trees
    temporary, latest, previous = (directory / name for name in ('pending.pth', 'last.pth', 'previous.pth'))
    with temporary.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if temporary.stat().st_size > 2**30:
        raise ValueError('Rolling checkpoint exceeds 1 GiB; pending file preserved')
    restored = torch.load(temporary, map_location='cpu', weights_only=True)
    if compare_trees(restored, payload) != 0:
        raise ValueError('Rolling serialization changed checkpoint state')
    if latest.exists():
        os.replace(latest, previous)
    os.replace(temporary, latest)
    return dict(path=str(latest), sha256=digest(latest), bytes=latest.stat().st_size,
                optimizer_updates=payload['metadata']['optimizer_updates'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rein-root', 'weights', 'data-report', 'fresh-report', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--formal-source-only', action='store_true')
    for name in ('runner-report', 'runner-log', 'runner-run'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    updates, save_interval, target_count = runner_budget(args.formal_source_only)
    microbatches = 4 * updates
    started = time.monotonic()
    created_run = False
    report = dict(ok=False, stage='preflight', phase=16,
                  purpose='20update_runner_integration_not_formal_training_or_accuracy',
                  git_sha=git_output(Path(__file__).resolve().parents[1], 'rev-parse', 'HEAD'),
                  seed=0, optimizer_updates=0, microbatches=microbatches, accumulation=4,
                  schedule_horizon_optimizer_updates=40000, formal_training_authorized=False,
                  target_labels_optimized=False, initialization='fresh_seed0_not_pilot_resume')
    try:
        if args.run_dir.exists():
            raise FileExistsError('Preserve existing run')
        if args.formal_source_only:
            if any(path is None for path in (args.runner_report, args.runner_log, args.runner_run)):
                raise ValueError('Formal source-only run requires complete saved runner evidence')
            from tools.audit_rein_runner import audit_files, REPORT_SHA
            report['runner_saved_audit'] = audit_files(args.runner_report, args.runner_log, args.runner_run)
            report.update(purpose='formal_source_only_rein_baseline_not_cqe_or_exact_paper_reproduction',
                          formal_training_authorized=True, runner_report_sha256=REPORT_SHA,
                          final_evaluation_samples=500, checkpoint_interval_updates=save_interval,
                          checkpoint_selection='fixed_final_update_no_target_selection', exact_resume_verified=False,
                          source='gta5', validation_dataset='cityscapes_val', backbone='dinov2_vitl14_rein_patch16',
                          rein_sha=REIN_SHA, decoder='Mask2Former', physical_batch_size=1, num_workers=0,
                          train_crop=[512,512], target_input=[1024,512], target_gt=[2048,1024],
                          slide_crop=[512,512], slide_stride=[341,341], cqe_enabled=False)
        for path, expected in ((args.weights, WEIGHT_SHA), (args.data_report, DATA_SHA),
                               (args.fresh_report, FRESH_REPORT_SHA)):
            if digest(path) != expected:
                raise ValueError('Input hash mismatch: ' + str(path))
        validate_fresh_report(json.loads(args.fresh_report.read_text()))
        if git_output(args.rein_root, 'rev-parse', 'HEAD') != REIN_SHA or git_output(args.rein_root, 'status', '--porcelain'):
            raise ValueError('Pinned clean REIN source required')
        if shutil.disk_usage(args.run_dir.parent).free < 2*2**30:
            raise OSError('Need 2 GiB free')
        if platform.python_version_tuple()[:2] != ('3', '10'):
            raise ValueError('Isolated Python 3.10 required')
        import torch
        import numpy as np
        from tools.rein_training_state import SourceSampler, capture_rng, cpu_tree
        from tools.check_rein_segmentor import validate_losses
        from tools.check_rein_slide_eval import restore_compact, evaluate_target
        versions = {name: str(importlib.import_module(name).__version__) for name in EXPECTED_VERSIONS}
        validate_versions(versions)
        if torch.version.cuda != '11.8' or not torch.cuda.is_available():
            raise RuntimeError('CUDA 11.8 GPU required')
        report.update(versions=versions, python=platform.python_version(), gpu=torch.cuda.get_device_name(0),
                      pretrained_sha256=WEIGHT_SHA, data_report_sha256=DATA_SHA,
                      fresh_report_sha256=FRESH_REPORT_SHA, dtype='float32')
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        random.seed(0)
        np.random.seed(0)
        sys.path.insert(0, str(args.rein_root.resolve()))
        with contextlib.redirect_stdout(sys.stderr):
            import rein  # noqa: F401
            from mmengine.registry import init_default_scope
            from mmengine.optim import build_optim_wrapper, PolyLR
            from mmseg.registry import DATASETS, MODELS, METRICS
            from tools.rein_protocol_adapter import register_data_transforms
            init_default_scope('mmseg')
            register_data_transforms()
            config = restore_protocol_config(json.loads(args.data_report.read_text()), args.rein_root)
            model = MODELS.build(config.model)
            model.decode_head.init_weights()
            weights = torch.load(args.weights, map_location='cpu', weights_only=True)
            keys = set(dict(model.backbone.named_parameters())) | set(dict(model.backbone.named_buffers()))
            loaded = model.backbone.load_state_dict(weights, strict=False)
            validate_load_keys(weights, keys, loaded.missing_keys, loaded.unexpected_keys)
            del weights
            model.train(True)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad != name.startswith(('backbone.reins.', 'decode_head.')):
                    raise ValueError('Unexpected trainability: ' + name)
            model.to('cuda:0')
            if args.formal_source_only:
                report['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
                if report['trainable_parameters'] != 23569877:
                    raise ValueError('Formal trainable parameter coverage changed')
            frozen = model.backbone.patch_embed.proj.weight.detach().clone()
            wrapper = build_optim_wrapper(model, config.optim_wrapper)
            if wrapper._accumulative_counts != 4:
                raise ValueError('Expected accumulation 4')
            wrapper.initialize_count_status(model, 0, microbatches)
            scheduler = PolyLR(wrapper, eta_min=0, power=.9, begin=0, end=40000, by_epoch=False)
            dataset = DATASETS.build(config.train_dataloader.dataset)
            if len(dataset) != 24966:
                raise ValueError('Source coverage changed')
            sampler = SourceSampler(len(dataset), seed=0)
            args.run_dir.mkdir()
            created_run = True
            (args.run_dir / 'metadata.json').write_text(json.dumps(report, indent=2))
            (args.run_dir / 'config.json').write_text(json.dumps(config.to_dict(), indent=2, default=str))
            report.update(stage='source_optimization', checkpoints=[])
            torch.cuda.reset_peak_memory_stats(0)
            losses_seen = []
            with (args.run_dir / 'train.jsonl').open('x') as trace:
                for iteration in range(1, microbatches + 1):
                    index = sampler.take(1)[0]
                    item = dataset[index]
                    batch = model.data_preprocessor(dict(inputs=[item['inputs']], data_samples=[item['data_samples']]), training=True)
                    if not (batch['data_samples'][0].gt_sem_seg.data != 255).any().item():
                        raise ValueError('No valid training pixels')
                    with wrapper.optim_context(model):
                        loss = validate_losses(model.loss(batch['inputs'], batch['data_samples']))
                    wrapper.backward(wrapper.scale_loss(loss))
                    norms = []
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            if not parameter.requires_grad or not torch.isfinite(parameter.grad).all().item():
                                raise FloatingPointError('Invalid gradient')
                            norms.append(parameter.grad.float().norm().item())
                    norm = math.sqrt(sum(value*value for value in norms))
                    if not math.isfinite(norm) or norm <= 0:
                        raise FloatingPointError('Invalid gradient norm')
                    update = wrapper.should_update()
                    if update != (iteration % 4 == 0):
                        raise ValueError('Accumulation boundary mismatch')
                    if update:
                        wrapper.step()
                        wrapper.zero_grad(set_to_none=True)
                        scheduler.step()
                        report['optimizer_updates'] += 1
                    if any(abs(group['lr']-scheduled_lr(report['optimizer_updates'])) > 1e-10 for group in wrapper.optimizer.param_groups):
                        raise ValueError('LR mismatch')
                    record = dict(iteration=iteration, dataset_index=index, loss=loss.item(), gradient_norm=norm,
                                  optimizer_update=update, lr=wrapper.optimizer.param_groups[0]['lr'])
                    losses_seen.append(record['loss'])
                    trace.write(json.dumps(record) + '\n')
                    trace.flush()
                    if not args.formal_source_only or iteration % 40 == 0 or iteration == 1:
                        print(json.dumps(record), file=sys.stderr)
                    del loss
                    if update and report['optimizer_updates'] % save_interval == 0:
                        if shutil.disk_usage(args.run_dir).free < 2*2**30:
                            raise OSError('Low disk space: preserve existing checkpoints and stop')
                        payload = dict(model=compact_state(model), optimizer=cpu_tree(wrapper.state_dict()),
                                       scheduler=cpu_tree(scheduler.state_dict()), sampler=sampler.state_dict(),
                                       rng=capture_rng(include_cuda=True), metadata={k: v for k, v in report.items() if k != 'checkpoints'})
                        report['checkpoints'].append(rolling_save(args.run_dir, payload))
                        del payload
            if report['optimizer_updates'] != updates or not torch.equal(frozen, model.backbone.patch_embed.proj.weight):
                raise ValueError('Update or frozen-weight mismatch')
            payload = torch.load(args.run_dir / 'last.pth', map_location='cpu', weights_only=True)
            previous = torch.load(args.run_dir / 'previous.pth', map_location='cpu', weights_only=True)
            if (previous['metadata']['optimizer_updates'] != updates-save_interval
                    or payload['metadata']['optimizer_updates'] != updates
                    or payload['sampler']['cursor'] != sampler.cursor
                    or payload['sampler']['epoch'] != sampler.epoch):
                raise ValueError('Rolling retention/cursor mismatch')
            del previous
            restore_compact(model, payload['model'])
            del payload
            report.update(stage='target_diagnostic', first_20_loss_mean=sum(losses_seen[:20])/20,
                          last_20_loss_mean=sum(losses_seen[-20:])/20, training_record_count=microbatches,
                          retained_checkpoint_count=2, sampler_cursor=sampler.cursor, sampler_epoch=sampler.epoch)
            report['validation'] = evaluate_target(model, DATASETS.build(config.val_dataloader.dataset), METRICS.build(config.val_evaluator), target_count,
                                                   formal_full=args.formal_source_only)
            if args.formal_source_only:
                report['validation']['miou'] = report['validation']['diagnostic_miou']
                report['validation']['iteration'] = updates
                report['stage'] = 'final_evaluation'
            report.update(ok=True, stage='complete', elapsed_seconds=round(time.monotonic()-started, 2),
                          peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30, 3))
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc(file=sys.stderr)
    if created_run:
        (args.run_dir / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
