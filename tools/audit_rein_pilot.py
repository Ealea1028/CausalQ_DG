"""Read-only saved pilot audit; no torch, datasets or GPU imports."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from tools.check_rein_runtime import EXPECTED_VERSIONS
from tools.rein_schedule_smoke import DATA_SHA, SLIDE_SHA, scheduled_lr
from tools.check_rein_backbone import WEIGHT_SHA

SOURCE_SHA = "b4e292d8da87d186f03ac4a4769f7464a2499ef8"
REPORT_SHA = "c90169ad57ea39a937bbd2760aff3122758399c28dc6138a6822689212fc789a"
LOG_SHA = "83ac1b99558f5fde68b18b5a7f39f5520569d083bf1a91319d472dcfa490ca43"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def audit(report, records):
    expected = dict(ok=True, stage='complete', git_sha=SOURCE_SHA, seed=0,
                    microbatches=2000, accumulation=4, optimizer_updates=500,
                    train_record_count=2000, schedule_horizon_optimizer_updates=40000,
                    target_labels_optimized=False, formal_training_authorized=False,
                    initialization='fresh_seed0_not_resume', dtype='float32',
                    all_losses_finite=True, pretrained_sha256=WEIGHT_SHA,
                    data_report_sha256=DATA_SHA, slide_report_sha256=SLIDE_SHA,
                    trainable_parameters=23569877)
    for key, value in expected.items():
        require(type(report.get(key)) is type(value) and report[key] == value, f'Unexpected {key}')
    require(report.get('versions') == EXPECTED_VERSIONS, 'Runtime versions mismatch')
    require(report.get('purpose') == '500update_source_pilot_with_50image_diagnostic_not_formal_accuracy',
            'Unexpected pilot purpose')
    require([r['iteration'] for r in records] == list(range(1, 2001)), 'Noncontiguous training records')
    for i, record in enumerate(records, 1):
        require(type(record['dataset_index']) is int and 0 <= record['dataset_index'] < 24966,
                'Invalid source index')
        require(record['optimizer_update'] is (i % 4 == 0), 'Accumulation boundary mismatch')
        require(all(finite(record[k]) for k in ('loss', 'gradient_norm', 'lr')), 'Nonfinite trace')
        require(record['loss'] > 0 and record['gradient_norm'] > 0, 'Invalid source loss/norm')
        require(abs(record['lr'] - scheduled_lr(i // 4)) <= 1e-10, 'Optimizer-update LR mismatch')
    require(len({r['dataset_index'] for r in records}) == 2000, 'Pilot source permutation changed')
    for key, subset in (('first_20_loss_mean', records[:20]), ('last_20_loss_mean', records[-20:])):
        require(finite(report[key]) and abs(report[key] - sum(r['loss'] for r in subset)/20) < 1e-6,
                'Loss summary mismatch')
    error = report['prediction_roundtrip_max_error']
    require(finite(error) and 0 <= error <= 1e-5, 'Checkpoint roundtrip failed')
    require(type(report['checkpoint_bytes']) is int and 0 < report['checkpoint_bytes'] <= 2**30,
            'Checkpoint budget failed')
    for key in ('peak_allocated_gib', 'peak_reserved_gib'):
        require(finite(report[key]) and report[key] > 0, 'Invalid memory measurement')
    require(report['peak_reserved_gib'] >= report['peak_allocated_gib'], 'Memory ordering mismatch')
    validation = report['validation']
    samples = validation['samples']
    require(validation['sample_count'] == 50 and [s['index'] for s in samples] == list(range(50)),
            'Incomplete target coverage')
    require(all(s['prediction_shape'] == [1024, 2048] and s['gt_shape'] == [1024, 2048]
                and type(s['valid_pixel_count']) is int and s['valid_pixel_count'] > 0 for s in samples),
            'Invalid target geometry/coverage')
    matrix = validation['confusion_matrix']
    require(len(matrix) == 19 and all(len(row) == 19 for row in matrix), 'Invalid confusion shape')
    require(all(type(v) is int and v >= 0 for row in matrix for v in row), 'Invalid confusion counts')
    total = sum(map(sum, matrix))
    require(total == validation['total_valid_pixels'] == sum(s['valid_pixel_count'] for s in samples),
            'Valid-pixel count mismatch')
    ious = []
    for c in range(19):
        union = sum(matrix[c]) + sum(row[c] for row in matrix) - matrix[c][c]
        ious.append(matrix[c][c]/union if union else None)
    require(len(validation['class_iou']) == 19, 'Class metric count mismatch')
    for calculated, saved in zip(ious, validation['class_iou']):
        require(saved is None if calculated is None else finite(saved) and abs(saved - calculated) < 1e-10,
                'Per-class confusion/metric mismatch')
    present = [v for v in ious if v is not None]
    require(len(present) == validation['valid_classes'] and bool(present), 'Valid class count mismatch')
    miou = sum(present)/len(present)
    require(finite(validation['diagnostic_miou']) and abs(miou-validation['diagnostic_miou']) < 1e-10,
            'Independent mIoU mismatch')
    official = validation['official_diagnostic_metrics']
    require(all(finite(v) for v in official.values()), 'Nonfinite official summary')
    require(abs(official['mIoU']/100-miou) <= 5.1e-5, 'Rounded official mIoU mismatch')
    return dict(pilot_saved_audit_ok=True, training_git_sha=SOURCE_SHA, optimizer_updates=500,
                microbatches=2000, target_sample_count=50, diagnostic_miou=miou,
                first_20_loss_mean=report['first_20_loss_mean'], last_20_loss_mean=report['last_20_loss_mean'],
                checkpoint_sha256=report['checkpoint_sha256'], checkpoint_bytes=report['checkpoint_bytes'],
                prediction_roundtrip_max_error=error, peak_reserved_gib=report['peak_reserved_gib'],
                accuracy_claim=False, formal_training_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--stderr-log', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        require(digest(args.report) == REPORT_SHA and digest(args.stderr_log) == LOG_SHA, 'Evidence SHA mismatch')
        report = json.loads(args.report.read_text(encoding='utf-8'))
        run = args.run_dir.resolve(strict=True)
        require(json.loads((run/'summary.json').read_text(encoding='utf-8')) == report, 'Saved summary mismatch')
        metadata = json.loads((run/'metadata.json').read_text(encoding='utf-8'))
        for key in ('git_sha', 'seed', 'microbatches', 'accumulation', 'initialization', 'versions',
                    'pretrained_sha256', 'data_report_sha256', 'slide_report_sha256'):
            require(metadata[key] == report[key], f'Metadata mismatch: {key}')
        checkpoint = (run/'checkpoint_500updates.pth').resolve(strict=True)
        require(checkpoint.parent == run and Path(report['checkpoint']).resolve(strict=True) == checkpoint,
                'Checkpoint path mismatch')
        require(checkpoint.stat().st_size == report['checkpoint_bytes']
                and digest(checkpoint) == report['checkpoint_sha256'], 'Checkpoint integrity mismatch')
        records = [json.loads(line) for line in (run/'train.jsonl').read_text(encoding='utf-8').splitlines()]
        print(json.dumps(audit(report, records), indent=2))
        return 0
    except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
        print(json.dumps(dict(pilot_saved_audit_ok=False, error=f'{type(exc).__name__}: {exc}'), indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
