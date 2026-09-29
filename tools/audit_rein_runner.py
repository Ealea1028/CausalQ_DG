"""Read-only hash-pinned integration audit before source-only 40k training."""

import argparse
import json
import math
from pathlib import Path

from tools.audit_rein_pilot import require, finite, digest
from tools.check_rein_runtime import EXPECTED_VERSIONS
from tools.check_rein_backbone import WEIGHT_SHA
from tools.rein_schedule_smoke import DATA_SHA, scheduled_lr

PRODUCER = '7a3a1c699f5b00ed4b6cc76055457beec374e05d'
REPORT_SHA = 'b584a6504f75351864922b5f43119950bee9621b2cf30cf75bbb09be6d833a97'
LOG_SHA = '22af8fbb574b10d70714716ad83dff186ec8c59708dcfeb0950abe79aa3bb5d5'


def audit(report, records):
    expected = dict(ok=True, stage='complete', phase=16, git_sha=PRODUCER, seed=0,
                    optimizer_updates=20, microbatches=80, accumulation=4, training_record_count=80,
                    schedule_horizon_optimizer_updates=40000, sampler_cursor=80, retained_checkpoint_count=2,
                    target_labels_optimized=False, formal_training_authorized=False, dtype='float32',
                    initialization='fresh_seed0_not_pilot_resume', pretrained_sha256=WEIGHT_SHA, data_report_sha256=DATA_SHA)
    for key, value in expected.items():
        require(type(report.get(key)) is type(value) and report[key] == value, 'Unexpected ' + key)
    require(report['versions'] == EXPECTED_VERSIONS, 'Runtime pins changed')
    require([r['iteration'] for r in records] == list(range(1, 81)), 'Noncontiguous trace')
    require(len({r['dataset_index'] for r in records}) == 80, 'Source permutation changed')
    for i, record in enumerate(records, 1):
        require(type(record['dataset_index']) is int and 0 <= record['dataset_index'] < 24966, 'Source index')
        require(record['optimizer_update'] is (i % 4 == 0), 'Accumulation boundary')
        require(all(finite(record[k]) for k in ('loss', 'gradient_norm', 'lr')), 'Nonfinite trace')
        require(record['loss'] > 0 and record['gradient_norm'] > 0, 'Loss/norm must be positive')
        require(abs(record['lr']-scheduled_lr(i//4)) <= 1e-10, 'LR drift')
    for key, subset in [('first_20_loss_mean', records[:20]), ('last_20_loss_mean', records[-20:])]:
        require(finite(report[key]) and abs(report[key]-sum(r['loss'] for r in subset)/20) < 1e-6, 'Loss summary')
    require([entry['optimizer_updates'] for entry in report['checkpoints']] == [5, 10, 15, 20], 'Save history')
    require(finite(report['peak_reserved_gib']) and report['peak_reserved_gib'] > 0, 'Memory measurement')
    validation = report['validation']
    samples = validation['samples']
    require(validation['sample_count'] == 5 and [s['index'] for s in samples] == list(range(5)), 'Target coverage')
    require(all(s['prediction_shape'] == s['gt_shape'] == [1024, 2048]
                and type(s['valid_pixel_count']) is int and s['valid_pixel_count'] > 0 for s in samples), 'Target geometry')
    matrix = validation['confusion_matrix']
    require(len(matrix) == 19 and all(len(row) == 19 for row in matrix), 'Confusion shape')
    require(all(type(v) is int and v >= 0 for row in matrix for v in row), 'Confusion values')
    require(sum(map(sum, matrix)) == validation['total_valid_pixels'] == sum(s['valid_pixel_count'] for s in samples), 'Pixel coverage')
    ious = []
    for c in range(19):
        union = sum(matrix[c]) + sum(row[c] for row in matrix) - matrix[c][c]
        ious.append(matrix[c][c]/union if union else None)
    require(len(validation['class_iou']) == 19, 'Class coverage')
    for expected_iou, saved in zip(ious, validation['class_iou']):
        require(saved is None if expected_iou is None else finite(saved) and abs(saved-expected_iou) < 1e-10, 'IoU consistency')
    present = [value for value in ious if value is not None]
    require(bool(present) and len(present) == validation['valid_classes'], 'Valid classes')
    mean = sum(present)/len(present)
    require(finite(validation['diagnostic_miou']) and abs(mean-validation['diagnostic_miou']) < 1e-10, 'Independent mIoU')
    official = validation['official_diagnostic_metrics']
    require(all(finite(v) for v in official.values()) and abs(official['mIoU']/100-mean) <= 5.1e-5, 'Official metric')
    return dict(runner_saved_audit_ok=True, producer_git_sha=PRODUCER, optimizer_updates=20,
                diagnostic_miou=mean, accuracy_claim=False)


def audit_files(report_path, log_path, run_dir):
    require(digest(report_path) == REPORT_SHA and digest(log_path) == LOG_SHA, 'Evidence hash mismatch')
    report = json.loads(report_path.read_text())
    run = run_dir.resolve(strict=True)
    require(json.loads((run/'summary.json').read_text()) == report, 'Summary mismatch')
    metadata = json.loads((run/'metadata.json').read_text())
    for key in ('git_sha', 'seed', 'microbatches', 'accumulation', 'initialization', 'versions',
                'pretrained_sha256', 'data_report_sha256', 'formal_training_authorized'):
        require(metadata[key] == report[key], 'Metadata mismatch: ' + key)
    require({p.name for p in run.glob('*.pth')} == {'last.pth', 'previous.pth'}, 'Retained files mismatch')
    for name, entry in [('last.pth', report['checkpoints'][-1]), ('previous.pth', report['checkpoints'][-2])]:
        path = (run/name).resolve(strict=True)
        require(path.parent == run, 'Checkpoint outside run')
        require(path.stat().st_size == entry['bytes'] and 0 < entry['bytes'] <= 2**30
                and digest(path) == entry['sha256'], 'Checkpoint integrity')
        require(Path(entry['path']).parent.resolve(strict=True) == run, 'Save history path')
    records = [json.loads(line) for line in (run/'train.jsonl').read_text().splitlines()]
    return audit(report, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'stderr-log', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit_files(args.report, args.stderr_log, args.run_dir)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        result = dict(runner_saved_audit_ok=False, error=f'{type(error).__name__}: {error}')
    print(json.dumps(result, indent=2))
    return 0 if result['runner_saved_audit_ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
