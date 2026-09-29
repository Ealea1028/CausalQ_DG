import copy

import pytest

from tools.audit_rein_runner import audit, PRODUCER
from tools.check_rein_backbone import WEIGHT_SHA
from tools.check_rein_runtime import EXPECTED_VERSIONS
from tools.rein_schedule_smoke import DATA_SHA, scheduled_lr
from tools.rein_runner_gate import runner_budget


def fixture():
    matrix = [[0]*19 for _ in range(19)]
    matrix[0][0] = 50
    validation = dict(sample_count=5, samples=[dict(index=i, prediction_shape=[1024,2048],
        gt_shape=[1024,2048], valid_pixel_count=10) for i in range(5)],
        confusion_matrix=matrix, total_valid_pixels=50, class_iou=[1.]+[None]*18,
        valid_classes=1, diagnostic_miou=1., official_diagnostic_metrics={'mIoU':100.})
    report = dict(ok=True, stage='complete', phase=16, git_sha=PRODUCER, seed=0,
        optimizer_updates=20, microbatches=80, accumulation=4, training_record_count=80,
        schedule_horizon_optimizer_updates=40000, sampler_cursor=80, retained_checkpoint_count=2,
        target_labels_optimized=False, formal_training_authorized=False, dtype='float32',
        initialization='fresh_seed0_not_pilot_resume', pretrained_sha256=WEIGHT_SHA,
        data_report_sha256=DATA_SHA, versions=EXPECTED_VERSIONS, peak_reserved_gib=4.,
        first_20_loss_mean=1., last_20_loss_mean=1., validation=validation,
        checkpoints=[dict(optimizer_updates=i) for i in (5,10,15,20)])
    records = [dict(iteration=i, dataset_index=i-1, loss=1., gradient_norm=1.,
                    optimizer_update=i%4==0, lr=scheduled_lr(i//4)) for i in range(1,81)]
    return report, records


def test_full_runner_evidence_required_not_only_exit_code():
    report, records = fixture()
    assert audit(report, records)['runner_saved_audit_ok']
    with pytest.raises(ValueError):
        audit(report, records[:-1])


@pytest.mark.parametrize('field,value', [('sampler_cursor',79), ('optimizer_updates',19),
    ('target_labels_optimized',True), ('formal_training_authorized',True), ('git_sha','wrong')])
def test_bad_runner_metadata_rejected(field,value):
    report, records = fixture()
    report[field] = value
    with pytest.raises(ValueError):
        audit(report, records)


def test_nan_absent_class_is_none_but_bad_finite_metrics_rejected():
    report, records = fixture()
    original = copy.deepcopy(report)
    report['validation']['class_iou'][1] = float('nan')
    with pytest.raises(ValueError):
        audit(report, records)
    original['validation']['total_valid_pixels'] += 1
    with pytest.raises(ValueError):
        audit(original, records)


def test_formal_budget_is_explicit_and_preserves_update_horizon():
    assert runner_budget(False) == (20,5,5)
    assert runner_budget(True) == (40000,1000,500)
    with pytest.raises(ValueError):
        runner_budget(40000)


def test_formal_sampler_budget_spans_source_epochs_without_exhaustion():
    from tools.rein_training_state import SourceSampler
    sampler = SourceSampler(24966, seed=0)
    indices = sampler.take(4 * runner_budget(True)[0])
    assert len(indices) == 160000
    assert sampler.epoch == 6 and sampler.cursor == 10204
    for start in range(0, 149796, 24966):
        assert sorted(indices[start:start+24966]) == list(range(24966))


def test_formal_full_evaluation_cannot_use_short_diagnostic_coverage():
    from tools.check_rein_slide_eval import evaluate_target
    with pytest.raises(ValueError, match='bounded'):
        evaluate_target(None, [None]*500, None, 50, formal_full=True)
    with pytest.raises(ValueError, match='bounded'):
        evaluate_target(None, [None]*499, None, 500, formal_full=True)


def test_existing_run_is_not_overwritten_on_preflight_failure(tmp_path, monkeypatch, capsys):
    from tools.rein_runner_gate import main
    saved = tmp_path / 'summary.json'
    saved.write_text('existing evidence')
    argv = ['runner']
    for name in ('rein-root', 'weights', 'data-report', 'fresh-report'):
        argv += ['--' + name, str(tmp_path/'unused')]
    argv += ['--run-dir', str(tmp_path)]
    monkeypatch.setattr('sys.argv', argv)
    assert main() == 1
    assert saved.read_text() == 'existing evidence'
    assert 'FileExistsError' in capsys.readouterr().out
