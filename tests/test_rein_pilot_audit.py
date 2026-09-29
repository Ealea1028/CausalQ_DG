import pytest

from tools.audit_rein_pilot import SOURCE_SHA, audit
from tools.check_rein_runtime import EXPECTED_VERSIONS
from tools.check_rein_backbone import WEIGHT_SHA
from tools.rein_schedule_smoke import DATA_SHA, SLIDE_SHA, scheduled_lr


def fixture_pilot():
    """Fabricated schema fixture, not a remote result."""
    matrix = [[50 if i == j == 0 else 0 for j in range(19)] for i in range(19)]
    report = dict(ok=True, stage='complete', git_sha=SOURCE_SHA, seed=0, microbatches=2000,
                  accumulation=4, optimizer_updates=500, train_record_count=2000,
                  schedule_horizon_optimizer_updates=40000, target_labels_optimized=False,
                  formal_training_authorized=False, initialization='fresh_seed0_not_resume',
                  dtype='float32', all_losses_finite=True, pretrained_sha256=WEIGHT_SHA,
                  data_report_sha256=DATA_SHA, slide_report_sha256=SLIDE_SHA,
                  trainable_parameters=23569877, versions=EXPECTED_VERSIONS.copy(),
                  purpose='500update_source_pilot_with_50image_diagnostic_not_formal_accuracy',
                  prediction_roundtrip_max_error=0., checkpoint_bytes=100,
                  checkpoint_sha256='fixture', peak_allocated_gib=3., peak_reserved_gib=4.,
                  first_20_loss_mean=2., last_20_loss_mean=2.,
                  validation=dict(sample_count=50, samples=[dict(index=i, prediction_shape=[1024, 2048],
                      gt_shape=[1024, 2048], valid_pixel_count=1) for i in range(50)],
                      confusion_matrix=matrix, total_valid_pixels=50, class_iou=[1.] + [None]*18,
                      valid_classes=1, diagnostic_miou=1., official_diagnostic_metrics=dict(mIoU=100.)))
    records = [dict(iteration=i, dataset_index=i-1, optimizer_update=(i % 4 == 0),
                    loss=2., gradient_norm=1000., lr=scheduled_lr(i//4)) for i in range(1, 2001)]
    return report, records


def test_audit_accepts_finite_preclip_norm_and_missing_class_nulls():
    result = audit(*fixture_pilot())
    assert result['pilot_saved_audit_ok'] and result['diagnostic_miou'] == 1.
    assert not result['accuracy_claim'] and not result['formal_training_authorized']


@pytest.mark.parametrize('damage', ['steps', 'lr', 'nan', 'counts', 'geometry', 'producer', 'loss_mean'])
def test_audit_rejects_incomplete_or_inconsistent_pilot(damage):
    report, records = fixture_pilot()
    if damage == 'steps':
        records[-1]['iteration'] = 1999
    elif damage == 'lr':
        records[1]['lr'] = 0.
    elif damage == 'nan':
        records[1]['loss'] = float('nan')
    elif damage == 'counts':
        report['validation']['confusion_matrix'][0][0] = 49
    elif damage == 'geometry':
        report['validation']['samples'][0]['gt_shape'] = [512, 512]
    elif damage == 'producer':
        report['git_sha'] = 'other'
    else:
        report['last_20_loss_mean'] = 1.
    with pytest.raises(ValueError):
        audit(report, records)
