import copy

import pytest

from tools.audit_rein_source40k import (
    ACCUMULATION,
    MICROBATCHES,
    PRODUCER,
    SOURCE_SIZE,
    UPDATES,
    audit,
    audit_records,
    audit_validation,
)
from tools.check_rein_backbone import WEIGHT_SHA
from tools.check_rein_runtime import EXPECTED_VERSIONS
from tools.rein_schedule_smoke import DATA_SHA
from tools.rein_schedule_smoke import scheduled_lr


def records_fixture():
    return [dict(iteration=i, dataset_index=(i - 1) % 4, loss=2., gradient_norm=3.,
                 optimizer_update=i % 2 == 0, lr=scheduled_lr(i // 2))
            for i in range(1, 13)]


def validation_fixture():
    matrix = [[0] * 19 for _ in range(19)]
    matrix[0][0] = 500
    return dict(sample_count=500,
                samples=[dict(index=i, prediction_shape=[1024, 2048],
                              gt_shape=[1024, 2048], valid_pixel_count=1)
                         for i in range(500)],
                confusion_matrix=matrix, total_valid_pixels=500,
                class_iou=[1.] + [None] * 18, valid_classes=1,
                diagnostic_miou=1., miou=1., iteration=40000,
                official_diagnostic_metrics=dict(mIoU=100.))


def test_record_audit_checks_accumulation_lr_and_epoch_permutations():
    result = audit_records(records_fixture(), updates=6, accumulation=2, source_size=4)
    assert result == dict(complete_source_epochs=3, partial_epoch_samples=0)


@pytest.mark.parametrize("damage", ["iteration", "lr", "index", "permutation", "finite"])
def test_record_audit_rejects_damaged_trace(damage):
    records = records_fixture()
    if damage == "iteration":
        records[-1]["iteration"] = 11
    elif damage == "lr":
        records[-1]["lr"] = 1e-4
    elif damage == "index":
        records[-1]["dataset_index"] = 4
    elif damage == "permutation":
        records[3]["dataset_index"] = 2
    else:
        records[-1]["loss"] = float("nan")
    with pytest.raises(ValueError):
        audit_records(records, updates=6, accumulation=2, source_size=4)


def test_validation_audit_recomputes_full_cityscapes_metric():
    validation = validation_fixture()
    assert audit_validation(validation) == 1.
    damaged = copy.deepcopy(validation)
    damaged["total_valid_pixels"] = 499
    with pytest.raises(ValueError):
        audit_validation(damaged)


def test_complete_formal_report_contract_is_explicit():
    validation = validation_fixture()
    records = [dict(iteration=i, dataset_index=(i - 1) % SOURCE_SIZE,
                    loss=2., gradient_norm=3., optimizer_update=i % ACCUMULATION == 0,
                    lr=scheduled_lr(i // ACCUMULATION))
               for i in range(1, MICROBATCHES + 1)]
    report = dict(ok=True, stage="complete", phase=16, git_sha=PRODUCER, seed=0,
                  optimizer_updates=UPDATES, microbatches=MICROBATCHES,
                  accumulation=ACCUMULATION, training_record_count=MICROBATCHES,
                  schedule_horizon_optimizer_updates=UPDATES,
                  target_labels_optimized=False, formal_training_authorized=True,
                  initialization="fresh_seed0_not_pilot_resume", dtype="float32",
                  pretrained_sha256=WEIGHT_SHA, data_report_sha256=DATA_SHA,
                  final_evaluation_samples=500, checkpoint_interval_updates=1000,
                  checkpoint_selection="fixed_final_update_no_target_selection",
                  exact_resume_verified=False, source="gta5",
                  validation_dataset="cityscapes_val", backbone="dinov2_vitl14_rein_patch16",
                  decoder="Mask2Former", physical_batch_size=1, num_workers=0,
                  train_crop=[512, 512], target_input=[1024, 512], target_gt=[2048, 1024],
                  slide_crop=[512, 512], slide_stride=[341, 341], cqe_enabled=False,
                  trainable_parameters=23569877, retained_checkpoint_count=2,
                  sampler_cursor=10204, sampler_epoch=6,
                  purpose="formal_source_only_rein_baseline_not_cqe_or_exact_paper_reproduction",
                  versions=EXPECTED_VERSIONS.copy(), first_20_loss_mean=2.,
                  last_20_loss_mean=2., elapsed_seconds=1., peak_reserved_gib=4.,
                  checkpoints=[dict(optimizer_updates=i, bytes=100,
                                    sha256=f"{i:064x}")
                               for i in range(1000, UPDATES + 1, 1000)],
                  validation=validation)
    result = audit(report, records)
    assert result["source40k_saved_audit_ok"]
    assert result["final_miou"] == 1. and result["complete_source_epochs"] == 6
    damaged = copy.deepcopy(validation)
    damaged["samples"][-1]["gt_shape"] = [512, 512]
    with pytest.raises(ValueError):
        audit_validation(damaged)
