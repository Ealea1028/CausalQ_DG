import pytest

from tools.report_classwise_iou import reduce_counts, build_report


def test_per_class_iou_counts_false_positives_and_false_negatives():
    matrix = [[0]*19 for _ in range(19)]
    matrix[0][0] = 8;matrix[0][1] = 2;matrix[1][0] = 3;matrix[1][1] = 7
    result = reduce_counts(matrix)
    assert result['class_iou'][:2] == pytest.approx([8/13, 7/12])
    assert result['miou'] == pytest.approx((8/13+7/12)/2)
    assert result['class_iou'][2] is None and result['valid_classes'] == 2
    assert result['gt_pixels'][:2] == [10, 10]


def test_absent_gt_with_false_positives_is_zero_iou_not_undefined():
    matrix = [[0]*19 for _ in range(19)];matrix[0][1] = 3
    result = reduce_counts(matrix)
    assert result['class_iou'][:2] == [0., 0.]
    assert result['miou'] == 0. and result['class_iou'][2] is None


def test_wrong_label_axis_or_ignore_column_rejected():
    with pytest.raises(ValueError):reduce_counts([[0]*20 for _ in range(19)])
    matrix = [[0]*19 for _ in range(19)];matrix[1][1] = -1
    with pytest.raises(ValueError):reduce_counts(matrix)


def test_complete_historical_models_have19_source_and_target_classes():
    r = build_report()
    assert len(r['models']) == 10 and r['class_order'][18]['name'] == 'bicycle'
    assert r['models']['CA']['target']['miou'] == pytest.approx(.6632195195146393)
    for model in r['models'].values():
        for domain in ('source', 'target'):
            assert model[domain]['valid_classes'] == 19
            assert len(model[domain]['class_iou']) == 19
