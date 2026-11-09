import numpy as np
import pytest

from tools.report_hpi_study import verify_metric


def test_final_reduction_rejects_incorrect_score():
    ids = [str(index) for index in range(500)]
    matrix = np.eye(19, dtype=np.int64)
    metric = {'ids':ids,'sample_count':500,'confusion_matrix':matrix.tolist(),
              'class_iou':[1.]*19,'miou':1.,'valid_pixels':19}
    assert verify_metric(metric,ids) == [1.]*19
    metric['miou'] = .999
    with pytest.raises(ValueError,match='score'):
        verify_metric(metric,ids)


def test_final_reduction_rejects_wrong_membership():
    metric = {'ids':['wrong'],'sample_count':500}
    with pytest.raises(ValueError,match='membership'):
        verify_metric(metric,['expected'])
