import numpy as np
import pytest
import torch

from tools.check_rein_slide_eval import confusion, evaluate_target, restore_compact
from tools.rein_schedule_smoke import compact_state


def test_target_evaluation_rejects_unbounded_or_incomplete_coverage():
    with pytest.raises(ValueError, match='bounded'):
        evaluate_target(None, [None] * 500, None, 500)
    with pytest.raises(ValueError, match='bounded'):
        evaluate_target(None, [None] * 50, None, 50)


def test_confusion_gt_rows_prediction_columns_ignore_excluded():
    label = np.array([[0, 1, 255], [1, 2, 2]])
    pred = np.array([[0, 2, 3], [1, 2, 0]])
    result = confusion(pred, label)
    assert result.shape == (19, 19) and result.sum() == 5
    assert result[1, 2] == 1 and result[2, 0] == 1 and result[3].sum() == 0
    with pytest.raises(ValueError):
        confusion(pred[:, :2], label)
    with pytest.raises(ValueError):
        confusion(np.full_like(pred, 19), label)


def test_compact_restore_exact_coverage_and_frozen_weights_untouched():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    model[0].requires_grad_(False)
    saved = compact_state(model)
    frozen = model[0].weight.detach().clone()
    with torch.no_grad():
        model[1].weight.add_(1)
    restore_compact(model, saved)
    assert torch.equal(model[1].weight, saved['1.weight'])
    assert torch.equal(model[0].weight, frozen)
    saved.pop('1.running_mean')
    with pytest.raises(ValueError, match='coverage'):
        restore_compact(model, saved)
