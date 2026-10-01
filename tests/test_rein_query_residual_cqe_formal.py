import numpy as np
import pytest

from tools.audit_rein_query_residual_cqe_formal import (
    _miou,
    paired_image_bootstrap,
)
from tools.rein_schedule_smoke import scheduled_lr
from tools.train_rein_query_residual_cqe_formal import (
    ACCUMULATION,
    UPDATES,
    objective,
)


def test_formal_poly_lr_matches_accepted_rein_schedule():
    assert scheduled_lr(0) == 1e-4
    assert scheduled_lr(1) < scheduled_lr(0)
    assert scheduled_lr(39998) > 0
    assert scheduled_lr(39999) == 0
    assert scheduled_lr(40000) == 0
    assert ACCUMULATION == 4 and UPDATES == 40000
    with pytest.raises(ValueError):
        scheduled_lr(-1)


def test_only_cqe_arm_adds_effect_loss():
    assert objective(2.0, 4.0, 0.25, enabled=False) == 3.0
    assert objective(2.0, 4.0, 0.25, enabled=True) == 3.25


def test_miou_ignores_classes_absent_from_union():
    matrix = np.zeros((19, 19), dtype=np.int64)
    matrix[0, 0] = 10
    matrix[1, 0] = 10
    assert _miou(matrix) == pytest.approx(0.25)


def test_paired_image_bootstrap_is_deterministic_and_paired():
    no_cqe, cqe = [], []
    for _ in range(500):
        control = np.zeros((19, 19), dtype=np.int64)
        candidate = np.zeros((19, 19), dtype=np.int64)
        control[0, 0] = 100
        candidate[0, 0] = 100
        no_cqe.append({"confusion_matrix": control.tolist()})
        cqe.append({"confusion_matrix": candidate.tolist()})
    first = paired_image_bootstrap(no_cqe, cqe, seed=9, replicates=40)
    second = paired_image_bootstrap(no_cqe, cqe, seed=9, replicates=40)
    assert first == second
    assert first["delta_miou_cqe_minus_control_ci95"] == [0.0, 0.0]


def test_paired_image_bootstrap_rejects_unpaired_coverage():
    with pytest.raises(ValueError, match="500 paired"):
        paired_image_bootstrap([], [])
