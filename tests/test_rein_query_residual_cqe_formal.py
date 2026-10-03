from pathlib import Path

import numpy as np
import pytest

from tools.audit_rein_query_residual_cqe_formal import (
    _miou,
    _optimizer_update_records,
    expected_augmentation_seed,
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


def test_formal_audit_summarizes_optimizer_boundaries_not_first_microbatches():
    records = [
        {
            "microbatch": microbatch,
            "optimizer_update": microbatch % 4 == 0,
            "loss": float(microbatch),
        }
        for microbatch in range(1, 13)
    ]
    selected = _optimizer_update_records(records)
    assert [record["microbatch"] for record in selected] == [4, 8, 12]
    assert [record["loss"] for record in selected] == [4.0, 8.0, 12.0]


def test_formal_augmentation_seed_is_controlled_by_training_seed():
    first = expected_augmentation_seed(20260931, 1)
    second = expected_augmentation_seed(20261001, 1)
    third = expected_augmentation_seed(20261002, 1)
    assert len({first, second, third}) == 3
    assert first == (20260931 * 1000003 + 1) % (2 ** 32)
    assert second == (20261001 * 1000003 + 1) % (2 ** 32)
    assert third == (20261002 * 1000003 + 1) % (2 ** 32)


def test_formal_launcher_uses_repository_module_entry_points():
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root / "scripts" / "run_rein_query_residual_cqe_formal.sh"
    ).read_text(encoding="utf-8")
    assert '"$PY" -u -m tools.train_rein_query_residual_cqe_formal' in launcher
    assert '"$PY" -u -m tools.audit_rein_query_residual_cqe_formal' in launcher
    assert '"$PY" -u tools/' not in launcher
    assert "import tools.train_rein_query_residual_cqe_formal" in launcher
    assert 'FORMAL_SEED=${FORMAL_SEED:-20260931}' in launcher
    assert '20261002) SEED_TAG=SEED2; ARTIFACT_TAG=seed2' in launcher
    assert '--seed "$FORMAL_SEED"' in launcher
    assert '--expected-seed "$FORMAL_SEED"' in launcher
