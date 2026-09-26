import pytest
import torch

from causalq.analysis import zero_effect_localization_values


def test_zero_effect_localization_separates_gt_and_non_gt_regions() -> None:
    effects = torch.zeros(1, 2, 2, 3)
    labels = torch.tensor([[[0, 0, 1], [0, 1, 255]]])
    effects[0, 0] = torch.tensor([[4.0, -2.0, 1.0], [3.0, 1.0, 100.0]])
    effects[0, 1] = torch.tensor([[2.0, 2.0, -6.0], [2.0, 4.0, 100.0]])

    values = zero_effect_localization_values(effects, labels)

    assert values["inside_absolute_mean"].tolist() == pytest.approx([3.0, 5.0])
    assert values["outside_absolute_mean"].tolist() == pytest.approx([1.0, 2.0])
    assert values["absolute_ratio"].tolist() == pytest.approx([3.0, 2.5])
    assert values["normalized_absolute_contrast"].tolist() == pytest.approx(
        [0.5, 3.0 / 7.0]
    )


def test_zero_effect_localization_preserves_signed_diagnostics() -> None:
    effects = torch.tensor([[[[-3.0, -1.0]], [[2.0, 4.0]]]])
    labels = torch.tensor([[[0, 1]]])

    values = zero_effect_localization_values(effects, labels)

    assert values["inside_signed_mean"].tolist() == pytest.approx([-3.0, 4.0])
    assert values["outside_signed_mean"].tolist() == pytest.approx([-1.0, 2.0])


def test_zero_effect_localization_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="BxCxHxW"):
        zero_effect_localization_values(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 3, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="no comparable"):
        zero_effect_localization_values(
            torch.zeros(1, 2, 1, 1),
            torch.zeros(1, 1, 1, dtype=torch.long),
        )
