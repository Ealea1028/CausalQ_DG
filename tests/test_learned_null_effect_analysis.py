import pytest
import torch

from causalq.analysis import paired_effect_localization_values


def test_paired_effect_localization_compares_aligned_class_maps() -> None:
    labels = torch.tensor([[[0, 0, 1], [0, 1, 255]]])
    zero = torch.zeros(1, 2, 2, 3)
    learned_null = torch.zeros_like(zero)
    zero[0, 0] = torch.tensor([[2.0, 2.0, 1.0], [2.0, 1.0, 100.0]])
    zero[0, 1] = torch.tensor([[2.0, 2.0, 4.0], [2.0, 4.0, 100.0]])
    learned_null[0, 0] = torch.tensor(
        [[4.0, 4.0, 1.0], [4.0, 1.0, 100.0]]
    )
    learned_null[0, 1] = torch.tensor(
        [[1.0, 1.0, 4.0], [1.0, 4.0, 100.0]]
    )

    values = paired_effect_localization_values(zero, learned_null, labels)

    assert values["zero_absolute_ratio"].tolist() == pytest.approx([2.0, 2.0])
    assert values["learned_null_absolute_ratio"].tolist() == pytest.approx(
        [4.0, 4.0]
    )
    assert values["learned_null_has_higher_absolute_ratio"].tolist() == [True, True]
    assert values["learned_null_has_lower_outside_absolute_mean"].tolist() == [
        False,
        True,
    ]


def test_paired_effect_localization_requires_matching_shapes() -> None:
    labels = torch.zeros(1, 2, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="identical shapes"):
        paired_effect_localization_values(
            torch.zeros(1, 2, 2, 2),
            torch.zeros(1, 2, 1, 2),
            labels,
        )
