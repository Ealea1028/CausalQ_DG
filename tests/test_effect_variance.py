import pytest
import torch

from causalq.analysis import (
    cross_style_effect_variance,
    cross_style_effect_variance_values,
)


def test_identical_style_effects_have_zero_variance() -> None:
    effect = torch.randn(2, 3, 4, 5)
    effects = effect[:, None].expand(-1, 3, -1, -1, -1)
    labels = torch.randint(0, 3, (2, 4, 5))

    assert cross_style_effect_variance(effects, labels).item() == pytest.approx(0.0)


def test_variance_ignores_void_pixels_and_absent_classes() -> None:
    effects = torch.zeros(1, 2, 3, 1, 3)
    effects[:, 0, 0, 0] = torch.tensor([1.0, 0.0, 100.0])
    effects[:, 1, 0, 0] = torch.tensor([0.0, 1.0, -100.0])
    effects[:, :, 2] = torch.randn_like(effects[:, :, 2]) * 1000
    labels = torch.tensor([[[0, 0, 255]]])

    values = cross_style_effect_variance_values(effects, labels)

    assert values.shape == (1,)
    assert values.item() == pytest.approx(0.5)


def test_positive_per_view_scaling_does_not_change_variance() -> None:
    effects = torch.randn(1, 3, 2, 3, 4)
    labels = torch.tensor([[[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]]])
    scales = torch.tensor([0.5, 2.0, 7.0]).view(1, 3, 1, 1, 1)

    first = cross_style_effect_variance(effects, labels)
    second = cross_style_effect_variance(effects * scales, labels)

    assert torch.allclose(first, second, atol=1e-6)


def test_effect_variance_rejects_invalid_inputs() -> None:
    labels = torch.zeros(1, 2, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="BxVxCxHxW"):
        cross_style_effect_variance(torch.zeros(1, 2, 2, 2), labels)
    with pytest.raises(ValueError, match="two style views"):
        cross_style_effect_variance(torch.zeros(1, 1, 2, 2, 2), labels)
    with pytest.raises(ValueError, match="no valid pixels"):
        cross_style_effect_variance(
            torch.zeros(1, 2, 2, 2, 2),
            torch.full_like(labels, 255),
        )
