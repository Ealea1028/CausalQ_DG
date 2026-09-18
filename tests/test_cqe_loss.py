import pytest
import torch

from causalq.losses import causal_query_effect_loss


def test_identical_effects_have_zero_loss() -> None:
    effect = torch.randn(2, 4, 3, 5)
    labels = torch.tensor(
        [
            [[0, 0, 1, 1, 255]] * 3,
            [[2, 2, 3, 3, 255]] * 3,
        ]
    )

    loss = causal_query_effect_loss(effect, effect, labels)

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_reference_stops_gradient_and_view_receives_gradient() -> None:
    view = torch.randn(1, 3, 2, 3, requires_grad=True)
    reference = torch.randn(1, 3, 2, 3, requires_grad=True)
    labels = torch.tensor([[[0, 0, 1], [0, 1, 1]]])

    causal_query_effect_loss(view, reference, labels).backward()

    assert view.grad is not None
    assert torch.isfinite(view.grad).all()
    assert reference.grad is None


def test_absent_classes_and_ignore_pixels_do_not_contribute() -> None:
    reference = torch.ones(1, 3, 1, 3)
    first = reference.clone()
    second = reference.clone()
    first[:, 2] = 100.0
    second[:, 2] = -100.0
    first[:, :, :, 2] = 50.0
    second[:, :, :, 2] = -50.0
    labels = torch.tensor([[[0, 0, 255]]])

    first_loss = causal_query_effect_loss(first, reference, labels)
    second_loss = causal_query_effect_loss(second, reference, labels)

    assert first_loss == pytest.approx(0.0, abs=1e-7)
    assert second_loss == pytest.approx(0.0, abs=1e-7)


def test_effect_normalization_removes_positive_magnitude_scaling() -> None:
    reference = torch.randn(1, 2, 3, 3)
    labels = torch.zeros(1, 3, 3, dtype=torch.long)

    loss = causal_query_effect_loss(reference * 7.0, reference, labels)

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_cqe_rejects_invalid_contract() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        causal_query_effect_loss(
            torch.randn(1, 2, 2, 2),
            torch.randn(1, 2, 3, 2),
            torch.zeros(1, 2, 2, dtype=torch.long),
        )
